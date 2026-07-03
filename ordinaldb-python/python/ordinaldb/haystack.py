"""Haystack document store and embedding retriever for OrdinalDB."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Iterable
import warnings

try:
    from haystack import Document, component
    try:
        from haystack.document_stores.types import DuplicatePolicy
    except ImportError:  # pragma: no cover
        from haystack import DuplicatePolicy
    from haystack.utils.filters import document_matches_filter
    from haystack.errors import FilterError
    try:
        from haystack.document_stores.errors import DuplicateDocumentError
    except ImportError:  # pragma: no cover
        from haystack.document_stores.errors.errors import DuplicateDocumentError
except ImportError as exc:  # pragma: no cover - exercised when extra is absent.
    raise ImportError(
        "ordinaldb.haystack requires the Haystack extra. "
        "Install with `pip install 'ordinaldb[haystack]'`."
    ) from exc

from .adapters import (
    AdapterPathWarning,
    AdapterRecord,
    AdapterStore,
    UnknownFilterKeyWarning,
    adapter_store_markers_exist,
)


class OrdinalDocumentStore:
    """Haystack document store backed by an OrdinalDB adapter directory."""

    def __init__(
        self,
        *,
        path: str | Path | None = None,
        dim: int | None = None,
        bits: int = 2,
        store: AdapterStore | None = None,
    ) -> None:
        """Open or create a Haystack document store over an adapter directory.

        If ``path`` already contains an adapter store it is loaded (and must
        have been written by the haystack adapter); otherwise an empty
        in-memory store is created. Call ``save`` to persist.
        """
        if store is not None:
            self._store = store
        elif adapter_store_markers_exist(path):
            self._store = AdapterStore.load(path, expected_adapter="haystack")
        else:
            self._store = AdapterStore(bits=bits, dim=dim, path=path, adapter_name="haystack")

    def count_documents(self) -> int:
        """Return the number of stored documents."""
        return len(self._store)

    def write_documents(
        self,
        documents: Iterable[Document],
        policy: Any = None,
    ) -> int:
        """Write embedded documents honoring the Haystack duplicate policy.

        FAIL (default) raises on duplicate ids, SKIP drops duplicates,
        OVERWRITE upserts (last write per id wins within the batch). Every
        document must carry an embedding. Returns the number written.
        Changes stay in memory until ``save``.
        """
        policy_name = _duplicate_policy_name(policy)
        effective_policy_name = "FAIL" if policy_name == "NONE" else policy_name
        incoming = list(documents)
        ids = [str(getattr(document, "id", "")) for document in incoming]
        existing = set(self._store.ids())

        duplicate_ids = _duplicate_ids(ids, existing)
        if effective_policy_name == "FAIL" and duplicate_ids:
            raise DuplicateDocumentError(
                f"duplicate document IDs for policy {policy_name}: {duplicate_ids}"
            )
        if effective_policy_name == "SKIP":
            seen: set[str] = set()
            filtered = []
            for document in incoming:
                doc_id = str(getattr(document, "id", ""))
                if doc_id in existing or doc_id in seen:
                    continue
                filtered.append(document)
                seen.add(doc_id)
            incoming = filtered
        elif effective_policy_name == "OVERWRITE":
            incoming = _keep_last_documents_by_id(incoming)
        elif effective_policy_name not in {"FAIL", "OVERWRITE"}:
            raise ValueError(f"unsupported duplicate policy {policy!r}")

        if not incoming:
            return 0

        ids = [str(getattr(document, "id", "")) for document in incoming]
        contents = [str(getattr(document, "content", "") or "") for document in incoming]
        embeddings = []
        metadatas = []
        for document in incoming:
            embedding = getattr(document, "embedding", None)
            if embedding is None:
                raise ValueError("Haystack documents must include embeddings for OrdinalDB")
            embeddings.append(embedding)
            metadatas.append(dict(getattr(document, "meta", {}) or {}))
        self._store.add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas,
            upsert=(policy_name == "OVERWRITE"),
        )
        return len(incoming)

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """Return the documents matching a Haystack filter dict (all if None).

        A filter that matches nothing and references ``meta`` keys present
        on no stored document emits ``UnknownFilterKeyWarning`` naming them
        (usually a typo in the key name).
        """
        documents = list(self._iter_matching_documents(filters))
        if filters and not documents:
            self._warn_unknown_filter_meta_keys(filters)
        return documents

    def delete_documents(self, document_ids: list[str]) -> None:
        """Delete documents by id in memory; missing ids are ignored."""
        self._store.delete(document_ids)

    def search_by_embedding(
        self,
        query_embedding: Any,
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Return the ``top_k`` documents most similar to a query embedding.

        Haystack filters are applied first; scores are raw OrdinalDB
        similarities (higher is better). A filter that matches nothing and
        references ``meta`` keys present on no stored document emits
        ``UnknownFilterKeyWarning`` naming them (usually a typo).
        """
        allowed = None
        if filters is not None:
            allowed = [
                record.u64_id
                for record in self._iter_matching_records(filters)
                if record.u64_id is not None
            ]
            if not allowed:
                self._warn_unknown_filter_meta_keys(filters)
                return []
        records = self._store.search_by_vector(
            query_embedding,
            k=top_k,
            allowed_u64_ids=allowed,
        )
        return [_document_from_record(record) for record in records]

    def save(self, path: str | Path | None = None) -> None:
        """Persist the store to disk atomically.

        See ``AdapterStore.save`` for the compare-and-swap semantics: if a
        concurrent writer committed since this store was loaded, an
        ``AdapterStoreError`` about a stale snapshot is raised.
        """
        self._store.save(path, adapter_name="haystack")

    @classmethod
    def load(cls, path: str | Path) -> "OrdinalDocumentStore":
        """Load a persisted haystack adapter directory from disk."""
        return cls(store=AdapterStore.load(path, expected_adapter="haystack"))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a Haystack component dict.

        WARNING: only the store's *path* is serialized, never its contents.
        Call ``save()`` before ``to_dict()`` (or before ``Pipeline.to_dict``/
        YAML export): deserializing a dict whose path holds no saved store
        reconstructs a structurally real but EMPTY document store, and
        ``from_dict`` warns about it.

        The ``type`` field must be the fully qualified class name (not just
        the bare class name): Haystack's generic deserialization
        (``default_from_dict``) only auto-reconstructs a nested
        init-parameter object -- such as this store when it is embedded in
        ``OrdinalEmbeddingRetriever.to_dict()`` -- if the serialized ``type``
        contains a dot it can import. Without it, a Pipeline round-tripped
        through ``to_dict``/``from_dict`` (or YAML) would silently leave the
        raw dict in place of a real ``OrdinalDocumentStore`` instance.
        """
        path = str(self._store.path) if self._store.path is not None else None
        qualified_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return {"type": qualified_name, "init_parameters": {"path": path}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrdinalDocumentStore":
        """Recreate a store from a Haystack component dict.

        ``to_dict`` serializes only the store's path, so the data must
        already have been persisted there with ``save()``. When the path is
        missing, holds no valid store markers, or was never set, this emits
        ``AdapterPathWarning`` and reconstructs an empty store.
        """
        params = data.get("init_parameters", {})
        path = params.get("path")
        if not adapter_store_markers_exist(path):
            if path is None:
                message = (
                    "OrdinalDocumentStore.from_dict: the serialized store has "
                    "no path — it was never bound to a directory, so there is "
                    "no saved data to reconstruct from. Call save(path) "
                    "before to_dict(). Reconstructing an empty in-memory "
                    "store."
                )
            else:
                message = (
                    f"OrdinalDocumentStore.from_dict: store path {str(path)!r} "
                    "has no saved data — did you save() before to_dict()? "
                    "to_dict() serializes only the store's path, so the "
                    "reconstructed store starts empty."
                )
            warnings.warn(message, AdapterPathWarning, stacklevel=2)
        return cls(path=path)

    def _iter_matching_records(
        self,
        filters: dict[str, Any] | None,
    ) -> Iterable[AdapterRecord]:
        for record in self._store.iter_records():
            if _matches_haystack_filter(record, filters):
                yield record

    def _iter_matching_documents(
        self,
        filters: dict[str, Any] | None,
    ) -> Iterable[Document]:
        for record in self._store.iter_records():
            document = _document_from_record(record)
            if _matches_haystack_document(document, filters):
                yield document

    def _warn_unknown_filter_meta_keys(self, filters: dict[str, Any]) -> None:
        """Warn when a zero-hit filter names meta keys no document carries.

        Parity hook for ``UnknownFilterKeyWarning``: the Haystack filter
        dialect goes through ``document_matches_filter`` and bypasses the
        shared warning in ``AdapterStore``. Only called on the zero-result
        path, so the scan over in-memory metadata costs nothing on
        successful queries. Haystack's ``meta.`` field prefix is stripped
        before comparing against stored metadata keys.
        """
        if len(self._store) == 0:
            return
        referenced = _filter_meta_keys(filters)
        if not referenced:
            return
        present: set[str] = set()
        for record in self._store.iter_records():
            present.update(referenced.intersection(record.metadata))
            if present == referenced:
                return
        unknown = sorted(referenced - present)
        if not unknown:
            return
        unknown_names = ", ".join(repr(key) for key in unknown)
        warnings.warn(
            f"filter matched 0 documents: filter key(s) {unknown_names} do "
            "not appear in any stored document's meta (possible typo in the "
            "key name?)",
            UnknownFilterKeyWarning,
            stacklevel=3,
        )


@component
class OrdinalEmbeddingRetriever:
    """Haystack retriever that runs vector search against `OrdinalDocumentStore`.

    Decorated with ``@component`` (and ``run`` with
    ``@component.output_types``) so instances carry the
    ``__haystack_input__``/``__haystack_output__`` sockets Haystack's
    ``Pipeline.add_component``/``connect``/``run`` require -- the same
    contract ``haystack.components.retrievers.in_memory.InMemoryEmbeddingRetriever``
    follows. Note: ``@component`` rebuilds the class object at decoration
    time, so a zero-arg ``super()`` call inside ``run`` would bind to the
    pre-decoration class and break; this class has no base class to call
    ``super()`` on, which sidesteps that trap entirely.
    """

    def __init__(
        self,
        *,
        document_store: OrdinalDocumentStore,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> None:
        """Create a retriever bound to a document store with default top_k/filters."""
        self.document_store = document_store
        self.top_k = top_k
        self.filters = filters

    @component.output_types(documents=list[Document])
    def run(
        self,
        *,
        query_embedding: Any,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, list[Document]]:
        """Run vector retrieval; returns ``{"documents": [...]}``.

        Explicit arguments override the retriever's defaults.
        """
        documents = self.document_store.search_by_embedding(
            query_embedding,
            top_k=top_k or self.top_k,
            filters=filters if filters is not None else self.filters,
        )
        return {"documents": documents}


def _duplicate_policy_name(policy: Any) -> str:
    """Return the caller-facing policy name (e.g. "FAIL", "NONE", "SKIP").

    Note this does NOT normalize "NONE" to "FAIL": callers that need the
    behavioral (FAIL-equivalent) treatment of NONE must do that themselves,
    so that error messages can still name the policy the caller actually
    passed.
    """
    if policy is None:
        policy = getattr(DuplicatePolicy, "FAIL", "FAIL")
    return str(getattr(policy, "name", policy)).upper()


def _duplicate_ids(ids: list[str], existing: set[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for doc_id in ids:
        if doc_id in existing or doc_id in seen:
            if doc_id not in duplicates:
                duplicates.append(doc_id)
        seen.add(doc_id)
    return duplicates


def _keep_last_documents_by_id(documents: list[Document]) -> list[Document]:
    order: list[str] = []
    by_id: dict[str, Document] = {}
    for document in documents:
        doc_id = str(getattr(document, "id", ""))
        if doc_id not in by_id:
            order.append(doc_id)
        by_id[doc_id] = document
    return [by_id[doc_id] for doc_id in order]


def _document_from_record(record: AdapterRecord) -> Document:
    kwargs = {
        "id": record.id,
        "content": record.document,
        "meta": dict(record.metadata),
    }
    if record.score is not None:
        kwargs["score"] = float(record.score)
    try:
        return Document(**kwargs)
    except TypeError:
        kwargs.pop("score", None)
        document = Document(**kwargs)
        try:
            document.score = record.score
        except Exception:
            pass
        return document


_DOCUMENT_FIELD_NAMES = frozenset(field.name for field in dataclasses.fields(Document))


def _filter_meta_keys(filters: Any) -> set[str]:
    """Collect the top-level ``meta`` keys a Haystack filter dict references.

    Mirrors ``haystack.utils.filters._comparison_condition`` field
    resolution: ``meta.<key>[...]`` addresses ``document.meta[<key>]``, a
    bare field that is not a real ``Document`` attribute is a legacy
    shorthand for a meta key, and everything else (``id``, ``content``,
    dotted non-``meta`` paths, ...) addresses document attributes and is
    ignored here. Logical conditions are walked recursively.
    """
    keys: set[str] = set()
    if not isinstance(filters, dict):
        return keys
    conditions = filters.get("conditions")
    if isinstance(conditions, list):
        for condition in conditions:
            keys.update(_filter_meta_keys(condition))
    field = filters.get("field")
    if isinstance(field, str) and field:
        if field.startswith("meta."):
            key = field.split(".")[1]
            if key:
                keys.add(key)
        elif "." not in field and field not in _DOCUMENT_FIELD_NAMES:
            keys.add(field)
    return keys


def _matches_haystack_filter(record: AdapterRecord, filters: dict[str, Any] | None) -> bool:
    return _matches_haystack_document(_document_from_record(record), filters)


def _matches_haystack_document(document: Document, filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    try:
        return document_matches_filter(filters, document)
    except KeyError as exc:
        operator = exc.args[0] if exc.args else exc
        raise FilterError(f"Unknown filter operator {operator!r}") from exc
