"""Agno vector database adapter for OrdinalDB."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import uuid

try:
    import agno as _agno  # noqa: F401
    from agno.knowledge.document import Document
except ImportError as exc:  # pragma: no cover - exercised when extra is absent.
    raise ImportError(
        "ordinaldb.agno requires the Agno extra. "
        "Install with `pip install 'ordinaldb[agno]'`."
    ) from exc

from .adapters import (
    AdapterRecord,
    AdapterStore,
    AdapterStoreError,
    adapter_store_markers_exist,
)


_CONTENT_ID_KEY = "__ordinaldb_agno_content_id__"
_NAME_KEY = "__ordinaldb_agno_name__"


class OrdinalDb:
    """Agno-compatible vector database backed by OrdinalDB."""

    def __init__(
        self,
        *,
        path: str | Path | None = None,
        embedder: Any | None = None,
        dim: int | None = None,
        bits: int = 2,
        store: AdapterStore | None = None,
        auto_save: bool = False,
    ) -> None:
        """Open or create an Agno vector database over an adapter directory.

        If ``path`` already contains an adapter store it is loaded (and must
        have been written by the agno adapter); otherwise an empty in-memory
        store is created. With ``auto_save=True`` every mutation persists
        immediately; otherwise call ``create`` to save — mutations are
        memory-only until then, and the first unsaved write to a path-bound
        store emits ``UnsavedWritesWarning``.

        A ``path`` that exists but holds no valid store markers emits
        ``AdapterPathWarning`` before starting fresh, and a ``path`` that is
        a plain file raises immediately (see ``AdapterStore`` for the full
        path validation matrix). Use ``load`` to open existing data
        fail-closed.
        """
        self.embedder = embedder
        self.auto_save = bool(auto_save)
        if store is not None:
            self._store = store
        elif adapter_store_markers_exist(path):
            self._store = AdapterStore.load(path, expected_adapter="agno")
        else:
            self._store = AdapterStore(bits=bits, dim=dim, path=path, adapter_name="agno")
        if self.auto_save:
            # Auto-save persists right after each mutation, so the
            # unsaved-writes reminder would be noise.
            self._store.warn_unsaved_writes = False

    def create(self) -> None:
        """Persist the current state to the store's path, if one is set.

        See ``AdapterStore.save`` for the compare-and-swap semantics on
        concurrent writers.
        """
        if self._store.path is not None:
            self._store.save(self._store.path, adapter_name="agno")

    async def async_create(self) -> None:
        """Async wrapper for ``create`` (runs synchronously)."""
        self.create()

    def upsert(
        self,
        content_hash: str | Iterable[Any],
        documents: Iterable[Any] | None = None,
        *,
        ids: list[str] | None = None,
        embeddings: list[Any] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[str]:
        """Upsert documents for a content hash; returns the written ids.

        Existing records with the same content id (or matching ``filters``)
        that are not rewritten by this batch are deleted.
        """
        content_id, docs = _resolve_content_hash_and_documents(content_hash, documents)
        replaced_ids = self._ids_for_content_id(content_id) if content_id is not None else []
        if filters:
            replaced_ids.extend(self._ids_for_metadata(filters))
        written_ids = self._write_documents(
            docs,
            content_id=content_id,
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            filters=filters,
            upsert=True,
        )
        stale_ids = _unique_values(
            string_id for string_id in replaced_ids if string_id not in written_ids
        )
        if stale_ids:
            self._store.delete(stale_ids)
        self._save_if_auto()
        return written_ids

    def insert(
        self,
        content_hash: str | Iterable[Any],
        documents: Iterable[Any] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Insert documents for a content hash; returns the written ids.

        Unlike ``upsert``, colliding string ids raise.
        """
        content_id, docs = _resolve_content_hash_and_documents(content_hash, documents)
        replaced_ids = self._ids_for_metadata(kwargs.get("filters")) if kwargs.get("filters") else []
        written_ids = self._write_documents(docs, content_id=content_id, upsert=False, **kwargs)
        stale_ids = _unique_values(
            string_id for string_id in replaced_ids if string_id not in written_ids
        )
        if stale_ids:
            self._store.delete(stale_ids)
        self._save_if_auto()
        return written_ids

    def delete(self, ids: list[str] | str | None = None) -> bool:
        """Delete the given ids, or every record when ``ids`` is None."""
        if ids is None:
            changed = len(self._store) > 0
            self.drop()
            return changed
        return self.delete_by_id(ids)

    def delete_by_id(self, ids: list[str] | str) -> bool:
        """Delete records by string id; returns True if anything was removed."""
        if isinstance(ids, str):
            ids = [ids]
        changed = self._store.delete([str(value) for value in ids])
        self._save_if_auto()
        return changed

    def delete_by_content_id(self, content_id: str) -> bool:
        """Delete all records written under the given content id."""
        return self._delete_where(lambda record: record.metadata.get(_CONTENT_ID_KEY) == content_id)

    def delete_by_content_hash(self, content_hash: str) -> bool:
        """Alias for ``delete_by_content_id``."""
        return self.delete_by_content_id(content_hash)

    def delete_by_name(self, name: str) -> bool:
        """Delete all records whose Agno document name matches."""
        return self._delete_where(lambda record: record.metadata.get(_NAME_KEY) == name)

    def delete_by_metadata(self, metadata: dict[str, Any]) -> bool:
        """Delete records whose metadata contains all given key/value pairs."""
        return self._delete_where(
            lambda record: all(record.metadata.get(key) == value for key, value in metadata.items())
        )

    def exists(self) -> bool:
        """Return True if the store has persisted markers or in-memory records."""
        if self._store.path is not None and Path(self._store.path, "adapter.json").exists():
            return True
        return len(self._store) > 0

    async def async_exists(self) -> bool:
        """Async wrapper for ``exists`` (runs synchronously)."""
        return self.exists()

    def id_exists(self, id: str) -> bool:
        """Return True if a record with this string id exists."""
        return bool(self._store.get([str(id)]))

    def content_hash_exists(self, content_hash: str) -> bool:
        """Return True if any record was written under this content hash."""
        return any(
            record.metadata.get(_CONTENT_ID_KEY) == content_hash
            for record in self._store.iter_records()
        )

    def content_id_exists(self, content_id: str) -> bool:
        """Alias for ``content_hash_exists``."""
        return self.content_hash_exists(content_id)

    def name_exists(self, name: str) -> bool:
        """Return True if any record carries this Agno document name."""
        return any(record.metadata.get(_NAME_KEY) == name for record in self._store.iter_records())

    def drop(self) -> None:
        """Delete every record (persists immediately when ``auto_save`` is set)."""
        self._store.delete(self._store.ids())
        self._save_if_auto()

    async def async_drop(self) -> None:
        """Async wrapper for ``drop`` (runs synchronously)."""
        self.drop()

    def get_count(self) -> int:
        """Return the number of stored records.

        Matches the accessor exposed by Agno's first-party vector dbs
        (LanceDb, ChromaDb, PgVector, ...), so callers never need to reach
        into the private ``_store``.
        """
        return len(self._store)

    async def async_get_count(self) -> int:
        """Async wrapper for ``get_count`` (runs synchronously)."""
        return self.get_count()

    def __len__(self) -> int:
        """Return the number of stored records (same as ``get_count``)."""
        return len(self._store)

    def get_supported_search_types(self) -> list[str]:
        """Return the supported search types (vector only)."""
        return ["vector"]

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Embed ``query`` with the configured embedder and run a vector search."""
        if self.embedder is None:
            raise ValueError("embedder is required for text search")
        return self.search_by_vector(
            _embed_query(self.embedder, query),
            limit=limit,
            filters=filters,
        )

    def search_by_vector(
        self,
        vector: Any,
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Return the ``limit`` documents most similar to a query vector.

        ``filters`` are exact-match metadata constraints (AND semantics).
        """
        return [
            _record_to_document(record)
            for record in self._store.search_by_vector(vector, k=limit, filter=filters)
        ]

    async def async_upsert(
        self,
        content_hash: str | Iterable[Any],
        documents: Iterable[Any] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Async wrapper for ``upsert`` (runs synchronously)."""
        return self.upsert(content_hash, documents, **kwargs)

    async def async_insert(
        self,
        content_hash: str | Iterable[Any],
        documents: Iterable[Any] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Async wrapper for ``insert`` (runs synchronously)."""
        return self.insert(content_hash, documents, **kwargs)

    async def async_search(self, query: str, **kwargs: Any) -> list[Document]:
        """Async wrapper for ``search`` (runs synchronously)."""
        return self.search(query, **kwargs)

    async def async_search_by_vector(self, vector: Any, **kwargs: Any) -> list[Document]:
        """Async wrapper for ``search_by_vector`` (runs synchronously)."""
        return self.search_by_vector(vector, **kwargs)

    def save(self, path: str | Path | None = None) -> None:
        try:
            self._store.save(path, adapter_name="agno")
        except AdapterStoreError as exc:
            raise ValueError(str(exc)) from exc

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        embedder: Any | None = None,
        auto_save: bool = False,
    ) -> "OrdinalDb":
        return cls(
            embedder=embedder,
            store=AdapterStore.load(path, expected_adapter="agno"),
            auto_save=auto_save,
        )

    async def async_delete_by_id(self, ids: list[str] | str) -> bool:
        return self.delete_by_id(ids)

    async def async_delete_by_content_id(self, content_id: str) -> bool:
        return self.delete_by_content_id(content_id)

    async def async_delete_by_content_hash(self, content_hash: str) -> bool:
        return self.delete_by_content_hash(content_hash)

    async def async_delete_by_name(self, name: str) -> bool:
        return self.delete_by_name(name)

    async def async_delete_by_metadata(self, metadata: dict[str, Any]) -> bool:
        return self.delete_by_metadata(metadata)

    async def async_id_exists(self, id: str) -> bool:
        return self.id_exists(id)

    async def async_content_hash_exists(self, content_hash: str) -> bool:
        return self.content_hash_exists(content_hash)

    async def async_content_id_exists(self, content_id: str) -> bool:
        return self.content_id_exists(content_id)

    async def async_name_exists(self, name: str) -> bool:
        return self.name_exists(name)

    def _write_documents(
        self,
        documents: Iterable[Any],
        *,
        content_id: str | None,
        ids: list[str] | None = None,
        embeddings: list[Any] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        filters: dict[str, Any] | None = None,
        upsert: bool,
    ) -> list[str]:
        docs = list(documents)
        if ids is not None and len(ids) != len(docs):
            raise ValueError("ids length must match documents length")
        resolved_ids = (
            [str(value) for value in ids]
            if ids is not None
            else [
                _document_id(document, content_id=content_id, offset=offset)
                for offset, document in enumerate(docs)
            ]
        )
        contents = [_document_content(document) for document in docs]
        if embeddings is not None and len(embeddings) != len(docs):
            raise ValueError("embeddings length must match documents length")
        vectors = (
            embeddings
            if embeddings is not None
            else [_doc_attr(document, "embedding") for document in docs]
        )
        if any(vector is None for vector in vectors):
            raise ValueError("embeddings are required for Agno OrdinalDb writes")
        if metadatas is not None and len(metadatas) != len(docs):
            raise ValueError("metadatas length must match documents length")
        metadata_inputs = metadatas if metadatas is not None else [None for _ in docs]
        metadata = [
            _document_metadata(document, content_id=content_id, explicit=explicit)
            for document, explicit in zip(docs, metadata_inputs)
        ]
        return self._store.add(
            ids=resolved_ids,
            embeddings=vectors,
            documents=contents,
            metadatas=metadata,
            upsert=upsert,
        )

    def _delete_where(self, predicate: Any) -> bool:
        ids = [record.id for record in self._store.iter_records() if predicate(record)]
        if not ids:
            return False
        changed = self._store.delete(ids)
        self._save_if_auto()
        return changed

    def _ids_for_content_id(self, content_id: str | None) -> list[str]:
        if content_id is None:
            return []
        return [
            record.id
            for record in self._store.iter_records()
            if record.metadata.get(_CONTENT_ID_KEY) == content_id
        ]

    def _ids_for_metadata(self, metadata: dict[str, Any] | None) -> list[str]:
        if not metadata:
            return []
        return [
            record.id
            for record in self._store.iter_records()
            if all(record.metadata.get(key) == value for key, value in metadata.items())
        ]

    def _save_if_auto(self) -> None:
        if self.auto_save and self._store.path is not None:
            self.save(self._store.path)


def _resolve_content_hash_and_documents(
    content_hash: str | Iterable[Any],
    documents: Iterable[Any] | None,
) -> tuple[str | None, Iterable[Any]]:
    if documents is None:
        if isinstance(content_hash, str):
            raise ValueError("documents must be provided when content_hash is a string")
        return None, content_hash  # type: ignore[return-value]
    return str(content_hash), documents


def _unique_values(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            unique.append(value)
            seen.add(value)
    return unique


def _doc_attr(document: Any, name: str) -> Any:
    if isinstance(document, dict):
        return document.get(name)
    return getattr(document, name, None)


def _document_id(document: Any, *, content_id: str | None, offset: int) -> str:
    explicit = _doc_attr(document, "id")
    if explicit:
        return str(explicit)
    document_content_id = content_id or _doc_attr(document, "content_id")
    if document_content_id is not None:
        return f"{document_content_id}:{offset}"
    return str(uuid.uuid4())


def _document_content(document: Any) -> str:
    return str(_doc_attr(document, "content") or _doc_attr(document, "text") or document)


def _document_metadata(
    document: Any,
    *,
    content_id: str | None,
    explicit: dict[str, Any] | None,
) -> dict[str, Any]:
    if explicit is not None:
        metadata = dict(explicit)
    else:
        metadata = dict(
            _doc_attr(document, "metadata")
            or _doc_attr(document, "meta")
            or _doc_attr(document, "meta_data")
            or {}
        )
    document_content_id = content_id or _doc_attr(document, "content_id")
    if document_content_id is not None:
        metadata[_CONTENT_ID_KEY] = str(document_content_id)
    name = _doc_attr(document, "name")
    if name is not None:
        metadata[_NAME_KEY] = str(name)
    return metadata


def _embed_query(embedder: Any, query: str) -> Any:
    for method_name in ("get_embedding", "embed_query", "get_query_embedding"):
        method = getattr(embedder, method_name, None)
        if method is not None:
            return method(query)
    raise TypeError("embedder must provide get_embedding, embed_query, or get_query_embedding")


def _record_to_document(record: AdapterRecord) -> Document:
    metadata = dict(record.metadata)
    content_id = metadata.pop(_CONTENT_ID_KEY, None)
    name = metadata.pop(_NAME_KEY, None)
    return Document(
        content=record.document,
        id=record.id,
        name=name,
        meta_data=metadata,
        reranking_score=record.score,
        content_id=content_id,
    )
