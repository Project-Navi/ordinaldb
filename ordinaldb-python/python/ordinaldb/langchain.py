"""LangChain vector store adapter for OrdinalDB."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
import uuid

try:
    from langchain_core.documents import Document
    from langchain_core.vectorstores import VectorStore
except ImportError as exc:  # pragma: no cover - exercised when extra is absent.
    raise ImportError(
        "ordinaldb.langchain requires the LangChain extra. "
        "Install with `pip install 'ordinaldb[langchain]'`."
    ) from exc

from .adapters import AdapterRecord, AdapterStore, adapter_store_markers_exist


class OrdinalDBVectorStore(VectorStore):
    """LangChain `VectorStore` backed by an OrdinalDB adapter directory."""

    def __init__(
        self,
        *,
        embedding: Any | None = None,
        path: str | Path | None = None,
        dim: int | None = None,
        bits: int = 2,
        store: AdapterStore | None = None,
    ) -> None:
        """Open or create a LangChain vector store over an adapter directory.

        If ``path`` already contains an adapter store it is loaded (and must
        have been written by the langchain adapter); otherwise an empty
        in-memory store is created. Call ``save_local`` to persist — writes
        (``add_texts``/``add_documents``/``delete``) are memory-only until
        then.

        A ``path`` that exists but holds no valid store markers emits
        ``AdapterPathWarning`` before starting fresh, and a ``path`` that is
        a plain file raises immediately (see ``AdapterStore`` for the full
        path validation matrix). Use ``load_local`` to open existing data
        fail-closed.
        """
        self.embedding = embedding
        if store is not None:
            self._store = store
        elif adapter_store_markers_exist(path):
            self._store = AdapterStore.load(path, expected_adapter="langchain")
        else:
            self._store = AdapterStore(bits=bits, dim=dim, path=path, adapter_name="langchain")

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Any,
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> "OrdinalDBVectorStore":
        """Build a store from raw texts by embedding and adding them."""
        store = cls(embedding=embedding, **kwargs)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store

    @classmethod
    def load_local(
        cls,
        path: str | Path,
        embeddings: Any | None = None,
        **kwargs: Any,
    ) -> "OrdinalDBVectorStore":
        """Load a persisted langchain adapter directory from disk."""
        store = AdapterStore.load(path, expected_adapter="langchain")
        return cls(embedding=embeddings, store=store, **kwargs)

    @classmethod
    def load(
        cls,
        folder_path: str | Path,
        embedding: Any | None = None,
        embeddings: Any | None = None,
        **kwargs: Any,
    ) -> "OrdinalDBVectorStore":
        """Alias for ``load_local`` accepting either embedding kwarg name."""
        return cls.load_local(
            folder_path,
            embeddings=embedding if embedding is not None else embeddings,
            **kwargs,
        )

    def save_local(self, path: str | Path | None = None) -> None:
        """Persist the store to disk atomically.

        See ``AdapterStore.save`` for the compare-and-swap semantics: if a
        concurrent writer committed since this store was loaded, an
        ``AdapterStoreError`` about a stale snapshot is raised.
        """
        self._store.save(path, adapter_name="langchain")

    def persist(self, path: str | Path | None = None) -> None:
        """Alias for ``save_local``."""
        self.save_local(path)

    def dump(self, folder_path: str | Path) -> None:
        """Alias for ``save_local`` with a required target path."""
        self.save_local(folder_path)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Iterable[dict[str, Any]] | None = None,
        ids: Iterable[str] | None = None,
        **_: Any,
    ) -> list[str]:
        """Embed and upsert texts; returns the assigned string ids.

        Ids default to fresh UUID4 strings; existing ids are replaced.

        WARNING: writes are held in memory only — data is NOT durable
        until ``save_local()``/``persist()`` is called. The first unsaved
        write to a store constructed with a ``path`` emits
        ``UnsavedWritesWarning`` as a reminder.
        """
        result_ids, texts, metadatas, ids = _prepare_langchain_batch(
            texts,
            metadatas,
            ids,
        )
        if not texts:
            return result_ids
        vectors = self._embed_documents(texts)
        self._store.add(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
            upsert=True,
        )
        return result_ids

    def add_documents(
        self,
        documents: Sequence[Document],
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Upsert LangChain ``Document`` objects; returns their ids.

        WARNING: writes are held in memory only — data is NOT durable
        until ``save_local()``/``persist()`` is called (see ``add_texts``).
        """
        texts = [document.page_content for document in documents]
        metadatas = [dict(document.metadata or {}) for document in documents]
        resolved_ids = ids or [
            str(getattr(document, "id", None) or uuid.uuid4()) for document in documents
        ]
        return self.add_texts(texts, metadatas=metadatas, ids=resolved_ids, **kwargs)

    def delete(
        self,
        ids: list[str] | None = None,
        *,
        filter: Any | None = None,
        **_: Any,
    ) -> None:
        """Delete records by id or by filter, in memory.

        Exactly one of ``ids`` or ``filter`` may be given. Missing ids are
        ignored. ``filter`` accepts the same forms as ``similarity_search``:
        a mapping of metadata key/value pairs (exact-match AND semantics)
        or a callable taking a ``Document``. A filter that matches nothing
        deletes nothing (and, for mappings whose keys appear on no record,
        emits ``UnknownFilterKeyWarning``). With both arguments ``None``
        this is a no-op.

        Changes stay in memory until ``save_local()``/``persist()``.

        Raises:
            ValueError: If both ``ids`` and ``filter`` are given.
            TypeError: If ``filter`` is neither a mapping nor a callable.
        """
        if ids is not None and filter is not None:
            raise ValueError("provide either ids or filter, not both")
        if ids is not None:
            self._store.delete(ids)
            return None
        if filter is not None:
            matches = self._records_matching_filter(filter)
            if matches:
                self._store.delete([record.id for record in matches])
        return None

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Return stored documents for the given ids; missing ids are skipped."""
        return [_document_from_record(record) for record in self._store.get(ids)]

    def get_count(self) -> int:
        """Return the number of stored records.

        Matches the public count accessor exposed by the Agno adapter
        (``OrdinalDb.get_count``), so callers never need to reach into the
        private ``_store``.
        """
        return len(self._store)

    def __len__(self) -> int:
        """Return the number of stored records (same as ``get_count``)."""
        return len(self._store)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Any | None = None,
        **_: Any,
    ) -> list[Document]:
        """Embed ``query`` and return the ``k`` most similar documents."""
        return [
            _document_from_record(record)
            for record in self._search_by_vector(self._embed_query(query), k=k, filter=filter)
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Any | None = None,
        **_: Any,
    ) -> list[tuple[Document, float]]:
        """Embed ``query`` and return ``(document, score)`` pairs.

        Scores are raw OrdinalDB similarity scores; higher is more similar.
        """
        return self.similarity_search_with_score_by_vector(
            self._embed_query(query), k=k, filter=filter
        )

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: Any | None = None,
        **_: Any,
    ) -> list[Document]:
        """Return the ``k`` documents most similar to a query embedding."""
        return [
            _document_from_record(record)
            for record in self._search_by_vector(embedding, k=k, filter=filter)
        ]

    def similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: Any | None = None,
        **_: Any,
    ) -> list[tuple[Document, float]]:
        """Return ``(document, score)`` pairs for a query embedding."""
        return [
            (_document_from_record(record), float(record.score or 0.0))
            for record in self._search_by_vector(embedding, k=k, filter=filter)
        ]

    def similarity_search_with_relevance_scores(self, *args: Any, **kwargs: Any) -> Any:
        """Not supported: OrdinalDB does not normalize relevance scores."""
        raise NotImplementedError(
            "OrdinalDB returns raw OrdinalDB similarity scores, not normalized relevance scores"
        )

    def max_marginal_relevance_search(self, *args: Any, **kwargs: Any) -> Any:
        """Not supported: OrdinalDB does not implement MMR search."""
        raise NotImplementedError("OrdinalDB does not implement MMR search")

    def max_marginal_relevance_search_by_vector(self, *args: Any, **kwargs: Any) -> Any:
        """Not supported: OrdinalDB does not implement MMR search."""
        raise NotImplementedError("OrdinalDB does not implement MMR search")

    def _embed_documents(self, texts: list[str]) -> Any:
        if self.embedding is None:
            raise ValueError("embedding is required to add text documents")
        if not hasattr(self.embedding, "embed_documents"):
            raise TypeError("embedding must provide embed_documents(texts)")
        return self.embedding.embed_documents(texts)

    def _embed_query(self, query: str) -> Any:
        if self.embedding is None:
            raise ValueError("embedding is required for text queries")
        if not hasattr(self.embedding, "embed_query"):
            raise TypeError("embedding must provide embed_query(query)")
        return self.embedding.embed_query(query)

    def _search_by_vector(
        self,
        embedding: Any,
        *,
        k: int,
        filter: Any | None,
    ) -> list[AdapterRecord]:
        if filter is None or isinstance(filter, Mapping):
            return self._store.search_by_vector(embedding, k=k, filter=filter)
        allowed_u64_ids = [
            record.u64_id
            for record in self._records_matching_filter(filter)
            if record.u64_id is not None
        ]
        if not allowed_u64_ids:
            return []
        return self._store.search_by_vector(
            embedding,
            k=k,
            allowed_u64_ids=allowed_u64_ids,
        )

    def _records_matching_filter(self, filter: Any) -> list[AdapterRecord]:
        """Resolve a LangChain-style filter to the matching adapter records.

        Mappings use the store's exact-match AND semantics. Callables take
        ``Document`` objects, which requires scanning adapter records.
        """
        if isinstance(filter, Mapping):
            return self._store.filter_records(filter)
        if callable(filter):
            predicate = _compile_filter(filter)
            return [
                record
                for record in self._store.iter_records()
                if predicate(_document_from_record(record))
            ]
        raise TypeError(
            f"unsupported filter type {type(filter).__name__}: pass a "
            "metadata key/value dict or a callable that accepts a Document"
        )


def _document_from_record(record: AdapterRecord) -> Document:
    try:
        return Document(
            page_content=record.document,
            metadata=dict(record.metadata),
            id=record.id,
        )
    except TypeError:
        document = Document(page_content=record.document, metadata=dict(record.metadata))
        try:
            document.id = record.id
        except Exception:
            pass
        return document


def _prepare_langchain_batch(
    texts: Iterable[str],
    metadatas: Iterable[dict[str, Any]] | None,
    ids: Iterable[str] | None,
) -> tuple[list[str], list[str], list[dict[str, Any]], list[str]]:
    texts = list(texts)
    metadatas = list(metadatas) if metadatas is not None else [{} for _ in texts]
    ids = list(ids) if ids is not None else [str(uuid.uuid4()) for _ in texts]
    if len(metadatas) != len(texts) or len(ids) != len(texts):
        raise ValueError("texts, metadatas, and ids must all have the same length")

    duplicate_ids = _duplicate_values(ids)
    if duplicate_ids:
        raise ValueError(f"duplicate string IDs in batch: {duplicate_ids}")
    result_ids = list(ids)
    return result_ids, texts, [dict(metadata or {}) for metadata in metadatas], ids


def _duplicate_values(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _compile_filter(filter: Callable[[Document], bool]) -> Callable[[Document], bool]:
    return filter
