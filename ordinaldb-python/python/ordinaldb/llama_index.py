"""LlamaIndex vector store adapter for OrdinalDB."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

try:
    from llama_index.core.schema import TextNode
    from llama_index.core.vector_stores.types import (
        BasePydanticVectorStore,
        VectorStoreQuery,
        VectorStoreQueryMode,
        VectorStoreQueryResult,
    )
    try:
        from llama_index.core.bridge.pydantic import PrivateAttr
    except ImportError:  # pragma: no cover
        from pydantic import PrivateAttr
except ImportError as exc:  # pragma: no cover - exercised when extra is absent.
    raise ImportError(
        "ordinaldb.llama_index requires the LlamaIndex extra. "
        "Install with `pip install 'ordinaldb[llama-index]'`."
    ) from exc

from .adapters import (
    AdapterRecord,
    AdapterStore,
    AdapterStoreError,
    adapter_store_markers_exist,
)


_NODE_PAYLOAD_KEY = "__ordinaldb_llama_index_node__"
_DEFAULT_PERSIST_FNAME = "default__vector_store.json"


class OrdinalDBVectorStore(BasePydanticVectorStore):
    """LlamaIndex vector store backed by an OrdinalDB adapter directory."""

    stores_text: bool = True
    is_embedding_query: bool = True
    bits: int = 2
    path: str | None = None

    _store: AdapterStore = PrivateAttr()

    def __init__(
        self,
        *,
        path: str | Path | None = None,
        dim: int | None = None,
        bits: int = 2,
        store: AdapterStore | None = None,
        **kwargs: Any,
    ) -> None:
        """Open or create a LlamaIndex vector store over an adapter directory.

        If ``path`` already contains an adapter store it is loaded (and must
        have been written by the llama-index adapter); otherwise an empty
        in-memory store is created. Call ``persist`` to save changes —
        nothing is written to disk before that.

        A ``path`` that exists but holds no valid store markers emits
        ``AdapterPathWarning`` before starting fresh, and a ``path`` that is
        a plain file raises immediately (see ``AdapterStore`` for the full
        path validation matrix). Use ``from_persist_dir``/``from_persist_path``
        to open existing data fail-closed.
        """
        super().__init__(bits=bits, path=str(path) if path is not None else None, **kwargs)
        if store is not None:
            self._store = store
        elif adapter_store_markers_exist(path):
            self._store = AdapterStore.load(path, expected_adapter="llama-index")
        else:
            self._store = AdapterStore(bits=bits, dim=dim, path=path, adapter_name="llama-index")

    @classmethod
    def class_name(cls) -> str:
        """Return the LlamaIndex-facing class name."""
        return "OrdinalDBVectorStore"

    @property
    def client(self) -> AdapterStore:
        """The underlying ``AdapterStore``."""
        return self._store

    def add(self, nodes: Sequence[Any], **_: Any) -> list[str]:
        """Add embedded nodes; returns their node ids.

        Every node must carry an embedding. Duplicate node ids raise.

        WARNING: changes stay in memory until ``persist``; the first
        unsaved write to a path-bound store emits ``UnsavedWritesWarning``.

        Following llama-index's ``node_to_metadata_dict`` convention, each
        node's ``ref_doc_id`` is stamped into the stored metadata under
        ``ref_doc_id``/``doc_id``/``document_id`` so that ``delete`` and
        metadata filters can find the nodes of a source document. Nodes
        without a source relationship keep caller-supplied values for
        those keys, defaulting to ``"None"``.
        """
        ids: list[str] = []
        documents: list[str] = []
        embeddings: list[Any] = []
        metadatas: list[dict[str, Any]] = []
        for node in nodes:
            embedding = getattr(node, "embedding", None)
            if embedding is None:
                raise ValueError("LlamaIndex nodes must have embeddings for OrdinalDB")
            node_id = _node_id(node)
            metadata = dict(getattr(node, "metadata", {}) or {})
            if _NODE_PAYLOAD_KEY in metadata:
                raise ValueError(f"LlamaIndex metadata key {_NODE_PAYLOAD_KEY!r} is reserved")
            metadata[_NODE_PAYLOAD_KEY] = _node_payload(node)
            _stamp_ref_doc_metadata(node, metadata)
            ids.append(node_id)
            documents.append(_node_text(node))
            embeddings.append(embedding)
            metadatas.append(metadata)
        return self._store.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            upsert=False,
        )

    async def async_add(self, nodes: Sequence[Any], **kwargs: Any) -> list[str]:
        """Async wrapper for ``add`` (runs synchronously)."""
        return self.add(nodes, **kwargs)

    def delete(self, ref_doc_id: str, **_: Any) -> None:
        """Delete a document and every node whose ``ref_doc_id`` matches.

        Missing ids are ignored. Changes stay in memory until ``persist``.
        """
        target = str(ref_doc_id)
        to_delete = [target]
        for record in self._store.iter_records():
            record_ref_doc_id = record.metadata.get("ref_doc_id")
            if record_ref_doc_id is not None and str(record_ref_doc_id) == target:
                to_delete.append(record.id)
        self._store.delete(to_delete, missing_ok=True)

    def delete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: Any | None = None,
        **_: Any,
    ) -> None:
        """Delete nodes by id and/or metadata filters.

        With both arguments None this is a no-op. Missing ids are ignored.
        """
        if node_ids is None and filters is None:
            return
        records = self._matching_records(node_ids=node_ids, filters=filters)
        self._store.delete([record.id for record in records], missing_ok=True)

    async def adelete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Async wrapper for ``delete_nodes`` (runs synchronously)."""
        self.delete_nodes(node_ids=node_ids, filters=filters, **kwargs)

    def get_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: Any | None = None,
    ) -> list[Any]:
        """Return stored nodes matching the given ids and/or filters."""
        return [
            _node_from_record(record)
            for record in self._matching_records(node_ids=node_ids, filters=filters)
        ]

    async def aget_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: Any | None = None,
    ) -> list[Any]:
        """Async wrapper for ``get_nodes`` (runs synchronously)."""
        return self.get_nodes(node_ids=node_ids, filters=filters)

    def clear(self) -> None:
        """Delete every stored node (in memory; ``persist`` to save)."""
        self._store.delete(self._store.ids(), missing_ok=True)

    async def aclear(self) -> None:
        """Async wrapper for ``clear`` (runs synchronously)."""
        self.clear()

    def query(self, query: VectorStoreQuery, **_: Any) -> VectorStoreQueryResult:
        """Run a top-k embedding query and return nodes with similarities.

        Only the default query mode is supported and ``query_embedding`` is
        required. Scores are raw OrdinalDB similarities (higher is better).
        """
        _require_default_query_mode(query)
        if query.query_embedding is None:
            raise ValueError("OrdinalDB requires query embeddings")
        filters = _llama_filters_to_callable(getattr(query, "filters", None))
        top_k = int(getattr(query, "similarity_top_k", 4) or 4)
        records = self._store.search_by_vector(
            query.query_embedding,
            k=top_k,
            filter=filters,
        )
        nodes = [_node_from_record(record) for record in records]
        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=[float(record.score or 0.0) for record in records],
            ids=[record.id for record in records],
        )

    async def aquery(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Async wrapper for ``query`` (runs synchronously)."""
        return self.query(query, **kwargs)

    def persist(
        self,
        persist_path: str | Path | None = None,
        fs: Any | None = None,
        **_: Any,
    ) -> None:
        """Persist the store to disk atomically.

        ``StorageContext.persist(persist_dir=X)`` hands vector stores a
        file-shaped ``persist_path`` (``X/default__vector_store.json``).
        OrdinalDB stores are directories, so a ``persist_path`` whose
        basename ends in ``.json`` (and is not itself an existing store
        directory) is mapped to its PARENT directory: the store is saved
        at ``X`` and can be reopened with ``OrdinalDBVectorStore(path=X)``
        or ``from_persist_dir(X)``. Any other ``persist_path`` is used as
        the store root directory directly.

        See ``AdapterStore.save`` for the compare-and-swap semantics: if a
        concurrent writer committed since this store was loaded, an
        ``AdapterStoreError`` about a stale snapshot is raised. Remote
        filesystems (``fs``) are not supported.
        """
        if fs is not None:
            raise NotImplementedError("OrdinalDB MVP persistence requires a local path")
        target = (
            _store_root_for_persist_path(persist_path)
            if persist_path is not None
            else self.path
        )
        self._store.save(target, adapter_name="llama-index")

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str | Path,
        fs: Any | None = None,
        **kwargs: Any,
    ) -> "OrdinalDBVectorStore":
        """Load a persisted store from a ``StorageContext`` persist dir.

        Looks for the adapter store at ``persist_dir`` itself, then falls
        back to the legacy nested layout at
        ``persist_dir/default__vector_store.json``. Fails closed: raises
        ``AdapterStoreError`` when no store is found instead of silently
        returning an empty store.
        """
        if fs is not None:
            raise NotImplementedError("OrdinalDB MVP persistence requires a local path")
        root = Path(persist_dir)
        if not adapter_store_markers_exist(root):
            nested = root / _DEFAULT_PERSIST_FNAME
            if adapter_store_markers_exist(nested):
                root = nested
            else:
                raise AdapterStoreError(
                    f"no OrdinalDB adapter store found at {root} (also checked "
                    f"the legacy nested location {nested}); pass the directory "
                    "given to StorageContext.persist(persist_dir=...)"
                )
        store = AdapterStore.load(root, expected_adapter="llama-index")
        return cls(store=store, path=str(root), **kwargs)

    @classmethod
    def from_persist_path(
        cls,
        persist_path: str | Path,
        fs: Any | None = None,
        **kwargs: Any,
    ) -> "OrdinalDBVectorStore":
        """Load a persisted store from a llama-index ``persist_path``.

        Accepts either a store root directory or the file-shaped path that
        ``StorageContext.persist`` uses (``X/default__vector_store.json``),
        which resolves to the store saved at ``X``. Fails closed: raises
        ``AdapterStoreError`` when no store is found.
        """
        if fs is not None:
            raise NotImplementedError("OrdinalDB MVP persistence requires a local path")
        path = Path(persist_path)
        if adapter_store_markers_exist(path):
            root = path
        elif path.suffix == ".json" and adapter_store_markers_exist(path.parent):
            root = path.parent
        else:
            raise AdapterStoreError(
                f"no OrdinalDB adapter store found at {path} (or, for a "
                "file-shaped persist path, in its parent directory)"
            )
        store = AdapterStore.load(root, expected_adapter="llama-index")
        return cls(store=store, path=str(root), **kwargs)

    def _matching_records(
        self,
        *,
        node_ids: list[str] | None,
        filters: Any | None,
    ) -> list[AdapterRecord]:
        records = (
            self._store.get([str(node_id) for node_id in node_ids])
            if node_ids is not None
            else list(self._store.iter_records())
        )
        filter_callable = _llama_filters_to_callable(filters)
        if filter_callable is None:
            return records
        return [
            record
            for record in records
            if filter_callable(record.document, record.metadata)
        ]


def _node_id(node: Any) -> str:
    node_id = getattr(node, "node_id", None) or getattr(node, "id_", None)
    if node_id is None:
        raise ValueError("LlamaIndex nodes must have node IDs for OrdinalDB")
    return str(node_id)


def _stamp_ref_doc_metadata(node: Any, metadata: dict[str, Any]) -> None:
    """Stamp llama-index's ref-doc convention keys into stored metadata.

    Mirrors ``llama_index.core.vector_stores.utils.node_to_metadata_dict``,
    which stamps the node's ``ref_doc_id`` under ``ref_doc_id`` (Weaviate),
    ``doc_id`` (Pinecone/Qdrant/Redis), and ``document_id`` (Chroma) so
    ``delete(ref_doc_id)`` and metadata filters work. When the node has no
    source relationship, caller-supplied values for those keys are kept
    (missing ones default to the conventional ``"None"``).
    """
    ref_doc_id = getattr(node, "ref_doc_id", None)
    if ref_doc_id is not None:
        stamp = str(ref_doc_id)
        metadata["ref_doc_id"] = stamp
        metadata["doc_id"] = stamp
        metadata["document_id"] = stamp
        return
    existing = metadata.get("ref_doc_id")
    stamp = str(existing) if existing is not None else "None"
    metadata.setdefault("ref_doc_id", stamp)
    metadata.setdefault("doc_id", stamp)
    metadata.setdefault("document_id", stamp)


def _store_root_for_persist_path(persist_path: str | Path) -> Path:
    """Map a llama-index ``persist_path`` to an adapter store root directory.

    File-shaped paths (basename ending in ``.json``) — the shape
    ``StorageContext.persist`` produces — map to their parent directory,
    unless the path is itself an existing store directory (the legacy
    nested layout), which keeps saving in place.
    """
    path = Path(persist_path)
    if path.suffix == ".json" and not adapter_store_markers_exist(path):
        return path.parent
    return path


def _node_payload(node: Any) -> dict[str, Any]:
    if hasattr(node, "model_dump"):
        try:
            payload = node.model_dump(mode="json")
        except TypeError:
            payload = node.model_dump()
    elif hasattr(node, "dict"):
        payload = node.dict()
    else:
        payload = {
            "id_": _node_id(node),
            "text": _node_text(node),
            "metadata": dict(getattr(node, "metadata", {}) or {}),
            "embedding": getattr(node, "embedding", None),
            "class_name": "TextNode",
        }
    if not isinstance(payload, dict):
        raise ValueError("LlamaIndex node payload must be a JSON object")
    return _json_compatible(payload)


def _node_from_record(record: AdapterRecord) -> Any:
    payload = record.metadata.get(_NODE_PAYLOAD_KEY)
    if isinstance(payload, dict):
        class_name = payload.get("class_name")
        if class_name in {None, "TextNode"}:
            return TextNode.from_dict(payload)
        raise NotImplementedError(f"unsupported LlamaIndex node class {class_name!r}")
    metadata = {
        key: value
        for key, value in record.metadata.items()
        if key != _NODE_PAYLOAD_KEY
    }
    return TextNode(
        id_=record.id,
        text=record.document,
        metadata=metadata,
        embedding=None,
    )


def _node_text(node: Any) -> str:
    if hasattr(node, "get_content"):
        return str(node.get_content())
    return str(getattr(node, "text", ""))


def _require_default_query_mode(query: VectorStoreQuery) -> None:
    mode = getattr(query, "mode", None)
    default = getattr(VectorStoreQueryMode, "DEFAULT", None)
    if mode is None or mode == default:
        return
    raise NotImplementedError(f"OrdinalDB supports only DEFAULT vector query mode, got {mode!r}")


def _llama_filters_to_callable(filters: Any | None) -> Callable[[str, dict[str, Any]], bool] | None:
    if filters is None:
        return None

    def _matches(_: str, metadata: dict[str, Any]) -> bool:
        return _matches_llama_filters(metadata, filters)

    return _matches


def _llama_filters_to_dict(filters: Any | None) -> dict[str, Any] | None:
    if filters is None:
        return None
    output: dict[str, Any] = {}
    for item in getattr(filters, "filters", []) or []:
        if hasattr(item, "filters"):
            raise NotImplementedError("nested LlamaIndex filters require callable translation")
        operator = _enum_value(getattr(item, "operator", "=="))
        if operator not in {"==", "eq"}:
            raise NotImplementedError("OrdinalDB LlamaIndex adapter supports exact-match filters")
        output[str(item.key)] = item.value
    condition = _enum_value(getattr(filters, "condition", "and"))
    if condition != "and":
        raise NotImplementedError("OrdinalDB LlamaIndex adapter supports AND exact-match filters")
    return output


def _matches_llama_filters(metadata: dict[str, Any], filters: Any) -> bool:
    items = list(getattr(filters, "filters", []) or [])
    condition = _enum_value(getattr(filters, "condition", "and"))
    if condition == "and":
        return all(_matches_llama_filter_item(metadata, item) for item in items)
    if condition == "or":
        return any(_matches_llama_filter_item(metadata, item) for item in items)
    if condition == "not":
        if len(items) != 1:
            raise ValueError("LlamaIndex NOT filters require exactly one condition")
        return not _matches_llama_filter_item(metadata, items[0])
    raise NotImplementedError(f"unsupported LlamaIndex filter condition {condition!r}")


def _matches_llama_filter_item(metadata: dict[str, Any], item: Any) -> bool:
    if hasattr(item, "filters"):
        return _matches_llama_filters(metadata, item)
    key = getattr(item, "key", None)
    if key is None:
        raise ValueError("LlamaIndex metadata filters require a key")
    actual = _nested_value(metadata, str(key))
    expected = getattr(item, "value", None)
    operator = _enum_value(getattr(item, "operator", "=="))
    if operator in {"==", "eq"}:
        return actual == expected
    if operator in {"!=", "ne"}:
        return actual != expected
    if operator == ">":
        return _compare_numeric(actual, expected, ">")
    if operator == ">=":
        return _compare_numeric(actual, expected, ">=")
    if operator == "<":
        return _compare_numeric(actual, expected, "<")
    if operator == "<=":
        return _compare_numeric(actual, expected, "<=")
    if operator == "in":
        return _contains(expected, actual)
    if operator in {"nin", "not in"}:
        return not _contains(expected, actual)
    if operator == "any":
        return _has_any(actual, expected)
    if operator == "all":
        return _has_all(actual, expected)
    raise NotImplementedError(f"unsupported LlamaIndex filter operator {operator!r}")


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value)).lower()


def _nested_value(metadata: dict[str, Any], path: str) -> Any:
    value: Any = metadata
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _compare_numeric(value: Any, expected: Any, operator: str) -> bool:
    if isinstance(value, bool) or isinstance(expected, bool):
        return False
    if not isinstance(value, Real) or not isinstance(expected, Real):
        return False
    if operator == ">":
        return value > expected
    if operator == ">=":
        return value >= expected
    if operator == "<":
        return value < expected
    return value <= expected


def _contains(container: Any, value: Any) -> bool:
    if isinstance(container, (str, bytes)) or not isinstance(container, Iterable):
        return False
    return value in container


def _has_any(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (str, bytes)) or not isinstance(actual, Iterable):
        return False
    if isinstance(expected, (str, bytes)) or not isinstance(expected, Iterable):
        return expected in actual
    return any(value in actual for value in expected)


def _has_all(actual: Any, expected: Any) -> bool:
    if isinstance(actual, (str, bytes)) or not isinstance(actual, Iterable):
        return False
    if isinstance(expected, (str, bytes)) or not isinstance(expected, Iterable):
        return expected in actual
    return all(value in actual for value in expected)


def _json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_compatible(value.tolist())
    if hasattr(value, "item"):
        return _json_compatible(value.item())
    return str(value)
