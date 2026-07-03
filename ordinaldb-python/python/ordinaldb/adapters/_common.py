"""Common persistence and search helpers for framework adapters.

The Rust `.odb` bundle remains a vector-only `OrdinalIndex` bundle here.
Framework text, metadata, string IDs, and stable numeric handles live in JSON
sidecars owned by the adapter directory.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
import warnings

import numpy as np

from ordinaldb._ordinaldb import _AdapterStateStore
from ordinaldb import OrdinalIndex


ADAPTER_SCHEMA_VERSION = "ordinaldb.adapter.v1"
ID_MAP_SCHEMA_VERSION = "ordinaldb.adapter.id_map.v1"
DOCUMENTS_SCHEMA_VERSION = "ordinaldb.adapter.documents.v1"
METADATA_SCHEMA_VERSION = "ordinaldb.adapter.metadata.v1"

ADAPTER_FILE = "adapter.json"
ID_MAP_FILE = "id_map.json"
DOCUMENTS_FILE = "documents.json"
METADATA_FILE = "metadata.json"
INDEX_DIR = "index.odb"
VECTORS_DIR = "vectors"
INITIAL_GENERATION_ID = 1
DEFAULT_INDEX_PATH = f"{VECTORS_DIR}/g{INITIAL_GENERATION_ID:012d}.odb"
ADAPTER_STORE_FILE = "adapter.redb"
WRITE_LOCK_FILE = ".ordinaldb.write.lock"

MAX_INPUT_MAGNITUDE = 1e16
MAX_ADAPTER_JSON_BYTES = 1024 * 1024
MAX_SIDECAR_JSON_BYTES = 64 * 1024 * 1024
_PUBLISH_TEST_HOOK: Callable[[str, Path], None] | None = None
_JSON_SCALAR_TYPES = (str, int, float, bool, type(None))

# redb's native lock-contention message ("Database already open. Cannot
# acquire lock.") — the signature of a concurrent adapter.redb reader
# (``ordinaldb adapter gc``, ``verify``/``inspect``/``stats``, or another
# loader) racing this process.
_REDB_LOCK_CONTENTION_MARKER = "Database already open"

# Frames from these top-level packages are treated as library internals when
# attributing warnings: ordinaldb itself, plus the framework wrappers that
# forward user calls into the adapters (e.g. LangChain's
# ``VectorStore.from_documents`` calling our ``from_texts``). Warnings should
# point at the user's call site, not at the glue in between.
_WARNING_INTERNAL_PACKAGES = frozenset(
    {
        "ordinaldb",
        "langchain",
        "langchain_core",
        "langchain_community",
        "llama_index",
        "haystack",
        "agno",
    }
)


def _external_warning_stacklevel() -> int:
    """Return the ``warnings.warn`` stacklevel for the caller of this helper
    that attributes the warning to the first frame outside the ordinaldb
    package (and outside the framework wrapper frames listed in
    ``_WARNING_INTERNAL_PACKAGES``).

    Must be called from the same function that calls ``warnings.warn``.
    Hand-rolled frame walk because ``skip_file_prefixes`` only exists on
    Python 3.12+ and this package supports 3.9.
    """
    level = 1
    frame = sys._getframe(1)  # the function that is about to warn
    while frame is not None:
        module = frame.f_globals.get("__name__", "")
        top_level = module.split(".", 1)[0]
        if top_level not in _WARNING_INTERNAL_PACKAGES:
            break
        if frame.f_back is None:
            break  # everything is internal; attribute to the outermost frame
        frame = frame.f_back
        level += 1
    return level


def _is_redb_lock_contention(error: BaseException) -> bool:
    return _REDB_LOCK_CONTENTION_MARKER in str(error)


class AdapterStoreError(ValueError):
    """Raised when adapter sidecars are malformed or internally inconsistent."""


class AdapterPathWarning(UserWarning):
    """Warns when a store is constructed against a suspicious path.

    Emitted by ``AdapterStore`` (and every framework adapter built on it)
    when the caller-supplied path exists but contains no valid store
    markers — e.g. a typo'd path, a directory of unrelated files, crash
    debris from an interrupted save, or a nested store one level below.
    """


class UnsavedWritesWarning(UserWarning):
    """Warns on the first unsaved write of each epoch to a path-bound store.

    Adapter mutations only touch memory; nothing reaches disk until the
    store is saved (LangChain ``save_local()``/``persist()``, LlamaIndex
    ``persist()``, Agno ``create()``/``save()``, ``AdapterStore.save()``).
    Warned once per unsaved-batch epoch: a successful save re-arms the
    warning, so the next unsaved write after a save warns again.
    """


class UnknownFilterKeyWarning(UserWarning):
    """Warns when a zero-hit filter references keys absent from every record.

    A metadata filter that matches nothing *and* names at least one key
    that no stored record carries is usually a typo (``doctype`` vs
    ``doc_type``); this warning names the unknown key(s).
    """


def adapter_store_markers_exist(path: str | os.PathLike[str] | None) -> bool:
    """Return True if ``path`` looks like an existing adapter store directory.

    Recognizes both the current ``adapter.redb`` layout and the legacy
    ``adapter.json`` sidecar layout. Framework adapters use this to decide
    between ``AdapterStore.load`` and creating a fresh store.
    """
    if path is None:
        return False
    root = Path(path)
    if (root / ADAPTER_STORE_FILE).is_file():
        return True
    return _valid_legacy_adapter_marker(root / ADAPTER_FILE)


def _valid_legacy_adapter_marker(path: Path) -> bool:
    try:
        adapter = _read_json(path)
        _require_exact_keys(
            adapter,
            {
                "schema_version",
                "adapter",
                "bits",
                "dim",
                "empty_lazy",
                "index_path",
                "sidecars",
            },
            ADAPTER_FILE,
        )
        if adapter["schema_version"] != ADAPTER_SCHEMA_VERSION:
            return False
        _require_string(adapter["adapter"], "adapter")
        _require_bits(adapter["bits"])
        if adapter["dim"] is not None:
            _require_positive_int(adapter["dim"], "dim")
        _require_bool(adapter["empty_lazy"], "empty_lazy")
        _validate_index_path(_require_string(adapter["index_path"], "index_path"))
        sidecars = _require_mapping(adapter["sidecars"], "sidecars")
        _require_exact_keys(sidecars, {ID_MAP_FILE, DOCUMENTS_FILE, METADATA_FILE}, "sidecars")
        for name in (ID_MAP_FILE, DOCUMENTS_FILE, METADATA_FILE):
            descriptor = _require_mapping(sidecars[name], f"sidecar descriptor for {name}")
            _require_exact_keys(
                descriptor,
                {"sha256", "file_size_bytes"},
                f"sidecar descriptor for {name}",
            )
            digest = _require_string(descriptor["sha256"], f"{name} sha256")
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                return False
            _require_non_negative_int(descriptor["file_size_bytes"], f"{name} file_size_bytes")
    except AdapterStoreError:
        return False
    return True


_STORE_ARTIFACT_NAMES = frozenset(
    {
        ADAPTER_STORE_FILE,
        ADAPTER_FILE,
        ID_MAP_FILE,
        DOCUMENTS_FILE,
        METADATA_FILE,
        INDEX_DIR,
        VECTORS_DIR,
        WRITE_LOCK_FILE,
    }
)


def _looks_like_store_debris(name: str) -> bool:
    if name in _STORE_ARTIFACT_NAMES:
        return True
    return name.startswith(".") and (".tmp-" in name or ".bak-" in name)


def _check_fresh_store_path(root: Path, adapter_name: str) -> None:
    """Validate a caller-supplied path before starting a fresh store there.

    Decision matrix (the fail-closed counterpart is ``AdapterStore.load``,
    which raises for every suspicious case):

    * nonexistent path or empty directory — legitimate fresh store, silent.
    * plain file / non-directory — raises ``AdapterStoreError`` immediately
      (not later at persist time).
    * directory with content but no valid store markers — emits
      ``AdapterPathWarning``; mentions crash debris (orphaned temp/backup
      files, stray store artifacts without a marker) and nested stores.
    * directory that already holds a valid store — emits
      ``AdapterPathWarning``: the fresh store shadows it, and saving over
      it will fail.
    """
    if not _path_exists_no_follow(root):
        return
    if not root.is_dir():
        raise AdapterStoreError(
            f"{adapter_name} adapter path {root} exists but is not a directory; "
            "adapter stores are directories, refusing to shadow the existing file"
        )
    try:
        entries = sorted(root.iterdir(), key=lambda entry: entry.name)
    except OSError as exc:
        raise AdapterStoreError(f"cannot inspect adapter path {root}: {exc}") from exc
    if not entries:
        return
    if adapter_store_markers_exist(root):
        warnings.warn(
            f"{adapter_name} adapter path {root} already contains an OrdinalDB "
            "adapter store, but this constructor starts a fresh, empty "
            "in-memory store and does NOT load it (use AdapterStore.load or "
            "the adapter's load classmethod to open it); saving this fresh "
            "store to the same path will fail rather than overwrite the "
            "existing data.",
            AdapterPathWarning,
            stacklevel=4,
        )
        return

    debris = sorted(entry.name for entry in entries if _looks_like_store_debris(entry.name))
    nested = [
        entry.name
        for entry in entries
        if entry.is_dir() and adapter_store_markers_exist(entry)
    ]
    names = [entry.name + ("/" if entry.is_dir() else "") for entry in entries]
    shown = ", ".join(names[:5])
    if len(names) > 5:
        shown += f", ... (+{len(names) - 5} more)"
    message = (
        f"{adapter_name} adapter path {root} already exists but contains no "
        f"valid OrdinalDB store markers ({ADAPTER_STORE_FILE} or legacy "
        f"{ADAPTER_FILE}); starting a fresh, empty in-memory store. "
        f"Existing entries: {shown}."
    )
    if debris:
        message += (
            " Some entries look like OrdinalDB store debris from an "
            f"interrupted save or crash: {', '.join(debris)}."
        )
    if nested:
        message += (
            f" A nested adapter store was found at {root / nested[0]}; did "
            "you mean to open that directory instead (for LlamaIndex, use "
            "OrdinalDBVectorStore.from_persist_dir)?"
        )
    message += (
        " If this path is a typo, fix it before saving; nothing on disk is "
        "modified until save."
    )
    warnings.warn(message, AdapterPathWarning, stacklevel=4)


@dataclass(frozen=True)
class AdapterRecord:
    """One stored document with its string id, metadata, and search score.

    Attributes:
        id: Caller-provided string identifier.
        document: Stored document text.
        metadata: JSON-scalar metadata mapping (a defensive copy).
        score: Similarity score; only set on records returned by
            ``search_by_vector`` (higher is more similar).
        u64_id: Internal stable numeric handle backing the string id.
    """

    id: str
    document: str
    metadata: dict[str, Any]
    score: float | None = None
    u64_id: int | None = None


@dataclass(frozen=True)
class _AdapterCommitToken:
    store_uuid: str | None
    active_generation_id: int | None
    active_generation_manifest_sha256: str | None
    commit_sequence: int | None
    index_path: str
    active_ids: int


class AdapterStore:
    """Adapter-owned text, metadata, and string-ID layer over `OrdinalIndex`.

    An ``AdapterStore`` is an in-memory working copy of an on-disk adapter
    directory. Mutations (``add``/``delete``) only touch memory; nothing is
    persisted until ``save`` is called.

    Concurrency model (compare-and-swap revisions):
        Every committed snapshot carries a commit token (store UUID, active
        generation id, manifest sha256, commit sequence). ``load`` records
        the token of the snapshot it read; ``save`` refuses to publish unless
        the target directory's current token still matches that base token,
        raising ``AdapterStoreError('stale adapter snapshot: ...')`` when
        another writer committed in between. The winning writer publishes
        atomically under an exclusive writer lock; losers must re-``load``,
        re-apply their changes, and ``save`` again. There is no merging.

    Path validation matrix (fresh construction with ``path=``):
        Constructing a fresh store bound to a path applies these rules
        (``load`` is the fail-closed counterpart and raises in every
        suspicious case):

        * path is None — pure in-memory store, no checks.
        * path does not exist — silently starts a fresh store (created on
          ``save``). To open an existing store and fail if it is missing,
          use ``load``.
        * path is an empty directory — silently starts a fresh store.
        * path is a plain file (or any non-directory) — raises
          ``AdapterStoreError`` immediately.
        * path is a directory with content but no valid store markers
          (``adapter.redb`` / legacy ``adapter.json``) — emits
          ``AdapterPathWarning`` (mentioning crash debris and nested
          stores when detected) and starts a fresh store.
        * path is a directory with valid store markers — emits
          ``AdapterPathWarning``: the existing store is NOT loaded by the
          plain constructor (use ``load``), and ``save`` will refuse to
          overwrite it.

    Unsaved-write warning:
        The first mutation (``add``/``delete``) of a path-bound store
        emits ``UnsavedWritesWarning`` once per unsaved-batch epoch,
        because changes stay in memory until ``save``. A successful save
        re-arms the warning: the next unsaved mutating write warns again
        (never twice within one epoch). Set ``warn_unsaved_writes = False``
        to opt out (done automatically by adapters that auto-save).
    """

    def __init__(
        self,
        *,
        bits: int = 2,
        dim: int | None = None,
        index: OrdinalIndex | None = None,
        string_to_u64: Mapping[str, int] | None = None,
        u64_to_slot: Mapping[int, int] | None = None,
        documents: Mapping[str, str] | None = None,
        metadata: Mapping[str, Mapping[str, Any]] | None = None,
        next_u64_id: int = 1,
        path: str | os.PathLike[str] | None = None,
        adapter_name: str = "common",
        base_generation_path: str | None = None,
        base_commit_token: _AdapterCommitToken | None = None,
        loaded_layout: str | None = None,
    ) -> None:
        bits = _require_bits(bits)
        if dim is not None:
            dim = _require_positive_int(dim, "dim")
            _validate_dim_compatible_with_bits(
                dim,
                bits,
                context=f"{adapter_name} adapter",
            )
        self.adapter_name = adapter_name
        self.path = Path(path) if path is not None else None
        self.warn_unsaved_writes = True
        self._unsaved_writes_warned = False
        if self.path is not None and loaded_layout is None:
            _check_fresh_store_path(self.path, adapter_name)
        self._index = index if index is not None else _new_index(dim=dim, bits=bits)
        self._string_to_u64 = {
            _require_string_id(key): _require_u64(value, "u64 id")
            for key, value in (string_to_u64 or {}).items()
        }
        self._u64_to_slot = {
            _require_u64(key, "u64 id"): _require_non_negative_int(value, "slot")
            for key, value in (u64_to_slot or {}).items()
        }
        self._documents = {
            _require_string_id(key): _require_document(value)
            for key, value in (documents or {}).items()
        }
        self._metadata = {
            _require_string_id(key): _require_metadata(value)
            for key, value in (metadata or {}).items()
        }
        self._next_u64_id = _require_u64(next_u64_id, "next_u64_id")
        self._base_generation_path = base_generation_path
        self._base_commit_token = base_commit_token
        self._loaded_layout = loaded_layout
        self._slot_to_u64 = _slot_map_from_u64_to_slot(self._u64_to_slot, len(self._index))
        self._u64_to_string = {
            u64_id: string_id for string_id, u64_id in self._string_to_u64.items()
        }
        self._validate_consistency()

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        *,
        expected_adapter: str | None = None,
    ) -> "AdapterStore":
        """Load and verify an adapter store directory.

        Supports the current ``adapter.redb`` layout and read-only migration
        from the legacy ``adapter.json`` sidecar layout. All payloads are
        integrity-checked before use.

        The returned store remembers the loaded snapshot's commit token as
        its CAS base: a later ``save`` back to the same directory succeeds
        only if no other writer has committed since this ``load``.

        Args:
            path: Adapter store directory.
            expected_adapter: If given, fail unless the store was written by
                this adapter name (e.g. ``"langchain"``).

        Returns:
            A new ``AdapterStore`` bound to ``path``.

        Raises:
            AdapterStoreError: If the directory is missing, a symlink,
                corrupt, fails verification, or belongs to a different
                adapter.
        """
        root = Path(path)
        if root.is_symlink():
            raise AdapterStoreError(f"adapter path must not be a symlink: {root}")
        if not root.is_dir():
            raise AdapterStoreError(f"adapter path {root} is not a directory")

        if _adapter_state_store_is_current(root):
            loaded_layout = "redb"
            adapter, id_map_payload, documents_payload, metadata_payload = _read_adapter_state(
                root,
                expected_adapter,
            )
        else:
            loaded_layout = "legacy-root-index"
            adapter = _read_json(root / ADAPTER_FILE)
            _require_exact_keys(
                adapter,
                {
                    "schema_version",
                    "adapter",
                    "bits",
                    "dim",
                    "empty_lazy",
                    "index_path",
                    "sidecars",
                },
                ADAPTER_FILE,
            )
            fallback_index_path = _require_string(adapter["index_path"], "index_path")
            _validate_index_path(fallback_index_path)
            if _requires_redb_state(root, fallback_index_path):
                raise AdapterStoreError(
                    f"{ADAPTER_STORE_FILE} is required for generation-layout adapter stores"
                )
            sidecars = adapter["sidecars"]
            _require_exact_keys(sidecars, {ID_MAP_FILE, DOCUMENTS_FILE, METADATA_FILE}, "sidecars")
            for name, expected in sidecars.items():
                _verify_sidecar(root / name, expected, name)

            id_map_payload = _read_json(root / ID_MAP_FILE)
            documents_payload = _read_json(root / DOCUMENTS_FILE)
            metadata_payload = _read_json(root / METADATA_FILE)

        if adapter["schema_version"] != ADAPTER_SCHEMA_VERSION:
            raise AdapterStoreError(
                f"unsupported adapter schema {adapter['schema_version']!r}"
            )
        adapter_name = _require_string(adapter["adapter"], "adapter")
        if expected_adapter is not None and adapter_name != expected_adapter:
            raise AdapterStoreError(
                f"adapter directory was written by {adapter_name!r}, not {expected_adapter!r}"
            )
        bits = _require_bits(adapter["bits"])
        dim = adapter["dim"]
        if dim is not None:
            dim = _require_positive_int(dim, "dim")
        empty_lazy = _require_bool(adapter["empty_lazy"], "empty_lazy")
        index_path = _require_string(adapter["index_path"], "index_path")
        _validate_index_path(index_path)
        string_to_u64, u64_to_slot, next_u64_id = _parse_id_map(id_map_payload)
        documents = _parse_documents(documents_payload)
        metadata = _parse_metadata(metadata_payload)
        base_commit_token = _current_adapter_commit(root)

        if empty_lazy:
            if dim is not None:
                raise AdapterStoreError("empty_lazy adapter sidecar must have dim=null")
            if string_to_u64 or u64_to_slot or documents or metadata:
                raise AdapterStoreError("empty_lazy adapter sidecar must not contain records")
            _reject_empty_lazy_vector_artifacts(root, index_path)
            index = OrdinalIndex(bits=bits)
        else:
            if dim is None:
                raise AdapterStoreError("non-empty adapter sidecar must have dim")
            generation_path = _validated_generation_path(root, index_path)
            index = OrdinalIndex.load(generation_path)
            if index.bits() != bits:
                raise AdapterStoreError("adapter bits do not match index bits")
            if dim is not None and index.dim() != dim:
                raise AdapterStoreError("adapter dim does not match index dim")

        return cls(
            bits=bits,
            dim=dim,
            index=index,
            string_to_u64=string_to_u64,
            u64_to_slot=u64_to_slot,
            documents=documents,
            metadata=metadata,
            next_u64_id=next_u64_id,
            path=root,
            adapter_name=adapter_name,
            base_generation_path=index_path,
            base_commit_token=base_commit_token,
            loaded_layout=loaded_layout,
        )

    def save(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        adapter_name: str | None = None,
    ) -> None:
        """Atomically publish the in-memory state to an adapter directory.

        Acquires the directory's exclusive writer lock, writes a new index
        generation, and commits the snapshot with compare-and-swap
        semantics: if the target already contains a store, its active
        commit token must equal the token this store loaded from (see
        class docstring). On success, this store's base token advances to
        the newly committed revision, so subsequent ``save`` calls keep
        working.

        Concurrent readers (``ordinaldb adapter gc``, diagnostics) racing
        this save can block the post-commit re-verification of
        ``adapter.redb`` even though the commit already published durably;
        when that outcome is provable, the failure is degraded to a
        ``UserWarning`` and the save reports success — do not re-add the
        batch (see ``_recover_contended_commit``).

        Args:
            path: Target directory; defaults to the directory this store
                was loaded from.
            adapter_name: Optionally rebrand the snapshot's adapter name.

        Raises:
            AdapterStoreError: If no target path is known, the writer lock
                cannot be acquired, the base revision is stale (concurrent
                writer won), a concurrent ``adapter.redb`` reader blocked
                the commit (the message says whether retrying is safe), or
                a legacy-layout store is saved onto itself (legacy stores
                must migrate to a new directory).
        """
        target = Path(path) if path is not None else self.path
        if target is None:
            raise AdapterStoreError("save requires a target path")
        if target.is_symlink():
            raise AdapterStoreError(f"adapter path must not be a symlink: {target}")
        if target.exists() and not target.is_dir():
            raise AdapterStoreError(f"cannot replace non-directory path {target}")
        if (
            self._loaded_layout == "legacy-root-index"
            and self.path is not None
            and Path(self.path).absolute() == target.absolute()
        ):
            raise AdapterStoreError(
                "legacy adapter stores must be migrated by saving to a different target"
            )

        self._validate_consistency()
        _ensure_directory_without_symlink(target)
        try:
            writer_lock = _AdapterStateStore.acquire_writer_lock(target)
        except ValueError as exc:
            raise AdapterStoreError(f"adapter writer lock failed: {exc}") from exc
        with writer_lock:
            target_has_redb = _adapter_state_store_is_current(target)
            self._require_fresh_base_generation(target)
            try:
                sidecar_payloads = self._sidecar_payloads()
                sidecar_digests = {
                    name: _payload_digest(payload)
                    for name, payload in sidecar_payloads.items()
                }
                empty_lazy = self._is_empty_lazy()
                index_path = DEFAULT_INDEX_PATH if empty_lazy else _next_index_path(target)
                if not empty_lazy:
                    _write_generation(self._index, target, index_path)
                _publish_hook("before_adapter_state_publish", target)

                adapter_payload = {
                    "schema_version": ADAPTER_SCHEMA_VERSION,
                    "adapter": adapter_name or self.adapter_name,
                    "bits": self.bits,
                    "dim": self.dim_opt,
                    "empty_lazy": empty_lazy,
                    "index_path": index_path,
                    "sidecars": sidecar_digests,
                }
                try:
                    committed_manifest = _write_adapter_state(
                        target,
                        adapter_payload,
                        sidecar_payloads,
                        expected_revision=(
                            self._base_commit_token if target_has_redb else None
                        ),
                    )
                except AdapterStoreError as exc:
                    committed_manifest = self._recover_contended_commit(
                        target, index_path, exc
                    )
                committed_token = _commit_token_from_manifest(committed_manifest)
                self.path = target
                self._base_generation_path = index_path
                self._base_commit_token = committed_token
                self._loaded_layout = "redb"
                # Re-arm the unsaved-writes latch: everything written so far
                # is durable now, so the NEXT unsaved mutating write starts a
                # new unsaved-batch epoch and warns again (still never twice
                # within one epoch).
                self._unsaved_writes_warned = False
                if adapter_name is not None:
                    self.adapter_name = adapter_name
                try:
                    _publish_hook("after_adapter_state_publish", target)
                    _write_compatibility_exports(target, adapter_payload, sidecar_payloads)
                except Exception:
                    # adapter.redb is authoritative. Once it is durably published,
                    # compatibility-export refresh failures are diagnostics, not a
                    # failed commit.
                    pass
            except Exception:
                raise

    @property
    def bits(self) -> int:
        """Quantization bit width of the backing index (1, 2, or 4)."""
        return int(self._index.bits())

    @property
    def dim_opt(self) -> int | None:
        """Vector dimensionality, or None while the store is empty-lazy."""
        return self._index.dim_opt()

    @property
    def dim(self) -> int:
        """Vector dimensionality; only valid once the dim is locked."""
        return int(self._index.dim())

    def __len__(self) -> int:
        """Number of stored records."""
        return len(self._string_to_u64)

    def ids(self) -> list[str]:
        """Return all stored string ids in insertion order."""
        return list(self._string_to_u64)

    def add(
        self,
        *,
        ids: Iterable[str],
        embeddings: Any,
        documents: Iterable[str],
        metadatas: Iterable[Mapping[str, Any] | None] | None = None,
        upsert: bool = False,
    ) -> list[str]:
        """Add (or upsert) a batch of records in memory.

        The whole batch is validated up front and applied atomically: on
        any error nothing is added. Changes are not persisted until
        ``save``.

        WARNING: writes are held in memory only. Nothing is durable until
        ``save`` is called; the first unsaved write to a path-bound store
        emits ``UnsavedWritesWarning``.

        Args:
            ids: Unique, non-empty string ids for the batch.
            embeddings: 2D float array-like of shape ``(len(ids), dim)``;
                values must be finite with ``|value| < 1e16``.
            documents: Document text per id.
            metadatas: Optional JSON-scalar metadata mapping per id
                (None entries become empty dicts).
            upsert: If True, existing ids are replaced. If False, an id
                collision raises.

        Returns:
            The list of string ids that were added.

        Raises:
            AdapterStoreError: On length mismatches, malformed ids,
                documents, metadata, or embeddings, or id collisions when
                ``upsert`` is False.
        """
        ids = [_require_string_id(value) for value in ids]
        if len(ids) == 0:
            return []
        documents = [_require_document(value) for value in documents]
        if metadatas is None:
            metadatas = [{} for _ in ids]
        metadata = [_require_metadata(value or {}) for value in metadatas]
        vectors = self._preflight_batch(ids, embeddings, documents, metadata, upsert=upsert)

        replacement_u64s = {
            string_id: self._string_to_u64[string_id]
            for string_id in ids
            if string_id in self._string_to_u64
        }
        new_u64s = [self._allocate_u64() for _ in ids]
        base_slot = len(self._slot_to_u64)

        self._index.add(vectors)
        for offset, u64_id in enumerate(new_u64s):
            slot = base_slot + offset
            self._slot_to_u64.append(u64_id)
            self._u64_to_slot[u64_id] = slot

        for string_id, u64_id, document, meta in zip(ids, new_u64s, documents, metadata):
            self._string_to_u64[string_id] = u64_id
            self._u64_to_string[u64_id] = string_id
            self._documents[string_id] = document
            self._metadata[string_id] = meta

        for old_u64 in replacement_u64s.values():
            self._remove_u64(old_u64)

        self._validate_consistency()
        self._note_unsaved_write()
        return ids

    def delete(self, ids: Iterable[str], *, missing_ok: bool = True) -> bool:
        """Delete records by string id in memory.

        Changes are not persisted until ``save``.

        Args:
            ids: A single string id or an iterable of ids.
            missing_ok: If False, raise when any id is absent; if True,
                absent ids are ignored.

        Returns:
            True if at least one record was removed.

        Raises:
            AdapterStoreError: If ``missing_ok`` is False and any id is
                not present (nothing is deleted in that case).
        """
        if isinstance(ids, str):
            ids = [ids]
        ids = [_require_string_id(value) for value in ids]
        missing = [string_id for string_id in ids if string_id not in self._string_to_u64]
        if missing and not missing_ok:
            raise AdapterStoreError(f"ids not present: {missing}")
        changed = False
        for string_id in ids:
            u64_id = self._string_to_u64.pop(string_id, None)
            if u64_id is None:
                continue
            self._documents.pop(string_id, None)
            self._metadata.pop(string_id, None)
            self._remove_u64(u64_id)
            changed = True
        self._validate_consistency()
        if changed:
            self._note_unsaved_write()
        return changed

    def get(self, ids: Iterable[str] | None = None) -> list[AdapterRecord]:
        """Fetch records by string id.

        Args:
            ids: A single id, an iterable of ids, or None for all records.
                Ids that are not present are silently skipped, so the
                result may be shorter than the request.

        Returns:
            Matching ``AdapterRecord`` objects (``score`` is None; stored
            embeddings are not returned).
        """
        if ids is None:
            ids = self._string_to_u64.keys()
        elif isinstance(ids, str):
            ids = [ids]
        records: list[AdapterRecord] = []
        for string_id in ids:
            string_id = _require_string_id(string_id)
            u64_id = self._string_to_u64.get(string_id)
            if u64_id is None:
                continue
            records.append(
                AdapterRecord(
                    id=string_id,
                    document=self._documents[string_id],
                    metadata=dict(self._metadata[string_id]),
                    u64_id=u64_id,
                )
            )
        return records

    def iter_records(self) -> Iterable[AdapterRecord]:
        """Yield every stored record lazily, in insertion order."""
        for string_id, u64_id in self._string_to_u64.items():
            yield AdapterRecord(
                id=string_id,
                document=self._documents[string_id],
                metadata=dict(self._metadata[string_id]),
                u64_id=u64_id,
            )

    def search_by_vector(
        self,
        query_embedding: Any,
        *,
        k: int = 4,
        filter: Mapping[str, Any] | Callable[[dict[str, Any]], bool] | None = None,
        allowed_u64_ids: Sequence[int] | None = None,
    ) -> list[AdapterRecord]:
        """Return the top-``k`` records most similar to a query embedding.

        Args:
            query_embedding: 1D float array-like of length ``dim`` (a
                single query).
            k: Maximum number of results; capped at the number of
                candidate records.
            filter: Either a metadata mapping matched with exact-equality
                AND semantics, or a callable predicate taking
                ``(metadata)`` or ``(document, metadata)`` and returning
                bool. Mutually exclusive with ``allowed_u64_ids``.
            allowed_u64_ids: Restrict the search to these internal u64
                handles (as found on ``AdapterRecord.u64_id``).

        Returns:
            Up to ``k`` ``AdapterRecord`` objects, best first, each with
            ``score`` set (higher is more similar).

        Raises:
            AdapterStoreError: If ``k`` is negative, both ``filter`` and
                ``allowed_u64_ids`` are given, the query is malformed, or
                an allowed u64 id is not present.
        """
        if k < 0:
            raise AdapterStoreError("k must be non-negative")
        if k == 0 or len(self) == 0:
            return []
        if self.dim_opt is None:
            return []

        query = _query_array(query_embedding, self.dim)
        if allowed_u64_ids is not None and filter is not None:
            raise AdapterStoreError("provide only one of filter or allowed_u64_ids")
        allowlist = (
            [_require_u64(value, "allowed u64 id") for value in allowed_u64_ids]
            if allowed_u64_ids is not None
            else self.filter_to_u64_allowlist(filter)
        )
        if allowlist is not None and not allowlist:
            return []

        mask = None
        if allowlist is not None:
            mask = np.zeros((len(self._slot_to_u64),), dtype=np.bool_)
            for u64_id in allowlist:
                if u64_id not in self._u64_to_slot:
                    raise AdapterStoreError(f"allowed u64 id {u64_id} is not present")
                mask[self._u64_to_slot[u64_id]] = True

        scores, slots = self._index.search(query, k=k, mask=mask)
        records: list[AdapterRecord] = []
        for score, slot in zip(scores.tolist(), slots.tolist()):
            slot = int(slot)
            if slot < 0 or slot >= len(self._slot_to_u64):
                raise AdapterStoreError(f"core search returned stale slot {slot}")
            u64_id = self._slot_to_u64[slot]
            string_id = self._string_id_for_u64(u64_id)
            records.append(
                AdapterRecord(
                    id=string_id,
                    document=self._documents[string_id],
                    metadata=dict(self._metadata[string_id]),
                    score=float(score),
                    u64_id=u64_id,
                )
            )
        return records

    def filter_to_u64_allowlist(
        self,
        filter: Mapping[str, Any] | Callable[[dict[str, Any]], bool] | None,
    ) -> list[int] | None:
        """Resolve a filter to the list of matching internal u64 handles.

        Args:
            filter: Same forms as in ``search_by_vector``; None means
                "no restriction".

        Returns:
            None when ``filter`` is None, otherwise the (possibly empty)
            list of matching u64 ids.
        """
        if filter is None:
            return None
        if not callable(filter):
            filter = _validate_portable_filter(filter)
        allowed: list[int] = []
        for string_id, u64_id in self._string_to_u64.items():
            document = self._documents[string_id]
            metadata = self._metadata[string_id]
            if _filter_matches(filter, metadata, document):
                allowed.append(u64_id)
        if not allowed and not callable(filter):
            self._warn_unknown_filter_keys(filter)
        return allowed

    def filter_records(
        self,
        filter: Mapping[str, Any] | Callable[[dict[str, Any]], bool] | None,
    ) -> list[AdapterRecord]:
        """Return the records matching a filter, without a vector search.

        Args:
            filter: Same forms as in ``search_by_vector``; None returns
                every record.

        Returns:
            Matching ``AdapterRecord`` objects in insertion order.
        """
        if filter is None:
            return list(self.iter_records())
        if not callable(filter):
            filter = _validate_portable_filter(filter)
        records = [
            record
            for record in self.iter_records()
            if _filter_matches(filter, record.metadata, record.document)
        ]
        if not records and not callable(filter):
            self._warn_unknown_filter_keys(filter)
        return records

    def _note_unsaved_write(self) -> None:
        """Warn once per unsaved-batch epoch, on the first unsaved write.

        A successful save re-arms the latch, so the next unsaved mutating
        write after a save warns again.
        """
        if (
            self.path is None
            or self._unsaved_writes_warned
            or not self.warn_unsaved_writes
        ):
            return
        self._unsaved_writes_warned = True
        warnings.warn(
            f"OrdinalDB {self.adapter_name} adapter: this write only updated "
            f"memory; nothing is persisted to {self.path} until you save "
            "(LangChain: save_local()/persist(); LlamaIndex: persist(); "
            "Agno: create()/save(); AdapterStore: save()). Unsaved data is "
            "lost when the process exits.",
            UnsavedWritesWarning,
            stacklevel=_external_warning_stacklevel(),
        )

    def _warn_unknown_filter_keys(self, filter: Mapping[str, Any]) -> None:
        """Warn when a zero-hit portable filter names keys no record has.

        Only called on the zero-result path, so the linear scan over the
        in-memory metadata keys costs nothing on successful queries.
        """
        if not self._metadata:
            return
        keys = set(filter)
        present: set[str] = set()
        for metadata in self._metadata.values():
            present.update(keys.intersection(metadata))
            if present == keys:
                return
        unknown = sorted(keys - present)
        if not unknown:
            return
        unknown_names = ", ".join(repr(key) for key in unknown)
        warnings.warn(
            f"filter matched 0 records: filter key(s) {unknown_names} do not "
            "appear in any record's metadata (possible typo in the key name?)",
            UnknownFilterKeyWarning,
            stacklevel=2,
        )

    def _write_sidecars(
        self,
        root: Path,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        payloads = self._sidecar_payloads()
        digests: dict[str, dict[str, Any]] = {}
        for name, payload in payloads.items():
            path = root / name
            _write_json(path, payload)
            digests[name] = _file_digest(path)
        return digests, payloads

    def _sidecar_payloads(self) -> dict[str, dict[str, Any]]:
        id_map_payload = {
            "schema_version": ID_MAP_SCHEMA_VERSION,
            "next_u64_id": self._next_u64_id,
            "string_to_u64": self._string_to_u64,
            "u64_to_slot": {str(key): value for key, value in self._u64_to_slot.items()},
        }
        documents_payload = {
            "schema_version": DOCUMENTS_SCHEMA_VERSION,
            "documents": self._documents,
        }
        metadata_payload = {
            "schema_version": METADATA_SCHEMA_VERSION,
            "metadata": self._metadata,
        }
        payloads = {
            ID_MAP_FILE: id_map_payload,
            DOCUMENTS_FILE: documents_payload,
            METADATA_FILE: metadata_payload,
        }
        return payloads

    def _preflight_batch(
        self,
        ids: Sequence[str],
        embeddings: Any,
        documents: Sequence[str],
        metadatas: Sequence[dict[str, Any]],
        *,
        upsert: bool,
    ) -> np.ndarray:
        if len(ids) != len(documents) or len(ids) != len(metadatas):
            raise AdapterStoreError("ids, documents, and metadatas must have the same length")
        duplicate_ids = _duplicate_values(ids)
        if duplicate_ids:
            raise AdapterStoreError(f"duplicate string IDs in batch: {duplicate_ids}")
        if not upsert:
            duplicates = [string_id for string_id in ids if string_id in self._string_to_u64]
            if duplicates:
                raise AdapterStoreError(f"IDs already present: {duplicates}")
        vectors = _vectors_array(embeddings)
        if vectors.shape[0] != len(ids):
            raise AdapterStoreError(
                f"embedding count {vectors.shape[0]} does not match id count {len(ids)}"
            )
        existing_dim = self.dim_opt
        if existing_dim is not None and vectors.shape[1] != existing_dim:
            raise AdapterStoreError(
                f"embedding dim mismatch: index dim={existing_dim}, got {vectors.shape[1]}"
            )
        if existing_dim is None:
            _validate_dim_compatible_with_bits(
                int(vectors.shape[1]),
                self.bits,
                context=f"{self.adapter_name} adapter embedding batch",
            )
        return vectors

    def _allocate_u64(self) -> int:
        while self._next_u64_id in self._u64_to_slot:
            self._next_u64_id += 1
        value = self._next_u64_id
        self._next_u64_id += 1
        return value

    def _remove_u64(self, u64_id: int) -> None:
        slot = self._u64_to_slot.pop(u64_id)
        self._u64_to_string.pop(u64_id, None)
        last_slot = len(self._slot_to_u64) - 1
        moved_u64 = self._slot_to_u64[last_slot]
        self._index.swap_remove(slot)
        if slot != last_slot:
            self._slot_to_u64[slot] = moved_u64
            self._u64_to_slot[moved_u64] = slot
        self._slot_to_u64.pop()

    def _string_id_for_u64(self, u64_id: int) -> str:
        try:
            return self._u64_to_string[u64_id]
        except KeyError as exc:
            raise AdapterStoreError(f"u64 id {u64_id} has no active string ID") from exc

    def _is_empty_lazy(self) -> bool:
        return len(self._slot_to_u64) == 0 and self.dim_opt is None

    def _validate_consistency(self) -> None:
        if len(self._slot_to_u64) != len(self._index):
            raise AdapterStoreError(
                f"id_map count {len(self._slot_to_u64)} does not match index len {len(self._index)}"
            )
        if set(self._documents) != set(self._string_to_u64):
            raise AdapterStoreError("documents sidecar keys do not match id_map string IDs")
        if set(self._metadata) != set(self._string_to_u64):
            raise AdapterStoreError("metadata sidecar keys do not match id_map string IDs")
        u64_values = list(self._string_to_u64.values())
        if len(set(u64_values)) != len(u64_values):
            raise AdapterStoreError("duplicate u64 IDs in string ID map")
        if set(self._u64_to_slot) != set(u64_values):
            raise AdapterStoreError("u64_to_slot keys do not match active string mappings")
        if self._u64_to_string != {
            u64_id: string_id for string_id, u64_id in self._string_to_u64.items()
        }:
            raise AdapterStoreError("u64_to_string keys do not match active string mappings")
        seen_slots: set[int] = set()
        for u64_id, slot in self._u64_to_slot.items():
            if slot < 0 or slot >= len(self._index):
                raise AdapterStoreError(f"u64 id {u64_id} points at stale slot {slot}")
            if slot in seen_slots:
                raise AdapterStoreError(f"duplicate vector slot {slot} in id_map")
            seen_slots.add(slot)
            if self._slot_to_u64[slot] != u64_id:
                raise AdapterStoreError("slot_to_u64 and u64_to_slot disagree")
        if u64_values and max(u64_values) >= self._next_u64_id:
            raise AdapterStoreError("next_u64_id must be greater than all allocated IDs")

    def _require_fresh_base_generation(self, target: Path) -> None:
        current_commit = _current_adapter_commit(target)
        if current_commit is None:
            return
        if self._base_commit_token is None:
            raise AdapterStoreError(
                "cannot save over an existing adapter store without a loaded base commit token"
            )
        if self.path is None or Path(self.path) != target:
            raise AdapterStoreError(
                "cannot save over an existing adapter store from a different path"
            )
        if current_commit != self._base_commit_token:
            raise AdapterStoreError(
                "stale adapter snapshot: target active commit changed from "
                f"{self._base_commit_token} to {current_commit}"
            )

    def _recover_contended_commit(
        self,
        target: Path,
        index_path: str,
        error: AdapterStoreError,
    ) -> dict[str, Any]:
        """Resolve a redb lock-contention failure raised mid-``save``.

        The adapter state write commits the snapshot durably (transaction +
        fsync) and then re-opens ``adapter.redb`` for a post-commit
        re-verification. A concurrent reader — ``ordinaldb adapter gc``,
        ``verify``/``inspect``/``stats``, or another loader — holding the
        database open at that instant makes the re-open fail with redb's raw
        "Database already open. Cannot acquire lock." even though the commit
        already succeeded: a false failure that invites double-inserting the
        batch on retry.

        Because this store holds the exclusive writer lock for the whole
        save, no other writer can have committed in the meantime, so the
        outcome is decidable: if the target's current commit token differs
        from this store's base token, the commit is ours and the failure is
        degraded to a ``UserWarning`` (returning the committed manifest);
        if the token is unchanged (or ``adapter.redb`` was never created),
        the commit did not publish and an ``AdapterStoreError`` saying that
        a retry is safe is raised. Only when the store cannot be re-read at
        all does this raise an ``AdapterStoreError`` spelling out that the
        outcome is unknown and a blind retry is unsafe.
        """
        if not _is_redb_lock_contention(error):
            raise error
        blame = (
            f"another process (e.g. `ordinaldb adapter gc` or a diagnostics "
            f"reader) had {ADAPTER_STORE_FILE} open while this writer was "
            "committing; never run gc or diagnostics against a store with a "
            "live writer"
        )
        if not _adapter_state_store_is_current(target):
            raise AdapterStoreError(
                f"adapter state store write failed: {error} ({blame}). The "
                "commit was NOT published: the store is unchanged, and the "
                "just-written vector generation is unreferenced debris that "
                "`ordinaldb adapter gc` can reclaim later. Retrying the save "
                "once the other process is gone is safe."
            ) from error
        manifest = _read_adapter_state_manifest_with_retry(target)
        if manifest is not None:
            committed_token = _commit_token_from_manifest(manifest)
            committed_ours = committed_token != self._base_commit_token and (
                self._is_empty_lazy() or committed_token.index_path == index_path
            )
            if committed_ours:
                warnings.warn(
                    f"OrdinalDB {self.adapter_name} adapter: save() committed "
                    "durably, but the post-commit re-verification of "
                    f"{ADAPTER_STORE_FILE} was blocked ({error}); {blame}. "
                    "The vector generation commit already succeeded and the "
                    "data IS durable — do not re-add this batch.",
                    UserWarning,
                    stacklevel=_external_warning_stacklevel(),
                )
                return manifest
            raise AdapterStoreError(
                f"adapter state store write failed: {error} ({blame}). The "
                "commit was NOT published; retrying the save once the other "
                "process is gone is safe."
            ) from error
        raise AdapterStoreError(
            f"adapter state store write failed: {error} ({blame}). Could not "
            "re-read the store to determine whether the commit published; the "
            "write may already be durable. Do NOT blindly re-add the batch "
            "and retry — re-load the store and check its contents first."
        ) from error


def _new_index(*, dim: int | None, bits: int) -> OrdinalIndex:
    if dim is None:
        return OrdinalIndex(bits=bits)
    return OrdinalIndex(dim=dim, bits=bits)


def _dim_multiple_for_bits(bits: int) -> int:
    codes_per_byte = 8 // bits
    buckets = 1 << bits
    return math.lcm(codes_per_byte, buckets)


def _validate_dim_compatible_with_bits(dim: int, bits: int, *, context: str) -> None:
    if dim < 2 or dim > 2**16 - 1:
        raise AdapterStoreError(
            f"{context} requires 2 <= dim <= 65535; got dim={dim}"
        )
    multiple = _dim_multiple_for_bits(bits)
    if dim % multiple != 0:
        raise AdapterStoreError(
            f"{context} requires dim divisible by {multiple} when bits={bits}; got dim={dim}"
        )


def _duplicate_values(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _vectors_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise AdapterStoreError("embeddings must be a 2D float32 array")
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array, dtype=np.float32)
    _validate_values(array)
    return array


def _query_array(values: Any, dim: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[0] != 1:
        raise AdapterStoreError("query_embedding must be a single vector")
    if array.shape[1] != dim:
        raise AdapterStoreError(f"query dim mismatch: index dim={dim}, got {array.shape[1]}")
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array, dtype=np.float32)
    _validate_values(array)
    return array


def _validate_values(array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        raise AdapterStoreError("embeddings must contain only finite values")
    if np.greater_equal(np.abs(array), MAX_INPUT_MAGNITUDE).any():
        raise AdapterStoreError(f"embeddings must have |value| < {MAX_INPUT_MAGNITUDE}")


def _filter_matches(
    filter: Mapping[str, Any] | Callable[[dict[str, Any]], bool],
    metadata: dict[str, Any],
    document: str,
) -> bool:
    if callable(filter):
        arity = _callable_filter_arity(filter)
        if arity == 1:
            return bool(filter(metadata))
        if arity == 2:
            return bool(filter(document, metadata))
        try:
            return bool(filter(metadata))
        except TypeError as exc:
            if not _looks_like_call_arity_error(exc):
                raise
            return bool(filter(document, metadata))
    filter = _validate_portable_filter(filter)
    for key, expected in filter.items():
        if key not in metadata:
            return False
        actual = metadata[key]
        if not _is_json_scalar(actual):
            raise AdapterStoreError(
                f"portable filter key {key!r} matched non-scalar metadata value"
            )
        if not _same_json_scalar_type(actual, expected):
            raise AdapterStoreError(
                f"portable filter key {key!r} mixed metadata type "
                f"{_json_scalar_type_name(actual)} with filter type "
                f"{_json_scalar_type_name(expected)}"
            )
        if actual != expected:
            return False
    return True


def _validate_portable_filter(filter: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(filter, Mapping):
        raise AdapterStoreError(
            "portable filters must be a mapping of metadata keys to JSON scalar values"
        )
    validated: dict[str, Any] = {}
    for key, value in filter.items():
        if not isinstance(key, str) or not key:
            raise AdapterStoreError("portable filter keys must be non-empty strings")
        if not _is_json_scalar(value):
            raise AdapterStoreError(
                f"portable filter value for {key!r} must be a JSON scalar"
            )
        validated[key] = value
    return validated


def _is_json_scalar(value: Any) -> bool:
    if not isinstance(value, _JSON_SCALAR_TYPES):
        return False
    if isinstance(value, float) and not math.isfinite(value):
        return False
    return True


def _same_json_scalar_type(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return not isinstance(left, bool) and not isinstance(right, bool)
    return type(left) is type(right)


def _json_scalar_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "number"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    return type(value).__name__


def _callable_filter_arity(filter: Callable[..., Any]) -> int | None:
    try:
        signature = inspect.signature(filter)
    except (TypeError, ValueError):
        return None

    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    required = [
        parameter
        for parameter in positional
        if parameter.default is inspect.Signature.empty
    ]
    has_varargs = any(
        parameter.kind is parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    max_args = float("inf") if has_varargs else len(positional)
    if len(required) <= 2 <= max_args:
        return 2
    if len(required) <= 1 <= max_args:
        return 1
    raise AdapterStoreError(
        "callable filters must accept metadata or document, metadata"
    )


def _looks_like_call_arity_error(exc: TypeError) -> bool:
    message = str(exc)
    markers = (
        "argument",
        "arguments",
        "positional",
        "required positional",
        "takes",
        "missing",
        "given",
    )
    return any(marker in message for marker in markers)


def _parse_id_map(payload: Mapping[str, Any]) -> tuple[dict[str, int], dict[int, int], int]:
    _require_exact_keys(
        payload,
        {"schema_version", "next_u64_id", "string_to_u64", "u64_to_slot"},
        ID_MAP_FILE,
    )
    if payload["schema_version"] != ID_MAP_SCHEMA_VERSION:
        raise AdapterStoreError(f"unsupported id_map schema {payload['schema_version']!r}")
    string_to_u64 = {
        _require_string_id(key): _require_u64(value, "u64 id")
        for key, value in _require_mapping(payload["string_to_u64"], "string_to_u64").items()
    }
    u64_to_slot = {
        _require_u64_key(key, "u64 id"): _require_non_negative_int(value, "slot")
        for key, value in _require_mapping(payload["u64_to_slot"], "u64_to_slot").items()
    }
    next_u64_id = _require_u64(payload["next_u64_id"], "next_u64_id")
    return string_to_u64, u64_to_slot, next_u64_id


def _parse_documents(payload: Mapping[str, Any]) -> dict[str, str]:
    _require_exact_keys(payload, {"schema_version", "documents"}, DOCUMENTS_FILE)
    if payload["schema_version"] != DOCUMENTS_SCHEMA_VERSION:
        raise AdapterStoreError(f"unsupported documents schema {payload['schema_version']!r}")
    return {
        _require_string_id(key): _require_document(value)
        for key, value in _require_mapping(payload["documents"], "documents").items()
    }


def _parse_metadata(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    _require_exact_keys(payload, {"schema_version", "metadata"}, METADATA_FILE)
    if payload["schema_version"] != METADATA_SCHEMA_VERSION:
        raise AdapterStoreError(f"unsupported metadata schema {payload['schema_version']!r}")
    return {
        _require_string_id(key): _require_metadata(value)
        for key, value in _require_mapping(payload["metadata"], "metadata").items()
    }


def _slot_map_from_u64_to_slot(mapping: Mapping[int, int], index_len: int) -> list[int]:
    slot_to_u64: list[int | None] = [None] * index_len
    for u64_id, slot in mapping.items():
        if slot < 0 or slot >= index_len:
            raise AdapterStoreError(f"u64 id {u64_id} points at stale slot {slot}")
        if slot_to_u64[slot] is not None:
            raise AdapterStoreError(f"duplicate vector slot {slot} in id_map")
        slot_to_u64[slot] = u64_id
    final_slot_map: list[int] = []
    for slot, u64_id in enumerate(slot_to_u64):
        if u64_id is None:
            raise AdapterStoreError(f"id_map does not cover vector slot {slot}")
        final_slot_map.append(u64_id)
    return final_slot_map


def _read_json(path: Path) -> dict[str, Any]:
    _reject_symlink_or_non_file(path, path.name)
    limit = _json_size_limit(path.name)
    try:
        data = path.read_bytes()
        if len(data) > limit:
            raise AdapterStoreError(
                f"{path.name} exceeds maximum JSON size {limit} bytes"
            )
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdapterStoreError(f"corrupt JSON in {path.name}: {exc}") from exc
    except OSError as exc:
        raise AdapterStoreError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdapterStoreError(f"{path.name} must contain a JSON object")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_bytes(_json_bytes(payload))


def _json_dumps(payload: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except ValueError as exc:
        raise AdapterStoreError("JSON payload contains non-finite number") from exc


def _payload_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    data = _json_bytes(payload)
    return {"sha256": hashlib.sha256(data).hexdigest(), "file_size_bytes": len(data)}


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (_json_dumps(payload) + "\n").encode("utf-8")


def _write_compatibility_exports(
    root: Path,
    adapter_payload: Mapping[str, Any],
    sidecar_payloads: Mapping[str, Mapping[str, Any]],
) -> None:
    for name in (ID_MAP_FILE, DOCUMENTS_FILE, METADATA_FILE):
        _write_json_atomic(root / name, sidecar_payloads[name])
        _publish_hook(f"after_export_{name}", root)
    _write_json_atomic(root / ADAPTER_FILE, adapter_payload)
    _publish_hook(f"after_export_{ADAPTER_FILE}", root)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temp_path.open("wb") as handle:
            handle.write(_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)
        _fsync_directory(path.parent)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _next_index_path(root: Path) -> str:
    max_generation_id = 0
    if (root / INDEX_DIR).exists():
        max_generation_id = max(max_generation_id, 1)
    vectors_root = root / VECTORS_DIR
    if vectors_root.is_dir():
        for child in vectors_root.iterdir():
            generation_id = _generation_id_from_dir_name(child.name)
            if generation_id is not None:
                max_generation_id = max(max_generation_id, generation_id)
    return _generation_index_path(max_generation_id + 1)


def _generation_index_path(generation_id: int) -> str:
    return f"{VECTORS_DIR}/g{generation_id:012d}.odb"


def _write_generation(index: OrdinalIndex, root: Path, index_path: str) -> None:
    final_path = root / index_path
    if final_path.exists():
        raise AdapterStoreError(f"generation already exists: {index_path}")
    _ensure_directory_without_symlink(final_path.parent)
    temp_path = final_path.with_name(
        f".{final_path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    try:
        index.write(temp_path)
        _publish_hook("after_generation_temp_write", root)
        _fsync_tree(temp_path)
        _publish_hook("after_generation_temp_fsync", root)
        loaded = OrdinalIndex.load(temp_path)
        if loaded.bits() != index.bits():
            raise AdapterStoreError("temporary generation bits mismatch")
        if loaded.dim() != index.dim():
            raise AdapterStoreError("temporary generation dim mismatch")
        _publish_hook("after_generation_temp_verified", root)
        temp_path.rename(final_path)
        _fsync_directory(final_path.parent)
        _publish_hook("after_generation_rename", root)
    except BaseException:
        shutil.rmtree(temp_path, ignore_errors=True)
        raise


def _validate_index_path(index_path: str) -> None:
    candidate = PurePosixPath(index_path)
    if not index_path or candidate.is_absolute() or str(candidate) != index_path:
        raise AdapterStoreError(f"unsupported index_path {index_path!r}")
    if any(part in ("", ".", "..") for part in candidate.parts):
        raise AdapterStoreError(f"unsupported index_path {index_path!r}")
    if index_path == INDEX_DIR:
        return
    if (
        len(candidate.parts) == 2
        and candidate.parts[0] == VECTORS_DIR
        and _is_generation_dir(candidate.parts[1])
    ):
        return
    raise AdapterStoreError(f"unsupported index_path {index_path!r}")


def _requires_redb_state(root: Path, index_path: str) -> bool:
    if PurePosixPath(index_path).parts[:1] == (VECTORS_DIR,):
        return True
    return _path_exists_no_follow(root / VECTORS_DIR)


def _is_generation_dir(name: str) -> bool:
    return _generation_id_from_dir_name(name) is not None


def _generation_id_from_dir_name(name: str) -> int | None:
    prefix = "g"
    suffix = ".odb"
    width = 12
    if not (name.startswith(prefix) and name.endswith(suffix)):
        return None
    digits = name[len(prefix) : -len(suffix)]
    if not (len(digits) == width and digits.isdecimal()):
        return None
    generation_id = int(digits)
    return generation_id if generation_id > 0 else None


def _reject_empty_lazy_vector_artifacts(root: Path, index_path: str) -> None:
    seen: set[Path] = set()
    for relative in (index_path, INDEX_DIR, VECTORS_DIR):
        artifact = root / relative
        if artifact in seen:
            continue
        seen.add(artifact)
        if _path_exists_no_follow(artifact):
            raise AdapterStoreError(
                f"empty_lazy adapter sidecar must not contain {relative}"
            )


def _write_adapter_state(
    root: Path,
    adapter_payload: Mapping[str, Any],
    sidecar_payloads: Mapping[str, Mapping[str, Any]],
    *,
    expected_revision: _AdapterCommitToken | None,
) -> dict[str, Any]:
    try:
        manifest_json = _AdapterStateStore.write_legacy_snapshot_with_existing_lock(
            root,
            _json_dumps(adapter_payload),
            _json_dumps(sidecar_payloads[ID_MAP_FILE]),
            _json_dumps(sidecar_payloads[DOCUMENTS_FILE]),
            _json_dumps(sidecar_payloads[METADATA_FILE]),
            _commit_token_json(expected_revision),
        )
    except ValueError as exc:
        raise AdapterStoreError(f"adapter state store write failed: {exc}") from exc
    return _loads_state_json(manifest_json, "adapter.redb manifest")


def _adapter_state_store_is_current(root: Path) -> bool:
    redb_path = root / ADAPTER_STORE_FILE
    try:
        metadata = redb_path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise AdapterStoreError(f"cannot stat {ADAPTER_STORE_FILE}: {exc}") from exc
    if redb_path.is_symlink():
        raise AdapterStoreError(f"{ADAPTER_STORE_FILE} must not be a symlink")
    if not redb_path.is_file():
        raise AdapterStoreError(f"{ADAPTER_STORE_FILE} must be a file")
    return True


def _current_adapter_commit(root: Path) -> _AdapterCommitToken | None:
    if _adapter_state_store_is_current(root):
        manifest = _read_adapter_state_manifest(root)
        return _commit_token_from_manifest(manifest)
    adapter_path = root / ADAPTER_FILE
    if not _path_exists_no_follow(adapter_path):
        return None
    adapter = _read_json(adapter_path)
    index_path = _require_string(adapter.get("index_path"), "index_path")
    _validate_index_path(index_path)
    id_map_payload = _read_json(root / ID_MAP_FILE)
    string_to_u64, _, _ = _parse_id_map(id_map_payload)
    return _AdapterCommitToken(
        store_uuid=None,
        active_generation_id=None,
        active_generation_manifest_sha256=None,
        commit_sequence=None,
        index_path=index_path,
        active_ids=len(string_to_u64),
    )


def _read_adapter_state_manifest(root: Path) -> dict[str, Any]:
    try:
        manifest_json = _AdapterStateStore.verify(root, None)
    except ValueError as exc:
        raise AdapterStoreError(f"adapter state store verification failed: {exc}") from exc
    return _loads_state_json(manifest_json, "adapter.redb manifest")


def _read_adapter_state_manifest_with_retry(root: Path) -> dict[str, Any] | None:
    """Read the manifest, briefly retrying past a transient concurrent reader.

    Returns None when the manifest stays unreadable (the concurrent reader
    is still holding ``adapter.redb`` open, or verification fails for any
    other reason).
    """
    for attempt in range(3):
        if attempt:
            time.sleep(0.05)
        try:
            return _read_adapter_state_manifest(root)
        except AdapterStoreError as exc:
            if not _is_redb_lock_contention(exc):
                return None
    return None


def _commit_token_from_manifest(manifest: Mapping[str, Any]) -> _AdapterCommitToken:
    active_generation_path = _require_string(
        manifest.get("active_generation_path"), "active_generation_path"
    )
    if active_generation_path:
        _validate_index_path(active_generation_path)
    return _AdapterCommitToken(
        store_uuid=_require_string(manifest.get("store_uuid"), "store_uuid"),
        active_generation_id=_require_u64(
            manifest.get("active_generation_id"), "active_generation_id"
        ),
        active_generation_manifest_sha256=_require_optional_string(
            manifest.get("active_generation_manifest_sha256"),
            "active_generation_manifest_sha256",
        ),
        commit_sequence=_require_u64(manifest.get("commit_sequence"), "commit_sequence"),
        index_path=active_generation_path,
        active_ids=_require_non_negative_int(
            manifest.get("active_id_count"), "active_id_count"
        ),
    )


def _commit_token_json(token: _AdapterCommitToken | None) -> str | None:
    if token is None:
        return None
    return _json_dumps(
        {
            "store_uuid": token.store_uuid,
            "commit_sequence": token.commit_sequence,
            "active_generation_id": token.active_generation_id,
            "active_generation_path": token.index_path,
            "active_generation_manifest_sha256": token.active_generation_manifest_sha256,
            "active_id_count": token.active_ids,
        }
    )


def _read_adapter_state(
    root: Path,
    expected_adapter: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        adapter_json, id_map_json, documents_json, metadata_json = (
            _AdapterStateStore.load_legacy_snapshot(root, expected_adapter)
        )
    except ValueError as exc:
        raise AdapterStoreError(f"adapter state store verification failed: {exc}") from exc
    return (
        _loads_state_json(adapter_json, ADAPTER_FILE),
        _loads_state_json(id_map_json, ID_MAP_FILE),
        _loads_state_json(documents_json, DOCUMENTS_FILE),
        _loads_state_json(metadata_json, METADATA_FILE),
    )


def _loads_state_json(payload: str, name: str) -> dict[str, Any]:
    limit = _json_size_limit(name)
    if len(payload.encode("utf-8")) > limit:
        raise AdapterStoreError(
            f"adapter state {name} exceeds maximum JSON size {limit} bytes"
        )
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_json,
        )
    except json.JSONDecodeError as exc:
        raise AdapterStoreError(f"corrupt JSON in adapter state {name}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdapterStoreError(f"adapter state {name} must contain a JSON object")
    return value


def _json_size_limit(name: str) -> int:
    return MAX_ADAPTER_JSON_BYTES if name == ADAPTER_FILE else MAX_SIDECAR_JSON_BYTES


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise AdapterStoreError(f"duplicate JSON key {key!r}")
        out[key] = value
    return out


def _reject_non_finite_json(value: str) -> None:
    raise AdapterStoreError(f"non-finite JSON number {value} is not supported")


def _file_digest(path: Path) -> dict[str, Any]:
    _reject_symlink_or_non_file(path, path.name)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "file_size_bytes": size}


def _verify_sidecar(path: Path, expected: Mapping[str, Any], name: str) -> None:
    _require_exact_keys(expected, {"sha256", "file_size_bytes"}, f"sidecars.{name}")
    try:
        actual = _file_digest(path)
    except AdapterStoreError as exc:
        raise AdapterStoreError(f"cannot read sidecar {name}: {exc}") from exc
    except OSError as exc:
        raise AdapterStoreError(f"cannot read sidecar {name}: {exc}") from exc
    if actual != expected:
        raise AdapterStoreError(f"sidecar integrity check failed for {name}")


def _writer_lock_path(target: Path) -> Path:
    return target / WRITE_LOCK_FILE


def _publish_hook(stage: str, root: Path) -> None:
    if _PUBLISH_TEST_HOOK is not None:
        _PUBLISH_TEST_HOOK(stage, root)


def _path_exists_no_follow(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _reject_symlink_or_non_file(path: Path, name: str) -> None:
    try:
        path.lstat()
    except OSError as exc:
        raise AdapterStoreError(f"cannot stat {name}: {exc}") from exc
    if path.is_symlink():
        raise AdapterStoreError(f"{name} must not be a symlink")
    if not path.is_file():
        raise AdapterStoreError(f"{name} must be a file")


def _allowed_platform_directory_symlink(path: Path) -> Path | None:
    if sys.platform != "darwin":
        return None
    allowed = {
        Path("/tmp"): Path("/private/tmp"),
        Path("/var"): Path("/private/var"),
    }
    expected = allowed.get(path)
    if expected is None:
        return None
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return None
    if resolved == expected:
        return resolved
    return None


def _ensure_directory_without_symlink(path: Path) -> None:
    current = Path(path.anchor) if path.is_absolute() else Path(".")
    parts = path.parts[1:] if path.is_absolute() else path.parts
    for part in parts:
        current = current / part
        if current.is_symlink():
            resolved = _allowed_platform_directory_symlink(current)
            if resolved is not None:
                current = resolved
                continue
            raise AdapterStoreError(
                f"directory path must not contain a symlink: {current}"
            )
        current.mkdir(exist_ok=True)
        if not current.is_dir():
            raise AdapterStoreError(
                f"directory path component is not a directory: {current}"
            )


def _validated_generation_path(root: Path, index_path: str) -> Path:
    _validate_index_path(index_path)
    current = root
    for part in PurePosixPath(index_path).parts:
        current = current / part
        if current.is_symlink():
            raise AdapterStoreError(
                f"active generation path must not contain a symlink: {current}"
            )
        if not current.exists():
            raise AdapterStoreError(f"active generation path is missing: {current}")
        if not current.is_dir():
            raise AdapterStoreError(
                f"active generation path component is not a directory: {current}"
            )
    manifest_path = current / "manifest.json"
    _reject_symlink_or_non_file(manifest_path, "generation manifest")
    return current


def _fsync_tree(root: Path) -> None:
    for current, _, files in os.walk(root):
        current_path = Path(current)
        for file_name in files:
            _fsync_file(current_path / file_name)
        _fsync_directory(current_path)


def _fsync_file(path: Path) -> None:
    flags = os.O_RDONLY
    if os.name == "nt":
        flags = os.O_RDWR | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise AdapterStoreError(f"cannot open {path} for sync: {exc}") from exc
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise AdapterStoreError(f"cannot open directory {path} for sync: {exc}") from exc
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _replace_directory(target: Path, source: Path) -> None:
    backup: Path | None = None
    if target.exists():
        backup = target.with_name(f".{target.name}.bak-{os.getpid()}")
        if backup.exists():
            shutil.rmtree(backup)
        target.rename(backup)
        _fsync_directory(target.parent)
    try:
        source.rename(target)
        _fsync_directory(target.parent)
    except Exception:
        if backup is not None and not target.exists():
            backup.rename(target)
            _fsync_directory(target.parent)
        raise
    finally:
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
            _fsync_directory(target.parent)


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if not isinstance(value, dict):
        raise AdapterStoreError(f"{name} must be a JSON object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise AdapterStoreError(f"{name} has invalid keys: {', '.join(details)}")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise AdapterStoreError(f"{name} must be a JSON object")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise AdapterStoreError(f"{name} must be a string")
    return value


def _require_optional_string(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, name)


def _require_string_id(value: Any) -> str:
    value = _require_string(value, "string ID")
    if not value:
        raise AdapterStoreError("string IDs must be non-empty")
    return value


def _require_document(value: Any) -> str:
    return _require_string(value, "document")


def _require_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AdapterStoreError("metadata must be a JSON object")
    return dict(value)


def _require_u64(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > 2**64 - 1:
        raise AdapterStoreError(f"{name} must be an unsigned 64-bit integer")
    return value


def _require_u64_key(value: Any, name: str) -> int:
    if isinstance(value, str):
        if not value.isdecimal():
            raise AdapterStoreError(f"{name} key must be an unsigned integer string")
        value = int(value)
    return _require_u64(value, name)


def _require_non_negative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AdapterStoreError(f"{name} must be a non-negative integer")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise AdapterStoreError(f"{name} must be a positive integer")
    return value


def _require_bits(value: Any) -> int:
    bits = _require_non_negative_int(value, "bits")
    if bits not in (1, 2, 4):
        raise AdapterStoreError("bits must be one of 1, 2, or 4")
    return bits


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise AdapterStoreError(f"{name} must be a boolean")
    return value
