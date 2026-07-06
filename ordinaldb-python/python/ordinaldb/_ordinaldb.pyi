"""Type stubs for the native `ordinaldb._ordinaldb` extension module."""

from os import PathLike
from types import TracebackType
from typing import final

import numpy as np
from numpy.typing import NDArray

_StrPath = str | PathLike[str]

__all__ = [
    "OrdinalIndex",
    "IdMapIndex",
    "_AdapterStateStore",
    "_AdapterWriteLock",
]

@final
class OrdinalIndex:
    """Slot-addressed ordinal-quantized vector index."""

    def __new__(
        cls,
        dim: int | None = None,
        bits: int | None = None,
        bit_width: int | None = None,
        sign: str = "optional",
    ) -> OrdinalIndex: ...
    @property
    def has_sign_sidecar(self) -> bool: ...
    def add(self, vectors: NDArray[np.float32]) -> None: ...
    def search(
        self,
        queries: NDArray[np.float32],
        k: int,
        mask: NDArray[np.bool_] | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]: ...
    def swap_remove(self, idx: int) -> int: ...
    def write(self, path: _StrPath) -> None: ...
    @classmethod
    def load(cls, path: _StrPath) -> OrdinalIndex: ...
    def len(self) -> int: ...
    def is_empty(self) -> bool: ...
    def dim(self) -> int: ...
    def dim_opt(self) -> int | None: ...
    def bits(self) -> int: ...
    def __len__(self) -> int: ...

@final
class IdMapIndex:
    """Vector index addressing rows by stable unsigned 64-bit identifiers."""

    def __new__(
        cls,
        dim: int | None = None,
        bits: int | None = None,
        bit_width: int | None = None,
        sign: str = "optional",
    ) -> IdMapIndex: ...
    @property
    def has_sign_sidecar(self) -> bool: ...
    def add_with_ids(
        self,
        vectors: NDArray[np.float32],
        ids: NDArray[np.uint64],
    ) -> None: ...
    def search(
        self,
        queries: NDArray[np.float32],
        k: int,
        allowlist: NDArray[np.uint64] | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.uint64]]: ...
    def remove(self, id: int) -> bool: ...
    def write(self, path: _StrPath) -> None: ...
    @classmethod
    def load(cls, path: _StrPath) -> IdMapIndex: ...
    def contains(self, id: int) -> bool: ...
    def len(self) -> int: ...
    def is_empty(self) -> bool: ...
    def dim(self) -> int: ...
    def dim_opt(self) -> int | None: ...
    def bits(self) -> int: ...
    def __len__(self) -> int: ...

@final
class _AdapterWriteLock:
    """Exclusive writer lock over an adapter store directory."""

    def __enter__(self) -> _AdapterWriteLock: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool: ...

@final
class _AdapterStateStore:
    """Low-level verified snapshot store used by the adapter layer."""

    @staticmethod
    def acquire_writer_lock(path: _StrPath) -> _AdapterWriteLock: ...
    @staticmethod
    def write_legacy_snapshot(
        path: _StrPath,
        adapter_json: str,
        id_map_json: str,
        documents_json: str,
        metadata_json: str,
        expected_revision_json: str | None = None,
    ) -> str: ...
    @staticmethod
    def write_legacy_snapshot_with_existing_lock(
        path: _StrPath,
        adapter_json: str,
        id_map_json: str,
        documents_json: str,
        metadata_json: str,
        expected_revision_json: str | None = None,
    ) -> str: ...
    @staticmethod
    def load_legacy_snapshot(
        path: _StrPath,
        expected_adapter: str | None = None,
    ) -> tuple[str, str, str, str]: ...
    @staticmethod
    def verify(
        path: _StrPath,
        expected_adapter: str | None = None,
    ) -> str: ...
