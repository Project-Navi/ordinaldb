#!/usr/bin/env python3
"""Deterministic hostile-input smoke test for OrdinalDB adapter storage."""

from __future__ import annotations

from pathlib import Path
import os
import random
import tempfile

from ordinaldb.adapters import AdapterStore, AdapterStoreError


SEED = 0x0DB03001


def main() -> int:
    rng = random.Random(SEED)
    cases = 0
    with tempfile.TemporaryDirectory(prefix="ordinaldb-hostile-inputs-") as tmp:
        root = Path(tmp)
        for index in range(8):
            case = root / f"corrupt-redb-{index}"
            case.mkdir()
            (case / "adapter.redb").write_bytes(_random_bytes(rng, rng.randrange(1, 129)))
            _expect_adapter_error(f"corrupt redb {index}", case)
            cases += 1

        for index in range(8):
            case = root / f"random-json-{index}"
            case.mkdir()
            (case / "adapter.json").write_bytes(_random_bytes(rng, rng.randrange(1, 129)))
            _expect_adapter_error(f"random adapter json {index}", case)
            cases += 1

        duplicate = root / "duplicate-json-key"
        duplicate.mkdir()
        (duplicate / "adapter.json").write_text(
            '{"schema_version": 1, "schema_version": 2}',
            encoding="utf-8",
        )
        _expect_adapter_error("duplicate adapter json key", duplicate, "duplicate")
        cases += 1

        non_finite = root / "non-finite-json"
        non_finite.mkdir()
        (non_finite / "adapter.json").write_text(
            '{"schema_version": NaN}',
            encoding="utf-8",
        )
        _expect_adapter_error("non-finite adapter json", non_finite, "non-finite")
        cases += 1

        if hasattr(os, "symlink"):
            outside = root / "outside.redb"
            outside.write_bytes(b"not a redb store")
            symlink_case = root / "symlink-redb"
            symlink_case.mkdir()
            os.symlink(outside, symlink_case / "adapter.redb")
            _expect_adapter_error("symlinked adapter redb", symlink_case, "symlink")
            cases += 1

    print(f"seed={SEED}")
    print(f"cases={cases}")
    print("result=passed")
    return 0


def _random_bytes(rng: random.Random, size: int) -> bytes:
    return bytes(rng.randrange(0, 256) for _ in range(size))


def _expect_adapter_error(label: str, path: Path, contains: str | None = None) -> None:
    try:
        AdapterStore.load(path)
    except AdapterStoreError as exc:
        message = str(exc)
        if contains is not None and contains.lower() not in message.lower():
            raise AssertionError(
                f"{label}: expected error containing {contains!r}, got {message!r}"
            ) from exc
        print(f"{label}: rejected: {message.splitlines()[0]}")
        return
    raise AssertionError(f"{label}: unexpectedly loaded")


if __name__ == "__main__":
    raise SystemExit(main())
