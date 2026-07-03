#!/usr/bin/env python3
"""Generate a persistence and filter limits report for OrdinalDB core and adapter storage."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np

from ordinaldb import OrdinalIndex
from ordinaldb.adapters import AdapterStore


SCHEMA_VERSION = "ordinaldb.limits_report.v1"
TARGET = "ordinaldb-v0.2.0"
SEED = 0x0DB30003


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    out = args.output
    work_dir = args.work_dir
    if args.clean and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    ordinaldb_bin = args.ordinaldb_bin.resolve()
    if not ordinaldb_bin.is_file():
        raise SystemExit(f"ordinaldb binary not found: {ordinaldb_bin}")

    rng = np.random.default_rng(SEED)
    results: list[dict[str, Any]] = []
    for size in args.sizes:
        vectors = rng.normal(size=(size, args.dim)).astype(np.float32)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-6)
        ids = [f"id-{index:06d}" for index in range(size)]
        documents = [f"document {index}" for index in range(size)]
        metadatas = [
            {
                "all": "yes",
                "half": "a" if index < size // 2 else "b",
                "pct1": "yes" if index < max(1, size // 100) else "no",
                "single": "yes" if index == 0 else "no",
            }
            for index in range(size)
        ]
        query = vectors[0]

        core_dir = work_dir / f"core-{size}.odb"
        adapter_dir = work_dir / f"adapter-{size}"
        if core_dir.exists():
            shutil.rmtree(core_dir)
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)

        core_metrics = run_core_case(
            size=size,
            dim=args.dim,
            bits=args.bits,
            vectors=vectors,
            query=query,
            core_dir=core_dir,
            ordinaldb_bin=ordinaldb_bin,
        )
        adapter_metrics = run_adapter_case(
            size=size,
            dim=args.dim,
            bits=args.bits,
            vectors=vectors,
            query=query,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            adapter_dir=adapter_dir,
            ordinaldb_bin=ordinaldb_bin,
        )
        results.append(
            {
                "rows": size,
                "dim": args.dim,
                "bits": args.bits,
                "core": core_metrics,
                "adapter": adapter_metrics,
            }
        )

    envelope = build_envelope(args=args, results=results)
    out.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    for result in results:
        print(
            "rows={rows} core_write={core_write:.6f}s adapter_save={adapter_save:.6f}s "
            "adapter_bytes={adapter_bytes}".format(
                rows=result["rows"],
                core_write=result["core"]["write_seconds"],
                adapter_save=result["adapter"]["save_seconds"],
                adapter_bytes=result["adapter"]["footprint_bytes"],
            )
        )
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--ordinaldb-bin", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10_000, 100_000])
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args(argv)


def run_core_case(
    *,
    size: int,
    dim: int,
    bits: int,
    vectors: np.ndarray,
    query: np.ndarray,
    core_dir: Path,
    ordinaldb_bin: Path,
) -> dict[str, Any]:
    index = OrdinalIndex(dim=dim, bits=bits)
    add_seconds = timed(lambda: index.add(vectors))
    write_seconds = timed(lambda: index.write(core_dir))
    loaded_holder: dict[str, Any] = {}
    cold_open_seconds = timed(lambda: loaded_holder.setdefault("index", OrdinalIndex.load(core_dir)))
    loaded = loaded_holder["index"]
    search_seconds = timed(lambda: loaded.search(query.reshape(1, -1), k=10))
    verify = run_cli(ordinaldb_bin, "verify", core_dir)
    inspect = run_cli(ordinaldb_bin, "inspect", core_dir)
    return {
        "add_seconds": add_seconds,
        "write_seconds": write_seconds,
        "cold_open_seconds": cold_open_seconds,
        "search_seconds": search_seconds,
        "footprint_bytes": path_size_bytes(core_dir),
        "verify_exit_code": verify["exit_code"],
        "inspect_exit_code": inspect["exit_code"],
    }


def run_adapter_case(
    *,
    size: int,
    dim: int,
    bits: int,
    vectors: np.ndarray,
    query: np.ndarray,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, str]],
    adapter_dir: Path,
    ordinaldb_bin: Path,
) -> dict[str, Any]:
    store = AdapterStore(bits=bits, dim=dim)
    add_seconds = timed(
        lambda: store.add(
            ids=ids,
            embeddings=vectors,
            documents=documents,
            metadatas=metadatas,
        )
    )
    save_seconds = timed(lambda: store.save(adapter_dir, adapter_name="limits-report"))
    loaded_holder: dict[str, Any] = {}
    cold_open_seconds = timed(lambda: loaded_holder.setdefault("store", AdapterStore.load(adapter_dir)))
    loaded = loaded_holder["store"]
    filters = measure_filters(loaded, query=query, size=size)
    verify = run_cli(ordinaldb_bin, "verify", adapter_dir)
    stats = run_cli(ordinaldb_bin, "stats", adapter_dir)
    return {
        "add_seconds": add_seconds,
        "save_seconds": save_seconds,
        "cold_open_seconds": cold_open_seconds,
        "footprint_bytes": path_size_bytes(adapter_dir),
        "verify_exit_code": verify["exit_code"],
        "stats_exit_code": stats["exit_code"],
        "filters": filters,
    }


def measure_filters(store: AdapterStore, *, query: np.ndarray, size: int) -> list[dict[str, Any]]:
    cases = [
        ("empty", {"single": "absent"}, 0),
        ("one_id", {"single": "yes"}, 1),
        ("pct1", {"pct1": "yes"}, max(1, size // 100)),
        ("pct50", {"half": "a"}, size // 2),
        ("pct100", {"all": "yes"}, size),
    ]
    results = []
    for name, filter_value, expected_count in cases:
        allowlist_holder: dict[str, Any] = {}
        allowlist_seconds = timed(
            lambda: allowlist_holder.setdefault(
                "allowlist",
                store.filter_to_u64_allowlist(filter_value),
            )
        )
        allowlist = allowlist_holder["allowlist"]
        if len(allowlist) != expected_count:
            raise AssertionError(
                f"{name} expected {expected_count} allowlist IDs, got {len(allowlist)}"
            )
        records_holder: dict[str, Any] = {}
        search_seconds = timed(
            lambda: records_holder.setdefault(
                "records",
                store.search_by_vector(query, k=10, filter=filter_value),
            )
        )
        records = records_holder["records"]
        expected_ids = {f"id-{index:06d}" for index in expected_indexes(name, size)}
        returned_ids = {record.id for record in records}
        if not returned_ids.issubset(expected_ids):
            raise AssertionError(f"{name} returned IDs outside expected filter set")
        results.append(
            {
                "name": name,
                "expected_matches": expected_count,
                "returned": len(records),
                "allowlist_seconds": allowlist_seconds,
                "search_seconds": search_seconds,
            }
        )
    return results


def expected_indexes(name: str, size: int) -> range:
    if name == "empty":
        return range(0)
    if name == "one_id":
        return range(1)
    if name == "pct1":
        return range(max(1, size // 100))
    if name == "pct50":
        return range(size // 2)
    if name == "pct100":
        return range(size)
    raise AssertionError(f"unknown filter case {name}")


def timed(action: Any) -> float:
    start = time.perf_counter()
    action()
    return time.perf_counter() - start


def run_cli(binary: Path, command: str, path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(binary), command, str(path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"{binary} {command} {path} failed with {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    return {
        "exit_code": completed.returncode,
        "stdout_bytes": len(completed.stdout.encode("utf-8")),
        "stderr_bytes": len(completed.stderr.encode("utf-8")),
    }


def path_size_bytes(root: Path) -> int:
    if root.is_symlink():
        raise AssertionError(f"refusing to size symlink {root}")
    if root.is_file():
        return root.stat().st_size
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise AssertionError(f"refusing to size symlink {path}")
        if path.is_file():
            total += path.stat().st_size
    return total


def build_envelope(*, args: argparse.Namespace, results: list[dict[str, Any]]) -> dict[str, Any]:
    dirty = bool(run_text(["git", "status", "--porcelain"]).strip())
    return {
        "schema_version": SCHEMA_VERSION,
        "report_id": "persistence_limits",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target": TARGET,
        "category": "persistence_limits",
        "status": "passed",
        "summary": "10K/100K core and adapter persistence, cold-open, footprint, verify, stats, and filter-selectivity measurements completed.",
        "claim_boundaries": [
            "Measurements are local single-process Linux x86_64 results from this runner.",
            "Embedding generation time is excluded; vectors are deterministic synthetic float32 arrays.",
            "Filter measurements use scan-based scalar equality before vector ranking.",
            "Results are capacity guidance for OrdinalDB, not cross-database benchmarks."
        ],
        "provenance": {
            "git_commit": run_text(["git", "rev-parse", "--short=7", "HEAD"]).strip(),
            "git_dirty": "dirty" if dirty else "clean",
            "commands": [
                {
                    "cmd": " ".join(sys.argv),
                    "exit_code": 0,
                }
            ],
            "os": {
                "name": platform.system().lower(),
                "release": platform.release(),
                "version": platform.version(),
            },
            "hardware": {
                "machine": platform.machine(),
                "processor": platform.processor(),
                "cpu_count": os.cpu_count(),
            },
            "dataset": {
                "seed": SEED,
                "sizes": args.sizes,
                "dim": args.dim,
                "bits": args.bits,
            },
        },
        "artifacts": [
            {
                "path": str(args.output),
            },
            {
                "path": str(args.work_dir),
            },
        ],
        "metrics": {
            "results": results,
        },
        "limits": recommended_limits(results),
    }


def recommended_limits(results: list[dict[str, Any]]) -> dict[str, Any]:
    largest = max(results, key=lambda item: item["rows"])
    return {
        "measured_rows_max": largest["rows"],
        "recommended_max_rows": largest["rows"],
        "mutation_model": "full-generation copy-on-write per committed save",
        "unsupported_claims": [
            "multi-writer throughput",
            "cross-process live sharing",
            "metadata index latency",
            "coverage beyond measured row counts",
        ],
    }


def run_text(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
