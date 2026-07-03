import argparse
import json
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np

from ordinaldb import IdMapIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="OrdinalDB local benchmark smoke")
    parser.add_argument("--vectors", type=int, default=10_000)
    parser.add_argument("--queries", type=int, default=64)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--bits", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()

    vectors = deterministic_matrix(args.vectors, args.dim)
    queries = deterministic_matrix(args.queries, args.dim)
    ids = np.arange(1_000_000, 1_000_000 + args.vectors, dtype=np.uint64)
    allowlist = ids[:: max(1, args.vectors // 512)].copy()

    rows = []
    for bits in args.bits:
        idx = IdMapIndex(dim=args.dim, bits=bits)

        start = perf_counter()
        idx.add_with_ids(vectors, ids)
        ingest_seconds = perf_counter() - start

        start = perf_counter()
        idx.search(queries, k=args.k)
        search_seconds = perf_counter() - start

        start = perf_counter()
        idx.search(queries, k=args.k, allowlist=allowlist)
        allowlist_seconds = perf_counter() - start

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"ordinaldb-b{bits}.odb"
            start = perf_counter()
            idx.write(path)
            write_seconds = perf_counter() - start
            bundle_bytes = directory_size(path)

            start = perf_counter()
            IdMapIndex.load(path)
            load_seconds = perf_counter() - start

        rows.append(
            {
                "bits": bits,
                "vectors": args.vectors,
                "queries": args.queries,
                "dim": args.dim,
                "k": args.k,
                "ingest_seconds": ingest_seconds,
                "search_seconds": search_seconds,
                "allowlist_seconds": allowlist_seconds,
                "write_seconds": write_seconds,
                "load_seconds": load_seconds,
                "bundle_bytes": bundle_bytes,
                "bundle_bytes_per_vector": bundle_bytes / max(1, args.vectors),
            }
        )

    print(json.dumps({"ordinaldb_benchmark_smoke": rows}, indent=2, sort_keys=True))


def deterministic_matrix(rows: int, dim: int):
    row = np.arange(rows, dtype=np.float32)[:, None]
    col = np.arange(dim, dtype=np.float32)[None, :]
    values = ((row + 5.0) * (col + 13.0) + row * 17.0 + col * 3.0) % 97.0
    return ((values - 48.0) / 49.0).astype(np.float32)


def directory_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


if __name__ == "__main__":
    main()
