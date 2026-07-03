import numpy as np

from ordinaldb import IdMapIndex


def main() -> None:
    vectors = np.arange(10 * 64, dtype=np.float32).reshape(10, 64)
    queries = vectors[:2].copy()
    ids = np.arange(500, 510, dtype=np.uint64)
    allowlist = np.array([501, 504, 509], dtype=np.uint64)

    idx = IdMapIndex(dim=64, bits=2)
    idx.add_with_ids(vectors, ids)
    _scores, found = idx.search(queries, k=10, allowlist=allowlist)

    assert found.shape == (6,)
    assert set(found.tolist()).issubset(set(allowlist.tolist()))


if __name__ == "__main__":
    main()
