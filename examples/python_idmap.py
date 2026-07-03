from pathlib import Path
import tempfile

import numpy as np

from ordinaldb import IdMapIndex


def main() -> None:
    vectors = np.arange(12 * 64, dtype=np.float32).reshape(12, 64)
    queries = vectors[:2].copy()
    ids = np.arange(1001, 1013, dtype=np.uint64)

    idx = IdMapIndex(dim=64, bits=2)
    idx.add_with_ids(vectors, ids)
    scores, found = idx.search(queries, k=3)

    assert scores.shape == (6,)
    assert found.dtype == np.uint64
    assert set(found.tolist()).issubset(set(ids.tolist()))

    idx.remove(1002)
    assert not idx.contains(1002)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "docs_ids.odb"
        idx.write(path)
        loaded = IdMapIndex.load(path)
        _, loaded_found = loaded.search(queries, k=11)

    assert 1002 not in loaded_found.tolist()


if __name__ == "__main__":
    main()
