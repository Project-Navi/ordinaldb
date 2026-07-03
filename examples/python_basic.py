from pathlib import Path
import tempfile

import numpy as np

from ordinaldb import OrdinalIndex


def main() -> None:
    vectors = np.arange(16 * 64, dtype=np.float32).reshape(16, 64)
    queries = vectors[:2].copy()

    idx = OrdinalIndex(dim=64, bits=2)
    idx.add(vectors)
    scores, indices = idx.search(queries, k=4)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "docs.odb"
        idx.write(path)
        loaded = OrdinalIndex.load(path)
        loaded_scores, loaded_indices = loaded.search(queries, k=4)

    assert scores.shape == (8,)
    assert indices.shape == (8,)
    np.testing.assert_array_equal(loaded_scores, scores)
    np.testing.assert_array_equal(loaded_indices, indices)


if __name__ == "__main__":
    main()
