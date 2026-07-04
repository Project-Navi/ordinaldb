"""Regression tests: filtered search must read NumPy filters zero-copy.

Before the fix, ``OrdinalIndex.search(mask=...)`` converted the mask via
``tolist()`` into a ``Vec<bool>`` and ``IdMapIndex.search(allowlist=...)``
copied the ids into a ``Vec<u64>`` -- boxing up to index-length elements
while HOLDING the GIL on every filtered call (tens of milliseconds at
1.26M rows, versus ~3ms for the unmasked search itself).

Now both paths pin the NumPy buffer via the buffer protocol (O(1) under
the GIL) and read it directly after the GIL is released. These tests
assert the fast path is equivalent to the legacy conversion path, that
dtype/contiguity validation still raises clean ``ValueError``s, and that
filtered calls still release the GIL.

Runs under both pytest and the CI ``unittest`` discovery.
"""

import array
import unittest

import numpy as np

from ordinaldb import IdMapIndex, OrdinalIndex
from test_gil_release import _main_thread_progress_during

RNG = np.random.default_rng(7)

DIM = 16
N_ROWS = 128

GIL_DIM = 256
GIL_N_ROWS = 100_000
GIL_N_QUERIES = 2_048
GIL_MIN_ITERATIONS = 30
GIL_MIN_CALL_SECONDS = 0.15


def _vectors(n: int, dim: int = DIM) -> np.ndarray:
    vectors = RNG.standard_normal((n, dim), dtype=np.float32)
    vectors *= 0.1
    return vectors


class _DuckMask:
    """Quacks like a 1D bool ndarray without exporting a buffer.

    Forces the bindings onto the legacy duck-typed ``tolist()`` fallback,
    letting the tests compare it against the zero-copy buffer fast path.
    """

    def __init__(self, mask: np.ndarray) -> None:
        self._mask = mask

    @property
    def ndim(self):
        return self._mask.ndim

    @property
    def flags(self):
        return self._mask.flags

    @property
    def dtype(self):
        return self._mask.dtype

    def tolist(self):
        return self._mask.tolist()


class OrdinalMaskEquivalenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = OrdinalIndex(dim=DIM, bits=2)
        self.index.add(_vectors(N_ROWS))
        self.queries = _vectors(4)

    def test_numpy_mask_matches_legacy_tolist_path(self):
        mask = RNG.random(N_ROWS) < 0.5
        self.assertEqual(mask.dtype, np.bool_)

        fast_scores, fast_indices = self.index.search(self.queries, k=8, mask=mask)
        slow_scores, slow_indices = self.index.search(
            self.queries, k=8, mask=_DuckMask(mask)
        )

        np.testing.assert_array_equal(fast_scores, slow_scores)
        np.testing.assert_array_equal(fast_indices, slow_indices)

    def test_numpy_mask_returns_only_allowed_slots(self):
        mask = np.zeros(N_ROWS, dtype=np.bool_)
        allowed = {3, 17, 64, 100}
        mask[list(allowed)] = True

        _, indices = self.index.search(self.queries, k=N_ROWS, mask=mask)

        self.assertTrue(set(indices.tolist()).issubset(allowed))
        self.assertEqual(indices.shape, (4 * len(allowed),))

    def test_readonly_mask_accepted(self):
        # The buffer is requested read-only; a write-protected array must
        # still take the zero-copy path.
        mask = RNG.random(N_ROWS) < 0.5
        mask.setflags(write=False)

        expected = self.index.search(self.queries, k=8, mask=_DuckMask(mask))
        actual = self.index.search(self.queries, k=8, mask=mask)

        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])

    def test_all_false_mask_matches_legacy_empty_results(self):
        mask = np.zeros(N_ROWS, dtype=np.bool_)

        fast = self.index.search(self.queries, k=8, mask=mask)
        slow = self.index.search(self.queries, k=8, mask=_DuckMask(mask))

        self.assertEqual(fast[0].shape, (0,))
        self.assertEqual(fast[1].shape, (0,))
        np.testing.assert_array_equal(fast[0], slow[0])
        np.testing.assert_array_equal(fast[1], slow[1])


class OrdinalMaskValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = OrdinalIndex(dim=DIM, bits=2)
        self.index.add(_vectors(N_ROWS))
        self.queries = _vectors(1)

    def test_int8_mask_raises_value_error(self):
        mask = np.ones(N_ROWS, dtype=np.int8)
        with self.assertRaisesRegex(ValueError, "dtype bool"):
            self.index.search(self.queries, k=1, mask=mask)

    def test_uint8_mask_raises_value_error(self):
        # Same byte width as bool but a different dtype: must not sneak
        # through the byte-buffer fast path.
        mask = np.ones(N_ROWS, dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "dtype bool"):
            self.index.search(self.queries, k=1, mask=mask)

    def test_two_dimensional_mask_raises_value_error(self):
        mask = np.ones((2, N_ROWS // 2), dtype=np.bool_)
        with self.assertRaisesRegex(ValueError, "1D"):
            self.index.search(self.queries, k=1, mask=mask)

    def test_non_contiguous_mask_raises_value_error(self):
        mask = np.ones(2 * N_ROWS, dtype=np.bool_)[::2]
        self.assertEqual(len(mask), N_ROWS)
        with self.assertRaisesRegex(ValueError, "C-contiguous"):
            self.index.search(self.queries, k=1, mask=mask)

    def test_wrong_length_mask_raises_value_error(self):
        mask = np.ones(N_ROWS + 1, dtype=np.bool_)
        with self.assertRaisesRegex(ValueError, "mask length"):
            self.index.search(self.queries, k=1, mask=mask)

    def test_python_list_mask_still_raises_value_error(self):
        # Back-compat: plain lists were never accepted as masks.
        with self.assertRaisesRegex(ValueError, "NumPy array"):
            self.index.search(self.queries, k=1, mask=[True] * N_ROWS)


class IdMapAllowlistZeroCopyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ids = np.arange(1000, 1000 + N_ROWS, dtype=np.uint64)
        self.index = IdMapIndex(dim=DIM, bits=2)
        self.index.add_with_ids(_vectors(N_ROWS), self.ids)
        self.queries = _vectors(4)

    def test_numpy_allowlist_returns_only_allowed_ids(self):
        allowed = self.ids[RNG.random(N_ROWS) < 0.25]
        self.assertGreater(len(allowed), 0)

        _, found = self.index.search(self.queries, k=N_ROWS, allowlist=allowed)

        self.assertTrue(set(found.tolist()).issubset(set(allowed.tolist())))
        self.assertEqual(found.shape, (4 * len(allowed),))

    def test_readonly_allowlist_accepted(self):
        allowlist = self.ids[:8].copy()
        allowlist.setflags(write=False)

        _, found = self.index.search(self.queries, k=8, allowlist=allowlist)

        self.assertTrue(set(found.tolist()).issubset(set(allowlist.tolist())))

    def test_array_array_allowlist_still_accepted(self):
        # Back-compat: any C-contiguous uint64 buffer was accepted before
        # the zero-copy change, not just ndarrays.
        allowlist = array.array("Q", self.ids[:3].tolist())

        _, found = self.index.search(self.queries, k=8, allowlist=allowlist)

        self.assertTrue(set(found.tolist()).issubset(set(self.ids[:3].tolist())))

    def test_empty_allowlist_returns_empty_results(self):
        scores, found = self.index.search(
            self.queries, k=8, allowlist=np.array([], dtype=np.uint64)
        )
        self.assertEqual(scores.shape, (0,))
        self.assertEqual(found.shape, (0,))

    def test_int64_allowlist_raises_value_error(self):
        allowlist = self.ids[:4].astype(np.int64)
        with self.assertRaisesRegex(ValueError, "uint64"):
            self.index.search(self.queries, k=1, allowlist=allowlist)

    def test_two_dimensional_allowlist_raises_value_error(self):
        allowlist = self.ids[:4].reshape(2, 2)
        with self.assertRaisesRegex(ValueError, "1D"):
            self.index.search(self.queries, k=1, allowlist=allowlist)

    def test_non_contiguous_allowlist_raises_value_error(self):
        allowlist = self.ids[::2]
        with self.assertRaisesRegex(ValueError, "C-contiguous"):
            self.index.search(self.queries, k=1, allowlist=allowlist)

    def test_python_list_allowlist_still_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "NumPy array"):
            self.index.search(self.queries, k=1, allowlist=self.ids[:4].tolist())


class FilteredSearchReleasesGilTests(unittest.TestCase):
    """The filtered paths must stay off the GIL like the unfiltered ones.

    Same strategy as test_gil_release: run the native call in a worker
    thread and require the main thread to keep making progress.
    """

    def _assert_progress(self, iterations: int, elapsed: float, call: str) -> None:
        if elapsed < GIL_MIN_CALL_SECONDS:
            self.skipTest(
                f"native call finished in {elapsed:.3f}s; too fast on this "
                "machine to measure GIL contention reliably"
            )
        self.assertGreaterEqual(
            iterations,
            GIL_MIN_ITERATIONS,
            f"main thread made only {iterations} iterations during a "
            f"{elapsed:.3f}s {call}: the GIL was not released",
        )

    def test_masked_search_releases_gil(self):
        idx = OrdinalIndex(dim=GIL_DIM, bits=2)
        idx.add(_vectors(GIL_N_ROWS, dim=GIL_DIM))
        queries = _vectors(GIL_N_QUERIES, dim=GIL_DIM)
        mask = RNG.random(GIL_N_ROWS) < 0.5

        iterations, elapsed = _main_thread_progress_during(
            lambda: idx.search(queries, k=10, mask=mask)
        )
        self._assert_progress(iterations, elapsed, "masked search")

    def test_allowlist_search_releases_gil(self):
        idx = IdMapIndex(dim=GIL_DIM, bits=2)
        ids = np.arange(GIL_N_ROWS, dtype=np.uint64)
        idx.add_with_ids(_vectors(GIL_N_ROWS, dim=GIL_DIM), ids)
        queries = _vectors(GIL_N_QUERIES, dim=GIL_DIM)
        allowlist = ids[RNG.random(GIL_N_ROWS) < 0.5]

        iterations, elapsed = _main_thread_progress_during(
            lambda: idx.search(queries, k=10, allowlist=allowlist)
        )
        self._assert_progress(iterations, elapsed, "allowlist search")


if __name__ == "__main__":
    unittest.main()
