"""Regression tests: malformed search input must raise catchable Python errors.

The native ``search`` paths previously routed through unchecked Rust code
(``search_with_mask`` / ``search_with_allowlist``) that ``panic!``-ed on
NaN/Inf/out-of-range query values, surfacing in Python as a raw
``pyo3_runtime.PanicException`` instead of a catchable ``ValueError``.

Every test in this module asserts that malformed input raises ``ValueError``
with a helpful message -- never a panic. Runs under both pytest and the CI
``unittest`` discovery.
"""

import unittest

import numpy as np

from ordinaldb import IdMapIndex, OrdinalIndex

VECTORS = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)
IDS = np.array([11, 22, 33], dtype=np.uint64)
GOOD_QUERY = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)


def _query_with(value: float) -> np.ndarray:
    query = GOOD_QUERY.copy()
    query[0, 2] = value
    return query


class OrdinalIndexSearchValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = OrdinalIndex(dim=4, bits=2)
        self.index.add(VECTORS)

    def test_nan_query_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "invalid query value"):
            self.index.search(_query_with(float("nan")), k=1)

    def test_inf_query_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "invalid query value"):
            self.index.search(_query_with(float("inf")), k=1)

    def test_negative_inf_query_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "invalid query value"):
            self.index.search(_query_with(float("-inf")), k=1)

    def test_overflow_magnitude_query_raises_value_error(self):
        # Values with |value| >= 1e16 are rejected by the core index.
        with self.assertRaisesRegex(ValueError, "invalid query value"):
            self.index.search(_query_with(1e17), k=1)

    def test_nan_query_with_mask_raises_value_error(self):
        mask = np.array([True, False, True], dtype=np.bool_)
        with self.assertRaisesRegex(ValueError, "invalid query value"):
            self.index.search(_query_with(float("nan")), k=1, mask=mask)

    def test_wrong_dim_query_raises_value_error(self):
        wrong_dim = np.ones((1, 8), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "query dim mismatch"):
            self.index.search(wrong_dim, k=1)

    def test_wrong_dtype_query_raises_value_error(self):
        float64_query = GOOD_QUERY.astype(np.float64)
        with self.assertRaisesRegex(ValueError, "float32"):
            self.index.search(float64_query, k=1)

    def test_valid_query_still_succeeds(self):
        scores, indices = self.index.search(GOOD_QUERY, k=2)
        self.assertEqual(scores.shape, (2,))
        self.assertEqual(indices.shape, (2,))


class IdMapIndexSearchValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = IdMapIndex(dim=4, bits=2)
        self.index.add_with_ids(VECTORS, IDS)

    def test_nan_query_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "invalid query value"):
            self.index.search(_query_with(float("nan")), k=1)

    def test_inf_query_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "invalid query value"):
            self.index.search(_query_with(float("inf")), k=1)

    def test_nan_query_with_allowlist_raises_value_error(self):
        allowlist = np.array([22], dtype=np.uint64)
        with self.assertRaisesRegex(ValueError, "invalid query value"):
            self.index.search(_query_with(float("nan")), k=1, allowlist=allowlist)

    def test_wrong_dim_query_raises_value_error(self):
        wrong_dim = np.ones((1, 8), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "query dim mismatch"):
            self.index.search(wrong_dim, k=1)

    def test_wrong_dtype_query_raises_value_error(self):
        float64_query = GOOD_QUERY.astype(np.float64)
        with self.assertRaisesRegex(ValueError, "float32"):
            self.index.search(float64_query, k=1)

    def test_never_present_allowlist_id_raises_value_error(self):
        allowlist = np.array([999], dtype=np.uint64)
        with self.assertRaisesRegex(ValueError, "not present"):
            self.index.search(GOOD_QUERY, k=1, allowlist=allowlist)

    def test_stale_allowlist_id_raises_value_error(self):
        # An id that existed but was removed must not panic the search path.
        self.assertTrue(self.index.remove(22))
        stale = np.array([22], dtype=np.uint64)
        with self.assertRaisesRegex(ValueError, "not present"):
            self.index.search(GOOD_QUERY, k=1, allowlist=stale)

    def test_valid_query_still_succeeds(self):
        scores, found = self.index.search(GOOD_QUERY, k=3)
        self.assertEqual(scores.shape, (3,))
        self.assertTrue(set(found.tolist()).issubset({11, 22, 33}))


if __name__ == "__main__":
    unittest.main()
