from pathlib import Path
import tempfile
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
QUERIES = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)


class OrdinalIndexTests(unittest.TestCase):
    def test_construct_add_search(self):
        idx = OrdinalIndex(dim=4, bits=2)
        idx.add(VECTORS)

        scores, indices = idx.search(QUERIES, k=2)

        self.assertEqual(len(idx), 3)
        self.assertEqual(idx.dim(), 4)
        self.assertEqual(idx.bits(), 2)
        self.assertEqual(scores.dtype, np.float32)
        self.assertEqual(indices.dtype, np.int64)
        self.assertEqual(scores.shape, (2,))
        self.assertEqual(indices.shape, (2,))
        self.assertTrue(set(indices.tolist()).issubset({0, 1, 2}))

    def test_lazy_index_dim_is_zero_before_first_add(self):
        # Documented contract: dim() mirrors the Rust accessor and returns
        # 0 (never raises) while a lazy index is unlocked; dim_opt()
        # distinguishes that case.
        idx = OrdinalIndex(bits=2)
        self.assertEqual(idx.dim(), 0)
        self.assertIsNone(idx.dim_opt())

    def test_lazy_index_locks_dim_on_add(self):
        idx = OrdinalIndex(bits=2)
        idx.add(VECTORS)

        self.assertEqual(idx.dim_opt(), 4)
        scores, indices = idx.search(QUERIES, k=10)
        self.assertEqual(scores.shape, (3,))
        self.assertEqual(indices.shape, (3,))

    def test_mask_returns_only_allowed_slots(self):
        idx = OrdinalIndex(dim=4, bits=2)
        idx.add(VECTORS)

        mask = np.array([False, True, False], dtype=np.bool_)
        _, indices = idx.search(QUERIES, k=10, mask=mask)

        self.assertEqual(indices.tolist(), [1])

    def test_invalid_bits_and_alias_rules(self):
        with self.assertRaisesRegex(ValueError, "bits 1, 2, or 4"):
            OrdinalIndex(dim=4, bits=3)

        idx = OrdinalIndex(dim=4, bit_width=2)
        self.assertEqual(idx.bits(), 2)

        with self.assertRaisesRegex(ValueError, "provide only one"):
            OrdinalIndex(dim=4, bits=2, bit_width=2)

    def test_non_contiguous_and_wrong_dim_raise_value_error(self):
        idx = OrdinalIndex(dim=4, bits=2)

        with self.assertRaisesRegex(ValueError, "C-contiguous"):
            idx.add(VECTORS[:, ::-1])

        idx.add(VECTORS)
        wrong_dim = np.ones((1, 8), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "query dim mismatch"):
            idx.search(wrong_dim, k=1)

    def test_write_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "docs.odb"
            idx = OrdinalIndex(dim=4, bits=2)
            idx.add(VECTORS)
            before = idx.search(QUERIES, k=2)

            idx.write(path)
            loaded = OrdinalIndex.load(path)
            after = loaded.search(QUERIES, k=2)

            self.assertEqual(loaded.dim(), 4)
            self.assertEqual(loaded.bits(), 2)
            self.assertEqual(len(loaded), 3)
            np.testing.assert_array_equal(after[0], before[0])
            np.testing.assert_array_equal(after[1], before[1])

    def test_lazy_write_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            idx = OrdinalIndex(bits=2)
            with self.assertRaisesRegex(ValueError, "cannot persist"):
                idx.write(Path(tmp) / "lazy.odb")

    def test_sign_policy_kwarg_mapping(self):
        # dim=64, bits=2 can carry a sign sidecar.
        self.assertTrue(OrdinalIndex(dim=64, bits=2).has_sign_sidecar)
        self.assertTrue(OrdinalIndex(dim=64, bits=2, sign="optional").has_sign_sidecar)
        self.assertTrue(OrdinalIndex(dim=64, bits=2, sign="required").has_sign_sidecar)
        self.assertFalse(OrdinalIndex(dim=64, bits=2, sign="disabled").has_sign_sidecar)

        # dim=4 cannot: "optional" constructs without one, "required" raises.
        self.assertFalse(OrdinalIndex(dim=4, bits=2).has_sign_sidecar)
        with self.assertRaisesRegex(ValueError, "sign policy Required"):
            OrdinalIndex(dim=4, bits=2, sign="required")
        with self.assertRaisesRegex(ValueError, "bits 4 never supports"):
            OrdinalIndex(bits=4, sign="required")

        with self.assertRaisesRegex(ValueError, "disabled"):
            OrdinalIndex(dim=64, bits=2, sign="bogus")

    def test_load_policy_can_reopen_intentionally_unsigned_bundle(self):
        vectors = np.zeros((2, 64), dtype=np.float32)
        vectors[0, 0] = 1.0
        vectors[1, 1] = 1.0

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unsigned.odb"
            idx = OrdinalIndex(dim=64, bits=2, sign="disabled")
            idx.add(vectors)
            idx.write(path)

            with self.assertRaisesRegex(ValueError, "requires a sign sidecar"):
                OrdinalIndex.load(path)

            backup = path.parent / f".{path.name}.bak-1-1"
            path.rename(backup)
            loaded = OrdinalIndex.load(path, sign="any")
            self.assertFalse(loaded.has_sign_sidecar)
            self.assertEqual(len(loaded), 2)
            self.assertTrue(path.is_dir())
            self.assertFalse(backup.exists())

            with self.assertRaisesRegex(ValueError, "require_if_supported"):
                OrdinalIndex.load(path, sign="bogus")

    def test_sign_required_lazy_index_raises_on_first_add(self):
        idx = OrdinalIndex(bits=2, sign="required")
        with self.assertRaisesRegex(ValueError, "sign policy Required"):
            idx.add(VECTORS)  # dim=4 cannot carry a sidecar
        self.assertIsNone(idx.dim_opt())
        self.assertEqual(len(idx), 0)


class IdMapIndexTests(unittest.TestCase):
    def test_add_search_delete_and_allowlist(self):
        idx = IdMapIndex(dim=4, bits=2)
        ids = np.array([1001, 1002, 1003], dtype=np.uint64)
        idx.add_with_ids(VECTORS, ids)

        _, found = idx.search(QUERIES, k=3)
        self.assertTrue(set(found.tolist()).issubset({1001, 1002, 1003}))

        allowlist = np.array([1002], dtype=np.uint64)
        _, found = idx.search(QUERIES, k=10, allowlist=allowlist)
        self.assertEqual(found.tolist(), [1002])

        self.assertTrue(idx.remove(1002))
        self.assertFalse(idx.contains(1002))
        _, found = idx.search(QUERIES, k=10)
        self.assertNotIn(1002, found.tolist())

    def test_lazy_index_dim_is_zero_before_first_add(self):
        idx = IdMapIndex(bits=2)
        self.assertEqual(idx.dim(), 0)
        self.assertIsNone(idx.dim_opt())

    def test_sign_policy_kwarg_mapping(self):
        self.assertTrue(IdMapIndex(dim=64, bits=2, sign="required").has_sign_sidecar)
        self.assertFalse(IdMapIndex(dim=64, bits=2, sign="disabled").has_sign_sidecar)
        self.assertFalse(IdMapIndex(dim=4, bits=2).has_sign_sidecar)
        with self.assertRaisesRegex(ValueError, "sign policy Required"):
            IdMapIndex(dim=4, bits=2, sign="required")
        with self.assertRaisesRegex(ValueError, "bits 4 never supports"):
            IdMapIndex(bits=4, sign="required")
        with self.assertRaisesRegex(ValueError, "disabled"):
            IdMapIndex(dim=64, bits=2, sign="bogus")

    def test_load_policy_can_reopen_intentionally_unsigned_bundle(self):
        vectors = np.zeros((2, 64), dtype=np.float32)
        vectors[0, 0] = 1.0
        vectors[1, 1] = 1.0
        ids = np.array([101, 202], dtype=np.uint64)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unsigned_ids.odb"
            idx = IdMapIndex(dim=64, bits=2, sign="disabled")
            idx.add_with_ids(vectors, ids)
            idx.write(path)

            with self.assertRaisesRegex(ValueError, "requires a sign sidecar"):
                IdMapIndex.load(path)

            backup = path.parent / f".{path.name}.bak-1-1"
            path.rename(backup)
            loaded = IdMapIndex.load(path, sign="any")
            self.assertFalse(loaded.has_sign_sidecar)
            self.assertTrue(loaded.contains(101))
            self.assertTrue(loaded.contains(202))
            self.assertTrue(path.is_dir())
            self.assertFalse(backup.exists())

    def test_duplicate_ids_rejected_without_partial_mutation(self):
        idx = IdMapIndex(dim=4, bits=2)
        ids = np.array([42, 42], dtype=np.uint64)

        with self.assertRaisesRegex(ValueError, "already stored"):
            idx.add_with_ids(VECTORS[:2], ids)

        self.assertEqual(len(idx), 0)

    def test_missing_allowlist_id_raises_value_error(self):
        idx = IdMapIndex(dim=4, bits=2)
        idx.add_with_ids(VECTORS, np.array([1, 2, 3], dtype=np.uint64))

        with self.assertRaisesRegex(ValueError, "not present"):
            idx.search(QUERIES, k=1, allowlist=np.array([99], dtype=np.uint64))

    def test_write_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "docs_ids.odb"
            idx = IdMapIndex(dim=4, bits=2)
            ids = np.array([1001, 1002, 1003], dtype=np.uint64)
            idx.add_with_ids(VECTORS, ids)
            before = idx.search(QUERIES, k=3)

            idx.write(path)
            loaded = IdMapIndex.load(path)
            after = loaded.search(QUERIES, k=3)

            self.assertEqual(loaded.dim(), 4)
            self.assertEqual(loaded.bits(), 2)
            self.assertEqual(len(loaded), 3)
            np.testing.assert_array_equal(after[0], before[0])
            np.testing.assert_array_equal(after[1], before[1])

            self.assertTrue(loaded.remove(1002))
            _, found = loaded.search(QUERIES, k=3)
            self.assertNotIn(1002, found.tolist())

    def test_wrong_bundle_loader_raises_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "docs_ids.odb"
            idx = IdMapIndex(dim=4, bits=2)
            idx.add_with_ids(VECTORS, np.array([1, 2, 3], dtype=np.uint64))
            idx.write(path)

            with self.assertRaisesRegex(ValueError, "IdMapIndex::load"):
                OrdinalIndex.load(path)


if __name__ == "__main__":
    unittest.main()
