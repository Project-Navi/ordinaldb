"""Regression tests for adapter warning-attribution and concurrency defects.

Covers:
1. ``UnsavedWritesWarning`` attribution: the warning must point at the first
   caller outside the ordinaldb package (and outside framework wrapper
   frames such as ``VectorStore.from_documents``), not at library internals.
2. LangChain ``OrdinalDBVectorStore`` public count accessors
   (``get_count()``/``__len__``), parity with the Agno adapter.
3. ``UnsavedWritesWarning`` epoch re-arm: warn once per unsaved-batch epoch;
   a successful save re-arms the latch so the next unsaved write warns again.
4. Haystack serialize-before-save trap: ``from_dict`` against a path with no
   saved store must warn instead of silently reconstructing an empty store.
5. Haystack filter parity with ``UnknownFilterKeyWarning``: zero-hit filters
   whose ``meta.``-prefixed keys exist on no stored document must warn.
6. ``save()`` racing a concurrent ``adapter.redb`` reader (``adapter gc``,
   diagnostics): post-commit redb lock contention must not surface as a
   false failure that invites double-insert-on-retry.
"""

import importlib.util
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

try:
    from _warning_hygiene import inoculate_lazy_module_aliases
except ImportError:  # invoked as `python -m unittest tests.test_...`
    from tests._warning_hygiene import inoculate_lazy_module_aliases

# Must be set before importing haystack -- telemetry reads these at import
# time. Keeps these tests offline/network-free like the rest of the suite.
os.environ.setdefault("HAYSTACK_TELEMETRY_ENABLED", "False")
os.environ.setdefault("HAYSTACK_DISABLE_TELEMETRY", "1")
os.environ.setdefault("POSTHOG_DISABLED", "1")


ALPHA = [1.0, 0.0, 0.0, 0.0]
BETA = [0.0, 1.0, 0.0, 0.0]

VECTORS = {
    "alpha": ALPHA,
    "beta": BETA,
}

REDB_CONTENTION_MESSAGE = "Database already open. Cannot acquire lock."


def _available(module_name):
    return importlib.util.find_spec(module_name) is not None


def _vector_for_text(text):
    return VECTORS.get(text, [0.5, 0.5, 0.5, 0.5])


def _fixed_embedding(seed, dim):
    """Deterministic pseudo-embedding; no model download needed."""
    values = [((seed * 9301 + 49297 * i) % 233280) / 233280.0 for i in range(dim)]
    norm = sum(v * v for v in values) ** 0.5 or 1.0
    return [v / norm for v in values]


class FakeLangChainEmbeddings:
    def embed_documents(self, texts):
        return [_vector_for_text(text) for text in texts]

    def embed_query(self, text):
        return _vector_for_text(text)


@unittest.skipUnless(_available("langchain_core"), "langchain-core not installed")
class UnsavedWritesWarningAttributionTests(unittest.TestCase):
    """Fix 1: the warning's .filename must be the caller's file."""

    def _unsaved(self, caught):
        from ordinaldb.adapters import UnsavedWritesWarning

        return [w for w in caught if issubclass(w.category, UnsavedWritesWarning)]

    def test_direct_add_texts_attributes_to_this_file(self):
        from ordinaldb.langchain import OrdinalDBVectorStore

        with tempfile.TemporaryDirectory() as tmp:
            store = OrdinalDBVectorStore(
                embedding=FakeLangChainEmbeddings(),
                path=Path(tmp) / "lc",
                dim=4,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                store.add_texts(["alpha"], ids=["a"])
        unsaved = self._unsaved(caught)
        self.assertEqual(len(unsaved), 1)
        self.assertEqual(unsaved[0].filename, __file__)

    def test_from_documents_attributes_to_this_file(self):
        from langchain_core.documents import Document
        from ordinaldb.langchain import OrdinalDBVectorStore

        with tempfile.TemporaryDirectory() as tmp:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                OrdinalDBVectorStore.from_documents(
                    [Document(page_content="alpha")],
                    FakeLangChainEmbeddings(),
                    path=Path(tmp) / "lc",
                    dim=4,
                )
        unsaved = self._unsaved(caught)
        self.assertEqual(len(unsaved), 1)
        # The extra langchain_core VectorStore.from_documents frame must not
        # steal the attribution (and neither must ordinaldb's own from_texts).
        self.assertEqual(unsaved[0].filename, __file__)


@unittest.skipUnless(_available("langchain_core"), "langchain-core not installed")
class LangChainCountAccessorTests(unittest.TestCase):
    """Fix 2: public count accessors, parity with the Agno adapter."""

    def test_get_count_len_and_after_delete(self):
        from ordinaldb.langchain import OrdinalDBVectorStore

        store = OrdinalDBVectorStore(embedding=FakeLangChainEmbeddings(), dim=4)
        self.assertEqual(store.get_count(), 0)
        self.assertEqual(len(store), 0)

        store.add_texts(["alpha", "beta"], ids=["a", "b"])
        self.assertEqual(store.get_count(), 2)
        self.assertEqual(len(store), 2)

        store.delete(ids=["a"])
        self.assertEqual(store.get_count(), 1)
        self.assertEqual(len(store), 1)


class UnsavedWritesEpochTests(unittest.TestCase):
    """Fix 3: warn once per unsaved-batch epoch, re-armed by each save."""

    def _unsaved(self, caught):
        from ordinaldb.adapters import UnsavedWritesWarning

        return [w for w in caught if issubclass(w.category, UnsavedWritesWarning)]

    def test_adapter_store_save_rearms_the_latch(self):
        from ordinaldb.adapters import AdapterStore

        with tempfile.TemporaryDirectory() as tmp:
            store = AdapterStore(bits=2, dim=4, path=Path(tmp) / "store")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                store.add(ids=["a"], embeddings=[ALPHA], documents=["alpha"], metadatas=[{}])
                store.add(ids=["b"], embeddings=[BETA], documents=["beta"], metadatas=[{}])
            self.assertEqual(len(self._unsaved(caught)), 1)

            store.save()

            # A mutating write after a successful save starts a new unsaved
            # epoch and must warn again -- including deletes.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                store.delete(["a"])
                store.delete(["b"])
            self.assertEqual(len(self._unsaved(caught)), 1)

    @unittest.skipUnless(_available("langchain_core"), "langchain-core not installed")
    def test_langchain_epochs_across_save_local_and_persist(self):
        from ordinaldb.langchain import OrdinalDBVectorStore

        with tempfile.TemporaryDirectory() as tmp:
            store = OrdinalDBVectorStore(
                embedding=FakeLangChainEmbeddings(),
                path=Path(tmp) / "lc",
                dim=4,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                store.add_texts(["alpha"], ids=["a"])
                store.add_texts(["beta"], ids=["b"])
            self.assertEqual(len(self._unsaved(caught)), 1)

            store.save_local()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                store.add_texts(["gamma"], ids=["c"])
                store.add_texts(["delta"], ids=["d"])
            self.assertEqual(len(self._unsaved(caught)), 1)

            store.persist()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                store.add_texts(["epsilon"], ids=["e"])
            self.assertEqual(len(self._unsaved(caught)), 1)


@unittest.skipUnless(_available("haystack"), "haystack-ai not installed")
class HaystackSerializeBeforeSaveTests(unittest.TestCase):
    """Fix 4 (haystack): to_dict() without save() must not round-trip silently."""

    def setUp(self):
        inoculate_lazy_module_aliases()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        self.addCleanup(self._tmpdir.cleanup)

    def _build_pipeline(self, path, *, save):
        from haystack import Document, Pipeline
        from ordinaldb.haystack import OrdinalDocumentStore, OrdinalEmbeddingRetriever

        store = OrdinalDocumentStore(path=path, dim=8, bits=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # UnsavedWritesWarning is not under test
            store.write_documents(
                [
                    Document(
                        id="d1",
                        content="alpha doc",
                        meta={"group": "x"},
                        embedding=_fixed_embedding(1, 8),
                    ),
                    Document(
                        id="d2",
                        content="beta doc",
                        meta={"group": "y"},
                        embedding=_fixed_embedding(2, 8),
                    ),
                ]
            )
        if save:
            store.save()
        pipeline = Pipeline()
        pipeline.add_component(
            "retriever", OrdinalEmbeddingRetriever(document_store=store, top_k=3)
        )
        return pipeline

    def test_from_dict_without_save_warns_about_missing_saved_data(self):
        from haystack import Pipeline
        from ordinaldb.adapters import AdapterPathWarning

        path = self.tmp / "hs-unsaved"
        serialized = self._build_pipeline(path, save=False).to_dict()

        with self.assertWarnsRegex(
            AdapterPathWarning, r"did you save\(\) before to_dict\(\)"
        ):
            restored = Pipeline.from_dict(serialized)
        restored_store = restored.get_component("retriever").document_store
        self.assertEqual(restored_store.count_documents(), 0)

    def test_from_dict_after_save_is_silent_and_data_is_present(self):
        from haystack import Pipeline
        from ordinaldb.adapters import AdapterPathWarning

        path = self.tmp / "hs-saved"
        serialized = self._build_pipeline(path, save=True).to_dict()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            restored = Pipeline.from_dict(serialized)
        path_warnings = [w for w in caught if issubclass(w.category, AdapterPathWarning)]
        self.assertEqual(path_warnings, [])
        restored_store = restored.get_component("retriever").document_store
        self.assertEqual(restored_store.count_documents(), 2)

    def test_from_dict_with_pathless_store_warns(self):
        from ordinaldb.adapters import AdapterPathWarning
        from ordinaldb.haystack import OrdinalDocumentStore

        store = OrdinalDocumentStore(dim=8, bits=2)
        serialized = store.to_dict()
        self.assertIsNone(serialized["init_parameters"]["path"])
        with self.assertWarnsRegex(AdapterPathWarning, r"save\("):
            OrdinalDocumentStore.from_dict(serialized)


@unittest.skipUnless(_available("haystack"), "haystack-ai not installed")
class HaystackUnknownFilterKeyParityTests(unittest.TestCase):
    """Fix 5 (haystack): typo'd meta filter keys must warn on zero hits."""

    def setUp(self):
        inoculate_lazy_module_aliases()

    def _build_store(self):
        from haystack import Document
        from ordinaldb.haystack import OrdinalDocumentStore

        store = OrdinalDocumentStore(dim=8, bits=2)
        store.write_documents(
            [
                Document(
                    id="d1",
                    content="alpha doc",
                    meta={"doc_type": "guide", "lang": "en"},
                    embedding=_fixed_embedding(1, 8),
                ),
                Document(
                    id="d2",
                    content="beta doc",
                    meta={"doc_type": "api", "lang": "en"},
                    embedding=_fixed_embedding(2, 8),
                ),
            ]
        )
        return store

    def test_typoed_meta_key_warns_via_filter_documents(self):
        from ordinaldb.adapters import UnknownFilterKeyWarning

        store = self._build_store()
        with self.assertWarnsRegex(UnknownFilterKeyWarning, "doctype"):
            results = store.filter_documents(
                {"field": "meta.doctype", "operator": "==", "value": "guide"}
            )
        self.assertEqual(results, [])

    def test_typoed_meta_key_warns_via_search_by_embedding(self):
        from ordinaldb.adapters import UnknownFilterKeyWarning

        store = self._build_store()
        with self.assertWarnsRegex(UnknownFilterKeyWarning, "doctype"):
            results = store.search_by_embedding(
                _fixed_embedding(1, 8),
                top_k=5,
                filters={"field": "meta.doctype", "operator": "==", "value": "guide"},
            )
        self.assertEqual(results, [])

    def test_valid_key_zero_hits_does_not_warn(self):
        from ordinaldb.adapters import UnknownFilterKeyWarning

        store = self._build_store()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = store.filter_documents(
                {"field": "meta.doc_type", "operator": "==", "value": "no-such"}
            )
        self.assertEqual(results, [])
        unknown = [w for w in caught if issubclass(w.category, UnknownFilterKeyWarning)]
        self.assertEqual(unknown, [])

    def test_nested_conditions_name_only_the_unknown_key(self):
        from ordinaldb.adapters import UnknownFilterKeyWarning

        store = self._build_store()
        with self.assertWarns(UnknownFilterKeyWarning) as ctx:
            store.filter_documents(
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "meta.lang", "operator": "==", "value": "fr"},
                        {"field": "meta.doctype", "operator": "==", "value": "guide"},
                    ],
                }
            )
        message = str(ctx.warning)
        self.assertIn("doctype", message)
        self.assertNotIn("'lang'", message)


class SaveRacingRedbReaderTests(unittest.TestCase):
    """Fix 6: adapter gc / diagnostics racing a live writer's save()."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.target = Path(self._tmpdir.name) / "store"
        self.addCleanup(self._tmpdir.cleanup)

    def _fresh_store(self):
        from ordinaldb.adapters import AdapterStore

        store = AdapterStore(bits=2, dim=4, path=self.target)
        store.warn_unsaved_writes = False
        store.add(ids=["a"], embeddings=[ALPHA], documents=["alpha"], metadatas=[{}])
        return store

    def _patched_state_store(self, *, commit_first, fail_verify=False):
        """Simulate a concurrent redb reader (e.g. ``adapter gc``).

        ``commit_first=True`` models the post-commit race: the snapshot
        commits durably, then the post-commit reacquire fails with redb's
        native lock-contention error. ``commit_first=False`` models the
        reader winning before the commit. ``fail_verify=True`` additionally
        keeps the reader holding the database open, so even the recovery
        read cannot get in.
        """
        from ordinaldb.adapters import _common

        real = _common._AdapterStateStore

        class ContendedStateStore:
            acquire_writer_lock = real.acquire_writer_lock
            load_legacy_snapshot = real.load_legacy_snapshot

            @staticmethod
            def verify(path, expected_adapter=None):
                if fail_verify:
                    raise ValueError(REDB_CONTENTION_MESSAGE)
                return real.verify(path, expected_adapter)

            @staticmethod
            def write_legacy_snapshot_with_existing_lock(*args, **kwargs):
                if commit_first:
                    real.write_legacy_snapshot_with_existing_lock(*args, **kwargs)
                raise ValueError(REDB_CONTENTION_MESSAGE)

        return mock.patch.object(_common, "_AdapterStateStore", ContendedStateStore)

    def test_post_commit_contention_degrades_to_warning_not_false_failure(self):
        from ordinaldb.adapters import AdapterStore

        store = self._fresh_store()
        with self._patched_state_store(commit_first=True):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                store.save()  # must NOT raise: the commit is already durable
        messages = [
            str(w.message)
            for w in caught
            if issubclass(w.category, UserWarning)
            and "Database already open" in str(w.message)
        ]
        self.assertEqual(len(messages), 1)
        self.assertRegex(messages[0], r"(?i)durable")
        self.assertRegex(messages[0], r"(?i)do not re-add")

        # The data really is durable...
        loaded = AdapterStore.load(self.target)
        self.assertEqual(loaded.ids(), ["a"])

        # ...and the surviving writer's commit token advanced, so it can
        # keep saving normally once the reader is gone.
        store.add(ids=["b"], embeddings=[BETA], documents=["beta"], metadatas=[{}])
        store.save()
        self.assertEqual(AdapterStore.load(self.target).ids(), ["a", "b"])

    def test_pre_commit_contention_still_raises_and_says_retry_is_safe(self):
        from ordinaldb.adapters import AdapterStoreError

        store = self._fresh_store()
        with self._patched_state_store(commit_first=False):
            with self.assertRaisesRegex(
                AdapterStoreError, r"NOT published"
            ) as ctx:
                store.save()
        self.assertIn(REDB_CONTENTION_MESSAGE, str(ctx.exception))
        self.assertFalse((self.target / "adapter.redb").exists())

        # Retrying the save once the reader is gone is safe and works.
        store.save()
        from ordinaldb.adapters import AdapterStore

        self.assertEqual(AdapterStore.load(self.target).ids(), ["a"])

    def test_unresolvable_contention_raises_do_not_blindly_retry(self):
        from ordinaldb.adapters import AdapterStoreError

        store = self._fresh_store()
        with self._patched_state_store(commit_first=True, fail_verify=True):
            with self.assertRaisesRegex(
                AdapterStoreError, r"determine whether the commit published"
            ) as ctx:
                store.save()
        self.assertIn(REDB_CONTENTION_MESSAGE, str(ctx.exception))
        self.assertRegex(str(ctx.exception), r"(?i)re-load")


if __name__ == "__main__":
    unittest.main()
