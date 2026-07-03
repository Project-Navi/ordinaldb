"""Regression tests for OrdinalDB's Haystack adapter integration defects.

These cover integration defects surfaced by driving the adapter through a
real Haystack 2.x ``Pipeline`` rather than through the store API alone:

* BLOCKER: `OrdinalEmbeddingRetriever` was not a Haystack `Component`, so
  `pipeline.add_component()` failed with
  `AttributeError: '__haystack_input__'`. Haystack's
  `Pipeline.add_component()` requires the `__haystack_input__`/
  `__haystack_output__` sockets that only exist once a class carries
  Haystack's `@component` decoration; the retriever must work in a real
  `Pipeline` directly, the way Haystack's own `InMemoryEmbeddingRetriever`
  does.
* MINOR: unknown filter operators (e.g. '~=', 'XOR') leaked a bare
  `KeyError` from `haystack.utils.filters` instead of `haystack.errors.FilterError`.
* MINOR: `DuplicatePolicy.FAIL` raised a plain `ValueError`, which
  Haystack-idiomatic code catching `haystack.document_stores.errors.DuplicateDocumentError`
  does not catch (it is not a `ValueError` subclass).
* PAPER-CUT: `DuplicatePolicy.NONE` duplicate errors said "policy FAIL"
  instead of naming the policy the caller actually passed.

The filter-operator matrix below exercises every documented Haystack filter
operator (comparison and logical) against the adapter.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

# Must be set before importing haystack -- telemetry reads these at import
# time. Keeps these tests offline/network-free like the rest of the suite.
os.environ.setdefault("HAYSTACK_TELEMETRY_ENABLED", "False")
os.environ.setdefault("HAYSTACK_DISABLE_TELEMETRY", "1")
os.environ.setdefault("POSTHOG_DISABLED", "1")


def _available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _fixed_embedding(seed: int, dim: int) -> list[float]:
    """Deterministic pseudo-embedding; no model download needed."""
    values = [((seed * 9301 + 49297 * i) % 233280) / 233280.0 for i in range(dim)]
    norm = sum(v * v for v in values) ** 0.5 or 1.0
    return [v / norm for v in values]


class _IsolatedHomeTestCase(unittest.TestCase):
    """Points HOME at a temp dir for the duration of each test.

    Matches the pattern used by the existing Haystack smoke tests so the
    adapter store's on-disk state never touches the real home directory.
    """

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmpdir.name)
        self._old_home = os.environ.get("HOME")
        os.environ["HOME"] = str(self.tmp)

    def tearDown(self) -> None:
        if self._old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = self._old_home
        self._tmpdir.cleanup()


@unittest.skipUnless(_available("haystack"), "haystack-ai not installed")
class OrdinalEmbeddingRetrieverPipelineTests(_IsolatedHomeTestCase):
    """Defect 1 (BLOCKER): the retriever must be a real Haystack Component."""

    @staticmethod
    def _make_fixed_text_embedder(vector: list[float]):
        """Tiny fake embedder returning a fixed vector; no model download."""
        from haystack import component

        @component
        class FixedTextEmbedder:
            @component.output_types(embedding=list[float])
            def run(self, text: str) -> dict[str, list[float]]:
                return {"embedding": vector}

        return FixedTextEmbedder()

    def test_add_component_accepts_the_retriever_without_attribute_error(self):
        # The historical failure mode: Pipeline.add_component() requires
        # __haystack_input__/__haystack_output__, which only exist once the
        # class carries Haystack's @component decoration.
        from ordinaldb.haystack import OrdinalDocumentStore, OrdinalEmbeddingRetriever
        from haystack import Pipeline

        store = OrdinalDocumentStore(dim=4, bits=2)
        retriever = OrdinalEmbeddingRetriever(document_store=store, top_k=2)
        pipeline = Pipeline()
        pipeline.add_component("retriever", retriever)
        self.assertTrue(hasattr(retriever, "__haystack_input__"))
        self.assertTrue(hasattr(retriever, "__haystack_output__"))

    def test_retriever_runs_end_to_end_inside_a_real_pipeline(self):
        from haystack import Document, Pipeline
        from ordinaldb.haystack import OrdinalDocumentStore, OrdinalEmbeddingRetriever

        dim = 384
        query_vector = _fixed_embedding(1, dim)

        store = OrdinalDocumentStore(dim=dim, bits=2)
        store.write_documents(
            [
                Document(
                    id="near",
                    content="near the query vector",
                    meta={"group": "keep"},
                    embedding=query_vector,
                ),
                Document(
                    id="far",
                    content="far from the query vector",
                    meta={"group": "drop"},
                    embedding=_fixed_embedding(999, dim),
                ),
            ]
        )

        pipeline = Pipeline()
        pipeline.add_component("embedder", self._make_fixed_text_embedder(query_vector))
        pipeline.add_component(
            "retriever", OrdinalEmbeddingRetriever(document_store=store, top_k=5)
        )
        pipeline.connect("embedder.embedding", "retriever.query_embedding")

        result = pipeline.run(
            {
                "embedder": {"text": "irrelevant -- the fake embedder ignores it"},
                "retriever": {
                    "filters": {"field": "meta.group", "operator": "==", "value": "keep"}
                },
            }
        )

        documents = result["retriever"]["documents"]
        self.assertEqual([doc.id for doc in documents], ["near"])

    def test_document_store_to_dict_uses_a_fully_qualified_type_name(self):
        # Haystack's generic deserialization (default_from_dict) only
        # auto-reconstructs a nested init-parameter object (like the
        # retriever's document_store) if its serialized "type" contains a
        # dot, e.g. "ordinaldb.haystack.OrdinalDocumentStore". A bare class
        # name silently leaves the raw dict in place instead of an instance.
        from ordinaldb.haystack import OrdinalDocumentStore

        store = OrdinalDocumentStore(dim=4, bits=2)
        data = store.to_dict()
        self.assertEqual(data["type"], "ordinaldb.haystack.OrdinalDocumentStore")

    def test_retriever_round_trips_through_pipeline_serialization(self):
        from haystack import Pipeline
        from ordinaldb.haystack import OrdinalDocumentStore, OrdinalEmbeddingRetriever

        path = self.tmp / "hs-pipeline-store"
        store = OrdinalDocumentStore(path=path, dim=4, bits=2)
        store.save(path)

        pipeline = Pipeline()
        pipeline.add_component(
            "retriever", OrdinalEmbeddingRetriever(document_store=store, top_k=3)
        )

        serialized = pipeline.to_dict()
        restored = Pipeline.from_dict(serialized)

        restored_retriever = restored.get_component("retriever")
        self.assertIsInstance(restored_retriever, OrdinalEmbeddingRetriever)
        self.assertIsInstance(restored_retriever.document_store, OrdinalDocumentStore)
        self.assertEqual(str(restored_retriever.document_store._store.path), str(path))
        self.assertEqual(restored_retriever.top_k, 3)


@unittest.skipUnless(_available("haystack"), "haystack-ai not installed")
class HaystackFilterOperatorMatrixTests(_IsolatedHomeTestCase):
    """Exercises the full documented Haystack filter-operator matrix."""

    def _build_store(self):
        from haystack import Document
        from ordinaldb.haystack import OrdinalDocumentStore

        def doc(doc_id, **meta):
            return Document(
                id=doc_id,
                content=f"content for {doc_id}",
                meta=meta,
                embedding=_fixed_embedding(hash(doc_id) % 1000, 8),
            )

        docs = [
            doc("f1", area="billing", rank=1, created="2026-01-10", tag="a"),
            doc("f2", area="billing", rank=3, created="2026-03-01", tag="b"),
            doc("f3", area="auth", rank=4, created="2026-05-20", tag="a"),
            doc("f4", area="auth", rank=2, created="2026-02-15", tag="c"),
            doc("f5", area="search", rank=4, created="2026-06-30", tag="b"),
            doc("f6", area="search", rank=1, created="2026-04-05", tag=None),
        ]
        store = OrdinalDocumentStore(path=str(self.tmp / "filters"), dim=8, bits=2)
        store.write_documents(docs)
        return store

    def test_filter_operator_matrix(self):
        store = self._build_store()

        cases = [
            ("== on meta field", {"field": "meta.area", "operator": "==", "value": "billing"}, ["f1", "f2"]),
            ("!= on meta field", {"field": "meta.area", "operator": "!=", "value": "billing"}, ["f3", "f4", "f5", "f6"]),
            ("> on numeric meta", {"field": "meta.rank", "operator": ">", "value": 2}, ["f2", "f3", "f5"]),
            (">= on numeric meta", {"field": "meta.rank", "operator": ">=", "value": 3}, ["f2", "f3", "f5"]),
            ("< on numeric meta", {"field": "meta.rank", "operator": "<", "value": 2}, ["f1", "f6"]),
            ("<= on numeric meta", {"field": "meta.rank", "operator": "<=", "value": 1}, ["f1", "f6"]),
            (
                ">= on ISO date-string meta",
                {"field": "meta.created", "operator": ">=", "value": "2026-04-05"},
                ["f3", "f5", "f6"],
            ),
            ("in on meta field", {"field": "meta.area", "operator": "in", "value": ["billing", "search"]}, ["f1", "f2", "f5", "f6"]),
            (
                "not in on meta field",
                {"field": "meta.area", "operator": "not in", "value": ["billing", "search"]},
                ["f3", "f4"],
            ),
            (
                "AND of two conditions",
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "meta.area", "operator": "==", "value": "search"},
                        {"field": "meta.rank", "operator": ">=", "value": 4},
                    ],
                },
                ["f5"],
            ),
            (
                "OR of two conditions",
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "meta.area", "operator": "==", "value": "billing"},
                        {"field": "meta.rank", "operator": "==", "value": 4},
                    ],
                },
                ["f1", "f2", "f3", "f5"],
            ),
            (
                "NOT wrapping a condition",
                {"operator": "NOT", "conditions": [{"field": "meta.area", "operator": "==", "value": "billing"}]},
                ["f3", "f4", "f5", "f6"],
            ),
            (
                "3-way nested AND(OR(..), ==)",
                {
                    "operator": "AND",
                    "conditions": [
                        {
                            "operator": "OR",
                            "conditions": [
                                {"field": "meta.area", "operator": "==", "value": "auth"},
                                {"field": "meta.area", "operator": "==", "value": "search"},
                            ],
                        },
                        {"field": "meta.rank", "operator": ">=", "value": 4},
                    ],
                },
                ["f3", "f5"],
            ),
            (
                "field without 'meta.' prefix (legacy shorthand)",
                {"field": "area", "operator": "==", "value": "billing"},
                ["f1", "f2"],
            ),
            ("filter on real Document field 'id' (not meta)", {"field": "id", "operator": "==", "value": "f3"}, ["f3"]),
            ("filter matching zero docs", {"field": "meta.area", "operator": "==", "value": "nonexistent-area"}, []),
            (
                "== against a None meta value (tag=None on f6)",
                {"field": "meta.tag", "operator": "==", "value": None},
                ["f6"],
            ),
        ]

        for label, filt, expected in cases:
            with self.subTest(label=label):
                got = sorted(doc.id for doc in store.filter_documents(filt))
                self.assertEqual(got, sorted(expected))

    def test_filter_matrix_also_holds_via_search_by_embedding_retriever_path(self):
        store = self._build_store()
        query = _fixed_embedding(1, 8)

        for label, filt, expected in [
            ("== on meta field", {"field": "meta.area", "operator": "==", "value": "billing"}, ["f1", "f2"]),
            ("in on meta field", {"field": "meta.area", "operator": "in", "value": ["billing", "search"]}, ["f1", "f2", "f5", "f6"]),
        ]:
            with self.subTest(label=label):
                hits = store.search_by_embedding(query, top_k=10, filters=filt)
                self.assertEqual(sorted(doc.id for doc in hits), sorted(expected))

    def test_unknown_comparison_operator_raises_filter_error_not_key_error(self):
        from haystack.errors import FilterError

        store = self._build_store()
        with self.assertRaisesRegex(FilterError, "~="):
            store.filter_documents({"field": "meta.area", "operator": "~=", "value": "billing"})

    def test_unknown_logical_operator_raises_filter_error_not_key_error(self):
        from haystack.errors import FilterError

        store = self._build_store()
        with self.assertRaisesRegex(FilterError, "XOR"):
            store.filter_documents({"operator": "XOR", "conditions": []})

    def test_unknown_operator_via_search_by_embedding_also_raises_filter_error(self):
        from haystack.errors import FilterError

        store = self._build_store()
        with self.assertRaises(FilterError):
            store.search_by_embedding(
                _fixed_embedding(1, 8),
                filters={"field": "meta.area", "operator": "~=", "value": "billing"},
            )


@unittest.skipUnless(_available("haystack"), "haystack-ai not installed")
class HaystackDuplicatePolicyErrorTests(_IsolatedHomeTestCase):
    """Defects 3 (MINOR) and 4 (PAPER-CUT): duplicate-id error handling."""

    def test_duplicate_document_error_is_not_a_value_error(self):
        from haystack.document_stores.errors import DuplicateDocumentError

        # This is exactly the trap: idiomatic Haystack code catches
        # DuplicateDocumentError, not ValueError. If it were a ValueError
        # subclass this assertion (and the adapter's old behavior) would be
        # fine, but it deliberately is not.
        self.assertFalse(issubclass(DuplicateDocumentError, ValueError))

    def test_fail_policy_raises_duplicate_document_error(self):
        from haystack import Document
        from haystack.document_stores.errors import DuplicateDocumentError
        from haystack.document_stores.types import DuplicatePolicy
        from ordinaldb.haystack import OrdinalDocumentStore

        store = OrdinalDocumentStore(dim=4, bits=2)
        doc = Document(id="a", content="alpha", embedding=[1.0, 0.0, 0.0, 0.0])
        store.write_documents([doc], policy=DuplicatePolicy.FAIL)

        with self.assertRaisesRegex(DuplicateDocumentError, r"duplicate document IDs for policy FAIL.*a"):
            store.write_documents([doc], policy=DuplicatePolicy.FAIL)

    def test_none_policy_error_message_names_none_not_fail(self):
        from haystack import Document
        from haystack.document_stores.errors import DuplicateDocumentError
        from haystack.document_stores.types import DuplicatePolicy
        from ordinaldb.haystack import OrdinalDocumentStore

        store = OrdinalDocumentStore(dim=4, bits=2)
        doc = Document(id="a", content="alpha", embedding=[1.0, 0.0, 0.0, 0.0])
        store.write_documents([doc], policy=DuplicatePolicy.FAIL)

        with self.assertRaisesRegex(DuplicateDocumentError, r"duplicate document IDs for policy NONE.*a"):
            store.write_documents([doc], policy=DuplicatePolicy.NONE)

        # And the FAIL-policy message must still say FAIL, not NONE.
        with self.assertRaisesRegex(DuplicateDocumentError, r"duplicate document IDs for policy FAIL.*a"):
            store.write_documents([doc], policy=DuplicatePolicy.FAIL)


if __name__ == "__main__":
    unittest.main()
