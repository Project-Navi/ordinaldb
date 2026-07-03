"""Idiomatic-path E2E tests on real embedding fixtures.

Every test here exists because a synthetic-vector or direct-API test
missed a real bug: the float-canonicalization corruption only triggered
on real MiniLM bit-patterns, delete_ref_doc was a silent no-op only via
the idiomatic LlamaIndex path, and the persist trap only manifested
through StorageContext. Vectors come from the committed fixture corpus
(tests/fixtures/real_embeddings/, see its generate.py for provenance);
no model download happens at test time.

Frameworks are optional test dependencies: each class skips cleanly when
its framework isn't installed, matching the repo's existing convention.
"""

import json
import math
import struct
import tempfile
import unittest
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "real_embeddings"
DIM = 384


def _load_matrix(name: str, rows: int) -> list[list[float]]:
    raw = (FIXTURES / name).read_bytes()
    assert len(raw) == rows * DIM * 4, f"{name} shape drifted"
    flat = struct.unpack(f"<{rows * DIM}f", raw)
    return [list(flat[r * DIM : (r + 1) * DIM]) for r in range(rows)]


CORPUS = json.loads((FIXTURES / "texts.json").read_text())
DOC_VECS = _load_matrix("minilm_docs_f32.bin", 32)
QUERY_VECS = _load_matrix("minilm_queries_f32.bin", 8)
ADVERSARIAL = json.loads((FIXTURES / "adversarial_floats.json").read_text())
TEXT_TO_VEC = {
    d["text"]: DOC_VECS[i] for i, d in enumerate(CORPUS["documents"])
} | {q["text"]: QUERY_VECS[i] for i, q in enumerate(CORPUS["queries"])}


def _vec_for(text: str) -> list[float]:
    hit = TEXT_TO_VEC.get(text)
    if hit is not None:
        return hit
    # Frameworks may prepend metadata to the embedded text (LlamaIndex's
    # metadata_mode=EMBED does); match the corpus text contained within.
    for corpus_text, vec in TEXT_TO_VEC.items():
        if corpus_text in text:
            return vec
    raise AssertionError(
        f"test text not in fixture corpus (first 60 chars): {text[:60]!r}"
    )


class LlamaIndexIdiomaticPath(unittest.TestCase):
    """VectorStoreIndex round trip — the path that hid two BLOCKERs."""

    @classmethod
    def setUpClass(cls):
        try:
            import llama_index.core  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("llama-index-core not installed")

    def _fixture_embed_model(self):
        from llama_index.core.embeddings import BaseEmbedding

        class FixtureEmbedding(BaseEmbedding):
            def _get_text_embedding(self, text):
                return _vec_for(text)

            def _get_query_embedding(self, query):
                return _vec_for(query)

            async def _aget_query_embedding(self, query):
                return _vec_for(query)

        return FixtureEmbedding(model_name="fixture-minilm")

    def _documents(self, count=8):
        from llama_index.core import Document

        return [
            Document(
                text=d["text"],
                doc_id=d["id"],
                metadata={"domain": d["domain"]},
            )
            for d in CORPUS["documents"][:count]
        ]

    def test_from_documents_query_delete_persist_reload(self):
        from llama_index.core import StorageContext, VectorStoreIndex
        from ordinaldb.adapters import (
            AdapterPathWarning,
            UnknownFilterKeyWarning,
            UnsavedWritesWarning,
        )
        from ordinaldb.llama_index import OrdinalDBVectorStore

        with tempfile.TemporaryDirectory() as tmp:
            store_dir = Path(tmp) / "paperstore"
            vector_store = OrdinalDBVectorStore(
                path=store_dir, dim=DIM, bits=2
            )
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store
            )
            index = VectorStoreIndex.from_documents(
                self._documents(),
                storage_context=storage_context,
                embed_model=self._fixture_embed_model(),
            )

            retriever = index.as_retriever(similarity_top_k=3)
            hits = retriever.retrieve(
                "how do I persist an index to disk and verify it"
            )
            self.assertEqual(len(hits), 3)
            self.assertEqual(hits[0].metadata["domain"], "docs")

            # delete_ref_doc must actually delete (previously a silent no-op).
            before = len(vector_store._store)
            index.delete_ref_doc("docs-001", delete_from_docstore=True)
            self.assertLess(len(vector_store._store), before)

            # Idiomatic persist: reopening at the SAME dir must find the
            # data (previously reopened silently empty because the store
            # nested one level below the persist dir). Scoped to
            # OrdinalDB's own adapter warning categories (not a blanket
            # simplefilter("error")) so an unrelated warning from an
            # upstream library during StorageContext.persist can't spuriously
            # fail this test; the intent is that persist must not emit an
            # AdapterPathWarning/UnsavedWritesWarning/UnknownFilterKeyWarning,
            # not that persist must be warning-free in general.
            with warnings.catch_warnings():
                for category in (
                    AdapterPathWarning,
                    UnknownFilterKeyWarning,
                    UnsavedWritesWarning,
                ):
                    warnings.filterwarnings("error", category=category)
                storage_context.persist(persist_dir=str(store_dir))
            reopened = OrdinalDBVectorStore.from_persist_dir(str(store_dir))
            self.assertGreater(len(reopened._store), 0)

    def test_real_embedding_in_metadata_survives_save(self):
        # The historical corruption trigger: a real embedding cached in node
        # metadata. Fails on pre-float_roundtrip builds with 'metadata
        # table does not match payload'.
        from ordinaldb.adapters import AdapterStore

        with tempfile.TemporaryDirectory() as tmp:
            store = AdapterStore(
                path=Path(tmp) / "meta-float", dim=DIM, bits=2
            )
            store.add(
                ids=["m-1"],
                embeddings=[DOC_VECS[0]],
                documents=[CORPUS["documents"][0]["text"]],
                metadatas=[{"embedding": DOC_VECS[0], "domain": "docs"}],
            )
            store.save()  # previously raised AdapterStoreError here
            reloaded = AdapterStore.load(Path(tmp) / "meta-float")
            [record] = reloaded.get(["m-1"])
            self.assertEqual(record.metadata["embedding"], DOC_VECS[0])


class LangChainIdiomaticPath(unittest.TestCase):
    """as_retriever + save_local/load_local with warning contracts."""

    @classmethod
    def setUpClass(cls):
        try:
            import langchain_core  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("langchain-core not installed")

    def _embeddings(self):
        from langchain_core.embeddings import Embeddings

        class FixtureEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return [_vec_for(t) for t in texts]

            def embed_query(self, text):
                return _vec_for(text)

        return FixtureEmbeddings()

    def test_retriever_save_load_filter_delete(self):
        from ordinaldb.langchain import OrdinalDBVectorStore

        docs = CORPUS["documents"][8:16]  # tickets domain
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tickets"
            store = OrdinalDBVectorStore(
                embedding=self._embeddings(), path=path, dim=DIM, bits=2
            )

            # Unsaved-writes warning contract: first mutating add warns.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                store.add_texts(
                    [d["text"] for d in docs],
                    metadatas=[
                        {"product_area": d["product_area"], "severity": d["severity"]}
                        for d in docs
                    ],
                    ids=[d["id"] for d in docs],
                )
            self.assertTrue(
                any("save" in str(w.message).lower() for w in caught),
                f"expected an unsaved-writes warning, got {[str(w.message) for w in caught]}",
            )

            retriever = store.as_retriever(search_kwargs={"k": 2})
            hits = retriever.invoke(
                "customers being billed twice for the same invoice"
            )
            self.assertEqual(hits[0].metadata["product_area"], "billing")

            # Filtered search stays top-k-within-filter on real vectors.
            filtered = store.similarity_search(
                "customers being billed twice for the same invoice",
                k=3,
                filter={"product_area": "authentication"},
            )
            self.assertTrue(
                all(h.metadata["product_area"] == "authentication" for h in filtered)
            )

            store.save_local()
            reopened = OrdinalDBVectorStore.load_local(
                path, embeddings=self._embeddings()
            )
            billing_query = "customers being billed twice for the same invoice"
            self.assertEqual(
                len(reopened.similarity_search(billing_query, k=8)), 8
            )

            # Filter-based delete (previously unsupported, now public API).
            reopened.delete(filter={"product_area": "billing"})
            remaining = reopened.similarity_search(billing_query, k=8)
            self.assertTrue(
                all(h.metadata["product_area"] != "billing" for h in remaining)
            )

    def test_adversarial_floats_in_metadata_roundtrip(self):
        # Every serialization boundary must survive the harvested and
        # curated pathological float values.
        from ordinaldb.langchain import OrdinalDBVectorStore

        hard_floats = (
            ADVERSARIAL["from_real_embeddings_f32_promoted"][:16]
            + ADVERSARIAL["curated_ieee754"]
        )
        # NaN/inf are rejected elsewhere; keep finite values only here.
        hard_floats = [f for f in hard_floats if f == f and abs(f) != float("inf")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hard-floats"
            store = OrdinalDBVectorStore(
                embedding=self._embeddings(), path=path, dim=DIM, bits=2
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                store.add_texts(
                    [CORPUS["documents"][0]["text"]],
                    metadatas=[{"hard": hard_floats, "ts": 1783014388.4746075}],
                    ids=["hard-1"],
                )
                store.save_local()
            reopened = OrdinalDBVectorStore.load_local(
                path, embeddings=self._embeddings()
            )
            hit = reopened.similarity_search(
                CORPUS["documents"][0]["text"], k=1
            )[0]
            self.assertEqual(hit.metadata["hard"], hard_floats)
            self.assertEqual(hit.metadata["ts"], 1783014388.4746075)

            # `==` cannot distinguish -0.0 from 0.0 (-0.0 == 0.0 is True in
            # Python), so the assertEqual above would pass vacuously even if
            # a regression canonicalized signed zero. adversarial_floats.json
            # curates -0.0 specifically to catch that; check sign bits
            # explicitly for every roundtripped zero.
            roundtripped = hit.metadata["hard"]
            for original, back in zip(hard_floats, roundtripped):
                if original == 0.0:
                    self.assertEqual(
                        math.copysign(1.0, back),
                        math.copysign(1.0, original),
                        f"signed zero not preserved: {original!r} -> {back!r}",
                    )


class AgnoIdiomaticPath(unittest.TestCase):
    """Agno OrdinalDb lifecycle with real vectors."""

    @classmethod
    def setUpClass(cls):
        try:
            import agno  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("agno not installed")

    def test_lifecycle_count_and_persistence(self):
        from ordinaldb.agno import OrdinalDb

        class FixtureEmbedder:
            dimensions = DIM

            def get_embedding(self, text):
                return _vec_for(text)

            def get_embedding_and_usage(self, text):
                return _vec_for(text), None

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory"
            db = OrdinalDb(
                path=str(path), embedder=FixtureEmbedder(), auto_save=True
            )
            db.create()
            from agno.knowledge.document import Document as AgnoDocument

            notes = CORPUS["documents"][16:24]
            db.insert(
                content_hash="batch-1",
                documents=[
                    AgnoDocument(
                        content=d["text"],
                        id=d["id"],
                        meta_data={"topic": d["topic"]},
                        embedding=_vec_for(d["text"]),
                    )
                    for d in notes
                ],
            )
            self.assertEqual(db.get_count(), 8)
            self.assertEqual(len(db), 8)

            results = db.search(
                "why is my database connection pool running out", limit=2
            )
            self.assertEqual(results[0].meta_data["topic"], "postgres")

            reopened = OrdinalDb(
                path=str(path), embedder=FixtureEmbedder(), auto_save=True
            )
            self.assertEqual(reopened.get_count(), 8)


if __name__ == "__main__":
    unittest.main()
