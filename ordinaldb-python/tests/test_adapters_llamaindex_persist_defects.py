"""Regression tests for LlamaIndex persistence and adapter-contract defects.

Covers:
1. LlamaIndex ``delete_ref_doc`` silently deleting nothing (add/delete
   ref_doc_id stamping symmetry per llama-index conventions).
2. LlamaIndex ``StorageContext.persist(persist_dir=X)`` nesting the store one
   level below ``X`` and reopening at ``X`` silently returning an empty store.
3. Adapter constructors silently creating fresh stores over missing/typo'd/
   invalid paths (path validation matrix in ``adapters/_common.py``).
4. Path-bound stores buffering writes in memory with no signal that data is
   lost without an explicit save.
5. Filters referencing metadata keys that exist on no record silently
   returning zero hits.
6. LangChain ``delete()`` supporting filter-based deletion.
7. Agno ``OrdinalDb`` exposing a public count accessor.
"""

import asyncio
import importlib.util
import tempfile
import unittest
import warnings
from pathlib import Path

try:
    from _warning_hygiene import inoculate_lazy_module_aliases
except ImportError:  # invoked as `python -m unittest tests.test_...`
    from tests._warning_hygiene import inoculate_lazy_module_aliases


ALPHA = [1.0, 0.0, 0.0, 0.0]
BETA = [0.0, 1.0, 0.0, 0.0]
GAMMA = [0.0, 0.0, 1.0, 0.0]

VECTORS = {
    "alpha": ALPHA,
    "beta": BETA,
    "gamma": GAMMA,
}


def _available(module_name):
    return importlib.util.find_spec(module_name) is not None


def _vector_for_text(text):
    return VECTORS.get(text, [0.5, 0.5, 0.5, 0.5])


class FakeLangChainEmbeddings:
    def embed_documents(self, texts):
        return [_vector_for_text(text) for text in texts]

    def embed_query(self, text):
        return _vector_for_text(text)


class FakeAgnoEmbedder:
    def get_embedding(self, text):
        return _vector_for_text(text)


@unittest.skipUnless(_available("llama_index"), "llama-index-core not installed")
class LlamaIndexRefDocDeleteTests(unittest.TestCase):
    """Defect 1: delete_ref_doc() must actually delete the doc's nodes."""

    def test_delete_ref_doc_removes_nodes_end_to_end(self):
        from llama_index.core import Document, StorageContext, VectorStoreIndex
        from llama_index.core.embeddings import MockEmbedding
        from ordinaldb.llama_index import OrdinalDBVectorStore

        store = OrdinalDBVectorStore(dim=8)
        storage_context = StorageContext.from_defaults(vector_store=store)
        documents = [
            Document(text="alpha doc", doc_id="doc-1"),
            Document(text="beta doc", doc_id="doc-2"),
        ]
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=MockEmbedding(embed_dim=8),
        )
        self.assertEqual(len(store.client), 2)

        index.delete_ref_doc("doc-1", delete_from_docstore=True)

        self.assertEqual(len(store.client), 1)
        remaining = list(store.client.iter_records())
        self.assertEqual(remaining[0].metadata.get("ref_doc_id"), "doc-2")

    def test_add_stamps_ref_doc_convention_keys(self):
        from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
        from ordinaldb.llama_index import OrdinalDBVectorStore

        node = TextNode(
            id_="n1",
            text="alpha",
            embedding=ALPHA,
            relationships={
                NodeRelationship.SOURCE: RelatedNodeInfo(node_id="doc-9"),
            },
        )
        store = OrdinalDBVectorStore(dim=4)
        store.add([node])
        record = store.client.get(["n1"])[0]
        # node_to_metadata_dict convention: ref doc id is stamped at the top
        # level under all three compatibility keys.
        self.assertEqual(record.metadata.get("ref_doc_id"), "doc-9")
        self.assertEqual(record.metadata.get("doc_id"), "doc-9")
        self.assertEqual(record.metadata.get("document_id"), "doc-9")

        store.delete("doc-9")
        self.assertEqual(len(store.client), 0)

    def test_user_supplied_ref_doc_id_metadata_still_honored(self):
        from llama_index.core.schema import TextNode
        from ordinaldb.llama_index import OrdinalDBVectorStore

        node = TextNode(
            id_="n1",
            text="alpha",
            metadata={"ref_doc_id": "doc-7"},
            embedding=ALPHA,
        )
        store = OrdinalDBVectorStore(dim=4)
        store.add([node])
        record = store.client.get(["n1"])[0]
        self.assertEqual(record.metadata.get("ref_doc_id"), "doc-7")

        store.delete("doc-7")
        self.assertEqual(len(store.client), 0)


@unittest.skipUnless(_available("llama_index"), "llama-index-core not installed")
class LlamaIndexPersistReloadTests(unittest.TestCase):
    """Defect 2: the idiomatic persist/reload pattern must round-trip."""

    def setUp(self):
        # Building a VectorStoreIndex probes llama-index's LangChain bridge,
        # which transitively imports transformers; keep later assertWarns
        # calls safe from its lazy alias modules.
        inoculate_lazy_module_aliases()

    def _build_index(self, store):
        from llama_index.core import Document, StorageContext, VectorStoreIndex
        from llama_index.core.embeddings import MockEmbedding

        storage_context = StorageContext.from_defaults(vector_store=store)
        documents = [
            Document(text="alpha doc", doc_id="doc-1"),
            Document(text="beta doc", doc_id="doc-2"),
        ]
        VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=MockEmbedding(embed_dim=8),
        )
        return storage_context

    def test_storage_context_persist_then_reopen_at_persist_dir(self):
        from ordinaldb.llama_index import OrdinalDBVectorStore

        with tempfile.TemporaryDirectory() as tmp:
            persist_dir = Path(tmp) / "storage"
            store = OrdinalDBVectorStore(path=persist_dir, dim=8)
            storage_context = self._build_index(store)
            storage_context.persist(persist_dir=str(persist_dir))

            with warnings.catch_warnings():
                warnings.simplefilter("error")
                reopened = OrdinalDBVectorStore(path=persist_dir)
            self.assertEqual(len(reopened.client), 2)

    def test_direct_persist_with_file_shaped_path(self):
        from ordinaldb.llama_index import OrdinalDBVectorStore

        with tempfile.TemporaryDirectory() as tmp:
            persist_dir = Path(tmp) / "storage"
            store = OrdinalDBVectorStore(dim=8)
            self._build_index(store)
            store.persist(persist_path=str(persist_dir / "default__vector_store.json"))

            reopened = OrdinalDBVectorStore(path=persist_dir)
            self.assertEqual(len(reopened.client), 2)

    def test_from_persist_dir_and_from_persist_path(self):
        from ordinaldb.llama_index import OrdinalDBVectorStore

        with tempfile.TemporaryDirectory() as tmp:
            persist_dir = Path(tmp) / "storage"
            store = OrdinalDBVectorStore(path=persist_dir, dim=8)
            storage_context = self._build_index(store)
            storage_context.persist(persist_dir=str(persist_dir))

            from_dir = OrdinalDBVectorStore.from_persist_dir(persist_dir)
            self.assertEqual(len(from_dir.client), 2)

            from_path = OrdinalDBVectorStore.from_persist_path(
                str(persist_dir / "default__vector_store.json")
            )
            self.assertEqual(len(from_path.client), 2)

    def test_from_persist_dir_fails_closed_when_no_store(self):
        from ordinaldb.llama_index import OrdinalDBVectorStore

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                OrdinalDBVectorStore.from_persist_dir(tmp)
            with self.assertRaises(ValueError):
                OrdinalDBVectorStore.from_persist_path(
                    str(Path(tmp) / "default__vector_store.json")
                )

    def test_legacy_nested_store_is_not_silently_shadowed(self):
        from ordinaldb.adapters import AdapterStore
        from ordinaldb.llama_index import OrdinalDBVectorStore

        with tempfile.TemporaryDirectory() as tmp:
            persist_dir = Path(tmp) / "storage"
            # The old buggy layout: a full store nested one level below the
            # persist dir, at the file-shaped persist path.
            inner = AdapterStore(bits=2, dim=4, adapter_name="llama-index")
            inner.add(
                ids=["n1"],
                embeddings=[ALPHA],
                documents=["alpha"],
                metadatas=[{}],
            )
            inner.save(persist_dir / "default__vector_store.json", adapter_name="llama-index")

            # Reopening at the persist dir must not silently return an empty
            # store; it must at least warn about the nested store.
            with self.assertWarnsRegex(UserWarning, "nested"):
                shadowed = OrdinalDBVectorStore(path=persist_dir)
            self.assertEqual(len(shadowed.client), 0)

            # And the conventional loader finds the nested legacy store.
            recovered = OrdinalDBVectorStore.from_persist_dir(persist_dir)
            self.assertEqual(len(recovered.client), 1)


class AdapterPathValidationTests(unittest.TestCase):
    """Defect 3: constructing against a suspicious path must warn or raise."""

    def test_nonexistent_path_creates_silently(self):
        from ordinaldb.adapters import AdapterStore

        with tempfile.TemporaryDirectory() as tmp:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                store = AdapterStore(bits=2, dim=4, path=Path(tmp) / "new-store")
            self.assertEqual(len(store), 0)

    def test_empty_existing_directory_creates_silently(self):
        from ordinaldb.adapters import AdapterStore

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "empty"
            target.mkdir()
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                store = AdapterStore(bits=2, dim=4, path=target)
            self.assertEqual(len(store), 0)

    def test_directory_with_unrelated_content_warns(self):
        from ordinaldb.adapters import AdapterPathWarning, AdapterStore

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "not-a-store"
            target.mkdir()
            (target / "notes.txt").write_text("unrelated")
            with self.assertWarnsRegex(AdapterPathWarning, "no valid OrdinalDB store markers"):
                AdapterStore(bits=2, dim=4, path=target)

    def test_plain_file_path_raises_at_construction(self):
        from ordinaldb.adapters import AdapterStore

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "store-file"
            target.write_text("not a directory")
            with self.assertRaisesRegex(ValueError, "not a directory"):
                AdapterStore(bits=2, dim=4, path=target)

    def test_crash_debris_is_called_out(self):
        from ordinaldb.adapters import AdapterPathWarning, AdapterStore

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "crashed"
            target.mkdir()
            (target / ".id_map.json.tmp-123-456").write_text("{}")
            (target / "vectors").mkdir()
            with self.assertWarnsRegex(AdapterPathWarning, "debris"):
                AdapterStore(bits=2, dim=4, path=target)

    def test_existing_store_not_loaded_warns(self):
        from ordinaldb.adapters import AdapterPathWarning, AdapterStore

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "existing"
            seed = AdapterStore(bits=2, dim=4)
            seed.add(ids=["a"], embeddings=[ALPHA], documents=["alpha"], metadatas=[{}])
            seed.save(target)
            with self.assertWarnsRegex(AdapterPathWarning, "existing"):
                AdapterStore(bits=2, dim=4, path=target)

    @unittest.skipUnless(_available("langchain_core"), "langchain-core not installed")
    def test_langchain_constructor_inherits_path_checks(self):
        from ordinaldb.adapters import AdapterPathWarning
        from ordinaldb.langchain import OrdinalDBVectorStore

        with tempfile.TemporaryDirectory() as tmp:
            foreign = Path(tmp) / "foreign"
            foreign.mkdir()
            (foreign / "README.md").write_text("hello")
            with self.assertWarns(AdapterPathWarning):
                OrdinalDBVectorStore(embedding=FakeLangChainEmbeddings(), path=foreign)

            plain_file = Path(tmp) / "plain-file"
            plain_file.write_text("x")
            with self.assertRaises(ValueError):
                OrdinalDBVectorStore(embedding=FakeLangChainEmbeddings(), path=plain_file)

    @unittest.skipUnless(_available("llama_index"), "llama-index-core not installed")
    def test_llama_index_constructor_inherits_path_checks(self):
        from ordinaldb.adapters import AdapterPathWarning
        from ordinaldb.llama_index import OrdinalDBVectorStore

        with tempfile.TemporaryDirectory() as tmp:
            foreign = Path(tmp) / "foreign"
            foreign.mkdir()
            (foreign / "README.md").write_text("hello")
            with self.assertWarns(AdapterPathWarning):
                OrdinalDBVectorStore(path=foreign, dim=4)

            plain_file = Path(tmp) / "plain-file"
            plain_file.write_text("x")
            with self.assertRaises(ValueError):
                OrdinalDBVectorStore(path=plain_file, dim=4)

    @unittest.skipUnless(_available("agno"), "agno not installed")
    def test_agno_constructor_inherits_path_checks(self):
        from ordinaldb.adapters import AdapterPathWarning
        from ordinaldb.agno import OrdinalDb

        with tempfile.TemporaryDirectory() as tmp:
            foreign = Path(tmp) / "foreign"
            foreign.mkdir()
            (foreign / "README.md").write_text("hello")
            with self.assertWarns(AdapterPathWarning):
                OrdinalDb(path=foreign, dim=4)

            plain_file = Path(tmp) / "plain-file"
            plain_file.write_text("x")
            with self.assertRaises(ValueError):
                OrdinalDb(path=plain_file, dim=4)


@unittest.skipUnless(_available("langchain_core"), "langchain-core not installed")
class UnsavedWriteWarningTests(unittest.TestCase):
    """Defect 4: first unsaved write to a path-bound store must warn."""

    def setUp(self):
        inoculate_lazy_module_aliases()

    def test_first_unsaved_write_warns_once(self):
        from ordinaldb.adapters import UnsavedWritesWarning
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
            unsaved = [w for w in caught if issubclass(w.category, UnsavedWritesWarning)]
            self.assertEqual(len(unsaved), 1)
            self.assertRegex(str(unsaved[0].message), r"save_local\(\)|persist\(\)")
            self.assertIn("memory", str(unsaved[0].message))

    def test_pathless_store_does_not_warn(self):
        from ordinaldb.adapters import UnsavedWritesWarning
        from ordinaldb.langchain import OrdinalDBVectorStore

        store = OrdinalDBVectorStore(embedding=FakeLangChainEmbeddings(), dim=4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            store.add_texts(["alpha"], ids=["a"])
        unsaved = [w for w in caught if issubclass(w.category, UnsavedWritesWarning)]
        self.assertEqual(unsaved, [])

    def test_save_local_rearms_warning_for_next_unsaved_epoch(self):
        # A successful save re-arms the latch: the store is durable now, so
        # the NEXT unsaved mutating write starts a fresh unsaved-batch epoch
        # and warns again (still never twice within one epoch).
        from ordinaldb.adapters import UnsavedWritesWarning
        from ordinaldb.langchain import OrdinalDBVectorStore

        with tempfile.TemporaryDirectory() as tmp:
            store = OrdinalDBVectorStore(
                embedding=FakeLangChainEmbeddings(),
                path=Path(tmp) / "lc",
                dim=4,
            )
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                store.add_texts(["alpha"], ids=["a"])
            store.save_local()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                store.add_texts(["beta"], ids=["b"])
                store.add_texts(["gamma"], ids=["c"])
            unsaved = [w for w in caught if issubclass(w.category, UnsavedWritesWarning)]
            self.assertEqual(len(unsaved), 1)

    def test_loaded_store_warns_on_first_unsaved_write(self):
        from ordinaldb.adapters import UnsavedWritesWarning
        from ordinaldb.langchain import OrdinalDBVectorStore

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lc"
            store = OrdinalDBVectorStore(
                embedding=FakeLangChainEmbeddings(), path=path, dim=4
            )
            store.add_texts(["alpha"], ids=["a"])
            store.save_local()

            loaded = OrdinalDBVectorStore.load_local(path, FakeLangChainEmbeddings())
            with self.assertWarns(UnsavedWritesWarning):
                loaded.add_texts(["beta"], ids=["b"])

    @unittest.skipUnless(_available("llama_index"), "llama-index-core not installed")
    def test_llama_index_add_warns(self):
        from llama_index.core.schema import TextNode
        from ordinaldb.adapters import UnsavedWritesWarning
        from ordinaldb.llama_index import OrdinalDBVectorStore

        with tempfile.TemporaryDirectory() as tmp:
            store = OrdinalDBVectorStore(path=Path(tmp) / "li", dim=4)
            with self.assertWarns(UnsavedWritesWarning):
                store.add([TextNode(id_="a", text="alpha", embedding=ALPHA)])

    @unittest.skipUnless(_available("agno"), "agno not installed")
    def test_agno_warns_unless_auto_save(self):
        from agno.knowledge.document import Document
        from ordinaldb.adapters import UnsavedWritesWarning
        from ordinaldb.agno import OrdinalDb

        with tempfile.TemporaryDirectory() as tmp:
            manual = OrdinalDb(path=Path(tmp) / "agno-manual", dim=4)
            with self.assertWarns(UnsavedWritesWarning):
                manual.upsert(
                    "hash-1",
                    [Document(id="a", content="alpha")],
                    embeddings=[ALPHA],
                )

            auto = OrdinalDb(path=Path(tmp) / "agno-auto", dim=4, auto_save=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                auto.upsert(
                    "hash-1",
                    [Document(id="a", content="alpha")],
                    embeddings=[ALPHA],
                )
            unsaved = [w for w in caught if issubclass(w.category, UnsavedWritesWarning)]
            self.assertEqual(unsaved, [])


@unittest.skipUnless(_available("langchain_core"), "langchain-core not installed")
class UnknownFilterKeyWarningTests(unittest.TestCase):
    """Defect 5: zero-hit filters with keys on no record must warn."""

    def setUp(self):
        inoculate_lazy_module_aliases()

    def _store(self):
        from ordinaldb.langchain import OrdinalDBVectorStore

        store = OrdinalDBVectorStore(embedding=FakeLangChainEmbeddings(), dim=4)
        store.add_texts(
            ["alpha", "beta"],
            metadatas=[
                {"doc_type": "guide", "lang": "en"},
                {"doc_type": "api", "lang": "en"},
            ],
            ids=["a", "b"],
        )
        return store

    def test_unknown_filter_key_warns(self):
        from ordinaldb.adapters import UnknownFilterKeyWarning

        store = self._store()
        with self.assertWarnsRegex(UnknownFilterKeyWarning, "doctype"):
            results = store.similarity_search("alpha", filter={"doctype": "guide"})
        self.assertEqual(results, [])

    def test_known_key_zero_hits_does_not_warn(self):
        from ordinaldb.adapters import UnknownFilterKeyWarning

        store = self._store()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = store.similarity_search("alpha", filter={"doc_type": "no-such"})
        self.assertEqual(results, [])
        unknown = [w for w in caught if issubclass(w.category, UnknownFilterKeyWarning)]
        self.assertEqual(unknown, [])

    def test_mixed_keys_names_only_unknown(self):
        from ordinaldb.adapters import UnknownFilterKeyWarning

        store = self._store()
        with self.assertWarns(UnknownFilterKeyWarning) as ctx:
            store.similarity_search("alpha", filter={"lang": "fr", "doctype": "guide"})
        message = str(ctx.warning)
        self.assertIn("doctype", message)
        self.assertNotIn("'lang'", message)

    def test_empty_store_does_not_warn(self):
        from ordinaldb.adapters import UnknownFilterKeyWarning
        from ordinaldb.langchain import OrdinalDBVectorStore

        store = OrdinalDBVectorStore(embedding=FakeLangChainEmbeddings(), dim=4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = store.similarity_search("alpha", filter={"doctype": "guide"})
        self.assertEqual(results, [])
        unknown = [w for w in caught if issubclass(w.category, UnknownFilterKeyWarning)]
        self.assertEqual(unknown, [])

    @unittest.skipUnless(_available("agno"), "agno not installed")
    def test_agno_filters_inherit_warning(self):
        from agno.knowledge.document import Document
        from ordinaldb.adapters import UnknownFilterKeyWarning
        from ordinaldb.agno import OrdinalDb

        db = OrdinalDb(embedder=FakeAgnoEmbedder(), dim=4)
        db.upsert(
            "hash-1",
            [Document(id="a", content="alpha", meta_data={"group": "x"})],
            embeddings=[ALPHA],
        )
        with self.assertWarnsRegex(UnknownFilterKeyWarning, "grup"):
            db.search("alpha", filters={"grup": "x"})


@unittest.skipUnless(_available("langchain_core"), "langchain-core not installed")
class LangChainFilterDeleteTests(unittest.TestCase):
    """Defect 6: delete() must support filter-based deletion."""

    def _store(self):
        from ordinaldb.langchain import OrdinalDBVectorStore

        store = OrdinalDBVectorStore(embedding=FakeLangChainEmbeddings(), dim=4)
        store.add_texts(
            ["alpha", "beta", "gamma"],
            metadatas=[{"group": "x"}, {"group": "y"}, {"group": "x"}],
            ids=["a", "b", "c"],
        )
        return store

    def test_delete_by_filter_mapping(self):
        store = self._store()
        store.delete(filter={"group": "x"})
        self.assertEqual(store._store.ids(), ["b"])

    def test_delete_filter_matches_nothing_is_noop(self):
        store = self._store()
        store.delete(filter={"group": "no-such-group"})
        self.assertEqual(set(store._store.ids()), {"a", "b", "c"})

    def test_delete_by_callable_filter(self):
        store = self._store()
        store.delete(filter=lambda document: document.metadata.get("group") == "y")
        self.assertEqual(set(store._store.ids()), {"a", "c"})

    def test_delete_with_both_ids_and_filter_raises(self):
        store = self._store()
        with self.assertRaisesRegex(ValueError, "not both"):
            store.delete(ids=["a"], filter={"group": "x"})
        self.assertEqual(set(store._store.ids()), {"a", "b", "c"})

    def test_delete_with_invalid_filter_type_raises(self):
        store = self._store()
        with self.assertRaises(TypeError):
            store.delete(filter="group == x")


@unittest.skipUnless(_available("agno"), "agno not installed")
class AgnoCountAccessorTests(unittest.TestCase):
    """Defect 7: OrdinalDb needs a public count accessor."""

    def test_get_count_len_and_async(self):
        from agno.knowledge.document import Document
        from ordinaldb.agno import OrdinalDb

        db = OrdinalDb(embedder=FakeAgnoEmbedder(), dim=4)
        self.assertEqual(db.get_count(), 0)
        self.assertEqual(len(db), 0)

        db.upsert(
            "hash-1",
            [
                Document(id="a", content="alpha"),
                Document(id="b", content="beta"),
            ],
            embeddings=[ALPHA, BETA],
        )
        self.assertEqual(db.get_count(), 2)
        self.assertEqual(len(db), 2)
        self.assertEqual(asyncio.run(db.async_get_count()), 2)

        db.delete_by_id("a")
        self.assertEqual(db.get_count(), 1)


if __name__ == "__main__":
    unittest.main()
