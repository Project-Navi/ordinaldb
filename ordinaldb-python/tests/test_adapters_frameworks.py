import asyncio
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest


ALPHA = [1.0, 0.0, 0.0, 0.0]
BETA = [0.0, 1.0, 0.0, 0.0]
GAMMA = [0.0, 0.0, 1.0, 0.0]
DELTA = [0.0, 0.0, 0.0, 1.0]


def _available(module_name):
    return importlib.util.find_spec(module_name) is not None


class FakeLangChainEmbeddings:
    def embed_documents(self, texts):
        return [_vector_for_text(text) for text in texts]

    def embed_query(self, text):
        return _vector_for_text(text)


class FakeAgnoEmbedder:
    def get_embedding(self, text):
        return _vector_for_text(text)


@unittest.skipUnless(_available("langchain_core"), "langchain-core not installed")
class LangChainAdapterSmokeTests(unittest.TestCase):
    def test_add_search_filter_persist_load_and_unsupported_scores(self):
        from collections import UserDict
        from ordinaldb.langchain import OrdinalDBVectorStore

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lc"
            with self.assertRaisesRegex(
                ValueError,
                "langchain adapter requires dim divisible by 4 when bits=2; got dim=6",
            ):
                OrdinalDBVectorStore(embedding=FakeLangChainEmbeddings(), dim=6)

            store = OrdinalDBVectorStore(embedding=FakeLangChainEmbeddings(), dim=4)
            self.assertEqual(
                store.add_texts(
                    ["alpha", "beta"],
                    metadatas=[{"group": "x"}, {"group": "y"}],
                    ids=["a", "b"],
                ),
                ["a", "b"],
            )

            docs = store.similarity_search("alpha", k=2, filter={"group": "y"})
            self.assertEqual([doc.page_content for doc in docs], ["beta"])
            mapping_docs = store.similarity_search("alpha", k=2, filter=UserDict({"group": "y"}))
            self.assertEqual([doc.page_content for doc in mapping_docs], ["beta"])
            self.assertEqual(
                store.add_texts(
                    (text for text in ["gamma"]),
                    metadatas=({"group": "z"} for _ in range(1)),
                    ids=(id_ for id_ in ["c"]),
                ),
                ["c"],
            )
            self.assertEqual(store.get_by_ids(["c"])[0].metadata["group"], "z")

            store.save_local(path)
            loaded = OrdinalDBVectorStore.load_local(path, embeddings=FakeLangChainEmbeddings())
            self.assertEqual(loaded.get_by_ids(["a"])[0].page_content, "alpha")
            self.assertEqual(loaded.get_by_ids(["missing"]), [])
            self.assertEqual(
                loaded.add_texts(
                    ["beta updated"],
                    metadatas=[{"group": "updated"}],
                    ids=["a"],
                ),
                ["a"],
            )
            self.assertEqual(loaded.get_by_ids(["a"])[0].page_content, "beta updated")
            with self.assertRaisesRegex(ValueError, "duplicate string IDs"):
                loaded.add_texts(
                    ["duplicate first", "duplicate second"],
                    metadatas=[{"group": "dup"}, {"group": "dup"}],
                    ids=["dup", "dup"],
                )
            callable_docs = loaded.similarity_search(
                "beta",
                k=2,
                filter=lambda doc: doc.id == "a"
                and doc.page_content == "beta updated"
                and doc.metadata["group"] == "updated",
            )
            self.assertEqual([doc.id for doc in callable_docs], ["a"])
            self.assertEqual(loaded.similarity_search("beta", k=2, filter=lambda doc: False), [])
            search_calls = []
            original_search = loaded._store.search_by_vector

            def recording_search(*args, **kwargs):
                search_calls.append(kwargs)
                return original_search(*args, **kwargs)

            loaded._store.search_by_vector = recording_search
            try:
                loaded.similarity_search("beta", k=2, filter=lambda doc: doc.id == "a")
            finally:
                loaded._store.search_by_vector = original_search
            self.assertIsNotNone(search_calls[-1].get("allowed_u64_ids"))
            self.assertIsNone(search_calls[-1].get("filter"))
            self.assertIsNone(loaded.delete(["missing"]))

            alias_path = Path(tmp) / "lc-alias"
            loaded.dump(alias_path)
            alias_loaded = OrdinalDBVectorStore.load(
                alias_path,
                embedding=FakeLangChainEmbeddings(),
            )
            self.assertEqual(alias_loaded.get_by_ids(["a"])[0].page_content, "beta updated")
            with self.assertRaisesRegex(ValueError, "not a directory"):
                OrdinalDBVectorStore.load_local(
                    Path(tmp) / "missing",
                    embeddings=FakeLangChainEmbeddings(),
                )

            with self.assertRaisesRegex(NotImplementedError, "normalized relevance"):
                loaded.similarity_search_with_relevance_scores("alpha")
            with self.assertRaisesRegex(NotImplementedError, "MMR"):
                loaded.max_marginal_relevance_search("alpha")

    def test_production_conformance_snapshot_mutation_and_filter_route(self):
        from ordinaldb.langchain import OrdinalDBVectorStore

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lc-conformance"
            store = OrdinalDBVectorStore(embedding=FakeLangChainEmbeddings(), dim=4)
            store.add_texts(
                ["alpha", "beta"],
                metadatas=[{"group": "x", "rank": 1}, {"group": "y", "rank": 2}],
                ids=["a", "b"],
            )

            routed = []
            original_search = store._store.search_by_vector

            def recording_search(*args, **kwargs):
                routed.append(kwargs)
                return original_search(*args, **kwargs)

            store._store.search_by_vector = recording_search
            try:
                filtered = store.similarity_search_by_vector(ALPHA, k=2, filter={"group": "y"})
            finally:
                store._store.search_by_vector = original_search
            self.assertEqual([doc.id for doc in filtered], ["b"])
            self.assertEqual(routed[-1].get("filter"), {"group": "y"})
            self.assertIsNone(routed[-1].get("allowed_u64_ids"))

            store.save_local(path)
            snapshot = OrdinalDBVectorStore.load_local(
                path,
                embeddings=FakeLangChainEmbeddings(),
            )
            store.add_texts(
                ["alpha updated"],
                metadatas=[{"group": "updated", "rank": 3}],
                ids=["a"],
            )
            store.save_local(path)

            self.assertEqual(snapshot.get_by_ids(["a"])[0].page_content, "alpha")
            fresh = OrdinalDBVectorStore.load_local(path, embeddings=FakeLangChainEmbeddings())
            self.assertEqual(fresh.get_by_ids(["a"])[0].page_content, "alpha updated")
            fresh.delete(["b"])
            fresh.save_local(path)
            after_delete = OrdinalDBVectorStore.load_local(
                path,
                embeddings=FakeLangChainEmbeddings(),
            )
            self.assertEqual(after_delete.get_by_ids(["b"]), [])
            self.assertEqual(snapshot.get_by_ids(["b"])[0].page_content, "beta")
            with self.assertRaisesRegex(ValueError, "portable filter value .* JSON scalar"):
                after_delete.similarity_search_by_vector(ALPHA, k=2, filter={"group": ["x"]})


@unittest.skipUnless(_available("llama_index"), "llama-index-core not installed")
class LlamaIndexAdapterSmokeTests(unittest.TestCase):
    def test_add_query_persist_load_and_reject_non_default_mode(self):
        from llama_index.core.schema import TextNode
        from llama_index.core.vector_stores.types import (
            FilterCondition,
            FilterOperator,
            MetadataFilter,
            MetadataFilters,
            VectorStoreQuery,
            VectorStoreQueryMode,
        )
        from ordinaldb.llama_index import OrdinalDBVectorStore, _llama_filters_to_dict

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "li"
            store = OrdinalDBVectorStore(dim=4)
            store.add(
                [
                    TextNode(
                        id_="a",
                        text="alpha",
                        metadata={"group": "x", "rank": 1, "ref_doc_id": "doc-1"},
                        embedding=ALPHA,
                    ),
                    TextNode(
                        id_="b",
                        text="beta",
                        metadata={"group": "y", "rank": 2, "ref_doc_id": "doc-1"},
                        embedding=BETA,
                    ),
                    TextNode(
                        id_="c",
                        text="gamma",
                        metadata={"group": "z", "rank": 3, "ref_doc_id": "doc-2"},
                        embedding=GAMMA,
                    ),
                ]
            )
            result = store.query(VectorStoreQuery(query_embedding=ALPHA, similarity_top_k=3))
            self.assertEqual(set(result.ids), {"a", "b", "c"})
            node_a = next(node for node in result.nodes if node.node_id == "a")
            self.assertEqual(node_a.get_content(), "alpha")
            self.assertEqual(node_a.metadata["group"], "x")
            self.assertEqual(node_a.embedding, ALPHA)
            self.assertEqual(store.get_nodes([]), [])
            store.delete_nodes(node_ids=[])
            self.assertEqual({node.node_id for node in store.get_nodes()}, {"a", "b", "c"})

            eq_filters = MetadataFilters(
                filters=[MetadataFilter(key="group", value="x", operator=FilterOperator.EQ)]
            )

            self.assertEqual(
                _llama_filters_to_dict(eq_filters),
                {"group": "x"},
            )
            query_calls = []
            original_query_search = store.client.search_by_vector

            def recording_query_search(*args, **kwargs):
                query_calls.append(kwargs)
                return original_query_search(*args, **kwargs)

            store.client.search_by_vector = recording_query_search
            try:
                store.query(
                    VectorStoreQuery(
                        query_embedding=ALPHA,
                        similarity_top_k=1,
                        filters=eq_filters,
                    )
                )
            finally:
                store.client.search_by_vector = original_query_search
            self.assertTrue(callable(query_calls[-1].get("filter")))
            self.assertIsNone(query_calls[-1].get("allowed_u64_ids"))
            comparison_result = store.query(
                VectorStoreQuery(
                    query_embedding=ALPHA,
                    similarity_top_k=3,
                    filters=MetadataFilters(
                        filters=[
                            MetadataFilter(
                                key="rank",
                                value=2,
                                operator=FilterOperator.GTE,
                            )
                        ]
                    ),
                )
            )
            self.assertEqual(set(comparison_result.ids), {"b", "c"})
            empty_filter_result = store.query(
                VectorStoreQuery(
                    query_embedding=ALPHA,
                    similarity_top_k=3,
                    filters=MetadataFilters(
                        filters=[
                            MetadataFilter(
                                key="group",
                                value="none",
                                operator=FilterOperator.EQ,
                            )
                        ]
                    ),
                )
            )
            self.assertEqual(empty_filter_result.ids, [])
            or_nodes = store.get_nodes(
                filters=MetadataFilters(
                    filters=[
                        MetadataFilter(key="group", value="x", operator=FilterOperator.EQ),
                        MetadataFilter(key="group", value="z", operator=FilterOperator.EQ),
                    ],
                    condition=FilterCondition.OR,
                )
            )
            self.assertEqual({node.node_id for node in or_nodes}, {"a", "c"})
            not_nodes = store.get_nodes(
                filters=MetadataFilters(
                    filters=[
                        MetadataFilter(key="group", value="z", operator=FilterOperator.EQ)
                    ],
                    condition=FilterCondition.NOT,
                )
            )
            self.assertEqual({node.node_id for node in not_nodes}, {"a", "b"})

            store.persist(path)
            loaded = OrdinalDBVectorStore(path=path)
            loaded_node = loaded.get_nodes(["a"])[0]
            self.assertEqual(loaded_node.get_content(), "alpha")
            self.assertEqual(loaded_node.metadata["rank"], 1)
            self.assertEqual(loaded.get_nodes(["missing"]), [])
            loaded.delete_nodes(node_ids=["missing"])
            self.assertEqual({node.node_id for node in loaded.get_nodes()}, {"a", "b", "c"})
            loaded.delete("doc-1")
            self.assertEqual(loaded.client.get(["a", "b"]), [])
            self.assertEqual(loaded.client.get(["c"])[0].document, "gamma")
            loaded.delete_nodes(node_ids=["c"])
            self.assertEqual(loaded.get_nodes(), [])
            loaded.add(
                [
                    TextNode(
                        id_="d",
                        text="delta",
                        metadata={"group": "x", "rank": 4},
                        embedding=ALPHA,
                    )
                ]
            )
            loaded.clear()
            self.assertEqual(loaded.get_nodes(), [])

            with self.assertRaisesRegex(NotImplementedError, "filter operator"):
                store.query(
                    VectorStoreQuery(
                        query_embedding=ALPHA,
                        similarity_top_k=1,
                        filters=MetadataFilters(
                            filters=[
                                MetadataFilter(
                                    key="group",
                                    value="x",
                                    operator=FilterOperator.TEXT_MATCH,
                                )
                            ]
                        ),
                    )
                )

            mmr = getattr(VectorStoreQueryMode, "MMR", None)
            if mmr is not None:
                with self.assertRaisesRegex(NotImplementedError, "DEFAULT"):
                    loaded.query(
                        VectorStoreQuery(
                            query_embedding=ALPHA,
                            similarity_top_k=1,
                            mode=mmr,
                        )
                    )

    def test_production_conformance_snapshot_mutation_and_filter_route(self):
        from llama_index.core.schema import TextNode
        from llama_index.core.vector_stores.types import (
            FilterOperator,
            MetadataFilter,
            MetadataFilters,
            VectorStoreQuery,
        )
        from ordinaldb.llama_index import OrdinalDBVectorStore

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "li-conformance"
            store = OrdinalDBVectorStore(dim=4)
            store.add(
                [
                    TextNode(
                        id_="a",
                        text="alpha",
                        metadata={"group": "x", "rank": 1},
                        embedding=ALPHA,
                    ),
                    TextNode(
                        id_="b",
                        text="beta",
                        metadata={"group": "y", "rank": 2},
                        embedding=BETA,
                    ),
                ]
            )

            eq_filters = MetadataFilters(
                filters=[MetadataFilter(key="group", value="y", operator=FilterOperator.EQ)]
            )
            routed = []
            original_search = store.client.search_by_vector

            def recording_search(*args, **kwargs):
                routed.append(kwargs)
                return original_search(*args, **kwargs)

            store.client.search_by_vector = recording_search
            try:
                result = store.query(
                    VectorStoreQuery(
                        query_embedding=ALPHA,
                        similarity_top_k=2,
                        filters=eq_filters,
                    )
                )
            finally:
                store.client.search_by_vector = original_search
            self.assertEqual(result.ids, ["b"])
            self.assertTrue(callable(routed[-1].get("filter")))
            self.assertIsNone(routed[-1].get("allowed_u64_ids"))

            store.persist(path)
            snapshot = OrdinalDBVectorStore(path=path)
            store.delete_nodes(node_ids=["a"])
            store.add(
                [
                    TextNode(
                        id_="a",
                        text="alpha updated",
                        metadata={"group": "updated", "rank": 3},
                        embedding=DELTA,
                    )
                ]
            )
            store.persist(path)

            self.assertEqual(snapshot.get_nodes(["a"])[0].get_content(), "alpha")
            fresh = OrdinalDBVectorStore(path=path)
            self.assertEqual(fresh.get_nodes(["a"])[0].get_content(), "alpha updated")
            with self.assertRaisesRegex(ValueError, "IDs already present"):
                fresh.add(
                    [
                        TextNode(
                            id_="a",
                            text="duplicate",
                            metadata={"group": "dup"},
                            embedding=ALPHA,
                        )
                    ]
                )
            fresh.delete_nodes(node_ids=["b"])
            fresh.persist(path)
            after_delete = OrdinalDBVectorStore(path=path)
            self.assertEqual(after_delete.get_nodes(["b"]), [])
            self.assertEqual(snapshot.get_nodes(["b"])[0].get_content(), "beta")


@unittest.skipUnless(_available("haystack"), "haystack-ai not installed")
class HaystackAdapterSmokeTests(unittest.TestCase):
    def test_write_filter_search_delete_persist_load_and_duplicate_policies(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_home = os.environ.get("HOME")
            os.environ["HOME"] = tmp
            try:
                from haystack import Document
                try:
                    from haystack.document_stores.types import DuplicatePolicy
                except ImportError:
                    from haystack import DuplicatePolicy
                from haystack.document_stores.errors import DuplicateDocumentError
                from haystack.utils.filters import FilterError
                import ordinaldb.haystack as haystack_adapter
                from ordinaldb.haystack import OrdinalDocumentStore, OrdinalEmbeddingRetriever

                path = Path(tmp) / "hs"
                store = OrdinalDocumentStore(dim=4)
                self.assertIsNone(store.to_dict()["init_parameters"]["path"])
                docs = [
                    Document(
                        id="a",
                        content="alpha",
                        meta={"group": "x", "rank": 1},
                        embedding=ALPHA,
                    ),
                    Document(
                        id="b",
                        content="beta",
                        meta={"group": "y", "rank": 2},
                        embedding=BETA,
                    ),
                ]
                self.assertEqual(store.write_documents(docs, policy=DuplicatePolicy.FAIL), 2)
                self.assertEqual(store.count_documents(), 2)
                with self.assertRaisesRegex(
                    DuplicateDocumentError, "duplicate document IDs for policy FAIL.*a"
                ):
                    store.write_documents(docs[:1], policy=DuplicatePolicy.FAIL)
                with self.assertRaisesRegex(
                    DuplicateDocumentError, "duplicate document IDs for policy NONE.*a"
                ):
                    store.write_documents(docs[:1], policy=DuplicatePolicy.NONE)
                duplicate_batch = [
                    Document(id="dup", content="alpha", embedding=ALPHA),
                    Document(id="dup", content="beta", embedding=BETA),
                ]
                with self.assertRaisesRegex(DuplicateDocumentError, "duplicate document IDs.*dup"):
                    store.write_documents(duplicate_batch, policy=DuplicatePolicy.FAIL)
                self.assertEqual(store.write_documents(duplicate_batch, policy=DuplicatePolicy.SKIP), 1)
                self.assertEqual(store.count_documents(), 3)
                overwrite_batch = [
                    Document(
                        id="dup-overwrite",
                        content="first",
                        meta={"winner": "first"},
                        embedding=ALPHA,
                    ),
                    Document(
                        id="dup-overwrite",
                        content="second",
                        meta={"winner": "last"},
                        embedding=BETA,
                    ),
                ]
                self.assertEqual(
                    store.write_documents(overwrite_batch, policy=DuplicatePolicy.OVERWRITE),
                    1,
                )
                self.assertEqual(
                    [
                        doc.content
                        for doc in store.filter_documents(
                            {"field": "meta.winner", "operator": "==", "value": "last"}
                        )
                    ],
                    ["second"],
                )
                store.delete_documents(["dup", "dup-overwrite"])
                self.assertEqual(store.count_documents(), 2)
                self.assertEqual(store.write_documents(docs[:1], policy=DuplicatePolicy.SKIP), 0)

                original_converter = haystack_adapter._document_from_record
                conversion_count = 0

                def counting_converter(record):
                    nonlocal conversion_count
                    conversion_count += 1
                    return original_converter(record)

                haystack_adapter._document_from_record = counting_converter
                try:
                    filtered = store.filter_documents(
                        {"field": "meta.rank", "operator": ">=", "value": 2}
                    )
                finally:
                    haystack_adapter._document_from_record = original_converter
                self.assertEqual([doc.id for doc in filtered], ["b"])
                self.assertEqual(conversion_count, 2)
                logical = store.filter_documents(
                    {
                        "operator": "OR",
                        "conditions": [
                            {"field": "meta.group", "operator": "==", "value": "x"},
                            {"field": "meta.rank", "operator": ">=", "value": 2},
                        ],
                    }
                )
                self.assertEqual([doc.id for doc in logical], ["a", "b"])
                with self.assertRaises(FilterError):
                    store.filter_documents(
                        {"field": "meta.group", "operator": ">", "value": False}
                    )

                retriever = OrdinalEmbeddingRetriever(document_store=store, top_k=2)
                result = retriever.run(
                    query_embedding=ALPHA,
                    filters={"field": "meta.group", "operator": "==", "value": "y"},
                )
                self.assertEqual([doc.id for doc in result["documents"]], ["b"])
                self.assertEqual(
                    store.search_by_embedding(
                        ALPHA,
                        filters={"field": "meta.group", "operator": "==", "value": "none"},
                    ),
                    [],
                )

                original_get = store._store.get
                store._store.get = lambda *args, **kwargs: self.fail(
                    "filtered paths should not copy all sidecar records"
                )
                try:
                    store.search_by_embedding(ALPHA, top_k=1)
                    filtered_again = store.search_by_embedding(
                        ALPHA,
                        top_k=1,
                        filters={"field": "meta.group", "operator": "==", "value": "y"},
                    )
                    self.assertEqual([doc.id for doc in filtered_again], ["b"])
                    self.assertEqual(
                        [
                            doc.id
                            for doc in store.filter_documents(
                                {"field": "meta.rank", "operator": ">=", "value": 2}
                            )
                        ],
                        ["b"],
                    )
                finally:
                    store._store.get = original_get

                store.save(path)
                loaded = OrdinalDocumentStore.load(path)
                with self.assertRaisesRegex(ValueError, "not a directory"):
                    OrdinalDocumentStore.load(Path(tmp) / "missing")
                loaded.delete_documents(["a"])
                self.assertEqual([doc.id for doc in loaded.filter_documents()], ["b"])
            finally:
                if old_home is None:
                    os.environ.pop("HOME", None)
                else:
                    os.environ["HOME"] = old_home

    def test_production_conformance_snapshot_mutation_and_filter_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_home = os.environ.get("HOME")
            os.environ["HOME"] = tmp
            try:
                from haystack import Document
                try:
                    from haystack.document_stores.types import DuplicatePolicy
                except ImportError:
                    from haystack import DuplicatePolicy
                from haystack.document_stores.errors import DuplicateDocumentError
                from ordinaldb.haystack import OrdinalDocumentStore

                path = Path(tmp) / "hs-conformance"
                store = OrdinalDocumentStore(dim=4)
                store.write_documents(
                    [
                        Document(
                            id="a",
                            content="alpha",
                            meta={"group": "x", "rank": 1},
                            embedding=ALPHA,
                        ),
                        Document(
                            id="b",
                            content="beta",
                            meta={"group": "y", "rank": 2},
                            embedding=BETA,
                        ),
                    ],
                    policy=DuplicatePolicy.FAIL,
                )

                routed = []
                original_search = store._store.search_by_vector

                def recording_search(*args, **kwargs):
                    routed.append(kwargs)
                    return original_search(*args, **kwargs)

                store._store.search_by_vector = recording_search
                try:
                    filtered = store.search_by_embedding(
                        ALPHA,
                        top_k=2,
                        filters={"field": "meta.group", "operator": "==", "value": "y"},
                    )
                finally:
                    store._store.search_by_vector = original_search
                self.assertEqual([doc.id for doc in filtered], ["b"])
                self.assertIsNone(routed[-1].get("filter"))
                self.assertIsNotNone(routed[-1].get("allowed_u64_ids"))

                store.save(path)
                snapshot = OrdinalDocumentStore.load(path)
                store.write_documents(
                    [
                        Document(
                            id="a",
                            content="alpha updated",
                            meta={"group": "updated", "rank": 3},
                            embedding=DELTA,
                        )
                    ],
                    policy=DuplicatePolicy.OVERWRITE,
                )
                store.save(path)

                self.assertEqual(
                    [doc.content for doc in snapshot.filter_documents() if doc.id == "a"],
                    ["alpha"],
                )
                fresh = OrdinalDocumentStore.load(path)
                self.assertEqual(
                    [doc.content for doc in fresh.filter_documents() if doc.id == "a"],
                    ["alpha updated"],
                )
                with self.assertRaisesRegex(DuplicateDocumentError, "duplicate document IDs.*a"):
                    fresh.write_documents(
                        [Document(id="a", content="duplicate", embedding=ALPHA)],
                        policy=DuplicatePolicy.FAIL,
                    )
                fresh.delete_documents(["b"])
                fresh.save(path)
                after_delete = OrdinalDocumentStore.load(path)
                self.assertEqual(
                    [doc.id for doc in after_delete.filter_documents() if doc.id == "b"],
                    [],
                )
                self.assertEqual(
                    [doc.content for doc in snapshot.filter_documents() if doc.id == "b"],
                    ["beta"],
                )
            finally:
                if old_home is None:
                    os.environ.pop("HOME", None)
                else:
                    os.environ["HOME"] = old_home


@unittest.skipUnless(_available("agno"), "agno not installed")
class AgnoAdapterSmokeTests(unittest.TestCase):
    def test_upsert_search_by_text_and_vector_persist_load(self):
        from agno.knowledge.document import Document
        from ordinaldb.agno import OrdinalDb

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agno"
            created = OrdinalDb(path=path, dim=4)
            self.assertFalse(created.exists())
            created.create()
            self.assertTrue(created.exists())
            self.assertTrue((path / "adapter.json").exists())
            self.assertEqual(OrdinalDb.load(path).search_by_vector(ALPHA), [])
            with self.assertRaisesRegex(ValueError, "not a directory"):
                OrdinalDb.load(Path(tmp) / "missing")

            db = OrdinalDb(embedder=FakeAgnoEmbedder(), dim=4)
            with self.assertRaisesRegex(ValueError, "documents"):
                db.upsert("content-hash-only")
            db.upsert(
                "content-hash-1",
                [
                    Document(
                        id="a",
                        name="alpha-doc",
                        content="alpha",
                        meta_data={"group": "x"},
                    ),
                    Document(
                        id="b",
                        name="beta-doc",
                        content="beta",
                        meta_data={"group": "y"},
                    ),
                ],
                embeddings=[ALPHA, BETA],
            )
            self.assertTrue(db.exists())
            self.assertTrue(db.id_exists("a"))
            self.assertTrue(db.name_exists("beta-doc"))
            self.assertTrue(db.content_hash_exists("content-hash-1"))
            self.assertTrue(db.content_id_exists("content-hash-1"))
            self.assertEqual(db.get_supported_search_types(), ["vector"])
            self.assertEqual([doc.id for doc in db.search("alpha", filters={"group": "y"})], ["b"])
            self.assertEqual(
                [doc.id for doc in db.search_by_vector(ALPHA, filters={"group": "x"})],
                ["a"],
            )
            self.assertEqual(db.search_by_vector(ALPHA, filters={"group": "none"}), [])
            generated_ids = asyncio.run(
                db.async_insert(
                    "content-hash-2",
                    [Document(content="alpha chunk"), Document(content="beta chunk")],
                    embeddings=[ALPHA, BETA],
                )
            )
            self.assertEqual(generated_ids, ["content-hash-2:0", "content-hash-2:1"])
            self.assertTrue(db.delete_by_content_id("content-hash-2"))
            db.upsert(
                "content-hash-3",
                [Document(id="m", content="alpha", meta_data={"group": "old"})],
                embeddings=[ALPHA],
                metadatas=[{}],
            )
            self.assertEqual(db.search_by_vector(ALPHA, filters={"group": "old"}), [])
            self.assertTrue(db.delete_by_content_id("content-hash-3"))

            db.upsert(
                "content-hash-rollback",
                [Document(id="rollback-old", content="alpha", meta_data={"group": "rollback"})],
                embeddings=[ALPHA],
            )
            with self.assertRaisesRegex(ValueError, "embedding dim mismatch"):
                db.upsert(
                    "content-hash-rollback",
                    [Document(id="rollback-new", content="bad")],
                    embeddings=[[1.0, 0.0]],
                )
            self.assertTrue(db.id_exists("rollback-old"))
            self.assertFalse(db.id_exists("rollback-new"))
            self.assertTrue(db.delete_by_content_hash("content-hash-rollback"))

            auto_path = Path(tmp) / "agno-auto"
            auto = OrdinalDb(path=auto_path, embedder=FakeAgnoEmbedder(), dim=4, auto_save=True)
            auto.insert(
                "content-hash-auto",
                [Document(id="auto", content="alpha", meta_data={"group": "auto"})],
                embeddings=[ALPHA],
            )
            self.assertTrue((auto_path / "adapter.json").exists())
            self.assertEqual(
                [
                    doc.id
                    for doc in OrdinalDb.load(auto_path, embedder=FakeAgnoEmbedder()).search(
                        "alpha",
                        filters={"group": "auto"},
                    )
                ],
                ["auto"],
            )

            with self.assertRaisesRegex(ValueError, "without a loaded base commit token"):
                db.save(path)
            populated_path = Path(tmp) / "agno-populated"
            db.save(populated_path)
            loaded = OrdinalDb.load(populated_path, embedder=FakeAgnoEmbedder())
            self.assertEqual([doc.id for doc in loaded.search("beta", filters={"group": "y"})], ["b"])
            self.assertTrue(loaded.delete_by_name("alpha-doc"))
            self.assertFalse(loaded.id_exists("a"))
            self.assertTrue(loaded.delete_by_content_id("content-hash-1"))
            self.assertFalse(loaded.content_hash_exists("content-hash-1"))
            with self.assertRaisesRegex(ValueError, "embedder"):
                OrdinalDb(dim=4).search("alpha")

    def test_production_conformance_snapshot_mutation_and_filter_route(self):
        from agno.knowledge.document import Document
        from ordinaldb.agno import OrdinalDb

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agno-conformance"
            db = OrdinalDb(embedder=FakeAgnoEmbedder(), dim=4)
            db.upsert(
                "content-a",
                [
                    Document(
                        id="a",
                        name="alpha-doc",
                        content="alpha",
                        meta_data={"group": "x", "rank": 1},
                    )
                ],
                embeddings=[ALPHA],
            )
            db.upsert(
                "content-b",
                [
                    Document(
                        id="b",
                        name="beta-doc",
                        content="beta",
                        meta_data={"group": "y", "rank": 2},
                    )
                ],
                embeddings=[BETA],
            )

            routed = []
            original_search = db._store.search_by_vector

            def recording_search(*args, **kwargs):
                routed.append(kwargs)
                return original_search(*args, **kwargs)

            db._store.search_by_vector = recording_search
            try:
                filtered = db.search_by_vector(ALPHA, filters={"group": "y"})
            finally:
                db._store.search_by_vector = original_search
            self.assertEqual([doc.id for doc in filtered], ["b"])
            self.assertEqual(routed[-1].get("filter"), {"group": "y"})
            self.assertIsNone(routed[-1].get("allowed_u64_ids"))

            db.save(path)
            snapshot = OrdinalDb.load(path, embedder=FakeAgnoEmbedder())
            db.upsert(
                "content-a",
                [
                    Document(
                        id="a",
                        name="alpha-doc",
                        content="alpha updated",
                        meta_data={"group": "updated", "rank": 3},
                    )
                ],
                embeddings=[DELTA],
            )
            db.save(path)

            self.assertEqual(
                [doc.content for doc in snapshot.search_by_vector(ALPHA, filters={"group": "x"})],
                ["alpha"],
            )
            fresh = OrdinalDb.load(path, embedder=FakeAgnoEmbedder())
            self.assertEqual(
                [
                    doc.content
                    for doc in fresh.search_by_vector(DELTA, filters={"group": "updated"})
                ],
                ["alpha updated"],
            )
            with self.assertRaisesRegex(ValueError, "duplicate string IDs"):
                fresh.upsert(
                    "content-dup",
                    [
                        Document(id="dup", content="duplicate first"),
                        Document(id="dup", content="duplicate second"),
                    ],
                    embeddings=[ALPHA, BETA],
                )
            self.assertTrue(fresh.delete_by_id("b"))
            fresh.save(path)
            after_delete = OrdinalDb.load(path, embedder=FakeAgnoEmbedder())
            self.assertEqual(after_delete.search_by_vector(ALPHA, filters={"group": "y"}), [])
            self.assertEqual(
                [doc.content for doc in snapshot.search_by_vector(BETA, filters={"group": "y"})],
                ["beta"],
            )
            with self.assertRaisesRegex(ValueError, "portable filter value .* JSON scalar"):
                after_delete.search_by_vector(ALPHA, filters={"group": ["x"]})


def _vector_for_text(text):
    return BETA if "beta" in text else ALPHA


if __name__ == "__main__":
    unittest.main()
