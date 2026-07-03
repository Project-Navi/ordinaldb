#!/usr/bin/env python
"""PaperScout: a local research-paper discovery tool built on OrdinalDB.

Run: python demo.py            (offline, bundled 40-paper corpus, default)
     python demo.py --live     (tries the live arXiv API first, falls back
                                 to the bundled corpus if it's unreachable)

Walks through the full flow an ML team would want from a metadata-rich
LlamaIndex vector store:
  1. load paper abstracts (bundled corpus by default, or --live for arXiv)
  2. index them through the LlamaIndex + OrdinalDBVectorStore adapter,
     using local sentence-transformers embeddings (no API keys)
  3. top-k semantic discovery queries
  4. metadata-filtered queries, including the full LlamaIndex
     FilterOperator dialect (GT/GTE/LT/NE/IN/NIN/ANY/ALL + AND/OR/NOT)
  5. node round-trip fidelity (text + metadata survive the round trip)
  6. the ref-doc lifecycle: delete_ref_doc, get_nodes/delete_nodes(filters=
     ...), clear()
  7. failure paths: empty index, missing filter key, bad store paths
  8. idiomatic persistence (StorageContext.persist + from_persist_dir)
     across a real process restart (subprocess, not just a fresh Python
     object)
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

from llama_index.core import Document
from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

from fetch_papers import load_papers
from ordinaldb.adapters import AdapterPathWarning, AdapterStoreError
from paperscout import (
    build_documents,
    build_index,
    configure_local_settings,
    open_vector_store,
)

PROJECT_DIR = Path(__file__).resolve().parent
STORE_DIR = PROJECT_DIR / "storage" / "paperscout_store"


def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def show_nodes(nodes, *, show_score: bool = True) -> None:
    if not nodes:
        print("  (no results)")
        return
    for node in nodes:
        meta = node.node.metadata
        prefix = f"score={node.score:.4f} " if show_score and node.score is not None else ""
        print(f"  - {prefix}[{meta.get('category')} {meta.get('year')}] {meta.get('title')}")


def main() -> None:
    live = "--live" in sys.argv[1:]
    t_start = time.time()

    header("STEP 0: load paper corpus")
    print(
        "Bundled corpus is the default (offline, deterministic, ~40 papers). "
        "Pass --live to fetch fresh abstracts from the arXiv API instead, "
        "falling back to the bundled corpus if arXiv isn't reachable."
    )
    papers, source = load_papers(live=live)
    print(f"Loaded {len(papers)} papers (source={source})")
    print("By category:", dict(Counter(p["category"] for p in papers)))
    print("Years >= 2024:", sum(1 for p in papers if p["year"] >= 2024))

    header("STEP 1: configure local embeddings (sentence-transformers, no API keys)")
    t0 = time.time()
    embed_model = configure_local_settings()
    print(f"HuggingFaceEmbedding(all-MiniLM-L6-v2, dim=384) loaded in {time.time() - t0:.2f}s")

    header("STEP 2: build the OrdinalDB-backed LlamaIndex VectorStoreIndex")
    if STORE_DIR.exists():
        shutil.rmtree(STORE_DIR)
    STORE_DIR.parent.mkdir(parents=True, exist_ok=True)

    documents = build_documents(papers)
    vector_store = open_vector_store(STORE_DIR)
    t0 = time.time()
    index = build_index(documents, vector_store, embed_model)
    build_elapsed = time.time() - t0
    print(
        f"Indexed {len(vector_store.client)} nodes into OrdinalDB "
        f"(dim=384, bits=2) in {build_elapsed:.2f}s "
        f"({build_elapsed / len(documents) * 1000:.1f} ms/doc)"
    )

    header("STEP 3: top-k semantic discovery query")
    query_text = "efficient fine-tuning of large language models on a single GPU"
    retriever = index.as_retriever(similarity_top_k=5)
    t0 = time.time()
    nodes = retriever.retrieve(query_text)
    print(f"Query: {query_text!r}  ({time.time() - t0:.3f}s)")
    show_nodes(nodes)

    header("STEP 3b: a second discovery query, via as_query_engine (no_text + MockLLM)")
    query_engine = index.as_query_engine(response_mode="no_text", similarity_top_k=5)
    query_text_2 = "approximate nearest neighbor search over billions of vectors"
    response = query_engine.query(query_text_2)
    print(f"Query: {query_text_2!r}")
    show_nodes(response.source_nodes)

    header("STEP 4: metadata filters -- pre-search allowlist, not post-filtering")
    print(
        "Every filter below resolves to an ID allowlist BEFORE the top-k "
        "vector search runs, so a filtered query returns the true top-k "
        "within the filtered set -- not a global top-k with non-matching "
        "hits discarded afterward."
    )

    header("STEP 4a: exact match AND (category == cs.IR AND year == 2024)")
    filters_and = MetadataFilters(
        filters=[
            MetadataFilter(key="category", value="cs.IR", operator=FilterOperator.EQ),
            MetadataFilter(key="year", value=2024, operator=FilterOperator.EQ),
        ],
        condition=FilterCondition.AND,
    )
    nodes = index.as_retriever(similarity_top_k=10, filters=filters_and).retrieve(
        "retrieval augmented generation"
    )
    print(f"{len(nodes)} match(es):")
    show_nodes(nodes)

    header("STEP 4b: exact match, single filter (category == cs.IR)")
    filters_ir = MetadataFilters(
        filters=[MetadataFilter(key="category", value="cs.IR", operator=FilterOperator.EQ)]
    )
    nodes = index.as_retriever(similarity_top_k=10, filters=filters_ir).retrieve(
        "text embeddings for semantic search"
    )
    print(f"{len(nodes)} match(es), all should be cs.IR:")
    show_nodes(nodes)
    assert all(n.node.metadata["category"] == "cs.IR" for n in nodes), "filter leaked non-cs.IR results"

    header("STEP 4c: operator >= (year >= 2024)")
    filters_gte = MetadataFilters(
        filters=[MetadataFilter(key="year", value=2024, operator=FilterOperator.GTE)]
    )
    nodes = index.as_retriever(similarity_top_k=10, filters=filters_gte).retrieve(
        "language model reasoning"
    )
    print(f"{len(nodes)} match(es), all should have year >= 2024:")
    show_nodes(nodes)
    assert all(n.node.metadata["year"] >= 2024 for n in nodes), "GTE filter returned an out-of-range year"

    header("STEP 4d: operator 'in' (category in [cs.IR, cs.DB])")
    filters_in = MetadataFilters(
        filters=[
            MetadataFilter(key="category", value=["cs.IR", "cs.DB"], operator=FilterOperator.IN)
        ]
    )
    nodes = index.as_retriever(similarity_top_k=10, filters=filters_in).retrieve(
        "database and retrieval systems"
    )
    print(f"{len(nodes)} match(es), all should be cs.IR or cs.DB:")
    show_nodes(nodes)
    assert all(n.node.metadata["category"] in ("cs.IR", "cs.DB") for n in nodes)

    header("STEP 4e: the full FilterOperator matrix (NE/GT/LT/LTE/NIN/ANY/ALL + OR/NOT)")
    print(
        "STEP 4a-4d above already exercised EQ+AND, GTE, and IN on the real "
        "corpus. This step exercises the rest of the documented dialect -- "
        "EQ/NE/GT/GTE/LT/LTE/IN/NIN/ANY/ALL, plus AND/OR/NOT composition -- "
        "against a small throwaway index with a synthetic list-valued "
        "'tags' field (ANY/ALL need a list to be meaningful)."
    )
    tag_dir = PROJECT_DIR / "storage" / "tag_filter_store"
    if tag_dir.exists():
        shutil.rmtree(tag_dir)
    tag_docs = [
        Document(
            text="a paper about databases and retrieval",
            metadata={"label": "a", "domain": "db", "rank": 1, "tags": ["retrieval", "db"]},
        ),
        Document(
            text="a paper about generative language models",
            metadata={"label": "b", "domain": "llm", "rank": 2, "tags": ["generative", "llm"]},
        ),
        Document(
            text="a paper bridging retrieval and language models",
            metadata={"label": "c", "domain": "hybrid", "rank": 3, "tags": ["retrieval", "llm"]},
        ),
    ]
    tag_store = open_vector_store(tag_dir)
    tag_index = build_index(tag_docs, tag_store, embed_model)

    def labels_for(filters: MetadataFilters) -> list[str]:
        nodes = tag_index.as_retriever(similarity_top_k=10, filters=filters).retrieve("paper")
        return sorted(n.node.metadata["label"] for n in nodes)

    matrix_cases = [
        ("NE domain != db", MetadataFilters(filters=[MetadataFilter(key="domain", value="db", operator=FilterOperator.NE)]), ["b", "c"]),
        ("GT rank > 1", MetadataFilters(filters=[MetadataFilter(key="rank", value=1, operator=FilterOperator.GT)]), ["b", "c"]),
        ("LT rank < 3", MetadataFilters(filters=[MetadataFilter(key="rank", value=3, operator=FilterOperator.LT)]), ["a", "b"]),
        ("LTE rank <= 1", MetadataFilters(filters=[MetadataFilter(key="rank", value=1, operator=FilterOperator.LTE)]), ["a"]),
        ("NIN domain not in [db, llm]", MetadataFilters(filters=[MetadataFilter(key="domain", value=["db", "llm"], operator=FilterOperator.NIN)]), ["c"]),
        ("ANY tags any of [db, generative]", MetadataFilters(filters=[MetadataFilter(key="tags", value=["db", "generative"], operator=FilterOperator.ANY)]), ["a", "b"]),
        ("ALL tags all of [retrieval, llm]", MetadataFilters(filters=[MetadataFilter(key="tags", value=["retrieval", "llm"], operator=FilterOperator.ALL)]), ["c"]),
        (
            "OR domain==db OR domain==llm",
            MetadataFilters(
                filters=[
                    MetadataFilter(key="domain", value="db", operator=FilterOperator.EQ),
                    MetadataFilter(key="domain", value="llm", operator=FilterOperator.EQ),
                ],
                condition=FilterCondition.OR,
            ),
            ["a", "b"],
        ),
        (
            "NOT domain==db",
            MetadataFilters(
                filters=[MetadataFilter(key="domain", value="db", operator=FilterOperator.EQ)],
                condition=FilterCondition.NOT,
            ),
            ["b", "c"],
        ),
    ]
    for name, filters, expected in matrix_cases:
        actual = labels_for(filters)
        status = "match" if actual == expected else "MISMATCH"
        print(f"  [{status}] {name}: {actual}")
        assert actual == expected, f"{name}: expected {expected}, got {actual}"
    shutil.rmtree(tag_dir, ignore_errors=True)

    header("STEP 5: node round-trip fidelity")
    # vector_store.client.get(ids) looks records up by the LlamaIndex
    # node_id (a fresh random UUID assigned when the Document is split into
    # nodes), NOT by our paper id / doc_id. To find "our" paper by its
    # original id we look up by metadata["ref_doc_id"], which
    # OrdinalDBVectorStore.add() stamps automatically.
    by_ref_doc_id = {
        record.metadata.get("ref_doc_id"): record for record in vector_store.client.iter_records()
    }
    sample = papers[:5] + papers[len(papers) // 2 : len(papers) // 2 + 5]
    for paper in sample:
        record = by_ref_doc_id.get(paper["id"])
        if record is None:
            print(f"  MISSING: {paper['id']} not found in store!")
            continue
        expected_text = f"{paper['title']}. {paper['abstract']}"
        text_ok = record.document == expected_text
        meta_ok = (
            record.metadata.get("category") == paper["category"]
            and record.metadata.get("year") == paper["year"]
            and record.metadata.get("authors_count") == paper["authors_count"]
        )
        status = "ok" if (text_ok and meta_ok) else "MISMATCH"
        print(f"  [{status}] {paper['id']}: text_ok={text_ok} metadata_ok={meta_ok}")
        assert text_ok and meta_ok, f"round-trip fidelity check failed for {paper['id']}"

    header("STEP 6: ref-doc lifecycle -- delete_ref_doc")
    target_paper = papers[0]
    print(f"Deleting ref_doc_id={target_paper['id']!r} ({target_paper['title']!r})")
    count_before = len(vector_store.client)
    index.delete_ref_doc(target_paper["id"], delete_from_docstore=True)
    count_after = len(vector_store.client)
    print(f"Record count: {count_before} -> {count_after}")
    still_present = [
        record
        for record in vector_store.client.iter_records()
        if record.metadata.get("ref_doc_id") == target_paper["id"]
    ]
    print(f"Still retrievable by ref_doc_id? {bool(still_present)}")
    assert count_after == count_before - 1, "delete_ref_doc did not remove exactly one record"
    assert not still_present, "deleted paper is still fetchable"

    header("STEP 6b: ref-doc lifecycle -- get_nodes / delete_nodes(filters=...) / clear()")
    scratch_dir = PROJECT_DIR / "storage" / "scratch_ops_store"
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_docs = [
        Document(text="scratch doc alpha", metadata={"group": "x"}, doc_id="scratch-a"),
        Document(text="scratch doc beta", metadata={"group": "x"}, doc_id="scratch-b"),
        Document(text="scratch doc gamma", metadata={"group": "y"}, doc_id="scratch-c"),
    ]
    scratch_store = open_vector_store(scratch_dir)
    build_index(scratch_docs, scratch_store, embed_model)
    group_x_filter = MetadataFilters(filters=[MetadataFilter(key="group", value="x", operator=FilterOperator.EQ)])
    print(
        "get_nodes(filters=group==x): "
        f"{sorted(n.metadata['group'] for n in scratch_store.get_nodes(filters=group_x_filter))}"
    )
    before = len(scratch_store.client)
    scratch_store.delete_nodes(filters=group_x_filter)
    after = len(scratch_store.client)
    print(f"delete_nodes(filters=group==x): {before} -> {after}")
    assert after == before - 2, "delete_nodes(filters=...) removed the wrong number of records"
    scratch_store.clear()
    print(f"clear(): {after} -> {len(scratch_store.client)}")
    assert len(scratch_store.client) == 0, "clear() did not empty the store"
    shutil.rmtree(scratch_dir, ignore_errors=True)

    header("STEP 7: failure paths")

    print("\n--- 7a. query against an empty index ---")
    empty_dir = PROJECT_DIR / "storage" / "paperscout_empty_store"
    if empty_dir.exists():
        shutil.rmtree(empty_dir)
    empty_store = open_vector_store(empty_dir)
    empty_index = build_index([], empty_store, embed_model)
    empty_nodes = empty_index.as_retriever(similarity_top_k=5).retrieve("anything at all")
    print(f"No exception. Returned {len(empty_nodes)} nodes.")
    shutil.rmtree(empty_dir, ignore_errors=True)

    print("\n--- 7b. filter on a metadata key that doesn't exist ---")
    bad_key_filters = MetadataFilters(
        filters=[MetadataFilter(key="does_not_exist", value="x", operator=FilterOperator.EQ)]
    )
    nodes = index.as_retriever(similarity_top_k=5, filters=bad_key_filters).retrieve("database")
    print(f"No exception. Returned {len(nodes)} nodes (silently excluded, not an error).")

    print("\n--- 7c. opening a path that's an existing plain file, not a directory ---")
    bad_file = PROJECT_DIR / "storage" / "not_a_directory.txt"
    bad_file.parent.mkdir(parents=True, exist_ok=True)
    bad_file.write_text("this is a regular file, not an OrdinalDB adapter store")
    try:
        open_vector_store(bad_file)
        print("No exception raised (unexpected).")
    except AdapterStoreError as exc:
        print(f"Raised immediately at construction: {type(exc).__name__}: {exc}")
    bad_file.unlink(missing_ok=True)

    print("\n--- 7d. opening an existing directory with unrelated stray content ---")
    stray_dir = PROJECT_DIR / "storage" / "stray_dir"
    if stray_dir.exists():
        shutil.rmtree(stray_dir)
    stray_dir.mkdir(parents=True)
    (stray_dir / "notes.txt").write_text("unrelated file that predates any OrdinalDB store")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        stray_store = open_vector_store(stray_dir)
        path_warnings = [w for w in caught if issubclass(w.category, AdapterPathWarning)]
    if path_warnings:
        print(f"AdapterPathWarning: {path_warnings[0].message} (len(store)={len(stray_store.client)})")
    else:
        print(f"No warning. len(store)={len(stray_store.client)}")
    shutil.rmtree(stray_dir, ignore_errors=True)

    header("STEP 8: idiomatic persistence across a real process restart")
    print(
        "index.storage_context.persist(persist_dir=X) is the pattern "
        "LlamaIndex's own docs teach. OrdinalDBVectorStore.persist() "
        "recognizes the namespaced file-shaped path LlamaIndex calls it "
        "with (f'{X}/default__vector_store.json') and maps it back to X "
        "itself, so a later reload at X finds the data."
    )
    if STORE_DIR.exists():
        shutil.rmtree(STORE_DIR)
    index.storage_context.persist(persist_dir=str(STORE_DIR))
    print(f"Persisted {len(vector_store.client)} records via storage_context.persist(persist_dir={STORE_DIR})")
    on_disk = sorted(p.name for p in STORE_DIR.iterdir())
    print("On-disk contents of STORE_DIR itself:", on_disk)
    nested_json = STORE_DIR / "default__vector_store.json"
    print(f"Nested 'default__vector_store.json' present as a directory? {nested_json.is_dir()}")

    reload_query = "vector database indexing for approximate nearest neighbor search"
    print("\n--- 8a. plain-constructor reload: OrdinalDBVectorStore(path=STORE_DIR) ---")
    print(f"Launching a brand-new OS process to reload and query: {reload_query!r}")
    result = subprocess.run(
        [sys.executable, str(PROJECT_DIR / "reload_check.py"), str(STORE_DIR), reload_query],
        capture_output=True,
        text=True,
        cwd=PROJECT_DIR,
        timeout=120,
    )
    print("--- subprocess stdout ---")
    print(result.stdout.strip())
    if result.returncode != 0:
        print("--- subprocess stderr ---")
        print(result.stderr.strip())
    assert result.returncode == 0, "reload subprocess failed"
    assert f"{len(vector_store.client)} records" in result.stdout, "record count mismatch after restart"

    print(f"\n--- 8b. from_persist_dir(STORE_DIR) classmethod reload ---")
    print(f"Launching a second brand-new OS process to reload via from_persist_dir and query: {reload_query!r}")
    result2 = subprocess.run(
        [
            sys.executable,
            str(PROJECT_DIR / "reload_check.py"),
            str(STORE_DIR),
            reload_query,
            "--from-persist-dir",
        ],
        capture_output=True,
        text=True,
        cwd=PROJECT_DIR,
        timeout=120,
    )
    print("--- subprocess stdout ---")
    print(result2.stdout.strip())
    if result2.returncode != 0:
        print("--- subprocess stderr ---")
        print(result2.stderr.strip())
    assert result2.returncode == 0, "from_persist_dir reload subprocess failed"
    assert f"{len(vector_store.client)} records" in result2.stdout, "record count mismatch after restart"

    header("DONE")
    print(f"Total demo runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
