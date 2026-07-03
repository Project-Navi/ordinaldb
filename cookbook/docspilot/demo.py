#!/usr/bin/env python
"""DocsPilot demo: ingest OrdinalDB's own docs, then answer retrieval
queries with source attribution.

Walks through the full pipeline: load a real markdown corpus -> chunk it ->
embed locally -> ingest into an OrdinalDB-backed LangChain vector store ->
query with source attribution -> filter by metadata -> delete + persist ->
reopen the store from a brand-new process to prove it survives a restart.

Run with: python demo.py   (from cookbook/docspilot, with the venv
from the README's Setup section active)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ordinaldb.langchain import OrdinalDBVectorStore  # noqa: E402
from ordinaldb.adapters import UnknownFilterKeyWarning, UnsavedWritesWarning  # noqa: E402

from docspilot.chunking import chunk_files  # noqa: E402
from docspilot.corpus import find_repo_root, load_corpus  # noqa: E402
from docspilot.embeddings import EMBEDDING_DIM, MiniLMEmbeddings  # noqa: E402
from docspilot.store import DEFAULT_BITS, build_store  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent
STORE_PATH = PROJECT_DIR / "data" / "adapter-store"

_START = time.perf_counter()


def banner(title: str) -> None:
    elapsed = time.perf_counter() - _START
    print(f"\n=== [{elapsed:6.2f}s] {title} ===")


def section_corpus() -> list:
    banner("CORPUS: locate repo root and load the real markdown files")
    repo_root = find_repo_root(PROJECT_DIR)
    corpus_files = load_corpus(repo_root)
    total_bytes = sum(f.absolute_path.stat().st_size for f in corpus_files)
    print(f"repo root: {repo_root}")
    print(f"{len(corpus_files)} files, {total_bytes:,} bytes total")
    for f in corpus_files:
        tag = "docs/" if f.is_docs_subtree else "root "
        print(f"  [{tag}] {f.relative_path} ({f.absolute_path.stat().st_size:,} bytes)")
    return corpus_files


def section_chunk(corpus_files):
    banner("CHUNKING: markdown-aware split with source/section metadata")
    chunks = chunk_files(corpus_files)
    print(f"{len(chunks)} chunks from {len(corpus_files)} files")
    sample = chunks[0]
    print("sample chunk:")
    print(f"  id={sample.id!r}")
    print(f"  metadata={sample.document.metadata}")
    print(f"  text[:120]={sample.document.page_content[:120]!r}")
    return chunks


def section_embedding_model() -> MiniLMEmbeddings:
    banner("EMBEDDING MODEL: sentence-transformers/all-MiniLM-L6-v2 (CPU)")
    t0 = time.perf_counter()
    embedding = MiniLMEmbeddings()
    print(f"model loaded in {time.perf_counter() - t0:.2f}s, dim={EMBEDDING_DIM}")
    return embedding


def section_ingest(embedding: MiniLMEmbeddings, chunks) -> OrdinalDBVectorStore:
    banner("INGEST: from_documents into a fresh persistent store")
    if STORE_PATH.exists():
        shutil.rmtree(STORE_PATH)
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        store = build_store(STORE_PATH, embedding, chunks)
    elapsed = time.perf_counter() - t0
    print(f"embedded + indexed {len(chunks)} chunks in {elapsed:.2f}s ({elapsed / len(chunks) * 1000:.1f}ms/chunk)")
    print(f"store is in-memory only so far -- not yet saved to {STORE_PATH}")

    unsaved = [w for w in caught if issubclass(w.category, UnsavedWritesWarning)]
    if unsaved:
        print(f"adapter warned about unsaved writes: {unsaved[0].message}")
    return store


def _print_hits(label: str, results) -> None:
    print(f"{label}: {len(results)} hit(s)")
    for doc, score in results:
        print(f"  score={score:.4f} source={doc.metadata['source']} section={doc.metadata['section']!r}")
        print(f"    {doc.page_content[:100]!r}")


def section_queries(store: OrdinalDBVectorStore) -> None:
    banner("QUERIES: similarity_search_with_score, with source attribution")
    queries = [
        "How do I persist a vector index to disk?",
        "What happens on a Raspberry Pi deployment?",
        "How does OrdinalDB handle metadata filters for LangChain?",
    ]
    for query in queries:
        results = store.similarity_search_with_score(query, k=3)
        _print_hits(f'query: "{query}"', results)


def section_filtered_query(store: OrdinalDBVectorStore) -> None:
    """The adapter resolves a metadata filter to an ID allowlist *before*
    running the vector search, instead of searching unfiltered and
    discarding non-matching hits afterward -- so a narrow filter over a
    large corpus doesn't cost you your `k` results to filtering noise.
    """
    banner("FILTERED QUERY: doc_type=docs (pre-search allowlist, not post-filtering)")
    results = store.similarity_search_with_score(
        "persistence and manifest verification",
        k=5,
        filter={"doc_type": "docs"},
    )
    _print_hits("filter={'doc_type': 'docs'}", results)
    root_hits = [doc for doc, _ in results if doc.metadata["doc_type"] != "docs"]
    print(f"root-level hits leaking through the filter: {len(root_hits)} (expect 0)")

    banner("FAILURE PATH: a typo'd filter key ('doctype' instead of 'doc_type')")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = store.similarity_search_with_score(
            "persistence and manifest verification",
            k=5,
            filter={"doctype": "docs"},
        )
    key_warnings = [w for w in caught if issubclass(w.category, UnknownFilterKeyWarning)]
    print(f"similarity_search with typo'd filter key -> {len(results)} hits")
    if key_warnings:
        print(f"adapter names the unrecognized key instead of silently matching nothing: {key_warnings[0].message}")


def section_retriever(store: OrdinalDBVectorStore) -> None:
    banner("AS_RETRIEVER: LangChain's standard retriever interface")
    retriever = store.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke("How does verified persistence work?")
    for doc in docs:
        print(f"  source={doc.metadata['source']} section={doc.metadata['section']!r}")


def section_delete_and_reupsert(store: OrdinalDBVectorStore, corpus_files) -> None:
    """Documents change over time. This simulates an edited docs/provenance.md:
    diff the old chunk ids against the new ones, delete what's stale, and
    upsert the rest -- the same workflow a real docs-sync job would run.
    """
    banner("DELETE + RE-UPSERT: simulate an edited docs/provenance.md")
    from docspilot.chunking import chunk_file
    from docspilot.corpus import CorpusFile

    provenance = next(f for f in corpus_files if f.relative_path == "docs/provenance.md")
    original_chunks = chunk_file(provenance)
    original_ids = {c.id for c in original_chunks}
    print(f"original docs/provenance.md: {len(original_chunks)} chunks")

    edited_text = (
        "# Provenance\n\n"
        "This revision replaces the historical provenance notes with a "
        "single short paragraph, on purpose -- shorter content means fewer "
        "chunks, which produces stale chunk ids to clean up below.\n"
    )
    edited_path = PROJECT_DIR / "data" / "_provenance_edited.md"
    edited_path.parent.mkdir(parents=True, exist_ok=True)
    edited_path.write_text(edited_text, encoding="utf-8")
    edited_corpus_file = CorpusFile(relative_path="docs/provenance.md", absolute_path=edited_path)
    new_chunks = chunk_file(edited_corpus_file)
    new_ids = {c.id for c in new_chunks}
    print(f"edited docs/provenance.md: {len(new_chunks)} chunk(s)")

    stale_ids = sorted(original_ids - new_ids)
    print(f"stale chunk ids to delete: {stale_ids}")
    if stale_ids:
        store.delete(ids=stale_ids)

    new_docs = [c.document for c in new_chunks]
    new_id_list = [c.id for c in new_chunks]
    store.add_documents(new_docs, ids=new_id_list)
    print(f"upserted {len(new_chunks)} chunk(s) for the edited file")
    edited_path.unlink()


def section_delete_by_filter(store: OrdinalDBVectorStore) -> None:
    banner("DELETE BY FILTER: remove every chunk from one source file")
    before = len(store)
    store.delete(filter={"file_name": "SECURITY.md"})
    after = len(store)
    print(f"records before/after deleting file_name=SECURITY.md: {before} -> {after}")
    remaining = store.similarity_search_with_score("SECURITY", k=3, filter={"file_name": "SECURITY.md"})
    print(f"SECURITY.md chunks still retrievable: {len(remaining)} (expect 0)")


def section_persist(store: OrdinalDBVectorStore) -> None:
    banner("PERSIST: save_local (add/upsert alone does not write to disk)")
    t0 = time.perf_counter()
    store.save_local(STORE_PATH)
    print(f"saved in {time.perf_counter() - t0:.3f}s")
    print(f"on-disk layout under {STORE_PATH.name}/:")
    for path in sorted(STORE_PATH.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(STORE_PATH)} ({path.stat().st_size} bytes)")


def section_reopen_new_process() -> None:
    banner("REOPEN IN A NEW PROCESS: cross-tool verify via reopen_check.py")
    query = "persistence and manifest verification"
    result = subprocess.run(
        [sys.executable, str(PROJECT_DIR / "reopen_check.py"), str(STORE_PATH), query],
        capture_output=True,
        text=True,
        cwd=PROJECT_DIR,
        check=False,
    )
    print(f"subprocess exit code: {result.returncode}")
    if result.stdout:
        print(f"subprocess stdout: {result.stdout.strip()}")
    if result.returncode != 0:
        print(f"subprocess stderr:\n{result.stderr}")
        raise RuntimeError(
            f"reopen_check.py failed with exit code {result.returncode}; "
            "cross-process persistence could not be verified"
        )

    payload = json.loads(result.stdout)
    hits = payload.get("hits", [])
    if not hits:
        raise RuntimeError(
            "reopen_check.py returned zero hits; cross-process persistence "
            "could not be verified"
        )
    print(f"reopened store returned {len(hits)} hit(s) -- persistence across restart verified")


def main() -> None:
    corpus_files = section_corpus()
    chunks = section_chunk(corpus_files)
    embedding = section_embedding_model()

    store = section_ingest(embedding, chunks)
    section_queries(store)
    section_filtered_query(store)
    section_retriever(store)
    section_delete_and_reupsert(store, corpus_files)
    section_delete_by_filter(store)
    section_persist(store)
    section_reopen_new_process()

    banner("DONE")
    print(f"total wall time: {time.perf_counter() - _START:.2f}s")


if __name__ == "__main__":
    main()
