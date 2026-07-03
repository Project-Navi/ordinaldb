"""Standalone reload script, run as a brand-new OS process by demo.py.

Proves persistence-across-restart for real: this process shares no Python
state with the process that built and persisted the index. It just opens
the on-disk store and queries it.

Usage: python reload_check.py <store_dir> <query text> [--from-persist-dir]

By default reloads with the plain constructor, OrdinalDBVectorStore(path=
store_dir) -- the natural thing to do since that's the same path used at
construction. Pass --from-persist-dir to instead reload with the
OrdinalDBVectorStore.from_persist_dir(store_dir) classmethod, which fails
closed (raises) instead of silently returning an empty store if the
directory doesn't hold one.
"""

from __future__ import annotations

import os
import sys

from ordinaldb.llama_index import OrdinalDBVectorStore
from paperscout import configure_local_settings, load_index_from_store, open_vector_store


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: reload_check.py <store_dir> <query text> [--from-persist-dir]", file=sys.stderr)
        raise SystemExit(2)
    use_from_persist_dir = "--from-persist-dir" in sys.argv[1:]
    args = [arg for arg in sys.argv[1:] if arg != "--from-persist-dir"]
    store_dir = args[0]
    query_text = " ".join(args[1:])

    embed_model = configure_local_settings()
    if use_from_persist_dir:
        vector_store = OrdinalDBVectorStore.from_persist_dir(store_dir)
        print(f"[reload_check] reloaded via OrdinalDBVectorStore.from_persist_dir({store_dir!r})")
    else:
        vector_store = open_vector_store(store_dir)
    record_count = len(vector_store.client)
    print(f"[reload_check pid={os.getpid()}] reopened store: {record_count} records")

    index = load_index_from_store(vector_store, embed_model)
    retriever = index.as_retriever(similarity_top_k=3)
    nodes = retriever.retrieve(query_text)
    print(f"[reload_check] query: {query_text!r}")
    for node in nodes:
        title = node.node.metadata.get("title")
        category = node.node.metadata.get("category")
        year = node.node.metadata.get("year")
        print(f"  - score={node.score:.4f} [{category} {year}] {title}")


if __name__ == "__main__":
    main()
