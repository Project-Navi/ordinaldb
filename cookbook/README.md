# OrdinalDB Cookbook

Four small, complete applications built on OrdinalDB. Each one is a real
use case, not a synthetic unit test: copy the directory, follow its
README's Quickstart, and it runs end to end with no API keys.

| Example | What it is | Read this when you want to see... |
| --- | --- | --- |
| [`docspilot/`](docspilot/) | Documentation Q&A retrieval over a real markdown corpus, via the LangChain adapter | metadata-filtered search, explicit persistence with warnings, and reopening a store from a fresh process |
| [`paperscout/`](paperscout/) | A research-paper discovery tool over ~40 real papers, via the LlamaIndex adapter | the full `FilterOperator` metadata-filter dialect and the LlamaIndex-native vector-store lifecycle (`delete_ref_doc`, `get_nodes`/`delete_nodes`, idiomatic `persist()` + reload) |
| [`keeper/`](keeper/) | Durable long-term memory for an AI agent, via the Agno adapter | crash-safe persistence, the `verify`/`adapter gc` operational commands, and tamper evidence |
| [`supportsearch/`](supportsearch/) | A support-knowledge-base search tool over ~120 KB articles/tickets, directly on the **experimental** `ordinaldb-hybrid`/`ordinaldb-ltr` Rust crates (no Python adapter) | BM25 vs. dense embedding search actually disagreeing on real queries, RRF fusion resolving it, and the LTR reranking serving path |

## Choosing one to read first

- New to OrdinalDB's **Python framework adapters**? Start with `docspilot/`
  -- it's the most straightforward retrieval pipeline (chunk -> embed ->
  store -> query) and the most honest look at what persistence actually
  requires.
- Care about **advanced metadata filtering**, or want to see the adapter's
  vector-store API used beyond `add`/`similarity_search`? `paperscout/`
  runs LlamaIndex's complete filter dialect and its document-lifecycle
  methods (`delete_ref_doc`, `get_nodes`, `delete_nodes`, `clear`).
- Care about **what happens when something goes wrong** (a crash, a
  corrupted file, routine disk cleanup) or day-2 operations? `keeper/`'s
  `durability_demo.py` walks through crash recovery, `verify`, and `gc`.
- Care about **hybrid (BM25 + dense) search or learning-to-rank**, in Rust,
  with no Python adapter involved? `supportsearch/` is the first real
  consumer of `ordinaldb-hybrid`/`ordinaldb-ltr` and says plainly where
  their experimental edges are.

## Running any of them

Each example assumes OrdinalDB is already built from source at the repo
root, per the top-level README's "Build from source" section. From there,
every example's own README has a self-contained Quickstart you can copy and
paste from that example's directory -- no example depends on another.
