# PaperScout

A local research-paper discovery tool for an ML team, built on OrdinalDB's
LlamaIndex adapter: index paper abstracts with local embeddings, then run
semantic discovery, rich metadata-filtered search, and idiomatic
persistence -- entirely offline, no API keys.

## The problem

An ML team tracking a fast-moving literature (RAG, retrieval, vector
databases) wants more than "search my papers." They want to slice by
category and year, delete a paper that turned out to be a duplicate, and
trust that closing their notebook and reopening it tomorrow doesn't lose
anything. PaperScout is a small discovery tool over ~40 real papers
spanning cs.IR, cs.LG, and cs.DB that exercises LlamaIndex's full metadata
filter dialect and its complete vector-store lifecycle -- not just
`add()` and `similarity_search()`.

## Quickstart

From this directory, with OrdinalDB already built from source at the repo
root (`cargo build --release` and
`maturin build --release -m ordinaldb-python/Cargo.toml`, per the top-level
README's "Build from source" section):

```bash
cd cookbook/paperscout
python3 -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install "$(ls ../../target/wheels/ordinaldb-*.whl)[llama-index]"
pip install -r requirements.txt
python demo.py
```

The wildcard has to resolve via command substitution before the
`[llama-index]` extras marker is appended -- see the top-level README's
"Build from source" section for why.

The first run downloads the `all-MiniLM-L6-v2` model weights (CPU-only, no
API key). `demo.py` sets `Settings.llm = MockLLM()` so the LlamaIndex
query-engine path works with no LLM and no API key at all -- without it,
even a retrieval-only `as_query_engine(response_mode="no_text")` tries to
resolve a default OpenAI LLM and raises an `ImportError`.

By default `demo.py` indexes the bundled 40-paper corpus (`corpus_data.py`)
-- fully offline, deterministic, no network required. Pass `--live` to
fetch fresh abstracts from the public arXiv API instead:

```bash
python demo.py --live
```

`--live` falls back to the bundled corpus automatically if arXiv is
unreachable or rate-limits the request, logging which path it took.

Vector data lands in `storage/` (gitignored, created on first run).

## What to look at in the code

```
corpus_data.py    -- the bundled 40-paper default corpus (cs.IR/cs.LG/cs.DB,
                      1996-2025), real papers with paraphrased abstracts
fetch_papers.py   -- load_papers(live=...): bundled by default, arXiv on
                      --live with automatic fallback and cross-listed-paper
                      dedup
paperscout.py     -- core library: embedding config, Document construction,
                      vector store helpers
demo.py           -- the runnable end-to-end walkthrough
reload_check.py   -- standalone script launched as a separate OS process by
                      demo.py, to prove persistence survives a restart
```

`demo.py` runs, in order: corpus load -> local embeddings -> build the
index -> two discovery queries (retriever and query-engine) -> five
metadata-filter steps, ending in the full `FilterOperator` matrix -> node
round-trip fidelity -> the ref-doc lifecycle (`delete_ref_doc`,
`get_nodes`/`delete_nodes(filters=...)`, `clear()`) -> four failure-path
probes -> idiomatic persistence with a real cross-process reload.

## Differentiators demonstrated

Both PaperScout and `docspilot/` are Python framework adapters over
OrdinalDB; PaperScout exercises the parts of the LlamaIndex integration
that DocsPilot's LangChain adapter doesn't have an equivalent for:

- **The full LlamaIndex `FilterOperator` dialect** -- not just exact-match
  AND. STEP 4 runs all nine operators (`EQ`/`NE`/`GT`/`GTE`/`LT`/`LTE`/
  `IN`/`NIN`/`ANY`/`ALL`) plus `AND`/`OR`/`NOT` composition, and prints the
  result of each. Every filter resolves to an ID allowlist **before** the
  top-k vector search runs, so a filtered query returns the true top-k
  *within the filtered set* -- not a global top-k with non-matching hits
  discarded afterward.
- **The ref-doc lifecycle** -- `index.delete_ref_doc()` removes a document
  by the id it was indexed under; `vector_store.get_nodes(filters=...)`,
  `.delete_nodes(filters=...)`, and `.clear()` operate directly on the
  adapter without going through the LlamaIndex retriever API at all.
- **Idiomatic persistence with a genuine reload** -- `demo.py` calls
  `index.storage_context.persist(persist_dir=...)`, the pattern
  LlamaIndex's own docs teach, and `reload_check.py` reopens the same
  store from a brand-new OS process twice: once via the plain
  `OrdinalDBVectorStore(path=...)` constructor and once via the
  `OrdinalDBVectorStore.from_persist_dir(...)` classmethod.
