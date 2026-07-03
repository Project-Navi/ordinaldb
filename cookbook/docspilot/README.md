# DocsPilot

A documentation Q&A retrieval tool: ingest a real markdown corpus, chunk it,
embed locally with `sentence-transformers`, store the vectors in
**OrdinalDB** via its LangChain adapter, and answer retrieval queries with
source attribution (file + section). There is no LLM generation step --
retrieval quality with sources is the product.

## The problem

Most "chat with your docs" demos use a handful of synthetic paragraphs and
never show what happens when you filter, delete, edit, or reopen the store
later. DocsPilot ingests OrdinalDB's own real documentation (its README plus
every project-level markdown doc) and walks through the parts of a retrieval
app that actually get exercised in production: metadata-filtered search,
updating a changed document, and reopening a persisted store from a fresh
process.

## Quickstart

From this directory, with OrdinalDB already built from source at the repo
root (`cargo build --release` and
`maturin build --release -m ordinaldb-python/Cargo.toml`, per the top-level
README's "Build from source" section):

```bash
cd cookbook/docspilot
python3 -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install "$(ls ../../target/wheels/ordinaldb-*.whl)[langchain]"
pip install -r requirements.txt
python demo.py
```

The wildcard has to resolve via command substitution before the `[langchain]`
extras marker is appended -- see the top-level README's "Build from source"
section for why.

The first run downloads the `all-MiniLM-L6-v2` model weights (CPU-only, no
API key). Vector data lands in `data/adapter-store/` (gitignored, created on
first run).

## What to look at in the code

```
src/docspilot/corpus.py     -- resolves the fixed markdown corpus against the repo root
src/docspilot/chunking.py   -- markdown-aware splitting, tags each chunk with
                                {source, section, doc_type, file_name}, and
                                assigns a stable positional id "{source}#{0000}"
src/docspilot/embeddings.py -- LangChain Embeddings wrapper around
                                sentence-transformers/all-MiniLM-L6-v2 (CPU, 384-dim)
src/docspilot/store.py      -- thin build_store()/open_store() wrappers around
                                ordinaldb.langchain.OrdinalDBVectorStore
demo.py                     -- the runnable end-to-end walkthrough
reopen_check.py             -- standalone script demo.py shells out to, to prove
                                the persisted store survives a real process restart
```

`demo.py` runs, in order: corpus load -> chunking -> local embedding ->
ingest -> similarity queries with source attribution -> a metadata-filtered
query (and what happens when the filter key is misspelled) -> the standard
LangChain retriever interface -> a simulated edited-file delete+re-upsert ->
a filter-based delete -> `save_local()` -> a real subprocess reopen-and-query
against the same on-disk store.

## Differentiators demonstrated

- **Metadata filtering with a pre-search allowlist** -- `similarity_search`
  resolves a metadata filter to an ID allowlist *before* running the vector
  search, instead of searching unfiltered and discarding non-matching hits
  afterward.
- **Explicit persistence with warnings** -- `add_documents`/`add_texts`
  never write to disk implicitly; the adapter warns the first time a
  path-bound store has unsaved writes so you don't lose them by forgetting
  `save_local()`. A typo'd metadata filter key gets its own warning naming
  the bad key, instead of silently matching nothing.
- **Cross-tool CLI verify** -- `reopen_check.py` is a small standalone script
  invoked as its own OS process (not an in-process function call) to prove
  the persisted store round-trips through a real process boundary, the way
  a second tool or a redeployed service would open it.
