# SupportSearch

A support-knowledge-base search tool over ~120 realistic KB articles and
tickets for a fictional API platform ("Meridian"), built directly on
**`ordinaldb-hybrid`** (BM25 mmap sparse index + RRF fusion) and
**`ordinaldb-ltr`**/`ordinaldb::hybrid`'s LTR reranking types. There is no
LLM generation step -- three query classes and their real, printed retrieval
results are the product.

## EXPERIMENTAL

This crate is the first real application ever built against
`ordinaldb-hybrid` and `ordinaldb-ltr`. Both ship behind Cargo feature flags
(`ordinaldb = { features = ["hybrid"] }`, `experimental-ltr`), are described
in the root README as "real, tested Rust crates" that are nonetheless
**experimental**, Rust-only, and not yet exposed through the Python
bindings. Nothing here should be read as "hybrid search in OrdinalDB is
production-ready" -- it should be read as "here is exactly what happens
when a real consumer uses it for the first time," including the rough
edges. See [Rough edges found here](#rough-edges-found-here) below.

## What this shows

- **Class 1 -- exact identifier**: the query `ERR_POOL_EXHAUSTED_5432`. BM25
  nails the one document containing that exact code, decisively (score
  15.2 vs. a distant second at 5.5); dense embedding search, faced with six
  near-duplicate connection-pool documents, ranks a *different* one first.
- **Class 2 -- paraphrase**: the query `can't log in after the cert change`
  against a KB article that deliberately shares zero exact terms with that
  phrasing (it says "authenticate," "certificate," "rotation," never "log,"
  "cert," or "change"). Dense embedding search finds it at rank 1; BM25
  doesn't find it in the top 5 at all, and instead surfaces a document that
  only *coincidentally* shares two literal words with the query.
- **Class 3 -- mixed identifier + semantics**: a query combining an error
  code, a version string, and a plain-language ask. The right document wins
  both signals independently, and RRF fusion confirms that agreement at
  rank 1 -- fusion doesn't need to rescue anything when both signals
  already agree, and it doesn't lose anything either.
- **LTR reranking (`--features experimental-ltr`)**: a hand-authored,
  three-stump tree ensemble, loaded through the real
  `TreeEnsembleReranker` verified-sidecar path and scored against genuine
  hybrid search features (BM25 score/rank, OrdVec rank-cosine, true dense
  cosine, RRF score, document/query length). This is where the
  first-consumer story gets most interesting -- see below.

## The problem

Two of these query classes are not contrived. "A support engineer pastes an
exact error code" and "a support engineer describes a symptom in their own
words" are the two most common ways a real person searches a knowledge
base, and they have almost opposite retrieval requirements: the first
needs exact term matching over rare tokens, the second needs semantic
understanding that survives a total vocabulary mismatch. Neither BM25 alone
nor dense embedding alone covers both well. SupportSearch builds one
corpus, salted with realistic identifiers (error codes, config keys, CLI
flags, version strings) precisely so both failure modes -- and hybrid's
answer to them -- are visible in the same run, not two cherry-picked demos.

## Quickstart

From this directory, with OrdinalDB already built from source at the repo
root (see the top-level README's "Build from source" section):

```bash
cd cookbook/supportsearch
cargo build --release
cargo run --release
```

The first run downloads `fastembed`'s `AllMiniLML6V2` ONNX model (~90MB,
384-dim, CPU-only) from Hugging Face -- a few seconds on a normal
connection, cached under `.fastembed_cache/` afterward. If that download
isn't possible in your environment, set `SUPPORTSEARCH_HASH_EMBEDDER=1` to
fall back to a deterministic, non-semantic hash embedder instead; note that
under the fallback, query class 2 no longer demonstrates dense search
*winning*, since a hash embedding has no notion of meaning -- that's the
whole point of it being a fallback, not an alternative.

For the LTR reranking step:

```bash
cargo run --release --features experimental-ltr
```

The KB corpus and the persisted hybrid store (`data/kb.odb/`) are rebuilt
from scratch on every run (gitignored, not checked in). Inspect or verify
the persisted bundle with the ops CLI from the repo root:

```bash
cargo run -p ordinaldb-cli -- inspect cookbook/supportsearch/data/kb.odb
cargo run -p ordinaldb-cli -- verify  cookbook/supportsearch/data/kb.odb
```

## What to look at in the code

```
src/corpus.rs    -- ~120 KB articles/tickets across 20 topics; three are
                    deliberately engineered for the three query classes
                    (see the module doc comment for exactly how and why)
src/embedder.rs  -- Embedder trait; fastembed (real, 384-dim) and a
                    deterministic hash fallback
src/index.rs     -- builds the dense (OrdVec) + sparse (BM25) sides and
                    writes them as one manifest-verified .odb bundle,
                    mirroring examples/downstream-smoke's pattern
src/queries.rs   -- the three demo queries paired with their gold doc ids
src/ltr_demo.rs  -- the hand-authored LTR model + rerank demo
                    (--features experimental-ltr only)
src/main.rs      -- runs all three query classes end to end and prints
                    dense / sparse / RRF-fused results side by side
```

## Differentiators demonstrated (when hybrid matters)

- **Exact identifiers beat semantics, and vice versa, depending on the
  query** -- this is the core hybrid-search pitch, and it is not a
  hypothetical here: both failure modes are real, measured `cargo run`
  output from this corpus, not asserted or cherry-picked after the fact.
- **RRF fusion as agreement-confirmation, not just rescue** -- the textbook
  RRF story is "a document ranked 2nd-3rd in both lists beats a document
  ranked 1st in only one list." Class 3 here shows the *other* legitimate
  case: when both signals already agree, RRF preserves that agreement
  cleanly instead of introducing noise.
- **LTR reranking is real, inference-only, and has a hard edge** -- the
  `TreeEnsembleReranker` verified-sidecar path works end to end against a
  hand-authored model (there is no trainer to call yet; see below), but
  building its input features requires an explicit score, from every
  configured signal, for every candidate in the fused set -- which a
  hybrid search's own candidate set does not naturally guarantee. See
  `ltr_demo.rs`'s test for a minimal, deterministic repro.

## Rough edges found here

Two issues in `ordinaldb-hybrid` directly shaped this crate's own code
(workarounds live here, not upstream -- neither crate was modified to build
this example):

- **`ordinaldb-hybrid`'s BM25 tokenizer can write a sparse index it cannot
  read back.** `normalize_term` strips one trailing `s` from any
  normalized token longer than 4 characters, but does not re-check whether
  the result still ends in `s`. Any word of 6+ letters ending in a double
  `s` -- `access`, `process`, `address`, `success`, `business`, and
  friends, i.e. ordinary English -- round-trips as a term that
  `Bm25MmapIndex::open`'s own validator then rejects as malformed.
  `src/corpus.rs` avoids every such word on purpose (see its module doc
  comment); that avoidance is a workaround for this crate's benefit, not a
  fix to `ordinaldb-hybrid` itself, which was left untouched.
- **Building LTR features requires exhaustive per-signal coverage over the
  RRF-fused candidate set.** A candidate found by only one retrieval mode
  -- normal for hybrid search, since that's the entire premise -- has no
  entry at all in the other mode's `RankedBatch`, and
  `LtrFeatureBatch::from_inputs` errors outright on the first such gap
  rather than substituting a sentinel. `ltr_demo.rs` works around this by
  searching both sides with `top_k` equal to the full corpus size purely to
  backfill LTR features, separate from the small `top_k` used for the
  headline hybrid results. `ltr_demo::tests` has a minimal, corpus-
  independent reproduction of the underlying error.
