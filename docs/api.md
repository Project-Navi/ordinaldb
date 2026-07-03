# OrdinalDB API

OrdinalDB exposes two MVP index types:

- `OrdinalIndex`: positional local vector index.
- `IdMapIndex`: stable `u64` external IDs layered over `OrdinalIndex`.

For b=2 indexes whose dimension supports OrdVec sign bitmaps, default
unfiltered search uses `SignBitmap` candidate generation followed by
`RankQuant` reranking. Filtered search uses exact `RankQuant` subset reranking
over the mask or allowlist slots so it never falls outside the requested filter.

The Rust and Python APIs are converging on the same names and semantics:

```rust
use ordinaldb::{IdMapIndex, OrdinalIndex};
```

```python
from ordinaldb import IdMapIndex, OrdinalIndex
```

The persistence extension for new OrdinalDB indexes is `.odb`. The new format
is a directory-backed bundle with an OrdVec `RankQuant` primary artifact and an
`ordvec-manifest` `manifest.json`. `IdMapIndex` bundles include an OrdinalDB
`ids.bin` sidecar for stable `u64` IDs. See [`persistence.md`](persistence.md).

Rust persistence:

```rust
idx.write("docs.odb")?;
let idx = OrdinalIndex::load("docs.odb")?;

ids.write("docs_ids.odb")?;
let ids = IdMapIndex::load("docs_ids.odb")?;
```

Python persistence:

```python
idx.write("docs.odb")
idx = OrdinalIndex.load("docs.odb")

ids.write("docs_ids.odb")
ids = IdMapIndex.load("docs_ids.odb")
```

Lazy indexes can be used for deferred dimension selection, but an index must
have a committed dimension before it can be persisted.

## Optional Python adapters

The base `ordinaldb` import does not import framework packages. Each adapter
module imports only its own optional dependency and raises an install-hint
`ImportError` when the extra is missing.

Supported entrypoints:

- `ordinaldb.langchain.OrdinalDBVectorStore`
- `ordinaldb.llama_index.OrdinalDBVectorStore`
- `ordinaldb.haystack.OrdinalDocumentStore`
- `ordinaldb.haystack.OrdinalEmbeddingRetriever`
- `ordinaldb.agno.OrdinalDb`

Adapter support is vector-search only. LangChain MMR and normalized relevance
score methods raise clear errors until honest normalization exists. LlamaIndex
supports default embedding queries and rejects hybrid, sparse, text-search,
semantic-hybrid, and MMR-style modes. Haystack supports duplicate policies
`FAIL`, `SKIP`, and `OVERWRITE`, plus comparison and logical metadata filters.
Agno text search requires an embedder; `search_by_vector(...)` is available for
callers that already have vectors.

Filter dialect richness is not uniform across adapters: LangChain and Agno
dict filters are exact-match AND only, Haystack accepts its full native filter
dialect (comparison and logical operators, including nesting) via
`document_matches_filter`, and LlamaIndex accepts the full `FilterOperator` set
(`EQ`, `NE`, `GT`, `GTE`, `LT`, `LTE`, `IN`, `NIN`, `ANY`, `ALL`) combined with
`AND`/`OR`/`NOT`. In every adapter, a filter is resolved to a pre-search
allowlist of matching records before the vector search runs, so filtered
results are the top-k **within that allowlist**, not a global top-k filtered
afterward. See the root [`README.md`](../README.md#filter-dialects) for the
per-adapter matrix.

Rust sparse/hybrid retrieval lives in the `ordinaldb-hybrid` crate: BM25 mmap
auxiliaries, allowlist-aware sparse search, RRF fusion, and gated LTR support.
None of this is exposed through the Python bindings yet — see
[Hybrid Search And LTR Feature Flags](#hybrid-search-and-ltr-feature-flags)
below.

Adapter dimensions follow the core RankQuant constraints: `bits=1` requires
dimensions divisible by 8, `bits=2` by 4, and `bits=4` by 16. Missing gets
return no result, missing deletes are no-ops unless the framework says
otherwise, and filtered-empty searches return an empty result without calling
core vector search. See [`edge-deployment.md`](edge-deployment.md) for local
embedding, offline install, and telemetry guidance.

### Adapter warnings

Common silent-failure mistakes warn loudly instead of no-oping. All three
classes are `UserWarning` subclasses exported from `ordinaldb.adapters` and
emitted by every framework adapter, so they can be silenced or escalated to
errors with Python's standard `warnings` filters:

| Warning | Emitted when |
| --- | --- |
| `AdapterPathWarning` | A store is constructed against a path that exists but contains no valid store markers — a typo'd path, a directory of unrelated files, crash debris, or a store nested one level below. |
| `UnsavedWritesWarning` | The first unsaved write of each epoch to a path-bound store. Adapter mutations only touch memory until an explicit save (LangChain `save_local()`/`persist()`, LlamaIndex `persist()`, Agno `create()`/`save()`, `AdapterStore.save()`); a successful save re-arms the warning for the next unsaved batch. |
| `UnknownFilterKeyWarning` | A metadata filter matches zero records *and* names at least one key that no stored record carries — usually a typo (`doctype` vs `doc_type`). The warning names the unknown key(s). |

## Hybrid Search And LTR Feature Flags

`ordinaldb-hybrid` and `ordinaldb-ltr` are real, tested Rust crates, but they
are experimental and off by default:

| Crate | Feature flag | Enables |
| --- | --- | --- |
| `ordinaldb` | `hybrid` | `ordinaldb::hybrid::*` — BM25 mmap index, allowlist-aware sparse search, RRF fusion |
| `ordinaldb` | `experimental-ltr` (implies `hybrid`) | LTR reranking types under `ordinaldb::hybrid` (`ordinaldb-hybrid`'s own `ltr` feature) |
| `ordinaldb-hybrid` | `ltr` | LTR-specific types within `ordinaldb-hybrid` itself |
| `ordinaldb-cli` | `experimental-ltr` | `ordinaldb ltr features/train/attach/inspect` subcommands |
| `ordinaldb-cli` | `experimental-ltr-local-train` (implies `experimental-ltr`) | local LTR training extras |

Status:

- Not exposed through the Python bindings (`ordinaldb-python`) yet — Rust
  only.
- **Serving-side LTR is implemented and tested**: `TreeEnsembleReranker`,
  `LtrFeatureBatch`, `rerank_fused_batch`, and the `ordinaldb-ltr`
  feature-cache write/read paths.
- **No training path exists anywhere in OrdinalDB.** `ordinaldb ltr train`,
  `ordinaldb ltr attach`, and `ordinaldb ltr inspect` are stubs that return
  a "not implemented yet" error rather than a silent no-op. Users bring an
  external trainer and must convert its output to `LtrTreeEnsembleRecord`'s
  JSON format themselves; the model header currently requires
  `training_objective` `"rank:pairwise"` and `booster` `"gbtree"`.
- `ordinaldb ltr features` (export a grouped feature cache from a verified
  bundle) is implemented and exports exactly the feature triple
  `[bm25_score, bm25_rank, query_len_chars]` — no dense or fused features
  yet.
- **`LtrFeatureBatch::from_inputs` requires an explicit score in every
  configured source for every fused row.** A candidate found by only one
  retrieval mode (the normal hybrid case) has no entry in the other mode's
  `RankedBatch`, and feature building errors on the first gap instead of
  substituting a sentinel. Workaround: backfill scores by running each side
  with a `top_k` large enough (up to the corpus size) that every fused
  candidate is covered, separate from the smaller `top_k` used for
  user-facing results. This coverage requirement (and the hardcoded model
  header above) is a documented 0.2.0 limitation, slated for a later
  release, not fixed here.
- `ordinaldb verify` structurally opens recognized hybrid/LTR sidecars
  (`ordinaldb.sparse_bm25`, `ordinaldb.ltr_model`, `ordinaldb.ltr_features`)
  through their domain loaders when built with `--features experimental-ltr`;
  without that feature it still checks manifest checksums and reports a
  warning that those sidecars were not structurally validated.
- See the root [`README.md`](../README.md#hybrid-search-and-ltr-experimental)
  for a narrative overview, the `ordinaldb-hybrid` crate-level rustdoc for a
  compile-tested sparse-sidecar + RRF example, and
  `examples/downstream-smoke/src/main.rs` for a working `--features hybrid`
  walkthrough (build/write a verified bundle with a BM25 sidecar, then dense
  search, sparse search, and RRF fuse the two).
