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
afterward. Each adapter's supported operators are documented in its own
module.

Rust sparse/hybrid retrieval lives in the `ordinaldb-hybrid` crate: BM25 mmap
auxiliaries, allowlist-aware sparse search, and RRF fusion. None of this is
exposed through the Python bindings yet — see [Hybrid Search](#hybrid-search)
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

## Hybrid Search

`ordinaldb-hybrid` is a real, tested Rust crate, but hybrid retrieval is
experimental and off by default:

| Crate | Feature flag | Enables |
| --- | --- | --- |
| `ordinaldb` | `hybrid` | `ordinaldb::hybrid::*` — BM25 mmap index, allowlist-aware sparse search, RRF fusion |

Status:

- Experimental and Rust-only — not exposed through the Python bindings
  (`ordinaldb-python`) yet.
- `ordinaldb verify` checks manifest checksums for recognized hybrid sidecars.
- See the root [`README.md`](../README.md) for a narrative overview, the
  `ordinaldb-hybrid` crate-level rustdoc for a compile-tested sparse-sidecar +
  RRF example, and `examples/downstream-smoke/src/main.rs` for a working
  `--features hybrid` walkthrough (build a verified bundle with a BM25 sidecar,
  then dense search, sparse search, and RRF fuse the two).
