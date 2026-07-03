# OrdinalDB

[![CI](https://github.com/Project-Navi/ordinaldb/actions/workflows/ci.yml/badge.svg)](https://github.com/Project-Navi/ordinaldb/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

OrdinalDB is an embedded vector database layer for local agent memory, RAG
pipelines, and enterprise-search prototypes that need verified vector retrieval
without running a server.

It pairs OrdVec ordinal quantization with a small persistence and adapter layer:
compact local indexes, stable IDs, inspectable `.odb` bundles, and framework
adapters that preserve the text and metadata semantics those frameworks expect.
The core stays vector-only. Integration state lives beside it, where it can be
validated and owned by the adapter.

## Why It Is Different

- **Tamper-evident vector artifacts, fail-closed verification:** every `.odb`
  bundle and adapter generation carries a sha256-verified `ordvec-manifest`
  manifest; `open_verified` refuses to load on any hash, path, or shape
  mismatch. Adapter control state (`adapter.redb`) is consistency-checked on
  every open — payloads, row tables, generation records, and the active
  generation's digest must all agree, and corruption surfaces as a structured
  error, never a crash — but the redb file as a whole is not yet covered by a
  cryptographic digest. See [`THREAT_MODEL.md`](THREAT_MODEL.md) for the exact
  scope.
- **Structured ops CLI:** `ordinaldb inspect`, `verify`, `stats`, and
  `adapter export-json` / `import-legacy` / `gc` give you machine-readable JSON
  reports for backup, garbage collection, and diagnostics out of the box.
- **Edge-capable Rust path:** Linux ARM64 is a first-class target with CI
  coverage for Raspberry Pi 4/5 class hardware, plus a Pi runtime profile job.
- **No training, no codebooks:** OrdVec `RankQuant` ordinal quantization builds
  compact 1, 2, and 4-bit indexes with no training step, no learned rotations,
  and no BLAS runtime. Add vectors and search immediately.
- **Local by default:** indexes are ordinary directories on disk, not a
  daemon, cluster, or hosted service.
- **Adapter-native state:** LangChain, LlamaIndex, Haystack, and Agno adapters
  keep documents, metadata, string IDs, and checkpoints in `adapter.redb`
  while vector state lives in immutable `vectors/g000000000001.odb/`-style
  generations.

Use OrdinalDB when you want a portable embedded retrieval layer that can sit
inside an agent runtime, desktop app, CLI, edge service, or integration test
without introducing a vector database server. It is especially useful when the
deployment boundary matters: local files, repeatable loads, explicit metadata
ownership, and no network dependency in the hot path.

## What OrdinalDB Is / Isn't (Yet)

OrdinalDB is alpha software (0.2.0) with a deliberately narrow scope, not an
accidentally incomplete one. It is:

- An embedded vector index and a small persistence/adapter layer — a library,
  not a service.
- Ordinal-retrieval only, powered by OrdVec `RankQuant`.
- Vector-only at the core; frameworks own text, metadata, and string IDs in
  the adapter layer, not the core `.odb` manifest.

It deliberately does **not** (yet) do the following:

- **No server or distributed mode.** There is no daemon, cluster, or network
  protocol. Every index is files on disk in one process.
- **Bring-your-own embeddings.** OrdinalDB never generates, downloads, or
  calls an embedding model. Your application, or the framework adapter
  wrapping OrdinalDB, computes vectors and passes them in as plain `f32`
  arrays.
- **No custom distance metrics.** Search uses OrdVec ordinal similarity only;
  there is no runtime choice of cosine, L2, or dot-product scoring.
- **Single-writer model.** One writer per adapter directory or `.odb` bundle
  at a time. There is no cross-process concurrent read/write sharing
  guarantee.
- **Hybrid search and learning-to-rank are experimental, not default.**
  BM25 + vector fusion and LTR reranking exist and are tested, but ship behind
  feature flags and are Rust-only today — see
  [Hybrid Search And LTR](#hybrid-search-and-ltr-experimental) below.

These are 0.2.0 scope decisions, not unnoticed gaps. See
[`docs/roadmap/0.2.0-feature-parity-spec.md`](docs/roadmap/0.2.0-feature-parity-spec.md),
[`docs/roadmap/0.3.0-api-async-streaming-spec.md`](docs/roadmap/0.3.0-api-async-streaming-spec.md),
and [`docs/roadmap/ltr-hybrid-production-spec.md`](docs/roadmap/ltr-hybrid-production-spec.md)
for what's next.

## Install

### Published packages (from the 0.2.0 release onward)

```bash
cargo add ordinaldb
```

```bash
pip install ordinaldb
```

Optional Python framework adapters are separate extras — see
[Framework Adapters](#framework-adapters) below.

### Build from source

Rust:

```bash
git clone https://github.com/Project-Navi/ordinaldb
cd ordinaldb
cargo build --release -p ordinaldb
```

Python (build the extension, install it, then it's importable):

```bash
git clone https://github.com/Project-Navi/ordinaldb
cd ordinaldb
python -m venv .venv
source .venv/bin/activate  # .venv\Scripts\activate on Windows
pip install maturin

# Editable install into the active virtualenv:
maturin develop --release -m ordinaldb-python/Cargo.toml

# Or build a wheel and install it explicitly:
maturin build --release -m ordinaldb-python/Cargo.toml
pip install target/wheels/ordinaldb-*.whl

python -c "import ordinaldb; print(ordinaldb.OrdinalIndex)"
```

To pull in an optional framework adapter (see
[Framework Adapters](#framework-adapters) below) from the same locally built
wheel, add the extras suffix to the resolved wheel path. The `*` glob has to
be expanded before the extras bracket is appended, so resolve it with command
substitution rather than writing the wildcard and the extras marker in one
unquoted string:

```bash
pip install "$(ls target/wheels/ordinaldb-*.whl)[langchain]"
```

`maturin develop` (or building and installing the wheel) is the step that
actually makes `import ordinaldb` work — running the Rust build alone only
produces a `.so`/`.pyd` artifact, not an installed package.

## Quick Start

Rust accepts flat row-major `f32` slices. Python accepts 2D C-contiguous `f32`
batches; IDs and allowlists are 1D C-contiguous `uint64` arrays.

Rust:

```rust
use ordinaldb::{IdMapIndex, OrdinalIndex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dim = 64;
    let vectors: Vec<f32> = (0..16 * dim)
        .map(|i| ((i % 23) as f32 - 11.0) / 11.0)
        .collect();
    let queries: Vec<f32> = vectors[..2 * dim].to_vec();
    let index_path = std::env::temp_dir().join("ordinaldb-quickstart.odb");
    let id_index_path = std::env::temp_dir().join("ordinaldb-quickstart-ids.odb");

    // Positional index: rows are addressed by insertion order.
    let mut idx = OrdinalIndex::new(dim, 2)?;
    idx.add_2d(&vectors, dim)?;
    let _results = idx.search_checked(&queries, 4)?;
    idx.write(&index_path)?;

    let loaded = OrdinalIndex::load(&index_path)?;
    let _ = loaded.search_checked(&queries, 4)?;

    // Stable-ID index: rows are addressed by caller-assigned u64 IDs.
    let mut ids = IdMapIndex::new(dim, 2)?;
    let external_ids: Vec<u64> = (0..16).map(|row| 1000 + row as u64).collect();
    ids.add_with_ids_2d(&vectors, dim, &external_ids)?;
    let (_scores, found_ids) = ids.search_checked(&queries, 4)?;
    let (_filtered_scores, _filtered_ids) =
        ids.search_checked_with_allowlist(&queries, 4, Some(&[1001, 1004, 1007]))?;
    ids.write(&id_index_path)?;

    println!("found {} unfiltered neighbor ids", found_ids.len());
    Ok(())
}
```

The quickstart above uses synthetic vectors so it has no external dependency.
For real local embeddings in Rust, the [`fastembed`](https://crates.io/crates/fastembed)
crate pairs well with OrdinalDB: its 384-dim MiniLM models satisfy the
`bits=2` packing constraint (dim divisible by 4) and stay on OrdVec's
`SignBitmap` fast path.

Python:

```python
import tempfile
from pathlib import Path

import numpy as np

from ordinaldb import IdMapIndex, OrdinalIndex

dim = 64
vectors = np.arange(16 * dim, dtype=np.float32).reshape(16, dim)
queries = vectors[:2].copy()
tmp_dir = Path(tempfile.gettempdir())
index_path = tmp_dir / "ordinaldb-quickstart.odb"
id_index_path = tmp_dir / "ordinaldb-quickstart-ids.odb"

# Positional index: rows are addressed by insertion order.
idx = OrdinalIndex(dim=dim, bits=2)
idx.add(vectors)
scores, indices = idx.search(queries, k=4)
idx.write(index_path)
loaded = OrdinalIndex.load(index_path)

# Stable-ID index: rows are addressed by caller-assigned uint64 IDs.
ids = np.arange(1000, 1016, dtype=np.uint64)
id_idx = IdMapIndex(dim=dim, bits=2)
id_idx.add_with_ids(vectors, ids)
scores, found_ids = id_idx.search(queries, k=4)
allowed_ids = np.array([1001, 1004, 1007], dtype=np.uint64)
filtered_scores, filtered_ids = id_idx.search(queries, k=4, allowlist=allowed_ids)
id_idx.write(id_index_path)
loaded_ids = IdMapIndex.load(id_index_path)

print(f"found {len(found_ids)} unfiltered neighbor ids")
```

Rust's `add_2d`/`search_checked`/`search_checked_with_allowlist` return a
`Result` instead of panicking on malformed input: non-finite query values, a
dimension mismatch, or an allowlist ID that isn't present in the index all
come back as a `DenseError`/`AddError`, not a panic. The Python bindings route
through the same checked paths and release the GIL during `add`/`search`, so
malformed input (wrong dtype/shape, non-finite values, an unknown allowlist
ID) raises a catchable `ValueError` and concurrent Python callers aren't
serialized behind the native search. See [`docs/api.md`](docs/api.md) for the
full API reference.

Default unfiltered search uses OrdVec `SignBitmap` candidate generation followed
by `RankQuant` b=2 reranking when the index shape supports sign bitmaps. Other
bit widths and dimensions use direct `RankQuant` search. Allowlist-filtered
search reranks exactly inside the allowed ID set.

Scores are OrdinalDB similarity scores. They are not advertised as cosine
scores or normalized relevance scores.

## Capabilities And Limits

| Area | 0.2.0 status |
| --- | --- |
| Vector index types | `OrdinalIndex` (positional), `IdMapIndex` (stable `u64` IDs, delete, allowlist search) |
| Persistence | Local `.odb` directory bundles, sha256-verified manifests, fail-closed `open_verified` loads; adapter.redb control state cross-verified on open (tamper-evidence scope in [`THREAT_MODEL.md`](THREAT_MODEL.md)) |
| Framework adapters | LangChain, LlamaIndex, Haystack, Agno — vector-search subset (see below) |
| Hybrid search | Experimental: BM25 + vector RRF fusion (`ordinaldb-hybrid`, `--features hybrid`) |
| Learning-to-rank | Experimental: feature cache + tree-ensemble reranking (`ordinaldb-ltr`, `--features experimental-ltr`) |
| Distance metrics | OrdVec ordinal similarity only — no cosine/L2/dot choice |
| Server / distributed mode | Not offered — embedded, single-process only |
| Concurrency | One writer per adapter directory or bundle; readers see stable immutable generations |
| Recommended planning limit | Up to ~100,000 rows per adapter directory (measured; see [`docs/limits.md`](docs/limits.md)) |
| Matryoshka (MRL) embeddings | Truncate-and-index with no renormalization — rank/sign codes are scale-invariant (measured up to 4096 dims; see [`docs/matryoshka.md`](docs/matryoshka.md)) |
| Platforms | Linux x86_64/ARM64 (incl. Raspberry Pi 4/5), macOS ARM64, Windows x64 |

See [`docs/limits.md`](docs/limits.md) for the full measured 10K/100K-row
timings and footprint numbers, and [`docs/operations.md`](docs/operations.md)
for the offline backup, verification, and generation-cleanup runbook.

## Framework Adapters

The base Python package remains framework-free and supports Python 3.9+.
Optional adapters require Python 3.10+ and install only the framework they wrap:

```bash
pip install 'ordinaldb[langchain]'
pip install 'ordinaldb[llama-index]'
pip install 'ordinaldb[haystack]'
pip install 'ordinaldb[agno]'
pip install 'ordinaldb[adapters]'
```

Adapter entrypoints:

- `ordinaldb.langchain.OrdinalDBVectorStore`
- `ordinaldb.llama_index.OrdinalDBVectorStore`
- `ordinaldb.haystack.OrdinalDocumentStore`
- `ordinaldb.haystack.OrdinalEmbeddingRetriever`
- `ordinaldb.agno.OrdinalDb`

Adapter support is vector-search focused. LangChain gets add/search/delete,
upsert, filters, and load/dump paths. LlamaIndex preserves node payloads and
supports default embedding queries. Haystack exposes a document store plus an
embedding retriever with duplicate policies and filter evaluation. Agno exposes
vector database lifecycle helpers and requires an embedder for text search.

Common silent-failure mistakes warn loudly instead of no-oping: adapters emit
`AdapterPathWarning` (suspicious store path), `UnsavedWritesWarning` (writes
buffered in memory with no save), and `UnknownFilterKeyWarning` (zero-hit
filter naming a key no record carries) — see
[`docs/api.md`](docs/api.md#adapter-warnings) for the full table.

All three snippets below use a throwaway hash-seeded local embedder so they
run with no network access and no model download — swap in
`sentence-transformers`, `fastembed`, or your framework's own embedding class
and keep `dim=384, bits=2` (384 is divisible by 4, which `bits=2` requires).

### LangChain

```python
import numpy as np
from langchain_core.embeddings import Embeddings
from ordinaldb.langchain import OrdinalDBVectorStore

class LocalEmbeddings(Embeddings):  # swap in sentence-transformers, etc.
    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]
    def embed_query(self, text):
        return self._embed(text)
    def _embed(self, text):
        seed = abs(hash(text)) % 2**32
        return np.random.default_rng(seed).standard_normal(384).astype("float32").tolist()

store = OrdinalDBVectorStore(embedding=LocalEmbeddings(), dim=384, bits=2, path="langchain-store")
store.add_texts(["local agent memory", "edge RAG pipeline"], ids=["a", "b"])
hits = store.similarity_search("agent memory", k=1)
store.save_local()  # writes stay in memory until save_local() persists them to `path`
```

### LlamaIndex

```python
import numpy as np
from llama_index.core import Settings
from llama_index.core.llms import MockLLM
from llama_index.core.schema import TextNode
from ordinaldb.llama_index import OrdinalDBVectorStore

Settings.llm = MockLLM()  # skip a real LLM call; embeddings still come from your app

def embed(text: str) -> list[float]:
    seed = abs(hash(text)) % 2**32
    return np.random.default_rng(seed).standard_normal(384).astype("float32").tolist()

vector_store = OrdinalDBVectorStore(dim=384, bits=2, path="llama-index-store")
vector_store.add([TextNode(id_="a", text="local agent memory", embedding=embed("local agent memory"))])
vector_store.persist()  # default path; safer than StorageContext.persist(persist_dir=...)
```

### Haystack

```python
import numpy as np
from haystack import Document
from haystack.document_stores.types import DuplicatePolicy
from ordinaldb.haystack import OrdinalDocumentStore, OrdinalEmbeddingRetriever

def embed(text: str) -> list[float]:
    seed = abs(hash(text)) % 2**32
    return np.random.default_rng(seed).standard_normal(384).astype("float32").tolist()

store = OrdinalDocumentStore(dim=384, bits=2, path="haystack-store")
docs = [Document(id="a", content="local agent memory", meta={"status": "approved"}, embedding=embed("local agent memory"))]
store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)
retriever = OrdinalEmbeddingRetriever(document_store=store, top_k=1)
result = retriever.run(query_embedding=embed("agent memory"), filters={"field": "meta.status", "operator": "==", "value": "approved"})
store.save()  # writes stay in memory until save() persists them to `path`
```

`OrdinalEmbeddingRetriever` is decorated with Haystack's `@component`, so it
can be wired into a `Pipeline` with `add_component`/`connect`/`run` like any
built-in Haystack retriever.

### Filter dialects

Filter richness differs per adapter because each one translates its
framework's native filter shape onto a shared exact-match core:

| Adapter | Filter dialect | Operators |
| --- | --- | --- |
| LangChain | Exact-match AND (dict), or an arbitrary Python callable over `Document` | `==` only for dict filters |
| LlamaIndex | Full `MetadataFilters`/`FilterOperator` dialect | `EQ`, `NE`, `GT`, `GTE`, `LT`, `LTE`, `IN`, `NIN`, `ANY`, `ALL`, combined with `AND`/`OR`/`NOT` |
| Haystack | Full Haystack filter dialect via `document_matches_filter` | Comparison and logical (`AND`/`OR`/`NOT`) operators, including nested filters |
| Agno | Exact-match AND (dict) | `==` only |

Every dialect above bottoms out in the same place: filtering computes a
pre-search allowlist of matching records, and the vector search then ranks
top-k **within that allowlist**. Filtered results are "top-k of the filtered
set," not a global top-k with non-matching rows filtered out afterward. See
[`docs/api.md`](docs/api.md) for the adapter API reference.

For edge and local deployments, see
[`docs/edge-deployment.md`](docs/edge-deployment.md). It covers local embedding
ownership, valid `bits`/`dim` combinations, explicit save boundaries, offline
wheelhouse installation, and telemetry controls for framework extras.

Some framework dependencies (Haystack, PostHog-backed telemetry, Agno) phone
home by default. For local-first deployments where no-egress matters, opt out
before importing the framework:

```bash
export HAYSTACK_TELEMETRY_ENABLED=False
export HAYSTACK_DISABLE_TELEMETRY=1
export POSTHOG_DISABLED=1
export AGNO_TELEMETRY=false
```

None of this is required for OrdinalDB itself — the adapters never call a
model or telemetry endpoint — but it keeps the underlying framework package
from making its own outbound calls in an offline or air-gapped environment.

Framework adapters are vector-search only: MMR, sparse search, text search,
hybrid search, and normalized relevance-score APIs are not implemented at the
adapter layer and fail clearly instead of returning misleading results.
Hybrid BM25+vector search does exist as an experimental, feature-gated Rust
capability — see the next section — but it is not yet wired into the Python
framework adapters.

## Hybrid Search And LTR (Experimental)

OrdinalDB ships two additional Rust crates for sparse/hybrid retrieval and
learning-to-rank reranking. Both are real, tested code — `ordinaldb-hybrid`
carries a compile-tested, runnable crate-level doctest of the full
sparse-sidecar + RRF path, and `examples/downstream-smoke` exercises it as a
downstream consumer — but they are **experimental and feature-gated**: off
by default, not yet exposed through the Python bindings, and with a serving
/ training boundary spelled out below.

- **`ordinaldb-hybrid`** — a BM25 mmap sparse index, allowlist-aware sparse
  search, and RRF (Reciprocal Rank Fusion) to combine BM25 with OrdVec vector
  search. Enable it on the `ordinaldb` crate:

  ```toml
  ordinaldb = { version = "0.2.0", features = ["hybrid"] }
  ```

  or build directly against the crate: `ordinaldb-hybrid = "0.2.0"`. Enabled
  types are re-exported at `ordinaldb::hybrid` (and auxiliary artifact names
  at `ordinaldb::artifacts`).

- **`ordinaldb-ltr`** — a learning-to-rank feature cache and tree-ensemble
  reranker on top of hybrid search results. The `ordinaldb` crate's
  `experimental-ltr` feature (which implies `hybrid`) unlocks the LTR
  reranking types under `ordinaldb::hybrid`:

  ```toml
  ordinaldb = { version = "0.2.0", features = ["experimental-ltr"] }
  ```

  The standalone `ordinaldb-ltr` crate (feature-cache export and training
  utilities) is what the CLI uses:

  ```bash
  cargo build -p ordinaldb-cli --features experimental-ltr
  cargo build -p ordinaldb-cli --features experimental-ltr-local-train
  ```

Where the LTR boundary is today:

- **Serving-side LTR is implemented and tested** — `TreeEnsembleReranker`,
  `LtrFeatureBatch`, `rerank_fused_batch`, and the feature-cache write/read
  paths all work end to end.
- **No training path exists anywhere in OrdinalDB.** `ordinaldb ltr train`,
  `ordinaldb ltr attach`, and `ordinaldb ltr inspect` are stubs that return a
  clear "not implemented" error rather than a silent no-op. You bring an
  external trainer and convert its output to `LtrTreeEnsembleRecord`'s JSON
  format yourself; the model header currently requires `training_objective`
  `"rank:pairwise"` and `booster` `"gbtree"`.
- `ordinaldb ltr features` exports a grouped feature cache from a verified
  bundle and works — with exactly the feature triple
  `[bm25_score, bm25_rank, query_len_chars]`.
- **Building LTR features requires exhaustive per-signal coverage.**
  `LtrFeatureBatch::from_inputs` errors if any fused candidate lacks an
  explicit score in any configured source — which candidates found by only
  one retrieval mode (the normal hybrid case) will. Workaround: backfill by
  searching each side with a `top_k` large enough (up to the corpus size)
  to cover every fused candidate, separate from the smaller `top_k` used
  for user-facing results. This limitation is documented, not fixed, in
  0.2.0.

See [`docs/api.md`](docs/api.md) for the full feature-flag matrix.

## Persistence Model

Core indexes persist as directory-backed `.odb` bundles:

```text
docs.odb/
    manifest.json
    index.ovrq
    sign.ovsb      # present when the sign stage is available
    ids.bin        # IdMapIndex only
```

New writes use OrdVec 0.5 `.ov*` artifact names. Manifest-verified loads remain
path-driven, so bundles whose manifest points at legacy `.tvrq` or `.tvsb`
artifacts continue to load.

Framework adapters persist to adapter directories:

```text
adapter-store/
    adapter.redb
    vectors/
        g000000000001.odb/
            manifest.json
            index.ovrq
            sign.ovsb      # present when the sign stage is available
```

`adapter.redb` is the canonical control-plane store for new adapter writes. It
binds the active vector generation, string-ID map, `u64` slot map, documents,
metadata, schema, and checkpoint tables. `adapter.json`, `id_map.json`,
`documents.json`, and `metadata.json` are compatibility exports of that state;
ordinary load and verify paths treat `adapter.redb` as authoritative. The
exports can also be recreated explicitly:

```bash
cargo run -p ordinaldb-cli -- adapter export-json adapter-store
cargo run -p ordinaldb-cli -- adapter import-legacy legacy-adapter-store --output imported-store
cargo run -p ordinaldb-cli -- adapter gc adapter-store --retain 2
```

The core `.odb` manifest verifies vector artifacts only. Adapter state is
verified by `adapter.redb`, the Python adapter layer, and `ordinaldb verify`.
See [`docs/persistence.md`](docs/persistence.md) for the full layout and
failure model.

Every adapter `save()` publishes a complete new vector generation — it
rewrites the whole current vector set, not just what changed since the last
save — and old generations are never garbage-collected automatically; run
`ordinaldb adapter gc --retain N` deliberately to reclaim them. Because that
rewrite cost scales with total store size, prefer batching writes over
per-item `auto_save`; see [`docs/persistence.md`](docs/persistence.md#adapter-directories)
for the measured numbers.

## Platform Support

Primary Rust targets:

- Linux x86_64
- Linux ARM64
- macOS ARM64
- Windows x64

Linux ARM64 is the Raspberry Pi class path. Use a Raspberry Pi 4 or Pi 5 with a
64-bit OS for the supported route. CI runs the Rust suite and downstream smoke
tests on a native `ubuntu-24.04-arm` runner, plus a Pi runtime profile job that
exercises persistence and Rayon worker counts.

The 32-bit Raspberry Pi OS target is best-effort until the project carries an
explicit `armv7-unknown-linux-gnueabihf` lane. See
[`docs/raspberry-pi.md`](docs/raspberry-pi.md) for Pi, QEMU/virt-manager, and
AWS Graviton testing guidance.

## Build And Inspect

Build the Rust workspace:

```bash
cargo build --release
```

Inspect or verify a core `.odb` bundle or adapter directory:

```bash
cargo run -p ordinaldb-cli -- inspect docs.odb
cargo run -p ordinaldb-cli -- verify adapter-store
```

Run examples (see [Install](#install) for the Python environment setup):

```bash
cargo run -p ordinaldb --example rust_basic
python examples/python_basic.py
python examples/python_idmap.py
python examples/python_allowlist.py
python examples/benchmark_smoke.py --vectors 10000 --queries 64 --dim 64 --k 10
```

For complete, runnable applications — a LangChain docs-Q&A tool, a
LlamaIndex research-paper navigator with advanced metadata filtering, and a
durable agent-memory store with a crash-recovery walkthrough — see the
[cookbook](cookbook/README.md).

## Project Status

OrdinalDB is alpha software. The 0.2.0 line lands the redb-backed adapter
control plane, immutable generation layout, and storage-hardening work while
depending on the published `ordvec` and `ordvec-manifest` 0.5.0 crates. See
[What OrdinalDB Is / Isn't (Yet)](#what-ordinaldb-is--isnt-yet) for the
current scope boundaries.

Project references:

- [`LICENSE`](LICENSE)
- [`CONTRIBUTING.md`](CONTRIBUTING.md)
- [`CHANGELOG.md`](CHANGELOG.md)
- [`SECURITY.md`](SECURITY.md)
- [`THREAT_MODEL.md`](THREAT_MODEL.md)
- [`docs/api.md`](docs/api.md)
- [`docs/edge-deployment.md`](docs/edge-deployment.md)
- [`docs/persistence.md`](docs/persistence.md)
- [`docs/limits.md`](docs/limits.md)
- [`docs/matryoshka.md`](docs/matryoshka.md)
- [`docs/operations.md`](docs/operations.md)
- [`docs/roadmap/0.2.0-feature-parity-spec.md`](docs/roadmap/0.2.0-feature-parity-spec.md)
- [`docs/roadmap/0.3.0-api-async-streaming-spec.md`](docs/roadmap/0.3.0-api-async-streaming-spec.md)
- [`docs/roadmap/ltr-hybrid-production-spec.md`](docs/roadmap/ltr-hybrid-production-spec.md)
- [`THIRD_PARTY.md`](THIRD_PARTY.md)
- [`NOTICE`](NOTICE)
- [`docs/provenance.md`](docs/provenance.md)
