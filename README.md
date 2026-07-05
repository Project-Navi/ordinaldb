# OrdinalDB

[![CI](https://github.com/Project-Navi/ordinaldb/actions/workflows/ci.yml/badge.svg)](https://github.com/Project-Navi/ordinaldb/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Vector search you can ship as a single verifiable file.**

OrdinalDB is a training-free, tamper-evident, embedded vector database. Bring
your own embeddings, index the very first vector with zero prior data, and
write the whole index as one self-contained, SHA-256-manifested `.odb` bundle
that scales past a million rows and opens verified anywhere. No server, no
BLAS, no learned codebook, no ANN graph — and no recall knob to get wrong.

## Why It's Different

- **One tamper-evident artifact.** The entire index is a self-contained `.odb`
  bundle with a SHA-256 manifest. `open_verified` catches a single flipped
  byte and refuses to load with an exact, structured error. The manifest hash
  *is* the version: copy it, hash it, sign it, carry it on a USB stick.
  Scope in [`THREAT_MODEL.md`](THREAT_MODEL.md).
- **Runs where servers can't.** Serves large corpora from a fraction of the
  raw-float RAM, with no daemon and no network in the hot path. Linux ARM64 is
  a first-class CI target — Raspberry Pi 4/5 class hardware included. See
  [`docs/edge-deployment.md`](docs/edge-deployment.md) and
  [`docs/raspberry-pi.md`](docs/raspberry-pi.md).
- **No build step, no training, no BLAS.** The substrate is
  [OrdVec](https://crates.io/crates/ordvec) ordinal + sign quantization — a
  per-vector rank/sign transform. No training pass, no learned rotations, no
  codebook to fit, no graph to construct, zero system dependencies. Ingestion
  is append-only and never refits.
- **Retrieval-quality parity, no fragile knob.** On OrdVec's public BEIR
  harness (scored against official qrels), OrdinalDB matches exact dense
  **nDCG@10** — lossless at `bits=4`, within bootstrap noise at `bits=2`. It
  deliberately does *not* reproduce float geometry: it returns
  different-but-equally-relevant neighbors (recovering ~80–90% of the exact
  float top-k), because exact ANN geometry is not the retrieval-relevant
  metric — and both are measured to show it. `bits ∈ {1,2,4}` is a transparent
  build-time size/quality dial; there is no `ef`, `nprobe`, or recall knob to
  set wrong at query time.
- **Cold-start instant, portable state.** A loaded index serves immediately —
  no warm-up, no rebuild. Bundles are ordinary files on disk that move between
  machines and architectures.
- **Verifiable ops out of the box.** `ordinaldb inspect`, `verify`, and
  `stats` emit machine-readable JSON for backup, auditing, and diagnostics.

## Scale, Measured

The core index holds **1M+ rows in a single verifiable file**. At 2 bits per
dimension the committed 100K-row benchmark writes a ~2.4 MB index for vectors
that occupy 25.6 MB as raw fp32 — about 10× smaller — with a verified cold
open in ~5 ms ([`docs/limits.md`](docs/limits.md)).

Framework adapter directories — which carry documents, metadata, and string
IDs alongside the vectors — have a measured planning guide of **~100,000 rows
per adapter directory**. That is an adapter-lane guideline, not the core
index's ceiling.

## Install

With the 0.2.0 release, packages publish to crates.io and PyPI:

```bash
cargo add ordinaldb
```

```bash
pip install ordinaldb
```

Until then — and for the latest development state — build from source.

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

To add an optional framework adapter extra to a locally built wheel, resolve
the wheel path before appending the extras bracket:

```bash
pip install "$(ls target/wheels/ordinaldb-*.whl)[langchain]"
```

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

Good to know:

- **Bit widths:** `bits` is 1, 2, or 4. The fast sign two-stage search path
  engages at `bits=2` with a dimension divisible by 64; other shapes use
  direct ordinal search. Allowlist-filtered search reranks exactly inside the
  allowed ID set.
- **Errors, not panics:** malformed input — non-finite values, dimension
  mismatches, unknown allowlist IDs — returns a typed `Result` in Rust and
  raises a catchable `ValueError` in Python. The bindings release the GIL
  during `add`/`search`.
- **Scores are ordinal similarity scores**, not cosine values or normalized
  relevance scores.
- **Real embeddings in Rust:** the
  [`fastembed`](https://crates.io/crates/fastembed) crate pairs well; its
  384-dim MiniLM models keep `bits=2` on the fast sign path (384 is divisible
  by 64).

Full API reference: [`docs/api.md`](docs/api.md).

## Framework Adapters

The base Python package is framework-free (Python 3.9+). Optional adapters
(Python 3.10+) install only the framework they wrap:

```bash
pip install 'ordinaldb[langchain]'
pip install 'ordinaldb[llama-index]'
pip install 'ordinaldb[haystack]'
pip install 'ordinaldb[agno]'
pip install 'ordinaldb[adapters]'
```

Entrypoints: `ordinaldb.langchain.OrdinalDBVectorStore`,
`ordinaldb.llama_index.OrdinalDBVectorStore`,
`ordinaldb.haystack.OrdinalDocumentStore` /
`OrdinalEmbeddingRetriever`, and `ordinaldb.agno.OrdinalDb`.

The snippets below use a throwaway hash-seeded local embedder so they run with
no network access and no model download — swap in `sentence-transformers`,
`fastembed`, or your framework's embedding class and keep `dim=384, bits=2`.

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

Adapter notes, briefly:

- **Filters are exact.** Every dialect computes a pre-search allowlist and
  ranks top-k *within* it — "top-k of the filtered set," never a global top-k
  filtered after the fact. Per-adapter dialects (LlamaIndex gets the full
  `MetadataFilters` operator set, Haystack the full nested filter dialect) are
  in [`docs/api.md`](docs/api.md).
- **Mistakes warn loudly.** Suspicious store paths, unsaved buffered writes,
  and zero-hit filters naming unknown keys raise warnings instead of no-oping
  ([`docs/api.md`](docs/api.md#adapter-warnings)).
- **Unsupported means unsupported.** MMR, sparse/text search, and normalized
  relevance scores fail clearly at the adapter layer instead of returning
  misleading results.
- **Offline stays offline.** OrdinalDB never calls a model or telemetry
  endpoint. Some wrapped frameworks phone home by default —
  [`docs/edge-deployment.md`](docs/edge-deployment.md) has the opt-out
  environment variables for air-gapped deployments.

Complete runnable applications — a LangChain docs-Q&A tool, a LlamaIndex
research-paper navigator, and a durable agent-memory store with crash
recovery — live in the [cookbook](cookbook/README.md).

## Persistence And Ops

Core indexes persist as `.odb` bundles:

```text
docs.odb/
    manifest.json
    index.ovrq
    sign.ovsb      # present when the sign stage is available
    ids.bin        # IdMapIndex only
```

Writes use the OrdVec `.ovrq`/`.ovsb` artifact names; manifest-verified loads
are path-driven, so bundles whose manifests point at legacy `.tvrq`/`.tvsb`
artifacts continue to load.

Framework adapters persist to adapter directories: `adapter.redb` is the
authoritative control plane (documents, metadata, string IDs, checkpoints),
and vector state lives in immutable `vectors/g000000000001.odb/`-style
generations. Writes take a cross-platform advisory writer lock — one writer
per bundle or adapter directory at a time.

```bash
ordinaldb inspect docs.odb
ordinaldb verify adapter-store
ordinaldb stats docs.odb --json
ordinaldb adapter export-json adapter-store
ordinaldb adapter gc adapter-store --retain 2
```

Each adapter `save()` publishes a complete new generation (a full rewrite of
the current vector set), and old generations are only reclaimed by an explicit
`gc` — batch your writes rather than saving per item. Full layout, failure
model, and runbook: [`docs/persistence.md`](docs/persistence.md) and
[`docs/operations.md`](docs/operations.md).

## What It's For — And Not For

Reach for OrdinalDB when the deployment boundary matters: agent memory inside
a runtime, RAG on a device or at the edge, air-gapped and offline search,
retrieval artifacts you need to hash, sign, and audit, or integration tests
that demand repeatable loads with no service dependency.

Deliberate non-goals in 0.2.0:

- **No server or distributed mode.** Embedded, single-process, files on disk.
- **Bring your own embeddings.** OrdinalDB never generates, downloads, or
  calls an embedding model — you pass in plain `f32` arrays.
- **No runtime metric choice.** Ordinal similarity only; no cosine/L2/dot
  switch.
- **Single-writer model.** One writer per bundle or adapter directory,
  enforced by an advisory lock.
- **Hybrid search is experimental.** BM25 + vector (RRF) fusion
  (`ordinaldb-hybrid`, `--features hybrid`) is tested but feature-gated,
  Rust-only, and not yet wired into the Python bindings or framework adapters.

Also worth knowing: Matryoshka (MRL) embeddings truncate-and-index with no
renormalization, because rank/sign codes are scale-invariant — see
[`docs/matryoshka.md`](docs/matryoshka.md).

## Platforms

Linux x86_64, Linux ARM64 (Raspberry Pi 4/5 class, CI-covered with a Pi
runtime profile job), macOS ARM64, and Windows x64. 32-bit Raspberry Pi OS is
best-effort — see [`docs/raspberry-pi.md`](docs/raspberry-pi.md).

## Project Status

OrdinalDB is alpha software (0.2.0) with a deliberately narrow scope, built on
the OrdVec (`ordvec`, `ordvec-manifest`) crates. The 0.2.0 line lands the
redb-backed adapter control plane, immutable generation layout, and
storage-hardening work.

Project references:

- [`LICENSE`](LICENSE)
- [`CONTRIBUTING.md`](CONTRIBUTING.md)
- [`CHANGELOG.md`](CHANGELOG.md)
- [`SECURITY.md`](SECURITY.md)
- [`THREAT_MODEL.md`](THREAT_MODEL.md)
- [`docs/api.md`](docs/api.md)
- [`docs/limits.md`](docs/limits.md)
- [`docs/persistence.md`](docs/persistence.md)
- [`docs/operations.md`](docs/operations.md)
- [`docs/edge-deployment.md`](docs/edge-deployment.md)
- [`docs/matryoshka.md`](docs/matryoshka.md)
- [`docs/raspberry-pi.md`](docs/raspberry-pi.md)
- [`docs/provenance.md`](docs/provenance.md)
- [`THIRD_PARTY.md`](THIRD_PARTY.md)
- [`NOTICE`](NOTICE)
