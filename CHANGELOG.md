# Changelog

## 0.2.0 - Unreleased

- Rewrote the adapter control plane on `ordinaldb-adapter-store`: a
  redb-backed store replaces adapter whole-directory rewrites with atomic,
  verified full-generation commits under `vectors/gNNNNNNNNNNNN.odb/`.
  `adapter.redb` is now the authoritative state; `adapter.json`,
  `id_map.json`, `documents.json`, and `metadata.json` are derived
  compatibility exports (`ordinaldb adapter export-json`).
- Added immutable, generation-based vector persistence: each committed write
  publishes a complete new vector generation instead of mutating one in
  place. Previous generations are retained until reclaimed by
  `ordinaldb adapter gc --retain N`.
- Added `ordinaldb-hybrid`, an experimental (`--features hybrid`) crate for
  sparse BM25 mmap indexing, allowlist-aware sparse search, and RRF
  (Reciprocal Rank Fusion) of BM25 with OrdVec vector search.
- Added `ordinaldb-ltr`, an experimental (`--features experimental-ltr`)
  crate for learning-to-rank feature caching and tree-ensemble reranking on
  top of hybrid search. CLI support currently covers
  `ordinaldb ltr features`; `train`, `attach`, and `inspect` are defined but
  not implemented yet.
- Hardened bundle storage I/O: atomic bundle replace on rewrite, Windows
  rename retries with backoff, symlink rejection on load, and
  checked-arithmetic overflow guards.
- Hardened `ordinaldb-adapter-store`: a cross-platform advisory writer lock,
  stale writer-lock recovery, and fail-closed opens on schema, table-count,
  active-generation digest, or ID-map mismatches.
- Expanded the `ordinaldb-cli` surface: `inspect`, `verify`, and `stats`
  (all with `--json`), plus `adapter export-json`, `adapter import-legacy`,
  and `adapter gc --retain`.
- Documented and shipped Python adapter subsets for LangChain, LlamaIndex,
  Haystack, and Agno, scoped to vector-search operations (add/search/delete,
  upsert, filters, and load/dump paths). MMR, sparse, text-search, and
  normalized relevance-score APIs remain intentionally unsupported at the
  adapter layer.
- Fixed the Python `add`/`search` bindings to route through the checked Rust
  API and release the GIL: malformed input (non-finite values, dtype/shape
  mismatches, an unknown allowlist ID) now raises a catchable `ValueError`
  instead of panicking, and concurrent Python callers are no longer
  serialized behind native search.
- Added full `///` rustdoc coverage across `ordinaldb`, `ordinaldb-adapter-store`,
  and `ordinaldb-ltr`, plus `#[pyo3(text_signature = ...)]` docstrings, a
  hand-written `_ordinaldb.pyi` stub, and a `py.typed` marker for the Python
  package.
- Added measured 10K/100K-row persistence, cold-open, and footprint numbers;
  see `docs/limits.md` for current planning limits and `docs/operations.md`
  for the offline backup and diagnostic runbook.
- Added a four-application cookbook (`cookbook/`): docs Q&A via the
  LangChain adapter (`docspilot/`), research-paper discovery with advanced
  metadata filtering via the LlamaIndex adapter (`paperscout/`), durable
  agent memory with a crash-recovery walkthrough via the Agno adapter
  (`keeper/`), and hybrid BM25+dense+LTR search on the experimental Rust
  crates (`supportsearch/`). Each runs end to end with no API keys.
- Added a nightly model-in-the-loop CI workflow
  (`nightly-real-embeddings.yml`) that runs the fixture-driven
  real-embedding suites and regenerates the committed fixtures with a live
  sentence-transformers model, comparing digests to catch model/library
  drift.
- Added `docs/matryoshka.md`: guidance for Matryoshka (MRL) and other
  high-dimensional embeddings — prefix truncation with no special handling
  required (no renormalization), how to choose a truncation dimension, and
  measured quality/footprint trade-offs on OrdinalDB's ordinal index.
- Developer experience: added Haystack `Pipeline` component support for
  `OrdinalEmbeddingRetriever` (proper `@component` sockets, plus a
  fully-qualified `to_dict`/`from_dict` type name so a `Pipeline` round-trips
  through serialization instead of leaving a raw dict in place of the store);
  fixed LlamaIndex delete/persist interop; added loud, explicit failures for
  unsaved-write and invalid-store-path mistakes instead of silent no-ops; and
  cleaned up `ordinaldb inspect`/`stats` JSON and text output so plain core
  `.odb` bundles no longer report adapter-only generation fields as
  misleading zeros.
- Durability fixes: metadata containing hard-to-round floats (for example
  real embedding vectors with near-zero components) no longer fails adapter
  saves with `metadata table does not match payload` — float parsing is now
  correctly rounded end to end (`serde_json` `float_roundtrip`). Crash debris
  from an interrupted generation replacement no longer permanently defeats
  `ordinaldb verify` and `adapter gc`: scratch names are always derived from
  the canonical generation name, malformed entries under `vectors/` are
  classified as reclaimable debris with structured warnings, and a store
  crashed during its very first save loads again instead of failing closed.
  Corruption inside `adapter.redb` now always surfaces as a structured
  `AdapterStoreError` instead of a raw storage-engine panic.
- Replaced all remaining Turbovec-derived code and prose; lineage
  acknowledgment in NOTICE.
- Upgrade note: stores written by a pre-0.2.0 build that contain a float the
  old parser read incorrectly-but-stably may now fail verification with
  `metadata table does not match payload`. The payload text is authoritative
  and the data is intact; re-save the store (or export and re-import) to
  refresh the derived tables.

## 0.1.0 MVP

- Forked the Turbovec local-index shell as OrdinalDB.
- Preserved upstream MIT attribution in `LICENSE`, `NOTICE`, and
  `THIRD_PARTY.md`.
- Renamed package metadata and workspace layout to `ordinaldb`.
- Replaced the inherited algorithm surface with OrdVec `RankQuant` indexing.
- Added default unfiltered `SignBitmap` candidate generation followed by
  `RankQuant` reranking for eligible b=2 indexes.
- Added stable `u64` ID mapping, delete, and allowlist search.
- Added Rust and Python `write`/`load` for `.odb` directory bundles backed by
  `ordvec-manifest`.
- Added persistence tests for roundtrip search, wrong bundle variants, loaded
  delete behavior, and malformed ID sidecars.
- Added adapter-directory persistence with JSON sidecars for framework text,
  metadata, string IDs, empty lazy stores, and stable monotonic numeric handles.
- Added optional LangChain, LlamaIndex, Haystack, and Agno adapter entrypoints.
- Added the separate `ordinaldb-cli` crate with `inspect` and `verify`
  commands for core bundles and adapter directories.
- Added Rust and Python examples for positional search, stable IDs, allowlists,
  delete, and persistence.
- Added correctness smoke tests and an honest local benchmark smoke script.

See `docs/provenance.md` for fork provenance and attribution boundaries.
