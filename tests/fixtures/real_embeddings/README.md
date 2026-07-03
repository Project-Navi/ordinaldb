# Real-Embedding Test Fixtures

Committed, real `all-MiniLM-L6-v2` embeddings used by the Rust and Python
test suites wherever a vector crosses a serialization or storage boundary.
They exist because synthetic vectors systematically miss real bugs: uniform
random floats never exercised the float-canonicalization path that real
embedding bit-patterns corrupt.

Contents:

- `texts.json` — 32 documents across four domains + 8 labeled queries.
  Row order is load-bearing; do not reorder.
- `minilm_docs_f32.bin`, `minilm_queries_f32.bin` — little-endian `f32`,
  row-major, 384-dim. Loaded via `np.fromfile` (Python) and
  `include_bytes!` (Rust). Never edit by hand.
- `adversarial_floats.json` — pathological float values harvested from the
  real embeddings plus curated IEEE-754 edge cases, for round-trip tests at
  serialization boundaries.
- `manifest.json` — model id, pinned revision, generator versions, and
  sha256 digests for every fixture file.
- `generate.py` — regeneration script and provenance documentation. The
  committed artifacts are authoritative; regenerate only if `texts.json`
  changes, and expect digest churn if model or library versions moved (the
  nightly workflow surfaces that drift on purpose).
