# arxiv-precomputed benchmark

Battle-test harness for OrdinalDB core against a corpus of **precomputed**
embeddings. No embedding model, GPU, or network access is involved: the
harness reads fp32 `.npy` files from disk, ingests them into an
`OrdinalIndex` (bits=2, sign sidecar on by default), writes and re-opens a
verified bundle, and measures search latency, throughput, and self-retrieval
recall.

This crate is deliberately **excluded from the workspace** (see the root
`Cargo.toml`), mirroring `benchmarks/beir-rust`: it behaves like a genuine
external consumer of the `ordinaldb` crate, with its own lockfile and none of
the workspace's inherited settings.

## No corpus data in this repository

Embeddings are **never committed**. The repository `.gitignore` excludes
`*.npy`, and this harness takes every input path on the command line — there
are no hard-coded corpus locations. Keep corpora outside the repository and
point the CLI at them.

## Corpus layout

The harness expects:

1. **Corpus embeddings** (`--corpus-npy`): a single 2-D NumPy `.npy` file,
   little-endian float32 (`<f4`), C order (`fortran_order: False`), shape
   `(rows, dim)`. Rows should be L2-normalized if you want the scores to
   behave as cosine similarity. The dimension is read from the header.
2. **Query sets** (repeatable `--queries-npy` / `--qids-jsonl` pairs, matched
   positionally):
   - `<set>_queries.npy` — 2-D `<f4` C-order array, shape `(n_queries, dim)`
     with the same `dim` as the corpus.
   - `<set>_qids.jsonl` — one JSON object per line, in query order, carrying
     a `"paper_id": <integer>` field. The integer is the 0-based **corpus row
     index** that is the query's ground-truth document (self-retrieval).

The query-set label used in output filenames and the JSON report is the
queries file stem with a trailing `_queries` stripped
(`title_queries.npy` → `title`).

The original run used the arXiv-1M corpus: 1.26M rows × 1024 dims of
normalized fp32 abstract embeddings, plus four 2,048-query self-retrieval
sets (`title`, `first_sentence`, `middle_sentence`, `paraphrase`) embedded
with the same model as the corpus.

## What it measures

- Ingest wall time, split into file read vs encode, and RSS after ingest.
- Verified bundle write time and on-disk size (vs raw fp32 bytes).
- Cold `open_verified` time (manifest + SHA-256 verification) and RSS.
- Per-query sequential latency (mean/p50/p95/p99) under the default `Auto`
  policy (resolving to `SignTwoStage`).
- Batched throughput: one `search_with_options` call over the whole set.
- `ExactRankQuant` full-scan latency on a 256-query subset.
- Self-retrieval recall@1 and recall@10 per query set.
- Peak RSS (`VmHWM`) for the whole run.

## Reproduction

```sh
cd benchmarks/arxiv-precomputed
cargo build --release
./target/release/ordinaldb-arxiv-precomputed-benchmark \
  --corpus-npy /data/arxiv-1m/embeddings.fp32.npy \
  --queries-npy /data/arxiv-1m/title_queries.npy \
  --qids-jsonl  /data/arxiv-1m/title_qids.jsonl \
  --queries-npy /data/arxiv-1m/first_sentence_queries.npy \
  --qids-jsonl  /data/arxiv-1m/first_sentence_qids.jsonl \
  --queries-npy /data/arxiv-1m/middle_sentence_queries.npy \
  --qids-jsonl  /data/arxiv-1m/middle_sentence_qids.jsonl \
  --queries-npy /data/arxiv-1m/paraphrase_queries.npy \
  --qids-jsonl  /data/arxiv-1m/paraphrase_qids.jsonl \
  --out-dir /data/arxiv-1m/bench-out
```

Outputs in `--out-dir`:

- `corpus.odb/` — the verified bundle written and re-opened by the run.
- `bench-results.json` — all measurements as JSON.
- `<set>_ordinal_top10.i64` — little-endian i64 top-10 row ids per query,
  for external exact-cosine baseline comparison (the original run checked
  these byte-identical against an fp32 cosine brute-force baseline).

Run on an otherwise idle machine; timings are wall-clock and single-process.

## Reference results

Measured numbers from the 1.26M × 1024 arXiv-1M run (AMD Ryzen 9 9950X,
local Linux x86_64) are recorded in [`docs/limits.md`](../../docs/limits.md).

## Resource limits note

Verification resource limits are manifest-derived: each auxiliary artifact
read is bounded by its manifest-declared, SHA-256-pinned size, so the sign
sidecar (`sign.ovsb`, `rows × dim / 8` bytes — ~161 MB at 1.26M × 1024)
loads with default options. ordinaldb 0.2.0 against ordvec 0.5.0 shipped a
flat 64 MB `max_auxiliary_artifact_bytes` default that had to be raised
manually at this scale (finding #1 of the original battle test).
