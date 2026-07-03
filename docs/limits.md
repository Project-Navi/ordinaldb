# OrdinalDB Supported Limits

**Target:** `ordinaldb-v0.2.0` (plus the pre-0.3 perf train where marked)
**Methods:**

- `scripts/limits_report.py` — local Linux x86_64, synthetic 64-dimensional
  float32 vectors, 2-bit OrdinalDB index, single process.
- `benchmarks/arxiv-precomputed` — Rust harness over the arXiv-1M corpus
  (1.26M real 1024-dimensional normalized float32 embeddings, four
  2,048-query self-retrieval sets), 2-bit index with sign sidecar, single
  process, AMD Ryzen 9 9950X, local Linux x86_64.

These are measured limits for these release targets, not broad performance
claims. Both benchmarks exclude embedding generation and competitor
comparisons.

## Measured Rows (synthetic, 64-dim)

| Rows | Core `.odb` write | Core cold open | Core footprint | Adapter save | Adapter cold open | Adapter footprint |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 0.0010s | 0.0007s | 241,016 B | 0.4709s | 0.1154s | 9,867,005 B |
| 100,000 | 0.0039s | 0.0053s | 2,401,020 B | 1.9065s | 1.1013s | 90,292,683 B |

## Measured 1.26M × 1024 (arXiv-1M, real embeddings)

Measured on the `benchmarks/arxiv-precomputed` harness (AMD Ryzen 9 9950X,
local Linux x86_64, single process, core `OrdinalIndex` only — no adapter).
"v0.2.0" is the published release; "perf train" is the pre-0.3 integration
heads (ordinaldb perf branches on ordvec `integration/full-stack`).

| Metric | v0.2.0 | perf train |
| --- | ---: | ---: |
| Batched throughput (one call, 2,048 queries) | 220 q/s | 10,189 q/s |
| Single-query latency, p50 (Auto → SignTwoStage) | 4.5 ms | 3.2 ms |
| Verified cold open (manifest + SHA-256) | 1.27 s | 0.38 s |
| Ingest, 1.26M rows (read + encode) | 4.9 s | 3.6 s |
| Peak RSS | 5,533 MB | 618 MB |

- **Recall:** top-10 row ids are byte-identical to an exact fp32 cosine
  brute-force baseline on all four 2,048-query self-retrieval sets (`title`,
  `first_sentence`, `middle_sentence`, `paraphrase`), for both builds. The
  perf train changes memory layout and scheduling, not ranking.
- **Artifact size:** the verified bundle is 483 MB vs 5.2 GB raw fp32 —
  10.7× smaller.
- **Sign sidecar:** `sign.ovsb` is `rows × dim / 8` bytes (~161 MB at
  1.26M × 1024).
- **Default limits:** verification resource limits are manifest-derived —
  each auxiliary artifact read is bounded by its manifest-declared,
  SHA-256-pinned size — so bundles with large sign sidecars open with
  default options on the perf train. v0.2.0 against ordvec 0.5.0 shipped a
  flat 64 MB `max_auxiliary_artifact_bytes` default that must be raised
  manually at this scale.

## Filter Measurements

Adapter filters are scan-based scalar equality filters evaluated before vector
ranking. At 100,000 rows, measured filter cases were:

| Selectivity | Expected matches | Returned | Allowlist time | Search time |
| --- | ---: | ---: | ---: | ---: |
| Empty | 0 | 0 | 0.0736s | 0.0709s |
| One ID | 1 | 1 | 0.0709s | 0.0709s |
| 1% | 1,000 | 10 | 0.0696s | 0.0684s |
| 50% | 50,000 | 10 | 0.0712s | 0.0718s |
| 100% | 100,000 | 10 | 0.0739s | 0.0753s |

The near-flat filter timings are expected for scan-based filtering: every
accepted portable filter scans adapter metadata to build an allowlist before
vector ranking.

## Guidance

- Recommended measured planning limit: up to 100,000 rows per adapter directory
  for this release target on comparable local storage and CPU. The 1.26M-row
  measurements above cover the core `OrdinalIndex` path only and do not raise
  the adapter-directory guidance.
- Mutation model: each committed save writes a complete replacement vector
  generation and publishes it through `adapter.redb`.
- Use closed-store backup or verified snapshot copy for backup.
- Measure your own workload before relying on larger collections, higher
  dimensions, slower disks, or high mutation rates.

## Explicit Non-Claims

- No multi-writer throughput claim.
- No cross-process live read/write sharing claim.
- No metadata-index latency claim.
- No guarantee beyond the measured row counts and local filesystem assumptions.
- No competitor benchmark claim.
