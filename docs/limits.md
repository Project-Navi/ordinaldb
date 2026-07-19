# OrdinalDB Measured Envelopes

**Target:** `ordinaldb-v0.2.0`
**Methods:**

- `scripts/limits_report.py` — local Linux x86_64, synthetic 64-dimensional
  float32 vectors, 2-bit OrdinalDB index, single process.
- `benchmarks/arxiv-precomputed` — Rust harness over the arXiv-1M corpus
  (1.26M real 1024-dimensional normalized float32 embeddings, four
  2,048-query self-retrieval sets), 2-bit index with sign sidecar, single
  process, AMD Ryzen 9 9950X, local Linux x86_64.

These are **measured envelopes, not ceilings**: each row count below is the
largest run a committed, reproducible harness has recorded — the point where
the benchmark stops, not where OrdinalDB stops. The only structural row
bound in the core index is `u32` slot indexing (~4.29 billion rows). Costs
beyond the measured points extrapolate linearly (scan-based search, SHA-256
verification over artifact bytes); measure your own workload before relying
on figures past them. The benchmarks exclude embedding generation and
competitor comparisons.

## Measured Rows (synthetic, 64-dim)

| Rows | Core `.odb` write | Core cold open | Core footprint | Adapter save | Adapter cold open | Adapter footprint |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 0.0010s | 0.0007s | 241,016 B | 0.4709s | 0.1154s | 9,867,005 B |
| 100,000 | 0.0039s | 0.0053s | 2,401,020 B | 1.9065s | 1.1013s | 90,292,683 B |

## Measured 1.26M × 1024 (arXiv-1M, real embeddings)

Measured on the `benchmarks/arxiv-precomputed` harness (AMD Ryzen 9 9950X,
local Linux x86_64, single process, core `OrdinalIndex` only — no adapter).
"pre-perf 0.2.0" is the 0.2.0 development state before the perf train;
"perf train" is the ordinaldb perf branches on ordvec
`integration/full-stack` at commit `522bade`, since merged source-identical
into ordvec `main` — which is what the workspace now pins, so the perf-train
column reflects current builds.

| Metric | pre-perf 0.2.0 | perf train |
| --- | ---: | ---: |
| Batched throughput (one call, 2,048 queries) | 220 q/s | 10,189 q/s |
| Single-query latency, p50 (Auto → SignTwoStage) | 4.5 ms | 3.2 ms |
| Verified cold open (manifest + SHA-256) | 1.27 s | 0.38 s |
| Ingest, 1.26M rows (read + encode) | 4.9 s | 3.6 s |
| Peak RSS | 5,533 MB | 618 MB |

- **Ranking stability:** both builds return identical rankings on all four
  2,048-query self-retrieval sets — the perf train changes memory layout
  and scheduling, not ranking. Retrieval quality itself is measured as
  nDCG@10 against exact dense on the public BEIR harness (see the README);
  exact-neighbor geometry overlap is deliberately not quoted as a quality
  metric.
- **Artifact size:** the verified bundle is 483 MB vs 5.2 GB raw fp32 —
  10.7× smaller.
- **Sign sidecar:** `sign.ovsb` is `rows × dim / 8` bytes (~161 MB at
  1.26M × 1024).
- **No recall knob.** There is no `ef`, `nprobe`, or recall dial to tune;
  the numbers here are size and latency measurements, not a recall
  benchmark.
- **Verification limits:** resource limits are manifest-derived — each
  auxiliary artifact read is bounded by its own manifest-declared,
  SHA-256-pinned size — so bundles with large sign sidecars open with
  default options. There is no flat auxiliary-artifact-size cap.

## Beyond the Measured Points

Nothing in the core path stops at 1.26M rows: ingestion is append-only,
search is a scan, and verification is bounded per-artifact by the manifest.
The OrdVec substrate has been exercised at substantially larger corpus
sizes upstream; those runs are not reproduced by an in-repo OrdinalDB
harness yet, so they are not quoted as OrdinalDB measurements here. Larger
committed runs extend this table — the harness takes any corpus size on the
command line.

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

- Adapter directories: ~100,000 rows per directory is the measured planning
  envelope for this release target on comparable local storage and CPU.
  This is a cost guideline, not a correctness cap — larger stores function,
  but each save writes a complete replacement vector generation and each
  filtered query scans metadata, so both costs grow linearly with store
  size. The envelope moves when incremental generation deltas land.
- The core single-file path is separate: 1.26M × 1024 real-embedding rows
  measured in one verifiable bundle (table above), with no structural limit
  below `u32` slot indexing.
- Mutation model: each committed adapter save writes a complete replacement
  vector generation and publishes it through `adapter.redb`.
- Use closed-store backup or verified snapshot copy for backup.
- Measure your own workload before relying on larger collections, higher
  dimensions, slower disks, or high mutation rates.

## Explicit Non-Claims

- No multi-writer throughput claim.
- No cross-process live read/write sharing claim.
- No metadata-index latency claim.
- No measurements beyond the row counts above — larger scales are expected
  to extrapolate linearly but are unmeasured here, not unsupported.
- No competitor benchmark claim.
