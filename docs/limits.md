# OrdinalDB Supported Limits

**Target:** `ordinaldb-v0.2.0`
**Method:** `scripts/limits_report.py`, local Linux x86_64, synthetic
64-dimensional float32 vectors, 2-bit OrdinalDB index, single process.

These are measured limits for this release target, not broad performance
claims. The benchmark excludes embedding generation and competitor
comparisons.

## Measured Rows

| Rows | Core `.odb` write | Core cold open | Core footprint | Adapter save | Adapter cold open | Adapter footprint |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 0.0010s | 0.0007s | 241,016 B | 0.4709s | 0.1154s | 9,867,005 B |
| 100,000 | 0.0039s | 0.0053s | 2,401,020 B | 1.9065s | 1.1013s | 90,292,683 B |

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
  for this release target on comparable local storage and CPU.
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
