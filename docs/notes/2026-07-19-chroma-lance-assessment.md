# Would you use OrdinalDB over Chroma or LanceDB? — assessment notes

**Date:** 2026-07-19
**Assessed at:** PR #23 tip (`c3f0458`, `feat/sign-load-policy`, which includes
PR #22 `feat/sign-policy`) — the latest substantive PR chain. PR #24 is a
Dependabot Actions bump and was excluded.
**Context:** requested ahead of a ~1M-subscriber newsletter feature going out
Monday 2026-07-21.

## What was independently verified at the PR tip

- `cargo test --workspace --all-features --locked` passes end to end
  (exit 0; the PR body's claim of 254 tests across 23 suites is consistent
  with what ran).
- A from-scratch smoke benchmark (not from the repo's own harnesses) was
  written against the public crate API and run on this container
  (Linux x86_64), 200,000 × 384-dim synthetic normalized vectors, `bits=2`:
  - ingest: ~0.8 s; bundle write: ~130 ms; **verified** cold open: ~60–70 ms
  - bundle size: 28.8 MB vs 307 MB raw fp32 (**10.7× smaller**, exactly the
    predicted 144 B/row for codes + sign sidecar)
  - single-query search: **p50 ~1.0 ms, p95 ~1.1 ms** at 200k rows
    (scan-based two-stage; extrapolates to roughly ~5 ms/query at 1M rows,
    single-threaded)
  - planted-neighbor recall (query = perturbed copy of an indexed row):
    **100/100 @1 and @10**. This is the only retrieval-shaped signal in the
    smoke test: when a genuinely similar item exists, it is found.
  - geometry non-reproduction check — **not a quality metric**: top-10
    overlap vs exact fp32 cosine came out low on this corpus, exactly as the
    README says it should ("does not reproduce float geometry"). i.i.d.
    random vectors are near-orthogonal, so top-10 membership there is
    tie-breaking noise with no relevance structure; per the project's own
    claims discipline, exact-ANN geometry overlap is a **banned** recall
    metric and the figure is deliberately not headlined here. It confirms
    the design statement, nothing more.
  - **Assessment gap, stated plainly:** the actual retrieval-quality claim —
    nDCG@10 parity with exact dense on BEIR (lossless at `bits=4`, within
    bootstrap noise at `bits=2`) — was **not** independently reproduced. It
    requires real embeddings and qrels; the public harness is in-repo at
    `benchmarks/beir-rust`. Until someone runs it, that claim is
    repo-reported, not third-party-verified.
- Tamper-evidence works as advertised: flipping **one byte** in
  `index.ovrq` of a written bundle makes `ordinaldb verify` report
  `artifact_sha256_mismatch` and exit 1, and `open_verified` refuses the
  load. This is the differentiator, and it is real.

## What OrdinalDB genuinely has that Chroma/LanceDB don't

- **Verified persistence as a first-class contract.** SHA-256 manifest over
  every artifact, fail-closed loads, a written THREAT_MODEL.md that is
  unusually honest about what is *not* covered (redb non-live bytes, hostile
  path actors), a Lean 4 model proof of the atomic-write protocol, and a
  roadmap for signed/sealed bundles. Neither Chroma nor LanceDB treats the
  index artifact as something you hash, sign, and audit.
- **Training-free, knob-free index.** No HNSW/IVF build step, no `ef` /
  `nprobe` recall dial, append-from-vector-one, instant cold start. The
  size/quality dial is a single build-time `bits ∈ {1,2,4}`.
- **Tiny footprint + edge posture.** ~10× compression measured; ARM64/
  Raspberry Pi are CI targets; no daemon, no network, no telemetry.
- **Documentation culture.** Measured-limits doc with explicit non-claims,
  loud adapter warnings instead of silent no-ops, exact filter semantics
  (top-k *within* the filter, never post-filtered).

## What Chroma/LanceDB have that OrdinalDB doesn't

- **You can install them today.** OrdinalDB 0.2.0 is *unreleased*: nothing
  on crates.io or PyPI; building requires Rust + maturin and a git-patched,
  unpublished `ordvec` 0.6.0 dependency. `pip install ordinaldb` does not
  work as of this assessment.
- **Maturity and bus factor.** The entire repo history is **16 days old**
  (first commit 2026-07-03), one human contributor plus AI/bot agents,
  self-described alpha. Chroma and LanceDB have years of production use,
  large communities, and funded companies behind them.
- **Sub-linear search.** OrdinalDB search is a (fast, SIMD-friendly) scan:
  ~1 ms at 200k rows, ~5 ms/query extrapolated at 1M, and it grows linearly.
  LanceDB's IVF-PQ serves millions-to-billions on disk; Chroma's HNSW is
  sub-ms at 1M. Above low-single-digit millions of rows, OrdinalDB's
  no-knob scan stops being a fair trade.
- **Scale figures are measurement floors, not caps.** Nothing in the code
  limits core rows below `u32` slot indexing (~4.29B); "1M+" is simply the
  largest in-repo verified run (the 1.26M arXiv battle test — whose detailed
  results table was dropped from `docs/limits.md` in the PR #13 refresh and
  should be restored; the arxiv harness README still points at it). The
  OrdVec substrate has been exercised at larger corpus sizes upstream, but
  per this repo's claims discipline only ordinaldb-run measurements are
  quotable. Extrapolating the perf-train numbers, 8M × 1024 would be a
  ~3.1 GB bundle, ~20 ms/query scan, ~2.4 s verified open — feasible, but
  unmeasured and therefore unclaimed.
- **The document/metadata lane is the bottleneck.** The 1M+ figure is for the
  vector-only core bundle. The adapter directories that hold documents,
  metadata, and string IDs — i.e., what a RAG app actually uses — have a
  measured planning limit of **~100k rows**, full-rewrite saves (~1.9 s at
  100k, cost scales with store size, explicit GC required), scan-based
  metadata filters (~70 ms at 100k regardless of selectivity), and a
  single-writer advisory lock.
- **Features:** no cosine/L2 scores (ordinal scores only — breaks any
  downstream that thresholds on similarity), no MMR, hybrid BM25+RRF is
  Rust-only and experimental, no multi-process sharing, no server option,
  no versioning/time-travel (LanceDB has Git-style branching; Chroma has
  cloud hosting, full-text search, multi-language clients).

## Verdict

**As a general-purpose replacement for Chroma or LanceDB: no, not today.**
It is 16 days old, unreleased, single-maintainer alpha with a ~100k-row
adapter lane and linear-scan search. For the median RAG app, Chroma (fastest
path) or LanceDB (scale + hybrid + versioning) remains the right default.

**For its actual niche: yes, and nothing else occupies it.** If the
deployment boundary is the problem — air-gapped/edge devices, agent memory
you need to hash/sign/audit, reproducible CI retrieval fixtures, tiny-RAM
ARM targets — the verified single-artifact model is genuinely novel, the
engineering quality is far above what the repo's age suggests, and every
claim I tested held up exactly as documented, including the self-limiting
ones (no float-geometry reproduction, by design; adapter-lane limits).

**Newsletter cautions (important for Monday):**

1. Do not print `pip install ordinaldb` / `cargo add ordinaldb` — neither
   works yet; readers must build from source, and the PyPI/crates.io names
   are not yet claimed by this project.
2. Frame it as a promising 0.2.0-alpha with a real differentiator, not a
   Chroma/Lance replacement; the repo itself never claims a competitor win
   ("No competitor benchmark claim" is an explicit non-claim).
3. The "1M+ rows" figure is the vector-only core path; quote the ~100k
   adapter-lane figure alongside it or readers will benchmark the wrong lane.
4. A 1M-reader spike on a 16-day-old, one-maintainer alpha is a real
   operational risk for the project itself; expect issues volume the
   maintainer may not absorb.
5. **Do not print any exact-cosine top-k overlap percentage as "recall" or
   "accuracy"** — including the one measured during this assessment. Exact
   ANN geometry is not the retrieval-relevant metric for this design (the
   repo's claims discipline explicitly bans it), and a bare overlap number
   will read as a quality score. The quotable quality claim is nDCG@10
   parity on BEIR — attribute it to the project, since it has not been
   independently reproduced here. The independently measured, safe-to-quote
   numbers are: footprint (10.7×), latency (p50 ~1 ms @ 200k rows), verified
   cold open (~60 ms), tamper detection (1 flipped byte → fail-closed), and
   planted-neighbor recall (100/100).

## Reproduction

The smoke benchmark used here (kept out of the assessed tree; reproduce by
dropping it into `ordinaldb/examples/bench_smoke.rs` at `c3f0458` and running
`cargo run --release --example bench_smoke -p ordinaldb`): 200k × 384
xorshift-seeded normalized vectors, queries = rows + 5% noise, k=10,
default (`Auto` → `SignTwoStage`) search, exact fp32 cosine brute force as
baseline. Tamper test: `printf '\x42' | dd of=<bundle>/index.ovrq bs=1
seek=500000 conv=notrunc` then `ordinaldb verify <bundle>`.

External references consulted for the competitor picture (July 2026):
Chroma's Rust core / HNSW / DuckDB-backed persistence and Q1-2026 cloud
launch; LanceDB's IVF-PQ/IVF-RQ disk indexes, Tantivy full-text + hybrid
search, automatic versioning and Git-style branching, and 1B+-vector
deployments on S3.
