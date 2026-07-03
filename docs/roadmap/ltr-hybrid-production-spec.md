# Hybrid + LTR: Production Spec

Status: **active — targeted inside the v0.2.0 launch window** (hybrid) with
LTR full-loop as a mid-window go/no-go. This supersedes the earlier
*deferral* ("maybe 0.3.0") for everything listed as in-scope below — it does
**not** change today's shipped state: hybrid and LTR remain
experimental-labeled, feature-gated, and Rust-only exactly as the README
says until the workstreams below land. README wording graduates in W6, when
the code has earned it, not before.

**Relationship to
[`0.3.0-api-async-streaming-spec.md`](0.3.0-api-async-streaming-spec.md):**
that spec governs *async and streaming* API surfaces and explicitly lists
"no full-text or hybrid streaming until those query types exist" as a
non-goal. This spec governs hybrid + LTR *correctness and
production-readiness* (feature export, training bridge, serving ergonomics,
eval) — synchronous, in-process, no streaming. The two are complementary:
this one ships a working synchronous hybrid/LTR loop; the 0.3.0 spec is
where async or paged/streaming search for hybrid would land later, if ever
proposed.

## Positioning

The differentiator: **reranking quality without the reranker tax.** External
research (maintainer's, XGBoost LambdaMART-family LTR vs. cross-encoder
reranking) shows tree-ensemble LTR matching CE quality in ~80% of cases at
three-plus orders of magnitude lower latency. OrdinalDB is unusually well
placed to own this story because the expensive part — serving — is already
implemented as a pure-Rust, allocation-light tree scorer over features the
hybrid pipeline produces natively. No Python in the hot path, no ONNX
runtime, no GPU: BM25 + ordinal dense + RRF + tree-ensemble rerank, all
inside one embedded library, all artifacts manifest-verified.

**Claims discipline:** the "matches CE in 80% of cases at 1000x less
latency" figure is *external research context*, not a product claim, until
W5's eval gate produces our own numbers on public data. Launch copy before
that gate says: "tree-ensemble reranking served in-process in microseconds"
(measured) and cites the research direction without asserting parity.

## Current state

Working today: BM25 mmap sparse index + allowlist-aware search; RRF fusion;
verified sidecar persistence; serving-side LTR (`TreeEnsembleReranker`,
`LtrFeatureBatch`, `rerank_fused_batch`); LTR feature-cache
write/attach/read/verify; CLI `ltr features` export (3 features).

Known gaps this spec closes: no training path anywhere; feature building
hard-errors on rows not present in every source; CLI `train/attach/inspect`
stubs; 3-of-9 feature export; model header requires fabricated
`rank:pairwise`/`gbtree` provenance; hybrid/LTR invisible to Python.

## Workstreams

### W1 — Hybrid API hardening (in flight)
Normalizer round-trip BLOCKER + property test; structural sidecar
validation in `verify`; first doctests; `open dense+sparse from one
manifest` convenience. Exit: a stranger builds hybrid search from rustdoc
alone; `verify` guarantee extends to every recognized auxiliary family.

### W2 — Feature pipeline correctness
1. **Missing-source semantics** (the design decision of this spec):
   `LtrFeatureBatch::from_inputs` currently requires every fused row to
   carry an explicit score in every configured source. Replace with
   explicit per-feature missing policy recorded in `LtrFeatureSchema`:
   `missing: neutral_fill(value)` (default: worst-rank / zero-score,
   matching what backfill produces today) or `missing: error` (current
   behavior, for callers that pre-backfill). XGBoost handles missing
   values natively (default-direction branches), so the trained-model path
   can also support `missing: native` where the scorer routes absent
   features down the tree's default branch — implement in
   `TreeEnsembleReranker` (small: each stump/node already knows its
   children; add default-direction). This removes the undocumented
   top_k=corpus-size backfill workaround entirely.
2. **Full feature export**: CLI `ltr features` exports all 9 supported
   features. Dense/RRF features need query vectors in addition to the text
   queries already required for lexical features. `--queries` keeps its
   current meaning unchanged — JSONL, one `{query_id, query}` (or
   `{id, text}`) object per line, exactly as `read_ltr_queries` parses it
   today (`ordinaldb-cli/src/main.rs`). Precomputed embeddings arrive
   through new, non-conflicting flags: `--query-vectors <path>` and
   `--query-vectors-dim <N>` (both required together; BYO-embeddings, no
   model invoked). Pairing contract: `--query-vectors` is a raw
   little-endian `f32`, row-major file whose row count must equal the
   number of non-empty lines in `--queries`, in the same order — row *i*
   (0-indexed) is the embedding for the *i*-th query line, matched by
   position, not by `query_id`. `ltr features` must hard-error before
   computing dense/RRF features if the file size is not an exact multiple
   of `--query-vectors-dim * 4` bytes, or if the resulting row count does
   not equal the query count.
3. Delete or wire the dead `dense_features_required` field (schema bump if
   needed — pre-release, no compat burden).

### W3 — Training bridge (the "fully working" keystone)
Decision: **do not write a trainer in Rust for this window.** XGBoost is
the trainer the research is based on and the one users already trust; the
product is the turnkey loop around it.
1. `ordinaldb-python`: `ordinaldb.ltr.export_xgboost(booster_or_json,
   schema) -> model sidecar JSON` — converts an XGBoost dump (exact
   supported versions pinned and tested) into `LtrTreeEnsembleRecord`,
   validating feature names against the schema and recording honest
   provenance.
2. Model header honesty: replace hardcoded `rank:pairwise`+`gbtree`
   acceptance with a recorded `provenance` block (`trainer`, `objective`,
   `source`) validated for *shape*, not fabricated values; hand-authored
   models declare themselves as such.
3. CLI `ltr attach` (real): attach a model sidecar to a bundle through the
   manifest with verification — the write path mirror of what serving
   already reads. CLI `ltr inspect`: dump model metadata, feature schema,
   tree stats.
4. A documented, tested end-to-end recipe: `ltr features` → pandas/XGBoost
   training script (shipped under `cookbook/supportsearch/train/`) →
   `export_xgboost` → `ltr attach` → serve. The cookbook's hand-authored
   stub model is replaced by a genuinely trained one.
5. `ltr train` stays a stub for 0.2.x — it now points at the recipe.
   In-process training is 0.3.0+ (candidate: vendor-free pure-Rust
   LambdaMART, only if demand shows).

### W4 — Serving ergonomics + latency numbers
One-call path on the W1 convenience: today `HybridBundle` (in
`ordinaldb/src/hybrid.rs`) exposes only `HybridBundle::open_verified`; this
workstream adds fused search plus an optional attached-model rerank. The
following is a **proposed method signature — not implemented, not call
syntax, and not yet part of the public API**:

```rust
// Proposed (does not exist yet):
impl HybridBundle {
    fn search(
        &self,
        query_vec: &[f32],
        query_text: &str,
        k: usize,
        rerank: bool,
    ) -> Result<FusedBatch, HybridBundleError>;
}
```

Benchmark the rerank overhead (p50/p95 µs per candidate set of 50/200/1000)
and publish alongside the existing search latency numbers — this is the
denominator of the "1000x less than a CE" story and must be first-party.

### W5 — Eval gate (unlocks the quality claim)
Small public-data eval proving LTR-over-RRF uplift: SciFact (BEIR's
smallest; prior harness evidence exists in-repo history) — nDCG@10 for
dense / BM25 / RRF / RRF+LTR, one page of methodology, seeds, and hardware.
Exit criteria: measured uplift over RRF, or an honest "no uplift on this
corpus" result (which caps the launch claim at the latency story). CE
comparison itself is post-launch scope; until then the CE figure remains
"external research" wording.

### W6 — Docs + cookbook integration
`docs/hybrid.md`: the full story in one page (when hybrid beats either
mode, the three-query-class demo, LTR loop diagram, feature table, missing
policy, provenance). supportsearch upgraded to the real trained-model
pipeline. README hybrid section graduates from "experimental, Rust-only"
to "hybrid: stable API at 0.2.x; LTR loop: supported via XGBoost bridge"
with flags unchanged.

### W7 — Explicitly out of this window (0.3.0 backlog)
Python serving bindings for hybrid/LTR; in-process trainer; adapter-layer
(LangChain et al.) hybrid exposure; CE head-to-head benchmark.

## Sequencing

Dependency order between workstreams — not a schedule:

- **W1 gates W2**: the feature pipeline builds on the hardened hybrid API.
  W2.1's missing-source semantics are a design decision reviewed before
  implementation. W3.1's XGBoost converter has no code dependency on
  either and can proceed in parallel against a pinned XGBoost version.
- **W2.2 (full feature export), W3.2–3.3 (attach/inspect), and W4
  (one-call API + latency bench)** are mutually independent once W2.1
  lands.
- **Go/no-go checkpoint**: when the full loop is green end-to-end on
  supportsearch, LTR ships in the next release; otherwise hybrid ships
  hardened on its own and the LTR bridge follows in a 0.2.x patch release.
- **W5 (eval gate) and W6 (docs + cookbook trained-model upgrade)** come
  last: W5 requires the complete loop, and W6's README graduation waits on
  W5's measured results. Nothing else in this spec waits on W5.

## Dependency policy note
W3.1 pins `xgboost` as a *dev/optional* Python dependency (extras group
`ltr-train`) — never required at serve time. No new Rust dependencies
anticipated; if W2.1's native-missing scoring needs none (expected), the
whole spec adds zero runtime deps.
