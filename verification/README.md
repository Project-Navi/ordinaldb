# Formal verification & security review

Formal-method and security-review artifacts for OrdinalDB. These are **models and
reviews**, not production code, and are held to an honest standard about what they
do and do not establish. Contributions here are reviewed, not blindly trusted.

## `AtomicWriteProof.lean`

A Lean 4 formal **model** of the `write_bundle` atomic-write protocol in
[`../ordinaldb/src/io.rs`](../ordinaldb/src/io.rs) — the
write-to-temp → verify → fsync → atomic-rename sequence — with theorems that the
canonical bundle path is never left holding unverified or partially-written
content, and is restored from backup if the final rename fails.

**Scope — what it does and does not establish (read before citing it):**

- It reasons over an **abstract state machine** whose transitions correspond to the
  Rust operations, *not* over the Rust source. The bridge from the implementation to
  this model (an Aeneas-style simulation lemma) is **future work**, flagged in the
  file's `Correspondence` comments and Section 8.
- OS `rename(2)` atomicity, SHA-256 collision-resistance, fsync durability,
  concurrent-writer exclusion, and the verify↔rename TOCTOU window are **explicit
  documented assumptions**, not theorems (Section 8).
- `write_bundle_no_partial_publish` is intentionally weak (a tautological witness).
  The load-bearing results are Theorems 2 and 3 and `write_bundle_path_invariant`.
- **Machine-check status:** being built against Lean 4 + Mathlib. Until that is
  confirmed green, treat this as a *reviewed draft*, not a checked proof.

To check it locally: install the toolchain with `elan`, create a Lake project that
depends on Mathlib (`lake exe cache get`), drop this file in, and `lake build`.

## Ed25519 sealing audit

See [`../docs/roadmap/ed25519-sealing-audit.md`](../docs/roadmap/ed25519-sealing-audit.md)
— a design review of the (draft, unscheduled) `ordinaldb-trust` Ed25519 sealing spec.
Seven concrete implementation traps (JSON canonicalization, message framing /
malleability, key-id oracle, seal-file TOCTOU, reserved-name collision, key rotation,
hybrid-artifact coverage). The findings are **actionable spec-level changes to make
before `ordinaldb-trust` is implemented**; they are not yet reflected in the spec.

## Provenance

Both artifacts are AI-assisted contributions (Perplexity), reviewed and incorporated
here with the scope and status stated above — deliberately not presented as
established guarantees until the Lean file machine-checks and the audit's findings
are folded into the spec.
