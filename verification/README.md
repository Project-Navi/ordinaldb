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
- `write_bundle_no_partial_publish` and `verified_temp_before_replace` are
  intentionally weak scaffolding (a tautological witness and an unconditional
  existential), marked as such in the file. The load-bearing safety results are
  `write_bundle_path_is_verified_on_success` (a successful write leaves the canonical
  path holding `Verified` content), `replace_bundle_restores_on_rename_failure` (the
  path is restored from backup on a failed rename), and `write_bundle_path_invariant`
  (the composed either/or).
- **Machine-check status: CLEAN.** Verified with **Lean 4.28.0 + Mathlib v4.28.0**
  (`lake build` → 0 errors; `#print axioms` on every declaration shows only Lean's
  foundational axioms — `propext`, `Quot.sound`, `Classical.choice` — and no
  `sorryAx`).
- **Repaired on incorporation (honest note):** the contributed version did **not**
  compile — three substantive theorems had broken proofs and
  `write_bundle_path_is_verified_on_success` was *mis-stated* (it keyed the backup on
  `fs temp` where `replace_bundle` keys it on `fs path`). These were corrected (the
  mis-stated theorem restated as the property `success ⇒ path = Verified`) so the file
  now checks; imports trimmed to `Option.Basic` + `Logic.Basic`.

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
here with the scope and status stated above. The Lean file's proofs were **broken as
received** (three theorems failed to typecheck, one was mis-stated) and had to be
repaired to compile — a concrete reminder that AI-generated proofs are *verified, not
trusted*. The audit's findings are accurate to the current code but are not yet folded
into the spec.
