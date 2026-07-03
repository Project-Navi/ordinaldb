# ordinaldb-trust: Cryptographically Sealed Index Bundles

Status: **draft spec — proposed, not scheduled**. Target: post-0.2.x (0.3.0
candidate). Explicitly out of scope for the 0.2.0 launch.

## Motivation

`ordvec-manifest` verification is structural: it checks paths, sizes, and
sha256 hashes before vector state loads. That fails closed against corruption
and accidental damage, but it does not establish *authenticity* — an attacker
with write access to a bundle can modify artifacts, recompute the hashes, and
rewrite `manifest.json`, and the store opens clean. The manifest proves the
bundle is self-consistent, not that it is the bundle you built.

`ordinaldb-trust` adds detached Ed25519 seals over bundles and adapter
generations so a caller can require, at open time, that vector state was
produced by a holder of a known signing key and has not been altered since.

## What a bundle physically contains (grounding)

A core `.odb` bundle holds `manifest.json`, `index.ovrq` (RankQuant ordinal
codes), `sign.ovsb` (sign bitmaps), and `ids.bin`. Full-precision embeddings
are never persisted — search operates on 1/2/4-bit ordinal state. This is an
inherent property worth stating in the threat model independently of this
crate: an attacker who obtains index files does not obtain the embedding
vectors, only heavily quantized rank/sign information. Quantization degrades
embedding-inversion attacks; it does not eliminate the signal, so "resistant"
is the honest claim, never "immune."

## Threats and what each mechanism actually buys

| Threat | Mechanism | Status |
|---|---|---|
| Index substitution / retrieval-corpus poisoning: attacker edits or swaps bundle contents to steer what a RAG pipeline or agent retrieves | Ed25519 seal verified at open; unauthorized modification fails closed | **This crate, phase 1** |
| Provenance for distributed artifacts: `.odb` bundles shipped to edge fleets / published for download | Same seal, supply-chain style: verify origin key before load | **This crate, phase 1** |
| Offline theft of index files → embedding inversion | Partially mitigated today (no raw vectors on disk, see above); completed by encryption at rest | Inherent today; **phase 2** for AEAD |
| Compromised runtime host (attacker inside the process boundary) | Not addressed — the process must hold verification (and any decryption) material | **Out of scope, permanently** |

The strongest near-term story is the first row. Retrieval corpora are becoming
part of the supply chain for agentic systems; a sealed index makes "what the
LLM retrieves" tamper-evident in the adversarial sense, not just the
corruption sense. Neither Chroma nor LanceDB offers this.

## Design sketch

**Crate:** `ordinaldb-trust`, workspace member, default-off feature in
`ordinaldb` (`--features trust`). Core stays dependency-minimal without it.

**Signing:** `ed25519-dalek` v2 (pure Rust, audited, fast on ARM — fits the
Pi/edge posture). Detached seal file (e.g. `manifest.sig`) containing the
signature over the canonical bytes of `manifest.json` plus a seal header
(signer key id, created-at, schema version). Because the manifest already
binds every artifact's path, size, and sha256, sealing the manifest seals the
bundle transitively. Detached file keeps full compatibility with existing
`ordvec-manifest` tooling and unsealed consumers.

**Adapter stores:** the immutable-generation model fits sealing naturally —
sign each committed generation (`vectors/gNNNNNNNNNNNN.odb/`) at commit time;
`adapter.redb` records the expected seal per generation. Mutable redb state
itself is *not* sealed in phase 1 (its integrity story remains the adapter
layer's CAS/revision checks); state that clearly in docs.

**Verification API (sketch):**

```rust
// seal at build/ship time
ordinaldb_trust::seal_bundle(path, &signing_key)?;

// open with a trust policy
let policy = TrustPolicy::RequireSeal(vec![verifying_key]);
let idx = OrdinalIndex::load_sealed(path, &policy)?;   // fails closed: no/invalid/unknown-key seal → error
```

`TrustPolicy::Unsealed` preserves current behavior. No TOFU mode — key lists
are explicit; ambient trust defeats the point.

**CLI:** `ordinaldb-cli seal <bundle> --key <file>`,
`ordinaldb-cli verify <bundle> --require-seal --pubkey <file>` — extends the
existing structured verify/inspect reports with a `seal` block.

**Runtime honesty:** verification happens at open (and via an explicit
`re-verify` call). Once state is loaded/mapped, a local writer with sufficient
privileges can still race the filesystem (see Path Boundary in
THREAT_MODEL.md). This is verify-at-load, not continuous attestation — docs
must say so.

**Key management (phase 1 = deliberately minimal):** callers supply key
material; generation is offered via CLI (`ordinaldb-cli keygen`). No keychain,
KMS, or HSM integration in-crate. Rotation = re-seal; document it.

## Phase 2 (separate decision): encryption at rest

Confidentiality is a different mechanism with real design cost: AEAD
(XChaCha20-Poly1305) over artifacts breaks lazy/mmap loading unless done at
block level, and introduces a key-distribution problem signing does not have.
Keep it a separate feature and a separate decision. Sealing does not require
it; do not conflate the two in docs or marketing.

## Claims discipline (for README / launch material)

- OK to claim: "raw embeddings are never written to disk"; "cryptographically
  sealed indexes (Ed25519), verified fail-closed at load"; "tamper-evident
  retrieval corpus for RAG/agent pipelines."
- Not OK: "encrypted" (until phase 2 ships), "immune to inversion," or any
  claim implying protection from a compromised host.

## Dependencies

`ed25519-dalek` v2 + `rand_core` (keygen only). Requires explicit maintainer
sign-off per project dependency policy before implementation starts.

## Non-goals

Access control, multi-tenant isolation, network auth, key storage/escrow,
sealing of mutable redb state (phase 1), host-compromise defense.

## Open questions

1. Seal the Python surface in 0.3.0 too, or Rust/CLI first?
2. Multiple signers / threshold verification (fleet scenarios) — v1 or later?
3. Should `open_verified` grow an env-var kill switch to *require* seals
   process-wide (defense against a call site forgetting the policy)?
4. Interaction with `ordinaldb-hybrid` auxiliary artifacts — seal them in the
   same manifest walk?
