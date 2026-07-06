# ordinaldb-trust: Cryptographically Sealed Index Bundles

Status: **reconciled design — 2026-07-05**. This revision folds the 2026-07-04
design review's seven findings into the spec body (see "Design-review
provenance" at the end) and aligns the seal with the deterministic-manifest
work in the v0.2.0 hardening wave. The crate itself remains post-0.2.x (0.3.0
candidate) and out of scope for the 0.2.0 launch — but its **preconditions
ship in 0.2.0**: the deterministic manifest and the reserved seal/provenance
file names.

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

## The identity/seal unification

The v0.2.0 hardening wave makes `manifest.json` **deterministic**:

- `manifest_id`, `created_at`, and `build.invocation_id` leave the manifest —
  and the bundle: v0.2.0 bundles are **manifest + artifacts only**. Volatile
  build provenance is not persisted anywhere in the bundle; authoritative
  created-at and signer identity arrive later in the seal envelope.
- Auxiliary artifact entries are written in sorted order, and the stored file
  bytes are the canonical form. Hashing and signing always operate on the
  bytes as stored on disk — never on a re-parsed / re-serialized document.

Consequence: `sha256(manifest.json)` is the bundle's stable content address —
"the manifest hash *is* the version" — and the Ed25519 seal signs exactly those
bytes. One byte surface serves three roles:

- **hash it** → the version / content address;
- **sign it** → the seal;
- **as a Merkle root** → it transitively binds every artifact and auxiliary
  sidecar through the SHA-256 digests embedded in it.

This resolves the review's canonicalization finding **by construction**: the
signer and the verifier never re-serialize JSON, so serializer drift cannot
invalidate old seals. A serialization or schema change is a manifest-schema
version event; bundles written earlier keep verifying because their stored
bytes are unchanged.

## What a bundle physically contains (grounding)

A core `.odb` bundle holds `manifest.json`, `index.ovrq` (RankQuant ordinal
codes), `sign.ovsb` (sign bitmaps), and `ids.bin`; a sealed bundle adds the
detached `manifest.sig` and nothing else (`provenance.json` is a reserved
name, not a shipped file — see Provenance). Full-precision embeddings are
never persisted — search
operates on 1/2/4-bit ordinal state. This is an inherent property worth
stating in the threat model independently of this crate: an attacker who
obtains index files does not obtain the embedding vectors, only heavily
quantized rank/sign information. Quantization degrades embedding-inversion
attacks; it does not eliminate the signal, so "resistant" is the honest claim,
never "immune."

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
corruption sense.

## Design

**Crate:** `ordinaldb-trust`, workspace member, default-off feature in
`ordinaldb` (`--features trust`). Core stays dependency-minimal without it.

**Signing:** `ed25519-dalek` v2 (pure Rust, audited, fast on ARM — fits the
Pi/edge posture). Detached seal file `manifest.sig`, keeping full compatibility
with existing `ordvec-manifest` tooling and unsealed consumers.

**Seal envelope (message framing).** The signature covers a length-prefixed,
domain-separated envelope — never a bare concatenation of manifest bytes and
header fields:

```text
ORDINALDB_SEAL_V1\0            16-byte domain prefix
u32 LE                         seal schema version
u64 LE                         created_at (unix seconds)
[32 bytes]                     Ed25519 verifying key, raw
u64 LE                         manifest length in bytes
[...]                          manifest.json bytes exactly as stored
```

Every field has an unambiguous boundary, the domain tag prevents
cross-protocol signature reuse, and embedding the verifying key binds the seal
to a specific key identity so a signature cannot be transplanted between trust
domains. The envelope's `created_at` is the authoritative timestamp — the
deterministic manifest no longer carries one.

**Key identity.** The verifying key recorded in the seal is **diagnostic
only**. Verification iterates the `TrustPolicy`'s explicit key list and never
selects a key based on what the seal claims; on failure, the recorded key
appears in the structured error ("sealed by X; policy trusts Y, Z"). Key id
format: the raw 32-byte verifying key in the envelope; lowercase-hex
fingerprints in reports and errors.

**Verification order (seal-swap TOCTOU).** Read `manifest.json` and
`manifest.sig` into memory in a single filesystem pass **before any
cryptographic operation**: capture both byte strings, then run structural
checks, then verify the Ed25519 signature over the captured manifest bytes,
and only then load artifacts from verified paths. Filesystem reads and crypto
are never interleaved, which eliminates the seal-file swap race. This composes
with the verify-once open path: verify + seal-check + load is a single hash
pass over the artifacts. Artifact reads after verification remain subject to
the documented local-writer TOCTOU window (see Runtime honesty).

**Reserved names.** `manifest.sig` and `provenance.json` join the bundle
reserved-file list (alongside `manifest.json`, `index.ovrq`, `sign.ovsb`,
`ids.bin`) in the v0.2.0 hardening wave — *before* this crate exists — so no
auxiliary artifact can ever claim, and thereby overwrite, either file. When a
trust policy requires a seal, a missing `manifest.sig` fails closed.

**Auxiliary and hybrid artifacts (transitive coverage).** The Ed25519 seal
covers `manifest.json` bytes only. Every registered auxiliary artifact —
including `ordinaldb-hybrid`/LTR sidecars, which are retrieval-policy
artifacts — is transitively covered because the sealed manifest contains its
SHA-256. The required verification chain is: seal over manifest → SHA-256 of
the auxiliary from the now-trusted manifest → bytes of the auxiliary. No
additional per-artifact sealing exists, and no implementation may shortcut
the chain.

**Provenance.** No provenance file ships in v0.2.0 — an unbound file inside a
"verifiable bundle" invites the assumption that it is verified, which muddies
exactly the story the bundle exists to tell. `provenance.json` is **reserved
as a name only**, keeping the option of a future informational sidecar open
without letting an auxiliary artifact squat on it. Authoritative time and
signer identity live in the seal envelope once this crate ships.

**Adapter stores and key rotation.** The immutable-generation model fits
sealing naturally — sign each committed generation
(`vectors/gNNNNNNNNNNNN.odb/`) at commit time. `adapter.redb` records, per
generation, the expected seal **and the signer key's fingerprint** (human-
auditable rotation history). `TrustPolicy::RequireSeal` holds the full
*accepted* key set; rotation is: add the new key to the accepted set, seal new
generations with it, re-seal old generations at leisure, and drop the retired
key only when no live generation's recorded fingerprint references it.
Retired-but-not-yet-GC'd generations verify against the accepted set, so GC
and export never block on a rotation in progress. Mutable redb state itself is
*not* sealed in phase 1 (its integrity story remains the adapter layer's
CAS/revision checks); state that clearly in docs.

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
existing structured verify/inspect reports (which carry the content digest as
of 0.2.0) with a `seal` block. Key generation via `ordinaldb-cli keygen`.

**Runtime honesty:** verification happens at open (and via an explicit
`re-verify` call). Once state is loaded/mapped, a local writer with sufficient
privileges can still race the filesystem (see Path Boundary in
THREAT_MODEL.md). This is verify-at-load, not continuous attestation — docs
must say so.

**Key management (phase 1 = deliberately minimal):** callers supply key
material; generation is offered via CLI (`ordinaldb-cli keygen`). No keychain,
KMS, or HSM integration in-crate. Rotation semantics are defined above;
re-sealing is the operator's action to schedule.

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
- The content-address claim ("the manifest hash is the version") becomes true
  at v0.2.0 via the deterministic manifest, independent of this crate — but
  sealing/authenticity claims are **not OK until this crate ships**.
- Not OK: "encrypted" (until phase 2 ships), "immune to inversion," or any
  claim implying protection from a compromised host.

## Dependencies

`ed25519-dalek` v2 + `rand_core` (keygen only). Requires explicit maintainer
sign-off per project dependency policy before implementation starts.

## Non-goals

Access control, multi-tenant isolation, network auth, key storage/escrow,
sealing of mutable redb state (phase 1), shipping a provenance sidecar
(`provenance.json` is a reserved name only), threshold/multi-signer
verification (open question), host-compromise defense.

## Open questions

1. Seal the Python surface in 0.3.0 too, or Rust/CLI first?
2. Multiple signers / threshold verification (fleet scenarios) — v1 or later?
3. Should `open_verified` grow an env-var kill switch to *require* seals
   process-wide (defense against a call site forgetting the policy)?

## Design-review provenance

A contributed design review (2026-07-04, AI-assisted via Perplexity; reviewed,
verified accurate against the code) found seven implementation traps in the
original draft: JSON canonicalization, signed-message framing, key-id oracle,
seal-file swap TOCTOU, `manifest.sig` reserved-name collision, key-rotation
semantics, and hybrid-artifact coverage. All seven are folded into the body
above — canonicalization is resolved structurally by the deterministic
manifest; rotation semantics are defined minimally; the rest are binding
design requirements. The standalone audit text is preserved in the history of
PR #15 (`ed25519-sealing-audit.md`, removed at reconciliation).
