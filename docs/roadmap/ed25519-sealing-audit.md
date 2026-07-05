# Ed25519 Sealing Design Audit — `ordinaldb-trust`

Source: `docs/roadmap/ordinaldb-trust-spec.md` (draft, not scheduled)
Auditor: Computer / Project-Navi design review
Date: 2026-07-04

---

## Executive Summary

The trust spec is architecturally sound and shows good threat-model discipline.
The claims-discipline section alone places it above most startup security
roadmaps. However, seven concrete implementation traps exist in the draft that
will cause subtle, hard-to-debug failures if not addressed before code is
written. Two are cryptographic-correctness issues. Three are
API-misuse-under-pressure issues. Two are operational security issues.

None of these invalidate the design. All are fixable at the spec level before
a line of implementation is written.

---

## Trap 1 — Canonicalization: "bytes of manifest.json" is underspecified

**Spec says:**
> "a seal header (signer key id, created-at, schema version). Because the
> manifest already binds every artifact's path, size, and sha256, sealing
> the manifest seals the bundle transitively."

**The trap:**
The spec says the signature covers "the canonical bytes of `manifest.json`."
But `manifest.json` is a JSON file. JSON serialization is not canonical.
Two semantically identical manifests can have different byte representations
(key ordering, whitespace, Unicode normalization of string values). If the
signing side and the verifying side use different JSON serializers — or
even different versions of the same serializer — they will produce different
byte strings over the same logical document. The signature will fail
verification on a bundle that was never tampered with.

This is not hypothetical: `ordvec-manifest` writes the manifest and
`ordinaldb-trust` reads it. If `ordvec-manifest` ever changes its
serialization (field ordering, indentation), any sealed bundle produced
before the change becomes unverifiable after.

**Concrete failure scenario:**
- Bundle sealed against `ordvec-manifest` 0.6.0 (writes fields in struct
  declaration order via `serde_json`).
- `ordvec-manifest` 0.7.0 adds a field, changes ordering.
- Caller upgrades. Old sealed bundles: signature verification fails, even
  though the bundle is clean and the vector content is identical.

**Fix:** Sign the raw bytes written to disk by `write_manifest_file`,
captured at write time, before any re-serialization can occur. The signer
must operate on the file bytes as written, not on a re-parsed/re-serialized
logical document. Alternatively, define a canonical serialization form
(e.g., JSON with sorted keys, no extra whitespace) and enforce it in
`ordvec-manifest` before the trust crate ships. Document which one you
chose and why.

---

## Trap 2 — Ed25519 Malleability and the Seal Header Binding

**Spec says:**
> "Detached seal file (e.g. `manifest.sig`) containing the signature over
> the canonical bytes of `manifest.json` plus a seal header (signer key id,
> created-at, schema version)."

**The trap:**
The spec says the signature covers "`manifest.json` plus a seal header." The
word "plus" hides a critical question: **how are they concatenated?**

If the message is `manifest_bytes || seal_header_bytes` with no length
prefix or separator, this is vulnerable to a **length-extension or
recomposition attack**. An attacker cannot forge Ed25519 signatures, but if
they can rearrange where one field ends and another begins (because there's
no framing), they can potentially construct a different seal header that
produces the same signed byte string as a legitimate one.

More concretely: Ed25519 via `ed25519-dalek` v2 signs an arbitrary byte
slice. If the signed payload is `manifest_bytes || signer_key_id_bytes ||
created_at_bytes || schema_version_bytes` with no framing, an attacker who
controls `manifest.json` content (a corpus poisoning scenario) could craft
a manifest whose suffix, when concatenated with a partial seal header,
produces a signing collision with a header from a different key.

**Fix:** Use a **domain-separated, length-prefixed message format**. A clean
approach is a structured envelope:

```
ORDINALDB_SEAL_V1\0          -- fixed domain prefix (16 bytes)
<4 bytes LE: schema version> -- prevents cross-version replay
<8 bytes LE: created_at_unix_secs>
<32 bytes: verifying key raw bytes>  -- binds seal to specific key identity
<8 bytes LE: manifest file size>     -- length prefix
<manifest file bytes>                -- exact bytes from disk
```

This ensures every field has an unambiguous boundary, the domain tag
prevents cross-protocol reuse of signatures, and the key binding means a
signature from key A cannot be transplanted to a bundle whose trust policy
only accepts key B.

Alternatively, use `ed25519-dalek`'s `Context` API (prehash + context
string) which provides domain separation at the library level. This is
cleaner but requires careful use of `SigningKey::sign_prehashed` with a
stable context string.

---

## Trap 3 — Key Identity: `signer key id` is not defined

**Spec says:**
> "a seal header (signer key id, created-at, schema version)"

**The trap:**
The spec records a "signer key id" in the seal header but does not define
what a key ID is, how it is derived, or how it is used at verification time.

There are at least three reasonable interpretations:
1. A raw Ed25519 verifying key (32 bytes) — self-describing but large.
2. A SHA-256 fingerprint of the verifying key — compact but requires a
   separate key registry lookup.
3. A human-assigned opaque label — user-controlled, no cryptographic binding.

Options 2 and 3 create a **key oracle problem** at verification time: the
verifier sees a key ID in the seal, must look it up against its trusted key
list, and must do so before calling `verify`. If the lookup uses the
seal-header key ID rather than the policy's explicit key list, an attacker
can forge a seal header with a spoofed key ID while providing a real
signature from a different key. The verifier looks up the wrong key, fails
to verify, but the failure mode may not be fail-closed if the error handling
is sloppy.

**Fix:** The key ID in the seal header must never be the *input* to key
selection at verify time. The verification API already sketches this
correctly — `TrustPolicy::RequireSeal(vec![verifying_key])` — the policy
holds the trusted keys and the seal header's key ID is only used for
**error reporting** (e.g., "seal was produced by key X, but policy only
trusts keys Y, Z"). The implementation must enforce: iterate the policy key
list, attempt verification with each key, succeed if any matches, fail
closed if none match. Never branch on the seal header's stated key ID.

Define the key ID format concretely (recommended: raw 32-byte verifying key
in the seal, or its lowercase hex SHA-256 fingerprint). Encode this in the
spec before implementation starts.

---

## Trap 4 — TOCTOU Between `verify_for_load` and Seal Verification

**Spec says:**
> "verification happens at open (and via an explicit `re-verify` call).
> Once state is loaded/mapped, a local writer with sufficient privileges
> can still race the filesystem."

**The trap:**
The trust spec correctly acknowledges the TOCTOU window at a high level,
but the implementation sketch has a specific race:

```rust
let policy = TrustPolicy::RequireSeal(vec![verifying_key]);
let idx = OrdinalIndex::load_sealed(path, &policy)?;
```

Inside `load_sealed`, the expected sequence is:
1. Verify manifest SHA-256 (via existing `verify_for_load`)
2. Verify Ed25519 seal over manifest bytes
3. Load vector artifacts from verified paths

Between steps 1 and 3, the files are not locked. An attacker with
filesystem write access can replace artifacts after SHA-256 verification
but before load. This is documented as out-of-scope, which is correct for
a local embedded library.

**However**, there is a subtler race specific to the *seal*: if the seal
file (`manifest.sig`) is read separately from `manifest.json`, and the
filesystem does not guarantee atomic read of both, an attacker can swap
`manifest.sig` between the two reads. This means `manifest.json` bytes
change but the seal file does not, causing a legitimate seal to
"cover" a tampered manifest.

**Fix:** Read both `manifest.json` and `manifest.sig` into memory in a
single filesystem traversal before performing any cryptographic operation.
Do not interleave filesystem reads with crypto. The sequence must be:
1. `bytes = fs::read(manifest_path)` — capture manifest bytes
2. `sig = fs::read(seal_path)` — capture seal
3. `verify_sha256_in_manifest(&bytes)` — structural check
4. `verify_ed25519_signature(&bytes, &sig, &policy)` — authenticity check
5. Only then return verified paths for artifact loading

This doesn't eliminate the TOCTOU on artifact reads, but it eliminates the
seal-file swap race, which is the one that matters for the security property
being claimed.

---

## Trap 5 — Seal File Naming and the `manifest.sig` Collision

**Spec says:**
> "Detached seal file (e.g. `manifest.sig`)"

**The trap:**
The `.sig` extension is a provisional suggestion. But the bundle layout in
`io.rs` uses a reserved file list:

```rust
if relative_path == Path::new(MANIFEST_FILE)   // "manifest.json"
    || relative_path == Path::new(INDEX_FILE)  // "index.ovrq"
    || relative_path == Path::new(SIGN_FILE)   // "sign.ovsb"
    || relative_path == Path::new(IDS_FILE)    // "ids.bin"
```

`manifest.sig` is not in this reserved list. If `ordinaldb-trust` adds
`manifest.sig` as a bundle artifact but does not add it to the reserved
file list in `copy_auxiliary_artifacts`, a caller can register an auxiliary
artifact named `manifest.sig` and overwrite the seal file. This is an
integrity bypass: the attacker writes a forged seal, registers it as an
auxiliary artifact with the name they choose, and OrdinalDB's existing
validation guards don't catch it because `manifest.sig` is not a known
reserved name.

**Fix:** Add `manifest.sig` (or whatever the seal file is named) to the
reserved file list in `copy_auxiliary_artifacts` and `validate_bundle_relative_path`
as part of the `ordinaldb-trust` implementation. Also register it as a
required auxiliary artifact in the manifest when a trust policy is active,
so verification will fail closed if the seal file is absent.

---

## Trap 6 — Key Rotation and Sealed Generation Lineage in Adapter Stores

**Spec says:**
> "sign each committed generation (`vectors/gNNNNNNNNNNNN.odb/`) at commit
> time; `adapter.redb` records the expected seal per generation."

**The trap:**
If a signing key is rotated, old generations were sealed under the old key.
The spec says "Rotation = re-seal; document it." But re-sealing old
generations requires:
1. The old verifying key to still be trusted (for validation of the re-seal
   operation itself), OR
2. Trusting that the re-sealing operator has the authority to vouch for
   old content they may not have originally produced.

Neither case is trivially safe. The spec does not say whether `TrustPolicy`
supports a key transition period (old key + new key both trusted), or
whether `adapter.redb`'s generation records store which key signed which
generation, or whether a seal mismatch during key rotation surfaces a
structured error versus a panic.

**Concrete failure:** an operator rotates the key, forgets to re-seal old
generations, and a subsequent `open_verified` with `RequireSeal([new_key])`
fails on every retired generation that gets referenced during GC or export
operations. If the GC queue uses generation seals for integrity checks,
this could block garbage collection entirely.

**Fix:** The spec needs to define:
1. Whether `TrustPolicy` accepts a transition key set (e.g.,
   `RequireSeal { current: [key_b], accepted: [key_a, key_b] }`).
2. Whether `adapter.redb` stores the key fingerprint alongside each
   generation seal (for human-auditable rotation history).
3. Whether retired (not-yet-GC'd) generations require valid seals or only
   the active generation does.

Define this before implementation. Retrofitting key rotation semantics into
`adapter.redb`'s schema is expensive.

---

## Trap 7 — The Hybrid Artifact Gap (Open Question 4 in the spec)

**Spec open question 4:**
> "Interaction with `ordinaldb-hybrid` auxiliary artifacts — seal them in
> the same manifest walk?"

**The trap:**
This is labeled as an open question, but it is actually a security decision
with a correct answer: **yes, they must be sealed in the same walk, and
failing to do so creates an integrity gap.**

The `ordinaldb-hybrid` crate ships LTR model artifacts as bundle sidecars
(auxiliary artifacts registered in `manifest.json`). The LTR model directly
affects what candidates surface from a hybrid search — it is a
retrieval-policy artifact. An attacker who can swap the LTR model sidecar
without breaking the Ed25519 seal over `manifest.json` can steer retrieval
decisions without triggering tamper detection.

This is possible if:
- The Ed25519 seal covers only `manifest.json` bytes, AND
- The auxiliary artifact SHA-256s in `manifest.json` are checked at
  load but the *manifest bytes that contain those hashes* are the signed
  surface.

Wait — this is actually fine IF the verification sequence is:
1. Verify Ed25519 seal over `manifest.json` bytes → confirms the SHA-256
   hashes in the manifest are authentic.
2. Verify each auxiliary artifact against the SHA-256 in the (now-trusted)
   manifest.

This is the correct design and it means LTR sidecars ARE covered transitively
by the seal, without needing to enumerate them in the seal walk.

**But the spec does not say this explicitly.** If an implementer reads
"sealing the manifest seals the bundle transitively" and assumes it applies
only to the core artifacts (which are enumerated), they may implement a
separate per-artifact signing walk that misses sidecars.

**Fix:** Add one sentence to the spec: "The Ed25519 seal covers
`manifest.json` bytes only. Auxiliary artifact integrity is transitively
covered because `manifest.json` contains SHA-256 hashes of all registered
auxiliary artifacts. No additional per-artifact sealing is required. An
implementer must verify: seal over manifest → SHA-256 of auxiliary from
manifest → bytes of auxiliary. No shortcut." Close open question 4 in the
spec.

---

## Summary Table

| # | Trap | Severity | Phase |
|---|------|----------|-------|
| 1 | JSON canonicalization undefined — signatures break on serializer change | High | Must fix before implementation |
| 2 | Seal header concatenation undefined — length-extension / recomposition risk | High | Must fix before implementation |
| 3 | Key ID undefined — key oracle attack if used as verification input | Medium | Must fix before implementation |
| 4 | Seal-file swap TOCTOU — read manifest + seal atomically before any crypto | Medium | Must fix before implementation |
| 5 | `manifest.sig` not in reserved file list — overwritable by auxiliary artifact | Medium | Fix during implementation |
| 6 | Key rotation semantics undefined — blocks GC and export on rotation | Medium | Spec decision required |
| 7 | Hybrid artifact gap — transitive coverage needs explicit documentation | Low | Close open question before implementation |

---

## What the Spec Gets Exactly Right

For the record, these design choices are correct and should not be changed:

- `TrustPolicy::RequireSeal(vec![key])` with no TOFU mode — explicit key
  lists are the only safe default for a security primitive.
- Separation of signing from encryption at rest — AEAD is a different
  problem with different key distribution cost.
- Detached `.sig` file — preserves compatibility with unsealed consumers.
- `ed25519-dalek` v2 (pure Rust, audited, constant-time) — correct choice.
- Verify-at-load, not continuous attestation — honest scope.
- Explicit phase 2 for encryption — not conflated with signing.
- Claims discipline section — "not OK: encrypted" is exactly the right
  guard.

The design is sound. The traps are implementation-level. Fix them in the
spec before the first commit of `ordinaldb-trust`.
