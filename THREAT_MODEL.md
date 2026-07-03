# OrdinalDB Threat Model

OrdinalDB is an embedded local vector index. It does not provide a network
server, authentication layer, authorization layer, multi-tenant isolation, or
remote execution surface.

## Trusted Inputs

- Embeddings supplied by the caller.
- Local paths selected by the caller, including their parent directories and
  filesystem namespace.
- Release artifacts produced by the repository release workflows.

## Untrusted Inputs

- `.odb` bundles and adapter directories from outside the caller's control.
- JSON sidecars in adapter directories.
- Python metadata and document text supplied by framework adapters.

## Current Controls

- Core `.odb` loads use `ordvec-manifest` verification for paths, hashes, sizes,
  and artifact metadata before loading vector state.
- Adapter sidecars are outside the core manifest and are validated by the
  adapter layer and CLI verifier.
- Store classification only uses authoritative sentinels: `adapter.redb`, a
  valid legacy `adapter.json`, or a core `manifest.json`. Generic directories
  such as `vectors/` are treated as debris or incomplete stores.
- Symlink and file-type checks reject common accidental misconfiguration and
  corrupted store contents.
- CI/CD workflows pin third-party Actions by commit SHA and use read-only
  default `GITHUB_TOKEN` permissions.
- Publishing is a manual, human-run step (`cargo publish`/`twine upload`);
  no registry tokens are stored in the repo or CI. See
  [RELEASING.md](RELEASING.md) for the planned migration to OIDC-based
  Trusted Publishing.

## Integrity Guarantees By Artifact

- **`.odb` bundles (core index and adapter generations)** — tamper-evident:
  `ordvec-manifest` sha256 digests cover every artifact; any byte flip in an
  artifact or its manifest fails closed on load and in `ordinaldb verify`.
- **`adapter.redb`, live state** — consistency-checked, not cryptographically
  tamper-evident: every open re-parses the JSON payloads and cross-checks them
  against the manifest, the expanded row tables, the generation records, and
  the active generation's sha256-verified bundle manifest. Corruption that
  intersects this live state fails closed with a structured error (never a
  panic).
- **`adapter.redb`, non-live bytes** — not covered: redb is a copy-on-write
  B-tree file, so most of its bytes (free pages, superseded page versions) are
  never read by verification; byte flips there pass verification silently
  (measured at ~99% of single-byte flips on a small store). redb's internal
  XXH3 checksums protect commit-slot headers for crash recovery; they are not
  cryptographic, and full page-checksum validation only runs during repair.
- **JSON sidecar exports (`adapter.json`, `documents.json`, `metadata.json`,
  `id_map.json`)** — export-only convenience copies, never verification inputs
  for redb-backed stores; tampering with them does not affect loads.

`ordinaldb verify` proves the data a load would actually use is consistent
and (for vector artifacts) authentic; it does not prove `adapter.redb` is
byte-identical to what OrdinalDB wrote. Whole-file attestation for
`adapter.redb` is a roadmap item (see `docs/roadmap/ordinaldb-trust-spec.md`).

## Path Boundary

OrdinalDB is scoped to trusted local filesystem roots. It does not currently
claim hostile-path resistance against a local actor that can replace ancestors,
swap path components during validation, or create platform-specific reparse
points while a store is being opened or written. The current checks fail closed
for direct symlinked store files and malformed store entries, but they are not a
descriptor-relative Unix open protocol or a Windows reparse-point-safe open
protocol.

## Out of Scope

- Protection against malicious code in caller-provided embedding models.
- Sandboxing arbitrary Python framework code.
- Distributed consistency, encryption at rest, or access-control policy.
- Defending against malicious mutation of caller-selected local paths by another
  process with write access to those paths or their ancestors.
