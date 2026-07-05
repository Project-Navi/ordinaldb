# Persistence

OrdinalDB persists indexes as directory-backed `.odb` bundles and uses
`ordvec-manifest` for manifest creation and verified loads.

```text
docs.odb/
    manifest.json
    index.ovrq
    sign.ovsb      # present when the two-stage sign stage is available
    ids.bin        # IdMapIndex only
```

The primary artifact is an OrdVec `RankQuant` file. When available, the
`SignBitmap` stage is stored as a required manifest auxiliary artifact so a
loaded index keeps the same default two-stage search path. ID-mapped indexes
store their external `u64` IDs in `ids.bin`, also declared as a required
auxiliary artifact.

New OrdinalDB writes use the OrdVec 0.5 artifact filenames `index.ovrq` and
`sign.ovsb`. Loads remain manifest-driven: existing bundles whose manifest
points at legacy `index.tvrq` or `sign.tvsb` files still load as long as the
manifest verifies those files.

`ordvec-manifest` owns the generic verifier work:

- manifest schema and size limits;
- relative path resolution and default path-escape rejection;
- primary artifact path, hash, size, metadata, and row-count checks;
- required auxiliary artifact path, hash, and size checks.

OrdinalDB owns only the database-layer semantics:

- positional vs. ID-mapped bundle policy;
- `ids.bin` binary format;
- ID count matching the loaded `RankQuant` length;
- duplicate ID rejection;
- `slot_to_id` and `id_to_slot` reconstruction;
- `io::ErrorKind::InvalidData` for malformed OrdinalDB sidecars.

OrdinalDB depends on the published `ordvec-manifest` 0.5.0 crate. This keeps
manifest verification in the OrdVec project instead of duplicating verifier
code inside OrdinalDB.

## Write semantics

OrdinalDB writes a bundle to a temporary directory, verifies the written
artifacts, syncs the temporary bundle tree, and then renames the bundle into
place. Rewrites move the previous target to a unique backup name first, sync the
parent directory around publication, and recover from a verified backup on load
if a crash left the requested target path absent between the backup rename and
the final publish rename.

This is still a directory-replacement protocol, not the long-term generation
catalog design. It protects normal rewrite failures and the missing-target crash
window, but it is not a blanket power-loss guarantee for every filesystem and
storage device. For Raspberry Pi deployments and other write-heavy edge devices,
prefer SSD or NVMe storage over microSD and use clean shutdowns.

## Implementation Notes

- OrdinalDB declares `sign.ovsb` and `ids.bin` as create-time manifest
  auxiliaries and then verifies the finished bundle before publishing it.
- Verified loads use the manifest plan as the trust boundary. The plan verifies
  artifact membership, paths, hashes, sizes, metadata, and required sidecars,
  and OrdinalDB immediately opens the verified artifacts for layout validation.
  It still does not pin file descriptors against hostile post-verification
  mutation.
- Raw vectors are never retained: fresh and loaded indexes alike maintain the
  `SignBitmap` sidecar incrementally via its `swap_remove` support, keeping the
  two-stage path active after delete at a memory footprint of codes + sidecar
  only.

Lazy indexes whose dimension has not been committed are not persisted. Calling
`write` on an uncommitted lazy index returns `io::ErrorKind::InvalidInput`.

## Adapter directories

Framework adapters persist to an adapter directory, not directly to a core
`.odb` bundle. The vector state remains in a vector-only `.odb` bundle;
framework text, metadata, string IDs, allocation state, and adapter checkpoints
live in `adapter.redb`:

```text
adapter-store/
    adapter.redb
    vectors/
        g000000000001.odb/
            manifest.json
            index.ovrq
            sign.ovsb      # present when the sign stage is available
```

`adapter.redb` is the canonical adapter control-plane store for new writes. It
stores a verified manifest, stable string-ID mappings, `u64` slot mappings,
documents, metadata JSON, generation checkpoints, empty pending-batch/tombstone
tables reserved for crash recovery and compaction, and an audit-log table.
Opening a redb-backed adapter store fails closed when the store schema,
completion status, table counts, active generation path, `.odb` manifest digest,
dimension, bit width, vector count, or ID-map bijections do not verify.

JSON sidecars are not a second live control plane. They are explicit exports
derived from `adapter.redb` when requested:

```bash
ordinaldb adapter export-json adapter-store
```

`export-json` writes `adapter.json`, `id_map.json`, `documents.json`,
`metadata.json`, and `adapter.redb.revision.json`. The revision export records
the redb schema version, store identity, commit sequence, active generation, and
active ID count so external tooling can tie the derived JSON files back to the
authoritative store revision.

New redb stores use a random UUIDv4 `store_uuid`, checked commit-sequence
increments, and an explicit manifest `origin`: `created`,
`imported_legacy_json`, or `upgraded_schema`. Fresh stores record
`migrated_from_json_sidecars: false`; legacy imports preserve their origin in
the redb manifest instead of pretending every new store was migrated.

Legacy JSON directories are imported explicitly:

```bash
ordinaldb adapter import-legacy legacy-adapter-store --output imported-store
```

The import target must not already exist. The importer rejects symlinked legacy
entries, copies the legacy directory, writes a redb snapshot, and leaves the
legacy JSON files as derived compatibility artifacts. A legacy directory without
`adapter.redb` still loads through the strict JSON verifier as a compatibility
path, but once `adapter.redb` exists, ordinary load and `ordinaldb verify`
validate redb plus the active vector generation.

Store detection is deliberately sentinel-based. `adapter.redb`, a valid legacy
`adapter.json`, or a core `manifest.json` classifies a path as a store; a
`vectors/` directory or `index.odb/manifest.json` without an adapter manifest is
reported as debris or an incomplete store.

`adapter.json` records the adapter schema, bit width, optional dimension,
`empty_lazy` sentinel state, and SHA-256/size checks for each JSON sidecar.
`id_map.json` stores a monotonic `next_u64_id`, string-to-`u64` mappings, and
the active `u64`-to-vector-slot map. `documents.json` and `metadata.json` store
framework text and metadata by string ID. In redb-backed stores these files are
export format only, not inputs to normal verification.

Adapter generation `manifest.json` files verify vector artifacts only.
Framework state is intentionally outside the core `.odb` manifest and is
verified by `adapter.redb`, the adapter layer, or by `ordinaldb verify`.

Lazy empty adapter stores can be persisted before a dimension is known. In that
case the redb manifest and `adapter.json` have `empty_lazy: true`, JSON exports
are empty, and `vectors/` has no active `.odb` generation until vectors are
added.

Adapter writes are in memory until the framework persistence method is called:
LangChain `persist(...)` / `save_local(...)`, LlamaIndex `persist(...)`,
Haystack `save(...)`, and Agno `save(...)`. Agno callers that want immediate
path-backed writes can opt into `auto_save=True`. See
[`edge-deployment.md`](edge-deployment.md) for local/edge deployment guidance
and [`operations.md`](operations.md) for the offline backup and diagnostic
runbook that builds on this layout.

Each save is a full rewrite, not an incremental append: every non-empty save
writes out the complete current vector set as a brand-new generation, so its
cost scales with total store size rather than with how many records changed
since the last save. Measured full-generation adapter save times in
[`limits.md`](limits.md) are approximately 0.47s at 10,000 rows and 1.91s at
100,000 rows. Calling `save()` (or Agno's `auto_save=True`) once per inserted
record therefore multiplies a cost that grows with the store, not a flat
per-call fee — batch writes and call `save()`/`persist()`/`save_local()` once
per batch instead of once per item.

Each non-empty adapter save writes a new immutable vector generation under
`vectors/` and publishes the active generation pointer in the stable
`adapter.redb` database. The same redb write transaction updates the manifest,
replaces the derived legacy JSON payload table, reconciles normalized lookup
tables row by row, retains completed generation records, preserves recovery/GC
tables, and appends an audit row keyed by the new commit sequence.
Generation records carry an explicit lifecycle state: the newly published
generation is `active`, and previously active generation rows are rewritten as
`retired` in the same transaction that publishes the new active pointer.

Adapter writes acquire a Rust-owned cross-platform advisory writer lock; PID
metadata in `.ordinaldb.write.lock` is diagnostic only and does not determine
ownership. Existing redb stores also require the caller's expected revision to
match the durable store revision before publication.
Direct Rust callers should use the CAS-shaped `commit(expected,
AdapterMutation)` API. The `ReplaceLegacySnapshot` mutation still feeds the
current compatibility bridge. `PatchMetadata` applies metadata-object updates
for existing IDs directly in redb, increments the revision and audit log, and
does not publish a new vector generation. The exported commit path keeps
revision comparison, generation validation, table mutation, and active-pointer
publication under the storage engine's writer lock.

The remaining storage-hardening work is to extend `AdapterMutation` with
document, ID-map, and vector-generation deltas so callers can stop rebuilding
full JSON payload snapshots for content changes.

Generation cleanup is explicit and never automatic: OrdinalDB does not run a
background or save-triggered GC, so retained and partial generations
accumulate on disk under `vectors/` until an operator deliberately reclaims
them:

```bash
ordinaldb adapter gc adapter-store --retain 2
```

`adapter gc` records reclaimable paths in `adapter.redb`, marks them `deleting`,
deletes only non-active generation directories beyond the retained-generation
budget plus abandoned partial temporary generation directories, then records
deleted paths in `adapter.redb`. GC records are constrained to known lifecycle states
(`active`, `retired`, `reclaimable`, `deleting`, and `deleted`) instead of
arbitrary strings. If a process exits after filesystem deletion but before the
final `deleted` event is recorded, the next non-dry-run `adapter gc` reconciles
the previous `deleting` event and records `deleted`; if the directory still
exists and is not pinned, it is deleted again before recording recovery. A
generation containing `.ordinaldb.pin` is protected from deletion. `ordinaldb
stats` reports generation lifecycle visibility: active completed generations,
retained completed generations, partial temporary generations, and currently
reclaimable partial generations are counted separately. See
[`operations.md`](operations.md#diagnostic-field-reference) for the exact
`inspect`/`stats` field names and how "reclaimable" in that output differs
from "reclaimable" in `adapter gc`'s own output.
