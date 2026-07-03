# OrdinalDB Operations

**Status:** Reference procedure for `ordinaldb-v0.2.0` and later.

This document describes the supported offline operations path. It does not
add online backup, automatic garbage collection, or cross-process live
sharing guarantees.

## Scope

Supported assumptions:

- local filesystem only;
- one writable process per adapter directory;
- no cross-process concurrent read/write guarantee;
- application-owned embeddings and adapter-owned payloads;
- core `.odb` bundles are vector-only artifacts;
- adapter directories use `adapter.redb` as the authoritative state and JSON
  sidecars as compatibility exports.

Do not run these procedures on network, removable, or otherwise unsupported
filesystems when making durability claims.

## Required Diagnostics

Before and after any offline operation, collect:

```bash
ordinaldb inspect --json <store-or-bundle>
ordinaldb verify --json <store-or-bundle>
ordinaldb stats --json <store-or-bundle>
```

`inspect` identifies the active layout and generation. `verify` checks hashes,
manifests, mappings, active-generation binding, and compatibility exports.
`stats` reports vector count, active IDs, generation count, byte footprint by
component, and orphan generation state.

Always pass `--json` when a procedure below says to "record" or "compare" a
field. The default (non-JSON) text output uses different, abbreviated field
labels (for example `orphan_generations:` for the count and one
`orphan_generation_path:` line per path, instead of the JSON keys named
below) and is meant for a human reading a terminal, not for diffing or
scripting.

## Diagnostic Field Reference

`inspect --json` and `stats --json` share a common field set for both core
`.odb` bundles and adapter directories: `kind`, `path`, `vector_count`, plus
(`inspect` only) `schema_version`, `adapter`, `bits`, `dim`, `empty_lazy`,
`sidecar_count`, `active_generation_id`, `active_generation_path`,
`active_generation_manifest_sha256`, and `active_generation_manifest_size_bytes`.
For a plain core bundle these adapter-shaped fields (`schema_version`,
`adapter`, `active_generation_*`) are simply `null` — a core bundle has no
adapter and no generation concept.

Generation-lifecycle fields, by contrast, are **adapter-directory only**:
`ordinaldb inspect`/`stats` against a plain core `.odb` bundle omits them from
JSON output entirely rather than reporting misleading zeros. They only appear
when the target is an adapter directory:

| Field | Meaning |
| --- | --- |
| `generation_count` | Total generation directories under `vectors/` (completed + partial). |
| `active_generation_count` | `1` if the currently active generation (per `adapter.redb`) is present on disk among the completed generations, `0` otherwise. `0` here means the active pointer's directory is missing — treat it as corruption, not a benign state. |
| `completed_generation_count` | Fully-written generation directories: the active one plus any older completed generations not yet reclaimed. |
| `retained_generation_count` / `retained_generation_paths` | Completed generations *other than* the active one — old generations still on disk, eligible for `adapter gc --retain N` once they exceed the retained budget. |
| `partial_generation_count` / `partial_generation_paths` | Incomplete/abandoned temporary generation directories left behind by an interrupted write (e.g. a crash mid-save). Never active, never valid to load. |
| `reclaimable_generation_count` / `reclaimable_generation_paths` | In `inspect`/`stats` output this is currently the same set as the partial generations above — generations that are always safe to reclaim regardless of `--retain N`. **This is a different, narrower sense of "reclaimable" than `adapter gc`'s own output** (below), which additionally treats completed generations beyond the `--retain N` budget as reclaimable. Don't assume the two `reclaimable` fields mean the same thing. |
| `orphan_generation_count` / `orphan_generation_paths` | The union of the retained and partial sets above — every generation directory that is not the current active one, regardless of crash history. This is broader than "crash debris": a store that has simply never been garbage-collected will show a nonzero `orphan_generation_count` from ordinary retained generations alone. |

`bytes_by_component` in `stats --json` uses different keys depending on the
target: `{"manifest", "total"}` for a core bundle, versus
`{"adapter_state", "compatibility_exports", "vectors", "total"}` for an
adapter directory.

`ordinaldb adapter gc --json` reports its own, differently-shaped fields:
`retained_generation_paths` (generations kept under the `--retain N` budget),
`reclaimable_generation_paths` (generations this run is about to delete —
completed-beyond-budget plus partial), `deleted_generation_paths`,
`pinned_generation_paths` (protected by `.ordinaldb.pin`), and
`redb_commit_sequence`. Treat `gc`'s vocabulary as the operative one when you
are about to delete something; treat `stats`'s `orphan_*`/`retained_*` fields
as informational context, not a to-delete list.

## Offline Backup

1. Stop the writer process, or otherwise prove the adapter directory or core
   bundle is closed for writes. OrdinalDB's own writer lock
   (`.ordinaldb.write.lock`) is advisory only and does not enforce this for
   you — see [`persistence.md`](persistence.md#adapter-directories) — so
   "closed for writes" has to be an operational guarantee (stopped process,
   held deployment lock, etc.), not something a diagnostic command can
   confirm on its own.
2. Run `ordinaldb verify --json <source>` and save the output with the backup
   record. A failing source verification blocks the backup claim.
3. Run `ordinaldb stats --json <source>` and save the output so the copy can be
   compared.
4. Copy the complete store directory with a platform tool that preserves regular
   files and directories. Do not dereference symlinks. A valid OrdinalDB
   store must not depend on symlinked artifacts.
5. Run `ordinaldb verify --json <copy>`. The backup is accepted only if the copy
   verifies.
6. Run `ordinaldb stats --json <copy>` and compare vector count, active ID
   count, generation count, orphan generation paths, and component byte totals
   against the source.
7. Restart the writer only after the source and copy evidence is recorded.

For a core `.odb` bundle, the same flow applies to the bundle directory. For an
adapter directory, copy the entire directory, including `adapter.redb`,
compatibility JSON exports, and every generation directory under `vectors/`.

Running `ordinaldb adapter export-json <source>` before the copy is optional,
not part of the required flow above: `adapter.redb` is the authoritative
control plane, and the JSON sidecars are derived exports, not a second live
state (see [`persistence.md`](persistence.md#adapter-directories)). Export
fresh JSON only when you specifically want a portable, human-readable
snapshot alongside the backup.

## Offline Restore

A backup is only as good as its restore path. To restore from a verified
backup:

1. Confirm the backup itself still verifies: `ordinaldb verify --json <backup>`.
   Do not restore from a backup that fails verification.
2. Copy the backup directory to the restore target with the same symlink-free
   copy discipline as step 4 of Offline Backup. Do not restore onto a path
   that still has an active writer.
3. Run `ordinaldb verify --json <restored>` and `ordinaldb stats --json
   <restored>`, and compare the `stats` output (vector count, active ID
   count, generation count, byte totals) against the evidence recorded at
   backup time. Treat any mismatch as a failed restore.
4. Only resume writing to the restored path after both checks in step 3 pass.

## Orphan Generation Maintenance

Crash recovery may leave complete or partial generation directories that are not
the active generation. These are orphans. They are safe to remove only under
exclusive ownership after verification proves the active generation is healthy.

Note that `orphan_generation_paths` (see
[Diagnostic Field Reference](#diagnostic-field-reference) above) is not
exclusively crash debris: a store that has simply never been garbage-collected
also reports its old, completed, non-active generations as orphans. Do not
assume every path in that list came from a crash before investigating.

1. Stop the writer process and hold exclusive ownership of the adapter
   directory. As in Offline Backup, `.ordinaldb.write.lock` is diagnostic
   only, not an enforcement mechanism — treat "exclusive ownership" as
   something you guarantee operationally.
2. Run `ordinaldb verify --json <store>`. Do not remove anything if verification
   fails.
3. Run `ordinaldb inspect --json <store>` and record `active_generation_path`.
   This is the one generation path you must never remove, and it is not
   included in `stats`'s `orphan_generation_paths`.
4. Run `ordinaldb stats --json <store>` and record `orphan_generation_paths`.
5. Confirm that each path to remove is listed as an orphan, is beneath the
   store's `vectors/` directory, and is not the `active_generation_path` from
   step 3.
6. Remove only the listed orphan generation directories.
7. Run `ordinaldb verify --json <store>` again.
8. Run `ordinaldb stats --json <store>` again and record that orphan state and
   byte totals changed as expected.

`ordinaldb adapter gc <path> --retain N` automates reclaiming completed
generations beyond the retained count, plus abandoned partial temporary
generations, under the same exclusive-ownership and verify-before-delete
discipline described above. Use the manual walkthrough in this section when
you want full manual control before deleting anything on a store you don't
fully trust yet, or to inspect orphans `adapter gc` does not consider safe to
reclaim on its own. See [`persistence.md`](persistence.md) for `adapter gc`'s
generation-lifecycle model.

> **Warning — never run `adapter gc` (or `verify`/`inspect`/`stats`) against
> a store with a live writer.** Step 1's exclusive-ownership requirement is
> not only about the files gc deletes. `adapter.redb` permits one open
> handle at a time, and the writer only holds its advisory lock for the
> duration of each save, so gc's own verification read can interleave with
> a live writer's save. If gc (or any diagnostic reader) has the database
> open at the wrong instant, the writer's save fails with redb's native
> "Database already open. Cannot acquire lock." — and when that contention
> hits *after* the writer's commit already published, the failure is false:
> the data is durable. The adapter degrades that specific case to a
> `UserWarning` when it can prove the commit landed, and otherwise raises an
> error telling you to re-load and inspect before retrying — because a
> caller that reacts to the "failed" save by re-adding the batch and saving
> again double-inserts its records. The race can also make the writer's
> save fail before publishing (leaving an unreferenced generation as
> debris), or make gc itself fail to acquire the writer lock. None of these
> outcomes corrupt the store, but all of them are avoidable: stop the
> writer first.

Two scope caveats when scheduling and interpreting these checks:

- **Plain loads do not audit `vectors/` for symlinks.** Opening a store
  through the adapters (`AdapterStore.load`, a framework constructor, or a
  `load_local`/`from_persist_dir`-style loader) validates only the active
  generation path it actually loads (rejecting symlinks along that path).
  It does not sweep the rest of `vectors/` for symlinked or foreign
  artifacts — only `verify`, `inspect`, `stats`, and `adapter gc` walk the
  full generation layout. A symlink smuggled into a non-active generation
  is therefore invisible to ordinary application traffic; schedule explicit
  `verify` runs (not just application loads) if you rely on that audit.
- **`verify` trusts non-active generations by name.** Hash and manifest
  verification covers the *active* generation only; retained and partial
  generations are classified by their directory-name layout, not by content
  digests. The orphan procedure's classifications above tell you where a
  directory sits in the generation lifecycle — they are not an authenticity
  claim about its contents. Do not treat a passing `verify` as an integrity
  guarantee for old generations you intend to keep or restore from.

## Failure Handling

- If `verify` fails before backup, do not claim the backup as valid.
- If the copy fails `verify`, discard it and keep the source unchanged.
- If `stats` reports an unexpected active ID count, generation count, or orphan
  path, record the output and investigate before deleting anything.
- If any diagnostic reports a symlink, path escape, malformed generation,
  mismatched digest, stale export, or missing active generation, treat the store
  as corrupt and fail closed.
