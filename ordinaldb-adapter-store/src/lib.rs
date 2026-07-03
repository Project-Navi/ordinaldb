#![warn(missing_docs)]

//! redb-authoritative document and metadata store for OrdinalDB adapters.
//!
//! An adapter root directory pairs one mutable redb database
//! ([`ADAPTER_STORE_FILE`], `adapter.redb`) with immutable vector index
//! generations (OrdinalDB bundle directories). The redb file is the source
//! of truth for everything except the vector artifacts themselves:
//!
//! - the four legacy JSON payloads ([`LegacyPayloads`]) plus their expanded
//!   row tables (documents, metadata, string/u64 ID maps, vector slots),
//! - generation lifecycle records and a garbage-collection queue,
//! - an append-only audit log keyed by commit sequence,
//! - a store manifest whose table counts are cross-checked on every open.
//!
//! # Directory layout
//!
//! ```text
//! <root>/
//!   adapter.redb                 # this store (authoritative)
//!   .ordinaldb.write.lock        # advisory writer lock file
//!   index.odb/                   # legacy generation bundle (generation id 1), or
//!   vectors/g000000000002.odb/   # 12-digit zero-padded generation bundles
//! ```
//!
//! # Locking
//!
//! All writes serialize on a non-blocking advisory file lock acquired via
//! [`acquire_writer_lock`] (see [`WriterLockGuard`] for the exact
//! mechanics). High-level entry points ([`commit`],
//! [`write_legacy_snapshot`], [`record_generation_gc`]) acquire and release
//! the lock internally; their `*_with_existing_lock` variants let a caller
//! hold one guard across a batch of operations and refuse to run unless the
//! lock is held by the current process through this crate. Reads
//! ([`open_verified`], [`generation_gc_events`]) do not take the lock.
//!
//! # Optimistic concurrency
//!
//! Mutating an existing store requires presenting the [`StoreRevision`]
//! the caller last observed. The mutation only proceeds when every revision
//! field matches the store's current manifest; otherwise it fails with a
//! "stale adapter snapshot" error and writes nothing. Creating a brand-new
//! store requires presenting no revision (`None`).
//!
//! # Generation lifecycle
//!
//! Vector generations move through the states `active`, `retired`,
//! `reclaimable`, `deleting`, and `deleted`. The `generations` table only
//! ever records `active` and `retired`: writing a new snapshot marks every
//! other generation `retired` (stamping `retired_by_commit_sequence`) and
//! a verified store contains exactly one `active` row — or zero rows for an
//! `empty_lazy` store that has no vectors yet. Later reclamation states are
//! recorded as [`GenerationGcUpdate`] events in the GC queue via
//! [`record_generation_gc`] and drained by the caller.
//!
//! # Crash debris
//!
//! A crash during generation replacement can leave scratch directories
//! (`.g….odb.tmp-<pid>-<nanos>`) or stray entries under `vectors/`. Debris
//! is *reclaimable, never fatal*: [`open_verified`] tolerates it,
//! [`scan_generation_directory`] classifies it with a structured warning
//! per entry, and [`remove_generation_debris`] deletes it under the writer
//! lock. Debris never carries a canonical `g<12 digits>.odb` name, so the
//! active generation and everything referenced by `adapter.redb` are
//! structurally excluded from reclamation. Symlinked entries and
//! canonically-named non-directories still fail closed.
//!
//! # Verification
//!
//! [`open_verified`] never trusts the database blindly: it re-parses the
//! payloads, cross-checks them against the manifest, the row tables, and the
//! generation records, and verifies the on-disk active generation bundle
//! manifest (including its SHA-256 digest) through the OrdVec manifest
//! pipeline before returning a [`VerifiedAdapterStore`].

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use fs4::{FileExt, TryLockError};
use getrandom::fill as fill_random;
use ordvec_manifest::{sha256_file, verify_for_load, ManifestIndexParams, VerifyOptions};
use redb::{Database, ReadableDatabase, ReadableTable, ReadableTableMetadata, TableDefinition};
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{json, Map, Number, Value};

/// File name of the redb database inside an adapter root directory.
pub const ADAPTER_STORE_FILE: &str = "adapter.redb";
/// Schema identifier required in the adapter store manifest
/// (`meta` table, `manifest` row).
pub const ADAPTER_STORE_SCHEMA_VERSION: &str = "ordinaldb.adapter_store.v1";
/// Schema identifier required in the adapter legacy JSON payload
/// ([`LegacyPayloads::adapter_json`]).
pub const ADAPTER_SCHEMA_VERSION: &str = "ordinaldb.adapter.v1";
/// Schema identifier required in the ID-map legacy JSON payload
/// ([`LegacyPayloads::id_map_json`]).
pub const ID_MAP_SCHEMA_VERSION: &str = "ordinaldb.adapter.id_map.v1";
/// Schema identifier required in the documents legacy JSON payload
/// ([`LegacyPayloads::documents_json`]).
pub const DOCUMENTS_SCHEMA_VERSION: &str = "ordinaldb.adapter.documents.v1";
/// Schema identifier required in the metadata legacy JSON payload
/// ([`LegacyPayloads::metadata_json`]).
pub const METADATA_SCHEMA_VERSION: &str = "ordinaldb.adapter.metadata.v1";

const WRITE_LOCK_FILE: &str = ".ordinaldb.write.lock";
const ADAPTER_FILE: &str = "adapter.json";
const ID_MAP_FILE: &str = "id_map.json";
const DOCUMENTS_FILE: &str = "documents.json";
const METADATA_FILE: &str = "metadata.json";
const INDEX_DIR: &str = "index.odb";
const VECTORS_DIR: &str = "vectors";
const MANIFEST_FILE: &str = "manifest.json";
const ROW_IDENTITY_KIND: &str = "row_id_identity";
const MAX_GENERATION_MANIFEST_BYTES: u64 = 1024 * 1024;
const GENERATION_STATE_ACTIVE: &str = "active";
const GENERATION_STATE_RETIRED: &str = "retired";
const GENERATION_STATE_RECLAIMABLE: &str = "reclaimable";
const GENERATION_STATE_DELETING: &str = "deleting";
const GENERATION_STATE_DELETED: &str = "deleted";

const KEY_FORMAT: &str = "string_or_u64_v1";
const PAYLOAD_FORMAT: &str = "json_v1";

const META: TableDefinition<&str, &str> = TableDefinition::new("meta");
const JSON_PAYLOADS: TableDefinition<&str, &str> = TableDefinition::new("json_payloads");
const STRING_TO_U64: TableDefinition<&str, u64> = TableDefinition::new("string_to_u64");
const U64_TO_STRING: TableDefinition<u64, &str> = TableDefinition::new("u64_to_string");
const U64_TO_SLOT: TableDefinition<u64, u64> = TableDefinition::new("u64_to_slot");
const DOCUMENTS: TableDefinition<u64, &str> = TableDefinition::new("documents");
const METADATA: TableDefinition<u64, &str> = TableDefinition::new("metadata");
const GENERATIONS: TableDefinition<u64, &str> = TableDefinition::new("generations");
const PENDING_BATCHES: TableDefinition<&str, &str> = TableDefinition::new("pending_batches");
const TOMBSTONES: TableDefinition<u64, &str> = TableDefinition::new("tombstones");
const GC_QUEUE: TableDefinition<u64, &str> = TableDefinition::new("gc_queue");
const AUDIT_LOG: TableDefinition<u64, &str> = TableDefinition::new("audit_log");

/// Error type returned by every fallible adapter store operation.
///
/// Carries a single human-readable message and exposes no structured
/// variants. Underlying failures from redb (open/transaction/table/commit/
/// storage), filesystem I/O, JSON parsing, and the system randomness source
/// are converted into this type via `From`, preserving only their message
/// text. Validation failures (stale revisions, manifest mismatches, invalid
/// payloads, lock contention) are reported the same way.
#[derive(Debug)]
pub struct AdapterStoreError(String);

impl AdapterStoreError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for AdapterStoreError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for AdapterStoreError {}

macro_rules! impl_from_display {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl From<$ty> for AdapterStoreError {
                fn from(err: $ty) -> Self {
                    Self::new(err.to_string())
                }
            }
        )+
    };
}

impl_from_display!(
    redb::CommitError,
    redb::DatabaseError,
    redb::StorageError,
    redb::TableError,
    redb::TransactionError,
    getrandom::Error,
);

impl From<std::io::Error> for AdapterStoreError {
    fn from(err: std::io::Error) -> Self {
        Self::new(err.to_string())
    }
}

impl From<serde_json::Error> for AdapterStoreError {
    fn from(err: serde_json::Error) -> Self {
        Self::new(err.to_string())
    }
}

/// Runs `operation`, converting any panic that escapes it into a structured
/// [`AdapterStoreError`].
///
/// The redb storage engine deserializes B-tree pages with internal
/// `unreachable!()` assertions; a corrupted or tampered `adapter.redb` can
/// therefore panic (`internal error: entered unreachable code`) instead of
/// returning a storage error. Every path that opens or reads the database
/// funnels through this guard so on-disk corruption always surfaces as an
/// error value, never as a panic (or as a `pyo3_runtime.PanicException`
/// through the Python bindings).
fn guard_against_storage_panics<T>(
    context: &str,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation)) {
        Ok(result) => result,
        Err(payload) => Err(AdapterStoreError::new(format!(
            "{ADAPTER_STORE_FILE} is corrupted or inconsistent ({context}): \
             storage engine panicked: {}",
            panic_payload_message(payload.as_ref())
        ))),
    }
}

fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> &str {
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        message
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.as_str()
    } else {
        "non-string panic payload"
    }
}

/// Result alias used by all public adapter store APIs; the error is always
/// [`AdapterStoreError`].
pub type Result<T> = std::result::Result<T, AdapterStoreError>;

/// The four canonical JSON documents that historically lived beside the
/// index as sidecar files and are now stored verbatim in the redb
/// `json_payloads` table.
///
/// Each payload must be a single JSON object with an exact key set and the
/// matching `schema_version` value; parsing is strict (duplicate object keys
/// and trailing data are rejected). The documents, metadata, and ID-map
/// payloads must agree with each other: documents and metadata must be keyed
/// by exactly the set of active string IDs in `string_to_u64`.
#[derive(Debug, Clone)]
pub struct LegacyPayloads {
    /// Adapter description (schema [`ADAPTER_SCHEMA_VERSION`]): keys
    /// `schema_version`, `adapter`, `bits` (1, 2, or 4), `dim` (positive or
    /// null), `empty_lazy`, `index_path` (root-relative), and `sidecars`.
    pub adapter_json: String,
    /// ID mappings (schema [`ID_MAP_SCHEMA_VERSION`]): keys `schema_version`,
    /// `string_to_u64`, `u64_to_slot`, and the `next_u64_id` allocation
    /// watermark (strictly greater than every allocated u64 ID).
    pub id_map_json: String,
    /// Document bodies keyed by string ID
    /// (schema [`DOCUMENTS_SCHEMA_VERSION`]); values must be strings.
    pub documents_json: String,
    /// Per-document metadata keyed by string ID
    /// (schema [`METADATA_SCHEMA_VERSION`]); values must be JSON objects.
    pub metadata_json: String,
}

/// A fully verified snapshot of an adapter store, returned by
/// [`open_verified`] and by every successful mutation.
///
/// Holding this value means the store manifest, the legacy JSON payloads,
/// the expanded row tables, the generation records, and the on-disk active
/// generation bundle manifest were all cross-checked and mutually consistent
/// at read time. The accessor methods read fields directly from
/// [`Self::manifest`] and return `None` when a field is absent or has an
/// unexpected JSON type.
#[derive(Debug, Clone)]
pub struct VerifiedAdapterStore {
    /// Parsed adapter store manifest (the `manifest` row of the `meta`
    /// table), already validated against [`ADAPTER_STORE_SCHEMA_VERSION`].
    pub manifest: Value,
    /// Verbatim legacy JSON payloads from the `json_payloads` table.
    pub payloads: LegacyPayloads,
}

impl VerifiedAdapterStore {
    /// Name of the adapter that wrote this store (manifest `adapter_name`).
    pub fn adapter_name(&self) -> Option<&str> {
        self.manifest.get("adapter_name").and_then(Value::as_str)
    }

    /// RankQuant quantization bit width; always 1, 2, or 4 for a verified
    /// store.
    pub fn bits(&self) -> Option<u8> {
        self.manifest
            .get("bits")
            .and_then(Value::as_u64)
            .and_then(|value| u8::try_from(value).ok())
    }

    /// Vector dimensionality of the active generation. `None` for an
    /// `empty_lazy` store, whose manifest records `dim` as null.
    pub fn dim(&self) -> Option<usize> {
        self.manifest
            .get("dim")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
    }

    /// Number of vectors in the active generation (0 for an `empty_lazy`
    /// store). Matches the active string-ID count in a verified store.
    pub fn vector_count(&self) -> Option<usize> {
        self.manifest
            .get("vector_count")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
    }

    /// Whether the store is `empty_lazy`: created before any vectors exist,
    /// with no generation bundle on disk, `dim` null, and no document,
    /// metadata, or ID rows. Defaults to `false` when the field is missing.
    pub fn empty_lazy(&self) -> bool {
        self.manifest
            .get("empty_lazy")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }

    /// Identifier of the active generation. Generation 1 corresponds to the
    /// legacy `index.odb` path; `vectors/g{id:012}.odb` paths carry their ID
    /// in the directory name. 0 for an `empty_lazy` store.
    pub fn active_generation_id(&self) -> Option<u64> {
        self.manifest
            .get("active_generation_id")
            .and_then(Value::as_u64)
    }

    /// Root-relative path of the active generation bundle (either
    /// `index.odb` or `vectors/g{id:012}.odb`). Empty string for an
    /// `empty_lazy` store.
    pub fn active_generation_path(&self) -> Option<&str> {
        self.manifest
            .get("active_generation_path")
            .and_then(Value::as_str)
    }

    /// Hex-encoded SHA-256 digest of the active generation's
    /// `manifest.json`, re-verified against the file on every open. `None`
    /// for an `empty_lazy` store.
    pub fn active_generation_manifest_sha256(&self) -> Option<&str> {
        self.manifest
            .get("active_generation_manifest_sha256")
            .and_then(Value::as_str)
    }

    /// Size in bytes of the active generation's `manifest.json` (capped at
    /// 1 MiB by validation). `None` for an `empty_lazy` store.
    pub fn active_generation_manifest_size_bytes(&self) -> Option<u64> {
        self.manifest
            .get("active_generation_manifest_size_bytes")
            .and_then(Value::as_u64)
    }
}

/// Compare-and-swap token identifying one committed state of an adapter
/// store.
///
/// Capture it from a [`VerifiedAdapterStore`] with
/// [`StoreRevision::from_manifest`] and pass it back as the `expected`
/// revision when mutating. A mutation only proceeds when **every** field
/// matches the store's current manifest exactly; otherwise the operation
/// fails with a "stale adapter snapshot" error and nothing is written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoreRevision {
    /// Random UUIDv4 minted when the store was first created; stable across
    /// commits, so it also detects a store being replaced wholesale.
    pub store_uuid: String,
    /// Monotonic commit counter, starting at 1 on creation and incremented
    /// by exactly 1 on every successful mutation.
    pub commit_sequence: u64,
    /// Identifier of the generation that was active at this revision.
    pub active_generation_id: u64,
    /// Root-relative bundle path of the generation that was active at this
    /// revision (empty string for an `empty_lazy` store).
    pub active_generation_path: String,
    /// Hex SHA-256 digest of the active generation's `manifest.json`;
    /// `None` for an `empty_lazy` store.
    pub active_generation_manifest_sha256: Option<String>,
    /// Number of active string IDs (rows) at this revision.
    pub active_id_count: u64,
}

/// How an adapter store came into existence.
///
/// Recorded in the manifest `origin` field when the store is first created
/// and preserved verbatim by all later commits (a later
/// [`write_legacy_snapshot_with_origin`] cannot change it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreOrigin {
    /// The store was created fresh (`"created"`).
    Created,
    /// The store was imported from pre-redb legacy JSON sidecar files
    /// (`"imported_legacy_json"`).
    ImportedLegacyJson,
    /// The store was produced by upgrading an older on-disk schema
    /// (`"upgraded_schema"`).
    UpgradedSchema,
}

impl StoreOrigin {
    fn as_str(self) -> &'static str {
        match self {
            Self::Created => "created",
            Self::ImportedLegacyJson => "imported_legacy_json",
            Self::UpgradedSchema => "upgraded_schema",
        }
    }
}

/// Result of a successful [`commit`]: the new revision token plus the full
/// post-commit manifest.
#[derive(Debug, Clone)]
pub struct CommittedRevision {
    /// Revision token identifying the newly committed state; pass it as
    /// `expected` to the next mutation.
    pub revision: StoreRevision,
    /// Complete post-commit adapter store manifest.
    pub manifest: Value,
}

/// A mutation applied atomically by [`commit`] in a single redb write
/// transaction under the writer lock.
#[derive(Debug, Clone)]
pub enum AdapterMutation {
    /// Replace the entire store contents with a new legacy snapshot. The
    /// payloads are validated against the on-disk active generation bundle;
    /// previously active generation rows are marked `retired` (stamped with
    /// `retired_by_commit_sequence`), the new generation is recorded as
    /// `active`, and a `legacy_snapshot_written` audit event is appended.
    ReplaceLegacySnapshot(LegacyPayloads),
    /// Patch per-document metadata in place. Must contain at least one
    /// entry; every patched ID must currently be active (present in
    /// `string_to_u64`), and each patch replaces that document's whole
    /// metadata object. Appends a `metadata_patched` audit event.
    PatchMetadata(Vec<MetadataPatch>),
}

/// Whole-object metadata replacement for a single document, used by
/// [`AdapterMutation::PatchMetadata`].
#[derive(Debug, Clone)]
pub struct MetadataPatch {
    /// String ID of the document; must be non-empty and currently active.
    pub id: String,
    /// New metadata object; replaces the document's previous metadata
    /// entirely (no per-key merge).
    pub metadata: Map<String, Value>,
}

/// A generation lifecycle transition to append to the GC queue via
/// [`record_generation_gc`].
#[derive(Debug, Clone)]
pub struct GenerationGcUpdate {
    /// Generation the update refers to, when known.
    pub generation_id: Option<u64>,
    /// Root-relative, normalized (forward-slash, no `..` or special
    /// components) path of the generation bundle. Must not target the
    /// legacy `index.odb` bundle.
    pub path: String,
    /// Target lifecycle state: one of `"active"`, `"retired"`,
    /// `"reclaimable"`, `"deleting"`, or `"deleted"`.
    pub state: String,
    /// Non-empty human-readable reason for the transition.
    pub reason: String,
}

/// A recorded GC queue entry, as returned by [`generation_gc_events`].
#[derive(Debug, Clone)]
pub struct GenerationGcEvent {
    /// Key of the row in the `gc_queue` table; assigned sequentially
    /// starting at 1 and strictly increasing across commits.
    pub row_id: u64,
    /// Generation the event refers to, when one was recorded.
    pub generation_id: Option<u64>,
    /// Root-relative path of the generation bundle the event targets.
    pub path: String,
    /// Lifecycle state recorded for the transition (one of the five states
    /// listed on [`GenerationGcUpdate::state`]).
    pub state: String,
    /// Reason recorded for the transition.
    pub reason: String,
    /// Wall-clock timestamp (Unix milliseconds) captured when the event was
    /// recorded, if present in the row.
    pub recorded_at_unix_ms: Option<u64>,
}

impl StoreRevision {
    /// Extracts the revision fields from an adapter store manifest (for
    /// example [`VerifiedAdapterStore::manifest`]).
    ///
    /// # Errors
    /// Fails when any required field is missing or has the wrong JSON type.
    pub fn from_manifest(manifest: &Value) -> Result<Self> {
        Ok(Self {
            store_uuid: required_string(manifest, "store_uuid")?.to_string(),
            commit_sequence: required_u64(manifest, "commit_sequence")?,
            active_generation_id: required_u64(manifest, "active_generation_id")?,
            active_generation_path: required_string(manifest, "active_generation_path")?
                .to_string(),
            active_generation_manifest_sha256: optional_string(
                manifest,
                "active_generation_manifest_sha256",
            )?,
            active_id_count: required_u64(manifest, "active_id_count")?,
        })
    }
}

/// Applies `mutation` to the existing adapter store at `root` under the
/// writer lock, guarded by a revision compare-and-swap.
///
/// Acquires the advisory writer lock (failing fast if it is already held),
/// verifies that `expected` matches the store's current revision on every
/// field, applies the mutation in one redb write transaction, increments
/// `commit_sequence` by 1, appends an audit log event, fsyncs the database
/// file and the root directory, and re-opens the store fully verified.
///
/// # Errors
/// Fails without writing when the lock is unavailable, the store does not
/// exist, `expected` is stale, or the mutation payload fails validation.
pub fn commit(
    root: impl AsRef<Path>,
    expected: StoreRevision,
    mutation: AdapterMutation,
) -> Result<CommittedRevision> {
    let root = root.as_ref();
    ensure_adapter_root_directory(root)?;
    let _writer_lock = WriterLockGuard::acquire(root)?;
    commit_inner(root, &expected, mutation)
}

/// Same as [`commit`], but for callers that already hold the writer lock
/// obtained from [`acquire_writer_lock`].
///
/// # Errors
/// In addition to the [`commit`] failure modes, fails when the writer lock
/// is not currently held by this process *through this crate* — a lock file
/// that merely exists on disk is not sufficient.
pub fn commit_with_existing_lock(
    root: impl AsRef<Path>,
    expected: StoreRevision,
    mutation: AdapterMutation,
) -> Result<CommittedRevision> {
    let root = root.as_ref();
    ensure_adapter_root_directory(root)?;
    require_writer_lock_held_by_current_process(root)?;
    commit_inner(root, &expected, mutation)
}

/// Writes a full legacy snapshot under the writer lock, creating the store
/// when it does not exist yet.
///
/// `expected` must be `None` when no `adapter.redb` exists at `root`, and
/// must be the store's current revision when overwriting an existing store
/// (both mismatches fail with a "stale adapter snapshot" error). The
/// payloads are validated against the on-disk active generation bundle
/// before anything is written. Equivalent to
/// [`write_legacy_snapshot_with_origin`] with [`StoreOrigin::Created`].
pub fn write_legacy_snapshot(
    root: impl AsRef<Path>,
    expected: Option<StoreRevision>,
    payloads: LegacyPayloads,
) -> Result<VerifiedAdapterStore> {
    write_legacy_snapshot_with_origin(root, expected, payloads, StoreOrigin::Created)
}

/// [`write_legacy_snapshot`] with an explicit [`StoreOrigin`].
///
/// `origin` is only recorded when the store is first created; a store that
/// already exists keeps the origin stamped at creation time regardless of
/// the value passed here.
pub fn write_legacy_snapshot_with_origin(
    root: impl AsRef<Path>,
    expected: Option<StoreRevision>,
    payloads: LegacyPayloads,
    origin: StoreOrigin,
) -> Result<VerifiedAdapterStore> {
    let root = root.as_ref();
    ensure_adapter_root_directory(root)?;
    let _writer_lock = WriterLockGuard::acquire(root)?;
    write_legacy_snapshot_inner(root, expected.as_ref(), payloads, origin)
}

/// Acquires the adapter root's advisory writer lock, creating the root
/// directory first (component by component, rejecting symlinks) when it
/// does not exist.
///
/// The lock is non-blocking: if any other holder exists — another process,
/// or this process via an earlier guard — this fails immediately with an
/// "adapter writer lock already held" error instead of waiting. Hold the
/// returned guard across a batch of `*_with_existing_lock` calls; the lock
/// is released when the guard is dropped.
pub fn acquire_writer_lock(root: impl AsRef<Path>) -> Result<WriterLockGuard> {
    let root = root.as_ref();
    ensure_adapter_root_directory(root)?;
    WriterLockGuard::acquire(root)
}

/// Same as [`write_legacy_snapshot`], but for callers that already hold the
/// writer lock obtained from [`acquire_writer_lock`].
///
/// # Errors
/// In addition to the [`write_legacy_snapshot`] failure modes, fails when
/// the writer lock is not currently held by this process through this
/// crate.
pub fn write_legacy_snapshot_with_existing_lock(
    root: impl AsRef<Path>,
    expected: Option<StoreRevision>,
    payloads: LegacyPayloads,
) -> Result<VerifiedAdapterStore> {
    let root = root.as_ref();
    ensure_adapter_root_directory(root)?;
    require_writer_lock_held_by_current_process(root)?;
    write_legacy_snapshot_inner(root, expected.as_ref(), payloads, StoreOrigin::Created)
}

fn commit_inner(
    root: &Path,
    expected: &StoreRevision,
    mutation: AdapterMutation,
) -> Result<CommittedRevision> {
    let verified = match mutation {
        AdapterMutation::ReplaceLegacySnapshot(payloads) => {
            write_legacy_snapshot_inner(root, Some(expected), payloads, StoreOrigin::Created)?
        }
        AdapterMutation::PatchMetadata(patches) => commit_metadata_patch(root, expected, patches)?,
    };
    let revision = StoreRevision::from_manifest(&verified.manifest)?;
    Ok(CommittedRevision {
        revision,
        manifest: verified.manifest,
    })
}

fn commit_metadata_patch(
    root: &Path,
    expected: &StoreRevision,
    patches: Vec<MetadataPatch>,
) -> Result<VerifiedAdapterStore> {
    if patches.is_empty() {
        return Err(AdapterStoreError::new(
            "metadata patch mutation must include at least one entry",
        ));
    }

    let current = current_verified_store(root, None)?
        .ok_or_else(|| AdapterStoreError::new("adapter store is required for metadata patch"))?;
    verify_expected_revision(Some(&current), Some(expected))?;
    let mut parsed = ParsedPayloads::parse(&current.payloads)?;

    for patch in &patches {
        require_non_empty_string_id(&patch.id)?;
        if !parsed.string_to_u64.contains_key(&patch.id) {
            return Err(AdapterStoreError::new(format!(
                "metadata patch ID is not active: {}",
                patch.id
            )));
        }
        parsed
            .metadata
            .insert(patch.id.clone(), Value::Object(patch.metadata.clone()));
    }
    parsed.validate_maps()?;

    let mut manifest = current.manifest.clone();
    let previous_commit_sequence = required_u64(&manifest, "commit_sequence")?;
    let commit_sequence = previous_commit_sequence
        .checked_add(1)
        .ok_or_else(|| AdapterStoreError::new("commit_sequence overflow"))?;
    let previous_audit_count = manifest_table_count(&manifest, "audit_log")?;
    let audit_count = previous_audit_count
        .checked_add(1)
        .ok_or_else(|| AdapterStoreError::new("audit_log table count overflow"))?;
    update_metadata_patch_manifest(&mut manifest, commit_sequence, audit_count)?;
    let manifest_json = canonical_json(&manifest)?;
    let metadata_json = metadata_payload_json(&parsed)?;

    guard_against_storage_panics("committing the metadata patch", || {
        let db = Database::open(root.join(ADAPTER_STORE_FILE))?;
        let write_txn = db.begin_write()?;
        {
            let mut meta = write_txn.open_table(META)?;
            meta.insert("manifest", manifest_json.as_str())?;
        }
        {
            let mut json_payloads = write_txn.open_table(JSON_PAYLOADS)?;
            json_payloads.insert("metadata", metadata_json.as_str())?;
        }
        {
            let mut metadata = write_txn.open_table(METADATA)?;
            for patch in &patches {
                let u64_id = parsed.u64_for_string(&patch.id)?;
                let metadata_value = parsed
                    .metadata
                    .get(&patch.id)
                    .ok_or_else(|| AdapterStoreError::new("metadata patch row missing"))?;
                let metadata_json = canonical_json(metadata_value)?;
                metadata.insert(u64_id, metadata_json.as_str())?;
            }
        }
        {
            let mut audit_log = write_txn.open_table(AUDIT_LOG)?;
            audit_log.insert(
                commit_sequence,
                canonical_json(&json!({
                    "event": "metadata_patched",
                    "commit_sequence": commit_sequence,
                    "patched_id_count": patches.len(),
                    "active_generation_id": required_u64(&manifest, "active_generation_id")?,
                    "active_generation_path": required_string(&manifest, "active_generation_path")?,
                    "created_at_unix_ms": now_ms(),
                }))?
                .as_str(),
            )?;
        }
        write_txn.commit()?;
        Ok(())
    })?;
    sync_file(&root.join(ADAPTER_STORE_FILE))?;
    sync_directory(root)?;

    open_verified(root, None)
}

fn update_metadata_patch_manifest(
    manifest: &mut Value,
    commit_sequence: u64,
    audit_count: u64,
) -> Result<()> {
    let object = manifest
        .as_object_mut()
        .ok_or_else(|| AdapterStoreError::new("adapter store manifest must be an object"))?;
    object.insert("commit_sequence".to_string(), Value::from(commit_sequence));
    object.insert(
        "updated_at_unix_ms".to_string(),
        Value::from(u64::try_from(now_ms()).unwrap_or(u64::MAX)),
    );
    let table_counts = object
        .get_mut("table_counts")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| AdapterStoreError::new("adapter store table_counts must be an object"))?;
    table_counts.insert("audit_log".to_string(), Value::from(audit_count));
    Ok(())
}

fn metadata_payload_json(parsed: &ParsedPayloads) -> Result<String> {
    let mut metadata = Map::new();
    let mut ids = parsed.metadata.keys().collect::<Vec<_>>();
    ids.sort();
    for id in ids {
        let value = parsed
            .metadata
            .get(id)
            .ok_or_else(|| AdapterStoreError::new("metadata payload row missing"))?;
        metadata.insert(id.clone(), value.clone());
    }
    canonical_json(&json!({
        "schema_version": METADATA_SCHEMA_VERSION,
        "metadata": metadata,
    }))
}

/// Appends generation lifecycle transitions to the GC queue under the
/// writer lock.
///
/// Every update is validated first (see the field requirements on
/// [`GenerationGcUpdate`]); all updates are then committed in a single redb
/// transaction together with one `generation_gc_recorded` audit event and a
/// single `commit_sequence` increment. GC queue rows are never consumed by
/// this crate — draining them (and deleting retired bundle directories) is
/// the caller's responsibility.
///
/// An empty `updates` slice writes nothing and returns the current verified
/// store. `expected`, when provided, is compare-and-swap-checked against
/// the store's current revision; the store must already exist either way.
pub fn record_generation_gc(
    root: impl AsRef<Path>,
    expected: Option<StoreRevision>,
    updates: &[GenerationGcUpdate],
) -> Result<VerifiedAdapterStore> {
    let root = root.as_ref();
    ensure_adapter_root_directory(root)?;
    let _writer_lock = WriterLockGuard::acquire(root)?;
    record_generation_gc_inner(root, expected.as_ref(), updates)
}

/// Same as [`record_generation_gc`], but for callers that already hold the
/// writer lock obtained from [`acquire_writer_lock`].
///
/// # Errors
/// In addition to the [`record_generation_gc`] failure modes, fails when
/// the writer lock is not currently held by this process through this
/// crate.
pub fn record_generation_gc_with_existing_lock(
    root: impl AsRef<Path>,
    expected: Option<StoreRevision>,
    updates: &[GenerationGcUpdate],
) -> Result<VerifiedAdapterStore> {
    let root = root.as_ref();
    ensure_adapter_root_directory(root)?;
    require_writer_lock_held_by_current_process(root)?;
    record_generation_gc_inner(root, expected.as_ref(), updates)
}

/// Reads every entry in the GC queue, ordered by ascending
/// [`GenerationGcEvent::row_id`].
///
/// Read-only: does not take the writer lock and performs none of the full
/// store verification done by [`open_verified`]. Each row is re-validated
/// on read; one corrupt row fails the whole call.
pub fn generation_gc_events(root: impl AsRef<Path>) -> Result<Vec<GenerationGcEvent>> {
    let root = root.as_ref();
    let store_file = root.join(ADAPTER_STORE_FILE);
    validate_adapter_store_file(&store_file)?;
    guard_against_storage_panics("reading the GC queue", || {
        let db = Database::open(store_file)?;
        let read_txn = db.begin_read()?;
        let gc_queue = read_txn.open_table(GC_QUEUE)?;
        let mut events = Vec::new();
        for entry in gc_queue.iter()? {
            let (key, value) = entry?;
            let row = parse_json_object(value.value(), "gc_queue row")?;
            let generation_id = optional_u64(&row, "generation_id")?;
            let path = required_string(&row, "path")?.to_string();
            let state = required_string(&row, "state")?.to_string();
            let reason = required_string(&row, "reason")?.to_string();
            let update = GenerationGcUpdate {
                generation_id,
                path: path.clone(),
                state: state.clone(),
                reason: reason.clone(),
            };
            validate_gc_update(&update)?;
            events.push(GenerationGcEvent {
                row_id: key.value(),
                generation_id,
                path,
                state,
                reason,
                recorded_at_unix_ms: optional_u64(&row, "recorded_at_unix_ms")?,
            });
        }
        Ok(events)
    })
}

fn ensure_adapter_root_directory(root: &Path) -> Result<()> {
    let mut current = if root.is_absolute() {
        PathBuf::new()
    } else {
        PathBuf::from(".")
    };

    for component in root.components() {
        match component {
            Component::Prefix(prefix) => {
                current.push(prefix.as_os_str());
            }
            Component::RootDir => {
                current.push(component.as_os_str());
            }
            Component::CurDir => {}
            Component::ParentDir | Component::Normal(_) => {
                current.push(component.as_os_str());
                current = ensure_adapter_root_component(&current)?;
            }
        }
    }

    Ok(())
}

fn ensure_adapter_root_component(component: &Path) -> Result<PathBuf> {
    match fs::symlink_metadata(component) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() {
                if let Some(resolved) = allowed_platform_directory_symlink(component) {
                    let resolved_metadata = fs::symlink_metadata(&resolved).map_err(|err| {
                        AdapterStoreError::new(format!(
                            "adapter root cannot be statted at {}: {err}",
                            resolved.display()
                        ))
                    })?;
                    if resolved_metadata.file_type().is_dir() {
                        return Ok(resolved);
                    }
                }
                return Err(AdapterStoreError::new(format!(
                    "adapter root must not be a symlink: {}",
                    component.display()
                )));
            }
            if !metadata.file_type().is_dir() {
                return Err(AdapterStoreError::new(format!(
                    "adapter root must be a directory: {}",
                    component.display()
                )));
            }
            Ok(component.to_path_buf())
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            match fs::create_dir(component) {
                Ok(()) => {}
                Err(create_err) if create_err.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(create_err) => {
                    return Err(AdapterStoreError::new(format!(
                        "adapter root directory cannot be created at {}: {create_err}",
                        component.display()
                    )));
                }
            }
            ensure_adapter_root_component(component)
        }
        Err(err) => Err(AdapterStoreError::new(format!(
            "adapter root cannot be statted at {}: {err}",
            component.display()
        ))),
    }
}

#[cfg(target_os = "macos")]
fn allowed_platform_directory_symlink(path: &Path) -> Option<PathBuf> {
    let expected = if path == Path::new("/tmp") {
        Path::new("/private/tmp")
    } else if path == Path::new("/var") {
        Path::new("/private/var")
    } else {
        return None;
    };
    let resolved = fs::canonicalize(path).ok()?;
    if resolved == expected {
        Some(resolved)
    } else {
        None
    }
}

#[cfg(not(target_os = "macos"))]
fn allowed_platform_directory_symlink(_path: &Path) -> Option<PathBuf> {
    None
}

fn write_legacy_snapshot_inner(
    root: &Path,
    expected: Option<&StoreRevision>,
    payloads: LegacyPayloads,
    origin: StoreOrigin,
) -> Result<VerifiedAdapterStore> {
    let parsed = ParsedPayloads::parse(&payloads)?;
    let generation = active_generation_checkpoint(root, &parsed, None)?;
    parsed.validate_against_generation(&generation)?;

    let previous_store = current_verified_store(
        root,
        (!parsed.empty_lazy).then_some(parsed.index_path.as_str()),
    )?;
    verify_expected_revision(previous_store.as_ref(), expected)?;
    let previous_manifest = previous_store.as_ref().map(|store| &store.manifest);
    let previous_generation_count = previous_manifest.map_or(Ok(0), |manifest| {
        manifest_table_count(manifest, "generations")
    })?;
    let previous_audit_count = previous_manifest.map_or(Ok(0), |manifest| {
        manifest_table_count(manifest, "audit_log")
    })?;
    let previous_pending_count = previous_manifest.map_or(Ok(0), |manifest| {
        manifest_table_count(manifest, "pending_batches")
    })?;
    let previous_tombstone_count = previous_manifest.map_or(Ok(0), |manifest| {
        manifest_table_count(manifest, "tombstones")
    })?;
    let previous_gc_count =
        previous_manifest.map_or(Ok(0), |manifest| manifest_table_count(manifest, "gc_queue"))?;
    let generation_already_recorded = previous_manifest
        .and_then(|manifest| manifest.get("active_generation_id"))
        .and_then(Value::as_u64)
        == Some(generation.id);
    let generation_count = if parsed.empty_lazy || generation_already_recorded {
        previous_generation_count
    } else {
        previous_generation_count
            .checked_add(1)
            .ok_or_else(|| AdapterStoreError::new("generation table count overflow"))?
    };
    let audit_count = previous_audit_count
        .checked_add(1)
        .ok_or_else(|| AdapterStoreError::new("audit_log table count overflow"))?;
    let table_counts = json!({
        "meta": 1,
        "json_payloads": 4,
        "string_to_u64": parsed.string_to_u64.len(),
        "u64_to_string": parsed.string_to_u64.len(),
        "u64_to_slot": parsed.u64_to_slot.len(),
        "documents": parsed.documents.len(),
        "metadata": parsed.metadata.len(),
        "generations": generation_count,
        "pending_batches": previous_pending_count,
        "tombstones": previous_tombstone_count,
        "gc_queue": previous_gc_count,
        "audit_log": audit_count,
    });
    let manifest = adapter_store_manifest(
        &parsed,
        &generation,
        table_counts,
        previous_manifest,
        origin,
    )?;
    let manifest_json = canonical_json(&manifest)?;

    let final_path = root.join(ADAPTER_STORE_FILE);
    guard_against_storage_panics("writing the snapshot", || {
        let db = if previous_store.is_some() {
            Database::open(&final_path)?
        } else {
            Database::create(&final_path)?
        };
        write_snapshot_transaction(
            &db,
            &payloads,
            &parsed,
            &generation,
            manifest_json.as_str(),
            required_u64(&manifest, "commit_sequence")?,
        )
    })?;
    sync_file(&final_path)?;
    sync_directory(root)?;

    open_verified(root, None)
}

/// Opens the adapter store at `root` and fully verifies it.
///
/// Verification cross-checks, in order: the manifest shape (schema version,
/// UUIDv4 `store_uuid`, key/payload formats, `build_status`/`complete`
/// flags), the four legacy JSON payloads against the manifest, every
/// table's row count against the manifest's `table_counts`, the expanded
/// row tables against the payloads, the generation records (exactly one
/// `active` row matching the manifest — or none for an `empty_lazy` store),
/// and finally the on-disk active generation bundle: its `manifest.json` is
/// verified through the OrdVec manifest pipeline and its SHA-256 digest and
/// size must match the store manifest. Any mismatch fails the open.
///
/// `expected_adapter`, when provided, additionally requires the manifest's
/// `adapter_name` to match it.
///
/// Read-only: does not take the writer lock.
pub fn open_verified(
    root: impl AsRef<Path>,
    expected_adapter: Option<&str>,
) -> Result<VerifiedAdapterStore> {
    let root = root.as_ref();
    open_verified_at_store_file(&root.join(ADAPTER_STORE_FILE), root, expected_adapter, None)
}

fn open_verified_at_store_file(
    store_file: &Path,
    root: &Path,
    expected_adapter: Option<&str>,
    allowed_prepared_generation: Option<&str>,
) -> Result<VerifiedAdapterStore> {
    validate_adapter_store_file(store_file)?;
    guard_against_storage_panics("opening and verifying the store", || {
        let db = Database::open(store_file)?;
        let read_txn = db.begin_read()?;
        let meta = read_txn.open_table(META)?;
        let manifest_json = meta
            .get("manifest")?
            .ok_or_else(|| AdapterStoreError::new("adapter store manifest missing"))?
            .value()
            .to_string();
        let manifest = parse_json_object(&manifest_json, "adapter store manifest")?;
        verify_manifest_shape(&manifest, expected_adapter)?;

        let payloads = {
            let json_payloads = read_txn.open_table(JSON_PAYLOADS)?;
            LegacyPayloads {
                adapter_json: required_payload(&json_payloads, "adapter")?,
                id_map_json: required_payload(&json_payloads, "id_map")?,
                documents_json: required_payload(&json_payloads, "documents")?,
                metadata_json: required_payload(&json_payloads, "metadata")?,
            }
        };
        let parsed = ParsedPayloads::parse(&payloads)?;

        verify_manifest_payloads(&manifest, &parsed)?;
        verify_table_counts(&read_txn, &manifest)?;
        let generation = active_generation_checkpoint(root, &parsed, allowed_prepared_generation)?;
        verify_tables_match_payloads(&read_txn, &parsed)?;
        verify_generation_records(&read_txn, &generation)?;
        parsed.validate_against_generation(&generation)?;
        verify_manifest_generation(&manifest, &generation)?;

        Ok(VerifiedAdapterStore { manifest, payloads })
    })
}

fn current_verified_store(
    root: &Path,
    allowed_prepared_generation: Option<&str>,
) -> Result<Option<VerifiedAdapterStore>> {
    let store_file = root.join(ADAPTER_STORE_FILE);
    match fs::symlink_metadata(&store_file) {
        Ok(_) => open_verified_at_store_file(
            &root.join(ADAPTER_STORE_FILE),
            root,
            None,
            allowed_prepared_generation,
        )
        .map(Some),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(err) => Err(AdapterStoreError::new(format!(
            "{ADAPTER_STORE_FILE} cannot be statted at {}: {err}",
            store_file.display()
        ))),
    }
}

fn verify_expected_revision(
    current: Option<&VerifiedAdapterStore>,
    expected: Option<&StoreRevision>,
) -> Result<()> {
    match (current, expected) {
        (None, None) => Ok(()),
        (None, Some(_)) => Err(AdapterStoreError::new(
            "stale adapter snapshot: expected an existing adapter store",
        )),
        (Some(_), None) => Err(AdapterStoreError::new(
            "expected adapter store revision is required for overwriting an existing store",
        )),
        (Some(store), Some(expected)) => {
            let actual = StoreRevision::from_manifest(&store.manifest)?;
            if &actual == expected {
                Ok(())
            } else {
                Err(AdapterStoreError::new(format!(
                    "stale adapter snapshot: expected revision {expected:?}, found {actual:?}"
                )))
            }
        }
    }
}

fn record_generation_gc_inner(
    root: &Path,
    expected: Option<&StoreRevision>,
    updates: &[GenerationGcUpdate],
) -> Result<VerifiedAdapterStore> {
    if updates.is_empty() {
        return current_verified_store(root, None)?
            .ok_or_else(|| AdapterStoreError::new("adapter store is required for generation GC"));
    }
    for update in updates {
        validate_gc_update(update)?;
    }
    let current = current_verified_store(root, None)?
        .ok_or_else(|| AdapterStoreError::new("adapter store is required for generation GC"))?;
    verify_expected_revision(Some(&current), expected)?;
    let mut manifest = current.manifest.clone();
    let previous_commit_sequence = required_u64(&manifest, "commit_sequence")?;
    let commit_sequence = previous_commit_sequence
        .checked_add(1)
        .ok_or_else(|| AdapterStoreError::new("commit_sequence overflow"))?;
    let previous_gc_count = manifest_table_count(&manifest, "gc_queue")?;
    let previous_audit_count = manifest_table_count(&manifest, "audit_log")?;
    let update_count =
        u64::try_from(updates.len()).map_err(|_| AdapterStoreError::new("too many GC updates"))?;
    let gc_count = previous_gc_count
        .checked_add(update_count)
        .ok_or_else(|| AdapterStoreError::new("gc_queue table count overflow"))?;
    let audit_count = previous_audit_count
        .checked_add(1)
        .ok_or_else(|| AdapterStoreError::new("audit_log table count overflow"))?;
    update_gc_manifest_counts(&mut manifest, commit_sequence, gc_count, audit_count)?;
    let manifest_json = canonical_json(&manifest)?;

    guard_against_storage_panics("recording generation GC updates", || {
        let db = Database::open(root.join(ADAPTER_STORE_FILE))?;
        let write_txn = db.begin_write()?;
        {
            let mut meta = write_txn.open_table(META)?;
            meta.insert("manifest", manifest_json.as_str())?;
        }
        {
            let mut gc_queue = write_txn.open_table(GC_QUEUE)?;
            for (idx, update) in updates.iter().enumerate() {
                let offset = u64::try_from(idx)
                    .map_err(|_| AdapterStoreError::new("too many GC updates"))?;
                let row_id = previous_gc_count
                    .checked_add(offset)
                    .and_then(|value| value.checked_add(1))
                    .ok_or_else(|| AdapterStoreError::new("gc_queue row id overflow"))?;
                gc_queue.insert(row_id, canonical_json(&gc_update_json(update))?.as_str())?;
            }
        }
        {
            let mut audit_log = write_txn.open_table(AUDIT_LOG)?;
            audit_log.insert(
                commit_sequence,
                canonical_json(&json!({
                    "event": "generation_gc_recorded",
                    "commit_sequence": commit_sequence,
                    "update_count": updates.len(),
                    "created_at_unix_ms": now_ms(),
                }))?
                .as_str(),
            )?;
        }
        write_txn.commit()?;
        Ok(())
    })?;
    sync_file(&root.join(ADAPTER_STORE_FILE))?;
    sync_directory(root)?;

    open_verified(root, None)
}

fn validate_gc_update(update: &GenerationGcUpdate) -> Result<()> {
    validate_relative_path(&update.path)?;
    if update.path == INDEX_DIR {
        return Err(AdapterStoreError::new(
            "generation GC must not target legacy index.odb",
        ));
    }
    if !is_valid_generation_lifecycle_state(&update.state) {
        return Err(AdapterStoreError::new(
            "generation GC state must be a known generation lifecycle state",
        ));
    }
    if update.reason.trim().is_empty() {
        return Err(AdapterStoreError::new(
            "generation GC reason must be non-empty",
        ));
    }
    Ok(())
}

fn is_valid_generation_record_state(state: &str) -> bool {
    matches!(state, GENERATION_STATE_ACTIVE | GENERATION_STATE_RETIRED)
}

fn is_valid_generation_lifecycle_state(state: &str) -> bool {
    matches!(
        state,
        GENERATION_STATE_ACTIVE
            | GENERATION_STATE_RETIRED
            | GENERATION_STATE_RECLAIMABLE
            | GENERATION_STATE_DELETING
            | GENERATION_STATE_DELETED
    )
}

fn retire_generation_record(raw: &str, commit_sequence: u64) -> Result<String> {
    let mut record = parse_json_object(raw, "generation table row")?;
    let state = record
        .get("state")
        .and_then(Value::as_str)
        .unwrap_or(GENERATION_STATE_ACTIVE);
    if !is_valid_generation_record_state(state) {
        return Err(AdapterStoreError::new(format!(
            "invalid generation lifecycle state: {state}"
        )));
    }
    if state == GENERATION_STATE_RETIRED {
        return canonical_json(&record);
    }

    let object = record
        .as_object_mut()
        .ok_or_else(|| AdapterStoreError::new("generation table row must be an object"))?;
    object.insert("state".to_string(), json!(GENERATION_STATE_RETIRED));
    object.insert(
        "retired_by_commit_sequence".to_string(),
        json!(commit_sequence),
    );
    canonical_json(&record)
}

fn verify_generation_records(
    read_txn: &redb::ReadTransaction,
    active_generation: &GenerationCheckpoint,
) -> Result<()> {
    let generations = read_txn.open_table(GENERATIONS)?;
    let mut active_rows = 0usize;
    for entry in generations.iter()? {
        let (key, value) = entry?;
        let generation_id = key.value();
        let record = parse_json_object(value.value(), "generation table row")?;
        if required_u64(&record, "generation_id")? != generation_id {
            return Err(AdapterStoreError::new(
                "generation table key does not match generation_id",
            ));
        }
        let state = required_string(&record, "state")?;
        if !is_valid_generation_record_state(state) {
            return Err(AdapterStoreError::new(format!(
                "invalid generation lifecycle state: {state}"
            )));
        }
        if state == GENERATION_STATE_ACTIVE {
            active_rows += 1;
            if generation_id != active_generation.id {
                return Err(AdapterStoreError::new(
                    "non-current generation is marked active",
                ));
            }
            if required_string(&record, "path")? != active_generation.path {
                return Err(AdapterStoreError::new(
                    "active generation record path mismatch",
                ));
            }
        } else if generation_id == active_generation.id {
            return Err(AdapterStoreError::new(
                "current generation is not marked active",
            ));
        }
    }

    if active_generation.empty_lazy {
        if active_rows != 0 {
            return Err(AdapterStoreError::new(
                "empty lazy store must not have active generation rows",
            ));
        }
    } else if active_rows != 1 {
        return Err(AdapterStoreError::new(
            "generation table must contain exactly one active generation",
        ));
    }
    Ok(())
}

fn update_gc_manifest_counts(
    manifest: &mut Value,
    commit_sequence: u64,
    gc_count: u64,
    audit_count: u64,
) -> Result<()> {
    let now = u64::try_from(now_ms()).unwrap_or(u64::MAX);
    let manifest_object = manifest
        .as_object_mut()
        .ok_or_else(|| AdapterStoreError::new("adapter store manifest must be an object"))?;
    manifest_object.insert("commit_sequence".to_string(), Value::from(commit_sequence));
    manifest_object.insert("updated_at_unix_ms".to_string(), Value::from(now));
    let table_counts = manifest_object
        .get_mut("table_counts")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| AdapterStoreError::new("adapter store table_counts must be an object"))?;
    table_counts.insert("gc_queue".to_string(), Value::from(gc_count));
    table_counts.insert("audit_log".to_string(), Value::from(audit_count));
    Ok(())
}

fn gc_update_json(update: &GenerationGcUpdate) -> Value {
    json!({
        "generation_id": update.generation_id,
        "path": update.path,
        "state": update.state,
        "reason": update.reason,
        "recorded_at_unix_ms": now_ms(),
    })
}

fn manifest_table_count(manifest: &Value, name: &str) -> Result<u64> {
    manifest
        .get("table_counts")
        .and_then(Value::as_object)
        .and_then(|counts| counts.get(name))
        .and_then(Value::as_u64)
        .ok_or_else(|| AdapterStoreError::new(format!("table_counts.{name} missing")))
}

fn write_snapshot_transaction(
    db: &Database,
    payloads: &LegacyPayloads,
    parsed: &ParsedPayloads,
    generation: &GenerationCheckpoint,
    manifest_json: &str,
    commit_sequence: u64,
) -> Result<()> {
    let write_txn = db.begin_write()?;
    {
        let mut meta = write_txn.open_table(META)?;
        meta.retain(|_, _| false)?;
        meta.insert("manifest", manifest_json)?;
    }
    {
        let mut json_payloads = write_txn.open_table(JSON_PAYLOADS)?;
        json_payloads.retain(|_, _| false)?;
        json_payloads.insert("adapter", payloads.adapter_json.as_str())?;
        json_payloads.insert("id_map", payloads.id_map_json.as_str())?;
        json_payloads.insert("documents", payloads.documents_json.as_str())?;
        json_payloads.insert("metadata", payloads.metadata_json.as_str())?;
    }
    {
        let mut string_to_u64 = write_txn.open_table(STRING_TO_U64)?;
        string_to_u64.retain(|string_id, _| parsed.string_to_u64.contains_key(string_id))?;
        for (string_id, u64_id) in &parsed.string_to_u64 {
            string_to_u64.insert(string_id.as_str(), *u64_id)?;
        }
    }
    {
        let mut u64_to_string = write_txn.open_table(U64_TO_STRING)?;
        let active_u64_ids = parsed
            .string_to_u64
            .values()
            .copied()
            .collect::<HashSet<_>>();
        u64_to_string.retain(|u64_id, _| active_u64_ids.contains(&u64_id))?;
        for (string_id, u64_id) in &parsed.string_to_u64 {
            u64_to_string.insert(*u64_id, string_id.as_str())?;
        }
    }
    {
        let mut u64_to_slot = write_txn.open_table(U64_TO_SLOT)?;
        u64_to_slot.retain(|u64_id, _| parsed.u64_to_slot.contains_key(&u64_id))?;
        for (u64_id, slot) in &parsed.u64_to_slot {
            u64_to_slot.insert(*u64_id, *slot as u64)?;
        }
    }
    {
        let mut documents = write_txn.open_table(DOCUMENTS)?;
        let document_ids = parsed
            .documents
            .keys()
            .map(|string_id| parsed.u64_for_string(string_id))
            .collect::<Result<HashSet<_>>>()?;
        documents.retain(|u64_id, _| document_ids.contains(&u64_id))?;
        for (string_id, document) in &parsed.documents {
            let u64_id = parsed.u64_for_string(string_id)?;
            documents.insert(u64_id, document.as_str())?;
        }
    }
    {
        let mut metadata = write_txn.open_table(METADATA)?;
        let metadata_ids = parsed
            .metadata
            .keys()
            .map(|string_id| parsed.u64_for_string(string_id))
            .collect::<Result<HashSet<_>>>()?;
        metadata.retain(|u64_id, _| metadata_ids.contains(&u64_id))?;
        for (string_id, metadata_value) in &parsed.metadata {
            let u64_id = parsed.u64_for_string(string_id)?;
            let metadata_json = canonical_json(metadata_value)?;
            metadata.insert(u64_id, metadata_json.as_str())?;
        }
    }
    {
        let mut generations = write_txn.open_table(GENERATIONS)?;
        if !parsed.empty_lazy {
            let mut retired_records = Vec::new();
            for entry in generations.iter()? {
                let (key, value) = entry?;
                let generation_id = key.value();
                if generation_id != generation.id {
                    retired_records.push((
                        generation_id,
                        retire_generation_record(value.value(), commit_sequence)?,
                    ));
                }
            }
            for (generation_id, record) in retired_records {
                generations.insert(generation_id, record.as_str())?;
            }
            generations.insert(generation.id, canonical_json(&generation.json)?.as_str())?;
        }
    }
    {
        let _ = write_txn.open_table(PENDING_BATCHES)?;
    }
    {
        let _ = write_txn.open_table(TOMBSTONES)?;
    }
    {
        let _ = write_txn.open_table(GC_QUEUE)?;
    }
    {
        let mut audit_log = write_txn.open_table(AUDIT_LOG)?;
        audit_log.insert(
            commit_sequence,
            canonical_json(&json!({
                "event": "legacy_snapshot_written",
                "commit_sequence": commit_sequence,
                "generation_id": generation.id,
                "active_generation_path": generation.path,
                "created_at_unix_ms": now_ms(),
            }))?
            .as_str(),
        )?;
    }
    write_txn.commit()?;
    Ok(())
}

fn validate_adapter_store_file(store_file: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(store_file).map_err(|err| {
        AdapterStoreError::new(format!(
            "{ADAPTER_STORE_FILE} cannot be statted at {}: {err}",
            store_file.display()
        ))
    })?;
    if metadata.file_type().is_symlink() {
        return Err(AdapterStoreError::new(format!(
            "{ADAPTER_STORE_FILE} must not be a symlink"
        )));
    }
    if !metadata.file_type().is_file() {
        return Err(AdapterStoreError::new(format!(
            "{ADAPTER_STORE_FILE} must be a file"
        )));
    }
    Ok(())
}

fn required_payload(table: &redb::ReadOnlyTable<&str, &str>, key: &str) -> Result<String> {
    Ok(table
        .get(key)?
        .ok_or_else(|| AdapterStoreError::new(format!("json payload {key:?} missing")))?
        .value()
        .to_string())
}

fn verify_manifest_shape(manifest: &Value, expected_adapter: Option<&str>) -> Result<()> {
    require_str(manifest, "schema_version", ADAPTER_STORE_SCHEMA_VERSION)?;
    validate_store_uuid(required_string(manifest, "store_uuid")?)?;
    let _ = required_u64(manifest, "commit_sequence")?;
    validate_store_origin(required_string(manifest, "origin")?)?;
    let _ = required_bool(manifest, "migrated_from_json_sidecars")?;
    require_str(manifest, "key_format", KEY_FORMAT)?;
    require_str(manifest, "payload_format", PAYLOAD_FORMAT)?;
    require_str(manifest, "build_status", "complete")?;
    if !required_bool(manifest, "complete")? {
        return Err(AdapterStoreError::new(
            "adapter store manifest is incomplete",
        ));
    }
    if let Some(expected) = expected_adapter {
        let actual = required_string(manifest, "adapter_name")?;
        if actual != expected {
            return Err(AdapterStoreError::new(format!(
                "adapter store was written by {actual:?}, not {expected:?}"
            )));
        }
    }
    Ok(())
}

fn validate_store_uuid(value: &str) -> Result<()> {
    let bytes = value.as_bytes();
    let valid = bytes.len() == 36
        && bytes[8] == b'-'
        && bytes[13] == b'-'
        && bytes[18] == b'-'
        && bytes[23] == b'-'
        && bytes[14] == b'4'
        && matches!(bytes[19], b'8' | b'9' | b'a' | b'b')
        && bytes
            .iter()
            .enumerate()
            .all(|(idx, byte)| matches!(idx, 8 | 13 | 18 | 23) || byte.is_ascii_hexdigit());
    if !valid {
        return Err(AdapterStoreError::new(
            "store_uuid must be a random UUIDv4 string",
        ));
    }
    Ok(())
}

fn validate_store_origin(value: &str) -> Result<()> {
    if matches!(
        value,
        "created" | "imported_legacy_json" | "upgraded_schema"
    ) {
        Ok(())
    } else {
        Err(AdapterStoreError::new(
            "origin must be one of created, imported_legacy_json, upgraded_schema",
        ))
    }
}

fn verify_manifest_payloads(manifest: &Value, parsed: &ParsedPayloads) -> Result<()> {
    if required_string(manifest, "adapter_name")? != parsed.adapter_name.as_str() {
        return Err(AdapterStoreError::new("adapter_name manifest mismatch"));
    }
    if required_u64(manifest, "bits")? != u64::from(parsed.bits) {
        return Err(AdapterStoreError::new("bits manifest mismatch"));
    }
    match (manifest.get("dim"), parsed.dim) {
        (Some(Value::Null), None) => {}
        (Some(value), Some(dim)) if value.as_u64() == Some(usize_to_u64(dim, "dim")?) => {}
        (Some(Value::Null), Some(_)) | (Some(_), None) => {
            return Err(AdapterStoreError::new("dim manifest mismatch"));
        }
        _ => return Err(AdapterStoreError::new("dim must be a u64 or null")),
    }
    if required_bool(manifest, "empty_lazy")? != parsed.empty_lazy {
        return Err(AdapterStoreError::new("empty_lazy manifest mismatch"));
    }
    if required_u64(manifest, "active_id_count")?
        != usize_to_u64(parsed.string_to_u64.len(), "active_id_count")?
    {
        return Err(AdapterStoreError::new("active_id_count manifest mismatch"));
    }
    if required_u64(manifest, "next_u64_id")? != parsed.next_u64_id {
        return Err(AdapterStoreError::new("next_u64_id manifest mismatch"));
    }
    if required_u64(manifest, "tombstone_count")? != 0 {
        return Err(AdapterStoreError::new("tombstone_count manifest mismatch"));
    }
    Ok(())
}

fn verify_manifest_generation(manifest: &Value, generation: &GenerationCheckpoint) -> Result<()> {
    if required_u64(manifest, "active_generation_id")? != generation.id {
        return Err(AdapterStoreError::new("active generation id mismatch"));
    }
    if required_string(manifest, "active_generation_path")? != generation.path {
        return Err(AdapterStoreError::new("active generation path mismatch"));
    }
    if required_u64(manifest, "vector_count")? != generation.vector_count as u64 {
        return Err(AdapterStoreError::new("vector count mismatch"));
    }
    if generation.empty_lazy {
        if !required_bool(manifest, "empty_lazy")? {
            return Err(AdapterStoreError::new("empty_lazy manifest mismatch"));
        }
        return Ok(());
    }
    let expected_digest = generation
        .manifest_sha256
        .as_ref()
        .ok_or_else(|| AdapterStoreError::new("generation digest missing"))?;
    if required_string(manifest, "active_generation_manifest_sha256")? != expected_digest {
        return Err(AdapterStoreError::new(
            "active generation manifest digest mismatch",
        ));
    }
    if required_u64(manifest, "active_generation_manifest_size_bytes")?
        != generation.manifest_size_bytes.unwrap_or(0)
    {
        return Err(AdapterStoreError::new(
            "active generation manifest size mismatch",
        ));
    }
    Ok(())
}

fn verify_table_counts(read_txn: &redb::ReadTransaction, manifest: &Value) -> Result<()> {
    let expected = manifest
        .get("table_counts")
        .and_then(Value::as_object)
        .ok_or_else(|| AdapterStoreError::new("adapter store table_counts must be an object"))?;
    let actual = [
        ("meta", read_txn.open_table(META)?.len()?),
        ("json_payloads", read_txn.open_table(JSON_PAYLOADS)?.len()?),
        ("string_to_u64", read_txn.open_table(STRING_TO_U64)?.len()?),
        ("u64_to_string", read_txn.open_table(U64_TO_STRING)?.len()?),
        ("u64_to_slot", read_txn.open_table(U64_TO_SLOT)?.len()?),
        ("documents", read_txn.open_table(DOCUMENTS)?.len()?),
        ("metadata", read_txn.open_table(METADATA)?.len()?),
        ("generations", read_txn.open_table(GENERATIONS)?.len()?),
        (
            "pending_batches",
            read_txn.open_table(PENDING_BATCHES)?.len()?,
        ),
        ("tombstones", read_txn.open_table(TOMBSTONES)?.len()?),
        ("gc_queue", read_txn.open_table(GC_QUEUE)?.len()?),
        ("audit_log", read_txn.open_table(AUDIT_LOG)?.len()?),
    ];
    for (name, actual_count) in actual {
        let expected_count = expected
            .get(name)
            .and_then(Value::as_u64)
            .ok_or_else(|| AdapterStoreError::new(format!("table_counts.{name} missing")))?;
        if expected_count != actual_count {
            return Err(AdapterStoreError::new(format!(
                "table count mismatch for {name}: expected={expected_count} actual={actual_count}"
            )));
        }
    }
    Ok(())
}

fn verify_tables_match_payloads(
    read_txn: &redb::ReadTransaction,
    parsed: &ParsedPayloads,
) -> Result<()> {
    let string_to_u64 = read_txn.open_table(STRING_TO_U64)?;
    let mut seen_strings = HashMap::new();
    for entry in string_to_u64.iter()? {
        let (key, value) = entry?;
        seen_strings.insert(key.value().to_string(), value.value());
    }
    if seen_strings != parsed.string_to_u64 {
        return Err(AdapterStoreError::new(
            "string_to_u64 table does not match payload",
        ));
    }

    let u64_to_string = read_txn.open_table(U64_TO_STRING)?;
    let mut seen_reverse = HashMap::new();
    for entry in u64_to_string.iter()? {
        let (key, value) = entry?;
        seen_reverse.insert(key.value(), value.value().to_string());
    }
    let expected_reverse = parsed
        .string_to_u64
        .iter()
        .map(|(string_id, u64_id)| (*u64_id, string_id.clone()))
        .collect::<HashMap<_, _>>();
    if seen_reverse != expected_reverse {
        return Err(AdapterStoreError::new(
            "u64_to_string table does not match payload",
        ));
    }

    let u64_to_slot = read_txn.open_table(U64_TO_SLOT)?;
    let mut seen_slots = HashMap::new();
    for entry in u64_to_slot.iter()? {
        let (key, value) = entry?;
        let slot = usize::try_from(value.value())
            .map_err(|_| AdapterStoreError::new("u64_to_slot value exceeds usize"))?;
        seen_slots.insert(key.value(), slot);
    }
    if seen_slots != parsed.u64_to_slot {
        return Err(AdapterStoreError::new(
            "u64_to_slot table does not match payload",
        ));
    }

    let documents = read_txn.open_table(DOCUMENTS)?;
    for (string_id, expected_doc) in &parsed.documents {
        let u64_id = parsed.u64_for_string(string_id)?;
        let actual = documents
            .get(u64_id)?
            .ok_or_else(|| AdapterStoreError::new("document table is missing active ID"))?
            .value()
            .to_string();
        if actual != *expected_doc {
            return Err(AdapterStoreError::new(
                "document table does not match payload",
            ));
        }
    }

    let metadata = read_txn.open_table(METADATA)?;
    for (string_id, expected_metadata) in &parsed.metadata {
        let u64_id = parsed.u64_for_string(string_id)?;
        let actual = metadata
            .get(u64_id)?
            .ok_or_else(|| AdapterStoreError::new("metadata table is missing active ID"))?
            .value()
            .to_string();
        if parse_json_object(&actual, "metadata table row")? != *expected_metadata {
            return Err(AdapterStoreError::new(
                "metadata table does not match payload",
            ));
        }
    }
    Ok(())
}

struct ParsedPayloads {
    adapter_name: String,
    bits: u8,
    dim: Option<usize>,
    empty_lazy: bool,
    index_path: String,
    string_to_u64: HashMap<String, u64>,
    u64_to_slot: HashMap<u64, usize>,
    documents: HashMap<String, String>,
    metadata: HashMap<String, Value>,
    next_u64_id: u64,
}

impl ParsedPayloads {
    fn parse(payloads: &LegacyPayloads) -> Result<Self> {
        let adapter = parse_json_object(&payloads.adapter_json, ADAPTER_FILE)?;
        let id_map = parse_json_object(&payloads.id_map_json, ID_MAP_FILE)?;
        let documents = parse_json_object(&payloads.documents_json, DOCUMENTS_FILE)?;
        let metadata = parse_json_object(&payloads.metadata_json, METADATA_FILE)?;

        require_exact_keys(
            &adapter,
            ADAPTER_FILE,
            &[
                "schema_version",
                "adapter",
                "bits",
                "dim",
                "empty_lazy",
                "index_path",
                "sidecars",
            ],
        )?;
        require_exact_keys(
            &id_map,
            ID_MAP_FILE,
            &[
                "schema_version",
                "next_u64_id",
                "string_to_u64",
                "u64_to_slot",
            ],
        )?;
        require_exact_keys(&documents, DOCUMENTS_FILE, &["schema_version", "documents"])?;
        require_exact_keys(&metadata, METADATA_FILE, &["schema_version", "metadata"])?;
        require_schema(&adapter, ADAPTER_SCHEMA_VERSION, ADAPTER_FILE)?;
        require_schema(&id_map, ID_MAP_SCHEMA_VERSION, ID_MAP_FILE)?;
        require_schema(&documents, DOCUMENTS_SCHEMA_VERSION, DOCUMENTS_FILE)?;
        require_schema(&metadata, METADATA_SCHEMA_VERSION, METADATA_FILE)?;

        let adapter_name = required_string(&adapter, "adapter")?.to_string();
        let bits = required_bits(&adapter)?;
        let dim = optional_dim(&adapter)?;
        let empty_lazy = required_bool(&adapter, "empty_lazy")?;
        let index_path = required_string(&adapter, "index_path")?.to_string();
        validate_index_path(&index_path)?;

        let string_to_u64 = parse_string_to_u64(required_object(&id_map, "string_to_u64")?)?;
        let u64_to_slot = parse_u64_to_slot(required_object(&id_map, "u64_to_slot")?)?;
        let next_u64_id = required_u64(&id_map, "next_u64_id")?;
        let documents = parse_documents(required_object(&documents, "documents")?)?;
        let metadata = parse_metadata(required_object(&metadata, "metadata")?)?;

        let parsed = Self {
            adapter_name,
            bits,
            dim,
            empty_lazy,
            index_path,
            string_to_u64,
            u64_to_slot,
            documents,
            metadata,
            next_u64_id,
        };
        parsed.validate_maps()?;
        Ok(parsed)
    }

    fn u64_for_string(&self, string_id: &str) -> Result<u64> {
        self.string_to_u64
            .get(string_id)
            .copied()
            .ok_or_else(|| AdapterStoreError::new("string ID missing from string_to_u64"))
    }

    fn validate_maps(&self) -> Result<()> {
        let string_keys: HashSet<_> = self.string_to_u64.keys().collect();
        if self.documents.keys().collect::<HashSet<_>>() != string_keys {
            return Err(AdapterStoreError::new(
                "documents keys do not match string_to_u64 keys",
            ));
        }
        if self.metadata.keys().collect::<HashSet<_>>() != string_keys {
            return Err(AdapterStoreError::new(
                "metadata keys do not match string_to_u64 keys",
            ));
        }
        let mut seen_u64 = HashSet::with_capacity(self.string_to_u64.len());
        for (string_id, u64_id) in &self.string_to_u64 {
            require_non_empty_string_id(string_id)?;
            if !seen_u64.insert(*u64_id) {
                return Err(AdapterStoreError::new(format!("duplicate u64 id {u64_id}")));
            }
            if *u64_id >= self.next_u64_id {
                return Err(AdapterStoreError::new(
                    "next_u64_id must be greater than all allocated IDs",
                ));
            }
        }
        if self.u64_to_slot.keys().copied().collect::<HashSet<_>>() != seen_u64 {
            return Err(AdapterStoreError::new(
                "u64_to_slot keys do not match string_to_u64 values",
            ));
        }
        let mut seen_slots = HashSet::with_capacity(self.u64_to_slot.len());
        for slot in self.u64_to_slot.values() {
            if !seen_slots.insert(*slot) {
                return Err(AdapterStoreError::new(format!(
                    "duplicate vector slot {slot}"
                )));
            }
        }
        Ok(())
    }

    fn validate_against_generation(&self, generation: &GenerationCheckpoint) -> Result<()> {
        if self.empty_lazy != generation.empty_lazy {
            return Err(AdapterStoreError::new("empty_lazy generation mismatch"));
        }
        if self.empty_lazy {
            if self.dim.is_some() {
                return Err(AdapterStoreError::new(
                    "empty_lazy adapter store must have dim=null",
                ));
            }
            if !self.string_to_u64.is_empty()
                || !self.u64_to_slot.is_empty()
                || !self.documents.is_empty()
                || !self.metadata.is_empty()
            {
                return Err(AdapterStoreError::new(
                    "empty_lazy adapter store must not contain records",
                ));
            }
            return Ok(());
        }
        if self.index_path != generation.path {
            return Err(AdapterStoreError::new("adapter index path mismatch"));
        }
        if self.bits != generation.bits {
            return Err(AdapterStoreError::new(
                "adapter bits do not match generation bits",
            ));
        }
        if self.dim != Some(generation.dim) {
            return Err(AdapterStoreError::new(
                "adapter dim does not match generation dim",
            ));
        }
        if self.string_to_u64.len() != generation.vector_count
            || self.u64_to_slot.len() != generation.vector_count
        {
            return Err(AdapterStoreError::new(format!(
                "adapter id count does not match generation vector count {}",
                generation.vector_count
            )));
        }
        for (u64_id, slot) in &self.u64_to_slot {
            if *slot >= generation.vector_count {
                return Err(AdapterStoreError::new(format!(
                    "u64 id {u64_id} points at stale slot {slot}"
                )));
            }
        }
        Ok(())
    }
}

struct GenerationCheckpoint {
    id: u64,
    path: String,
    empty_lazy: bool,
    bits: u8,
    dim: usize,
    vector_count: usize,
    manifest_sha256: Option<String>,
    manifest_size_bytes: Option<u64>,
    json: Value,
}

fn active_generation_checkpoint(
    root: &Path,
    parsed: &ParsedPayloads,
    allowed_prepared_generation: Option<&str>,
) -> Result<GenerationCheckpoint> {
    if parsed.empty_lazy {
        reject_empty_lazy_vector_artifacts(root, &parsed.index_path, allowed_prepared_generation)?;
        return Ok(GenerationCheckpoint {
            id: 0,
            path: String::new(),
            empty_lazy: true,
            bits: parsed.bits,
            dim: 0,
            vector_count: 0,
            manifest_sha256: None,
            manifest_size_bytes: None,
            json: json!({
                "generation_id": 0,
                "empty_lazy": true,
                "path": null,
                "vector_count": 0,
                "state": GENERATION_STATE_ACTIVE,
            }),
        });
    }

    let manifest_path = active_generation_manifest_path(root, &parsed.index_path)?;
    validate_generation_manifest_file(&manifest_path)?;
    let plan = verify_for_load(&manifest_path, VerifyOptions::default())
        .map_err(|err| AdapterStoreError::new(err.to_string()))?;
    let metadata = plan.metadata();
    let ManifestIndexParams::RankQuant { bits } = metadata.params else {
        return Err(AdapterStoreError::new(
            "active generation must describe a RankQuant index",
        ));
    };
    if plan.row_identity().kind() != ROW_IDENTITY_KIND {
        return Err(AdapterStoreError::new(format!(
            "active generation must use {ROW_IDENTITY_KIND} row identity"
        )));
    }
    let generation_id = generation_id_from_index_path(&parsed.index_path)?;
    let digest = sha256_file(&manifest_path)?;
    let json = json!({
        "generation_id": generation_id,
        "path": parsed.index_path,
        "state": GENERATION_STATE_ACTIVE,
        "manifest_sha256": digest.sha256,
        "manifest_size_bytes": digest.size_bytes,
        "bits": bits,
        "dim": metadata.dim,
        "vector_count": metadata.vector_count,
        "row_identity": ROW_IDENTITY_KIND,
    });
    Ok(GenerationCheckpoint {
        id: generation_id,
        path: parsed.index_path.clone(),
        empty_lazy: false,
        bits,
        dim: metadata.dim,
        vector_count: metadata.vector_count,
        manifest_sha256: Some(digest.sha256),
        manifest_size_bytes: Some(digest.size_bytes),
        json,
    })
}

fn adapter_store_manifest(
    parsed: &ParsedPayloads,
    generation: &GenerationCheckpoint,
    table_counts: Value,
    previous_manifest: Option<&Value>,
    origin: StoreOrigin,
) -> Result<Value> {
    let now = u64::try_from(now_ms()).unwrap_or(u64::MAX);
    let store_uuid = previous_manifest
        .and_then(|manifest| manifest.get("store_uuid"))
        .and_then(Value::as_str)
        .map(str::to_owned)
        .map(Ok)
        .unwrap_or_else(random_store_uuid)?;
    let created_at_unix_ms = previous_manifest
        .and_then(|manifest| manifest.get("created_at_unix_ms"))
        .and_then(Value::as_u64)
        .unwrap_or(now);
    let commit_sequence = match previous_manifest
        .and_then(|manifest| manifest.get("commit_sequence"))
        .and_then(Value::as_u64)
    {
        Some(sequence) => sequence
            .checked_add(1)
            .ok_or_else(|| AdapterStoreError::new("commit_sequence overflow"))?,
        None => 1,
    };
    let previous_generation_id = previous_manifest
        .and_then(|manifest| manifest.get("active_generation_id"))
        .and_then(Value::as_u64)
        .map_or(Value::Null, Value::from);
    let origin = previous_manifest
        .and_then(|manifest| manifest.get("origin"))
        .cloned()
        .unwrap_or_else(|| Value::String(origin.as_str().to_string()));
    let migrated_from_json_sidecars = previous_manifest
        .and_then(|manifest| manifest.get("migrated_from_json_sidecars"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    Ok(json!({
        "schema_version": ADAPTER_STORE_SCHEMA_VERSION,
        "store_uuid": store_uuid,
        "adapter_name": parsed.adapter_name,
        "created_at_unix_ms": created_at_unix_ms,
        "updated_at_unix_ms": now,
        "commit_sequence": commit_sequence,
        "bits": parsed.bits,
        "dim": parsed.dim,
        "empty_lazy": parsed.empty_lazy,
        "active_generation_id": generation.id,
        "active_generation_path": generation.path,
        "active_generation_manifest_sha256": generation.manifest_sha256,
        "active_generation_manifest_size_bytes": generation.manifest_size_bytes,
        "vector_count": generation.vector_count,
        "active_id_count": parsed.string_to_u64.len(),
        "tombstone_count": 0,
        "next_u64_id": parsed.next_u64_id,
        "table_counts": table_counts,
        "key_format": KEY_FORMAT,
        "payload_format": PAYLOAD_FORMAT,
        "build_status": "complete",
        "complete": true,
        "previous_generation_id": previous_generation_id,
        "writer_version": env!("CARGO_PKG_VERSION"),
        "origin": origin,
        "migrated_from_json_sidecars": migrated_from_json_sidecars,
    }))
}

fn random_store_uuid() -> Result<String> {
    let mut bytes = [0_u8; 16];
    fill_random(&mut bytes)?;
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    Ok(format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0],
        bytes[1],
        bytes[2],
        bytes[3],
        bytes[4],
        bytes[5],
        bytes[6],
        bytes[7],
        bytes[8],
        bytes[9],
        bytes[10],
        bytes[11],
        bytes[12],
        bytes[13],
        bytes[14],
        bytes[15],
    ))
}

fn parse_json_object(input: &str, name: &str) -> Result<Value> {
    let mut deserializer = serde_json::Deserializer::from_str(input);
    let value = NoDuplicateValue::deserialize(&mut deserializer)
        .map_err(|err| AdapterStoreError::new(format!("failed to parse JSON in {name}: {err}")))?
        .0;
    deserializer
        .end()
        .map_err(|err| AdapterStoreError::new(format!("trailing data in {name}: {err}")))?;
    if !value.is_object() {
        return Err(AdapterStoreError::new(format!(
            "{name} must be a JSON object"
        )));
    }
    Ok(value)
}

struct NoDuplicateValue(Value);

impl<'de> Deserialize<'de> for NoDuplicateValue {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer
            .deserialize_any(NoDuplicateValueVisitor)
            .map(NoDuplicateValue)
    }
}

struct NoDuplicateValueVisitor;

impl<'de> Visitor<'de> for NoDuplicateValueVisitor {
    type Value = Value;

    fn expecting(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("JSON without duplicate object keys")
    }

    fn visit_bool<E>(self, value: bool) -> std::result::Result<Self::Value, E> {
        Ok(Value::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> std::result::Result<Self::Value, E> {
        Ok(Value::Number(Number::from(value)))
    }

    fn visit_u64<E>(self, value: u64) -> std::result::Result<Self::Value, E> {
        Ok(Value::Number(Number::from(value)))
    }

    fn visit_f64<E>(self, value: f64) -> std::result::Result<Self::Value, E>
    where
        E: de::Error,
    {
        Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E> {
        Ok(Value::String(value.to_string()))
    }

    fn visit_string<E>(self, value: String) -> std::result::Result<Self::Value, E> {
        Ok(Value::String(value))
    }

    fn visit_none<E>(self) -> std::result::Result<Self::Value, E> {
        Ok(Value::Null)
    }

    fn visit_unit<E>(self) -> std::result::Result<Self::Value, E> {
        Ok(Value::Null)
    }

    fn visit_some<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        NoDuplicateValue::deserialize(deserializer).map(|value| value.0)
    }

    fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = seq.next_element::<NoDuplicateValue>()? {
            values.push(value.0);
        }
        Ok(Value::Array(values))
    }

    fn visit_map<A>(self, mut access: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut object = Map::new();
        while let Some((key, value)) = access.next_entry::<String, NoDuplicateValue>()? {
            if object.contains_key(&key) {
                return Err(de::Error::custom(format!("duplicate JSON key {key:?}")));
            }
            object.insert(key, value.0);
        }
        Ok(Value::Object(object))
    }
}

fn canonical_json(value: &Value) -> Result<String> {
    serde_json::to_string(value).map_err(AdapterStoreError::from)
}

fn require_schema(value: &Value, expected: &str, name: &str) -> Result<()> {
    let actual = required_string(value, "schema_version")?;
    if actual != expected {
        return Err(AdapterStoreError::new(format!(
            "{name} has unsupported schema {actual:?}"
        )));
    }
    Ok(())
}

fn require_exact_keys(value: &Value, name: &str, expected: &[&str]) -> Result<()> {
    let object = value
        .as_object()
        .ok_or_else(|| AdapterStoreError::new(format!("{name} must be a JSON object")))?;
    let actual: HashSet<&str> = object.keys().map(String::as_str).collect();
    let expected: HashSet<&str> = expected.iter().copied().collect();
    if actual == expected {
        return Ok(());
    }
    let mut missing: Vec<_> = expected.difference(&actual).copied().collect();
    let mut extra: Vec<_> = actual.difference(&expected).copied().collect();
    missing.sort_unstable();
    extra.sort_unstable();
    Err(AdapterStoreError::new(format!(
        "{name} has invalid keys: missing={missing:?}, extra={extra:?}"
    )))
}

fn require_str(value: &Value, field: &str, expected: &str) -> Result<()> {
    let actual = required_string(value, field)?;
    if actual != expected {
        return Err(AdapterStoreError::new(format!(
            "{field} must be {expected:?}, got {actual:?}"
        )));
    }
    Ok(())
}

fn required_string<'a>(value: &'a Value, field: &str) -> Result<&'a str> {
    value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| AdapterStoreError::new(format!("{field} must be a string")))
}

fn optional_string(value: &Value, field: &str) -> Result<Option<String>> {
    match value.get(field) {
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(Value::Null) | None => Ok(None),
        Some(_) => Err(AdapterStoreError::new(format!(
            "{field} must be a string or null"
        ))),
    }
}

fn required_bool(value: &Value, field: &str) -> Result<bool> {
    value
        .get(field)
        .and_then(Value::as_bool)
        .ok_or_else(|| AdapterStoreError::new(format!("{field} must be a bool")))
}

fn required_u64(value: &Value, field: &str) -> Result<u64> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| AdapterStoreError::new(format!("{field} must be a u64")))
}

fn optional_u64(value: &Value, field: &str) -> Result<Option<u64>> {
    match value.get(field) {
        Some(Value::Null) | None => Ok(None),
        Some(value) => value
            .as_u64()
            .map(Some)
            .ok_or_else(|| AdapterStoreError::new(format!("{field} must be null or a u64"))),
    }
}

fn required_bits(value: &Value) -> Result<u8> {
    let bits = required_u64(value, "bits")?;
    if !matches!(bits, 1 | 2 | 4) {
        return Err(AdapterStoreError::new("bits must be one of 1, 2, or 4"));
    }
    Ok(bits as u8)
}

fn optional_dim(value: &Value) -> Result<Option<usize>> {
    match value.get("dim") {
        Some(Value::Null) | None => Ok(None),
        Some(dim) => {
            let dim = dim
                .as_u64()
                .ok_or_else(|| AdapterStoreError::new("dim must be a positive integer or null"))?;
            if dim == 0 {
                return Err(AdapterStoreError::new("dim must be positive"));
            }
            usize::try_from(dim)
                .map(Some)
                .map_err(|_| AdapterStoreError::new("dim exceeds usize"))
        }
    }
}

fn required_object<'a>(value: &'a Value, field: &str) -> Result<&'a Map<String, Value>> {
    value
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| AdapterStoreError::new(format!("{field} must be a JSON object")))
}

fn parse_string_to_u64(value: &Map<String, Value>) -> Result<HashMap<String, u64>> {
    let mut out = HashMap::with_capacity(value.len());
    for (key, value) in value {
        require_non_empty_string_id(key)?;
        let u64_id = value
            .as_u64()
            .ok_or_else(|| AdapterStoreError::new("string_to_u64 values must be u64"))?;
        out.insert(key.clone(), u64_id);
    }
    Ok(out)
}

fn parse_u64_to_slot(value: &Map<String, Value>) -> Result<HashMap<u64, usize>> {
    let mut out = HashMap::with_capacity(value.len());
    for (key, value) in value {
        let u64_id = key
            .parse::<u64>()
            .map_err(|_| AdapterStoreError::new(format!("u64_to_slot key {key:?} is not u64")))?;
        let slot = value
            .as_u64()
            .ok_or_else(|| AdapterStoreError::new("u64_to_slot values must be slots"))?;
        out.insert(
            u64_id,
            usize::try_from(slot)
                .map_err(|_| AdapterStoreError::new("u64_to_slot value exceeds usize"))?,
        );
    }
    Ok(out)
}

fn parse_documents(value: &Map<String, Value>) -> Result<HashMap<String, String>> {
    let mut out = HashMap::with_capacity(value.len());
    for (key, value) in value {
        require_non_empty_string_id(key)?;
        let document = value
            .as_str()
            .ok_or_else(|| AdapterStoreError::new("document values must be strings"))?;
        out.insert(key.clone(), document.to_string());
    }
    Ok(out)
}

fn parse_metadata(value: &Map<String, Value>) -> Result<HashMap<String, Value>> {
    let mut out = HashMap::with_capacity(value.len());
    for (key, value) in value {
        require_non_empty_string_id(key)?;
        if !value.is_object() {
            return Err(AdapterStoreError::new(
                "metadata values must be JSON objects",
            ));
        }
        out.insert(key.clone(), value.clone());
    }
    Ok(out)
}

fn require_non_empty_string_id(value: &str) -> Result<()> {
    if value.is_empty() {
        return Err(AdapterStoreError::new("string IDs must be non-empty"));
    }
    Ok(())
}

fn validate_relative_path(value: &str) -> Result<()> {
    if value.is_empty() {
        return Err(AdapterStoreError::new(
            "active generation path must not be empty",
        ));
    }
    if value.contains('\\') || value.contains("//") || value.ends_with('/') {
        return Err(AdapterStoreError::new(
            "active generation path must use normalized forward-slash components",
        ));
    }
    let path = Path::new(value);
    if path.is_absolute() {
        return Err(AdapterStoreError::new(
            "active generation path must be relative",
        ));
    }
    for component in path.components() {
        match component {
            Component::Normal(_) => {}
            _ => {
                return Err(AdapterStoreError::new(
                    "active generation path must not contain parent or special components",
                ))
            }
        }
    }
    Ok(())
}

fn validate_index_path(value: &str) -> Result<()> {
    validate_relative_path(value)?;
    if value == INDEX_DIR {
        return Ok(());
    }

    let mut components = Path::new(value).components();
    let Some(Component::Normal(first)) = components.next() else {
        return Err(invalid_index_path(value));
    };
    let Some(Component::Normal(second)) = components.next() else {
        return Err(invalid_index_path(value));
    };
    if components.next().is_some() {
        return Err(invalid_index_path(value));
    }
    if first != VECTORS_DIR || parse_generation_dir(second.to_string_lossy().as_ref()).is_none() {
        return Err(invalid_index_path(value));
    }
    Ok(())
}

fn generation_id_from_index_path(value: &str) -> Result<u64> {
    validate_index_path(value)?;
    if value == INDEX_DIR {
        return Ok(1);
    }
    let mut components = Path::new(value).components();
    let _ = components.next();
    let Some(Component::Normal(generation_dir)) = components.next() else {
        return Err(invalid_index_path(value));
    };
    parse_generation_dir(generation_dir.to_string_lossy().as_ref())
        .ok_or_else(|| invalid_index_path(value))
}

fn validate_generation_manifest_file(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() {
        return Err(AdapterStoreError::new(format!(
            "generation manifest must not be a symlink: {}",
            path.display()
        )));
    }
    if !metadata.file_type().is_file() {
        return Err(AdapterStoreError::new(format!(
            "generation manifest must be a file: {}",
            path.display()
        )));
    }
    if metadata.len() > MAX_GENERATION_MANIFEST_BYTES {
        return Err(AdapterStoreError::new(format!(
            "generation manifest too large: {} bytes exceeds {}",
            metadata.len(),
            MAX_GENERATION_MANIFEST_BYTES
        )));
    }
    Ok(())
}

fn active_generation_manifest_path(root: &Path, index_path: &str) -> Result<PathBuf> {
    validate_index_path(index_path)?;
    let mut current = root.to_path_buf();
    for component in Path::new(index_path).components() {
        let Component::Normal(part) = component else {
            return Err(invalid_index_path(index_path));
        };
        current.push(part);
        let metadata = fs::symlink_metadata(&current).map_err(|err| {
            if err.kind() == std::io::ErrorKind::NotFound {
                AdapterStoreError::new(format!(
                    "active generation path is missing: {}",
                    current.display()
                ))
            } else {
                AdapterStoreError::new(format!(
                    "cannot stat active generation path component {}: {err}",
                    current.display()
                ))
            }
        })?;
        if metadata.file_type().is_symlink() {
            return Err(AdapterStoreError::new(format!(
                "active generation path must not contain a symlink: {}",
                current.display()
            )));
        }
        if !metadata.file_type().is_dir() {
            return Err(AdapterStoreError::new(format!(
                "active generation path component must be a directory: {}",
                current.display()
            )));
        }
    }
    Ok(current.join(MANIFEST_FILE))
}

fn invalid_index_path(value: &str) -> AdapterStoreError {
    AdapterStoreError::new(format!(
        "adapter index_path must be {INDEX_DIR:?} or a vectors/g000000000001.odb-style 12-digit generation path, got {value:?}"
    ))
}

fn parse_generation_dir(name: &str) -> Option<u64> {
    const PREFIX: &str = "g";
    const SUFFIX: &str = ".odb";
    const WIDTH: usize = 12;
    let digits = name
        .strip_prefix(PREFIX)
        .and_then(|value| value.strip_suffix(SUFFIX))?;
    if digits.len() != WIDTH || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    match digits.parse::<u64>() {
        Ok(id) if id > 0 => Some(id),
        _ => None,
    }
}

/// Recovers the generation id from a crash-debris entry name under
/// `vectors/`, when the name is a recognizable interrupted-replacement
/// scratch directory: one or more leading dots, the canonical
/// `g<12 digits>.odb` name, then one or more `.tmp-*` or `.bak-*`
/// decorations (a pre-fix writer could stack the temp decoration, producing
/// names like `..g000000000005.odb.tmp-<pid>-<ns>.tmp-<pid>-<ns>`).
fn parse_generation_debris_name(name: &str) -> Option<u64> {
    let trimmed = name.trim_start_matches('.');
    if trimmed.len() == name.len() {
        return None;
    }
    let (generation, suffix) = trimmed
        .split_once(".tmp-")
        .or_else(|| trimmed.split_once(".bak-"))?;
    if suffix.is_empty() {
        return None;
    }
    parse_generation_dir(generation)
}

/// A reclaimable entry found under `vectors/` that is not a committed
/// generation bundle: an interrupted-replacement scratch directory, a stray
/// file, or any other entry whose name does not parse as a canonical
/// `g<12 digits>.odb` generation name.
///
/// Because every generation that `adapter.redb` can reference — including
/// the active generation — carries a canonical name (enforced on every
/// write and verify), debris can never alias live data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationDebrisEntry {
    /// Root-relative, forward-slash path of the entry (for example
    /// `vectors/.g000000000005.odb.tmp-1234-5678`).
    pub path: String,
    /// Generation id recovered from the entry name when the debris is a
    /// recognizable interrupted-replacement scratch directory, else `None`.
    pub generation_id: Option<u64>,
    /// Human-readable warning describing why the entry is debris and how to
    /// reclaim it. Suitable for surfacing verbatim from `verify` tooling.
    pub warning: String,
}

/// Result of [`scan_generation_directory`]: the committed generation
/// bundles and the reclaimable debris found under `vectors/`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GenerationDirectoryScan {
    /// Root-relative paths of committed generation bundle directories
    /// (canonical `vectors/g<12 digits>.odb` names), sorted.
    pub generation_paths: Vec<String>,
    /// Reclaimable debris entries, sorted by path. Debris never carries a
    /// canonical generation name, so the active generation and every
    /// generation referenced by `adapter.redb` can never appear here.
    pub debris: Vec<GenerationDebrisEntry>,
}

/// Scans `<root>/vectors/`, classifying every entry as either a committed
/// generation bundle or reclaimable debris.
///
/// Crash debris — scratch directories left by an interrupted generation
/// replacement, stray files — is *never* a fatal error: it is reported as
/// [`GenerationDirectoryScan::debris`] with a structured warning per entry
/// so `verify`-style tooling can warn and `gc`-style tooling can reclaim
/// (see [`remove_generation_debris`]). Real integrity violations still fail
/// closed: symlinked entries and non-directory entries that carry a
/// canonical generation name return an error.
///
/// A missing `vectors/` directory yields an empty scan. Read-only: does not
/// take the writer lock.
pub fn scan_generation_directory(root: impl AsRef<Path>) -> Result<GenerationDirectoryScan> {
    let root = root.as_ref();
    let vectors_root = root.join(VECTORS_DIR);
    let metadata = match fs::symlink_metadata(&vectors_root) {
        Ok(metadata) => metadata,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Ok(GenerationDirectoryScan::default());
        }
        Err(err) => {
            return Err(AdapterStoreError::new(format!(
                "cannot stat {}: {err}",
                vectors_root.display()
            )));
        }
    };
    if metadata.file_type().is_symlink() {
        return Err(AdapterStoreError::new(format!(
            "{} must not be a symlink",
            vectors_root.display()
        )));
    }
    if !metadata.file_type().is_dir() {
        return Err(AdapterStoreError::new(format!(
            "{} must be a directory",
            vectors_root.display()
        )));
    }

    let mut scan = GenerationDirectoryScan::default();
    for entry in fs::read_dir(&vectors_root)? {
        let entry = entry?;
        let file_name = entry.file_name();
        let name = file_name.to_string_lossy();
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            return Err(AdapterStoreError::new(format!(
                "generation entry must not be a symlink: {}",
                entry.path().display()
            )));
        }
        let relative = format!("{VECTORS_DIR}/{name}");
        if let Some(generation_id) = parse_generation_dir(&name) {
            if !file_type.is_dir() {
                return Err(AdapterStoreError::new(format!(
                    "generation bundle must be a directory: {} (generation {generation_id})",
                    entry.path().display()
                )));
            }
            scan.generation_paths.push(relative);
        } else {
            let generation_id = parse_generation_debris_name(&name);
            let warning = match generation_id {
                Some(id) => format!(
                    "reclaimable debris in {VECTORS_DIR}/: {name:?} is a scratch directory \
                     left by an interrupted replacement of generation {id}; delete it via gc"
                ),
                None => format!(
                    "reclaimable debris in {VECTORS_DIR}/: {name:?} is not a generation \
                     bundle; delete it via gc"
                ),
            };
            scan.debris.push(GenerationDebrisEntry {
                path: relative,
                generation_id,
                warning,
            });
        }
    }
    scan.generation_paths.sort();
    scan.debris.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(scan)
}

/// Deletes one reclaimable debris entry under `<root>/vectors/`, as
/// classified by [`scan_generation_directory`].
///
/// Guards, in order:
/// - the caller must hold the writer lock through this crate
///   ([`acquire_writer_lock`]);
/// - `debris_path` must be a normalized root-relative path of exactly the
///   form `vectors/<name>`;
/// - `<name>` must **not** parse as a canonical generation name, so the
///   active generation and anything `adapter.redb` references (which are
///   canonical by construction) can never be deleted through this API;
/// - a symlinked entry is refused.
///
/// Deleting an entry that no longer exists is a no-op, so interrupted GC
/// runs can safely be re-driven.
pub fn remove_generation_debris(root: impl AsRef<Path>, debris_path: &str) -> Result<()> {
    let root = root.as_ref();
    require_writer_lock_held_by_current_process(root)?;
    validate_relative_path(debris_path)
        .map_err(|err| AdapterStoreError::new(format!("invalid debris path: {err}")))?;

    let mut components = Path::new(debris_path).components();
    let (Some(Component::Normal(first)), Some(Component::Normal(name)), None) =
        (components.next(), components.next(), components.next())
    else {
        return Err(AdapterStoreError::new(format!(
            "debris path must be directly under {VECTORS_DIR}/, got {debris_path:?}"
        )));
    };
    if first != VECTORS_DIR {
        return Err(AdapterStoreError::new(format!(
            "debris path must be directly under {VECTORS_DIR}/, got {debris_path:?}"
        )));
    }
    let name = name.to_string_lossy().into_owned();
    if parse_generation_dir(&name).is_some() {
        return Err(AdapterStoreError::new(format!(
            "refusing to delete committed generation {debris_path:?}: only \
             non-generation debris is reclaimable through this API"
        )));
    }

    let path = root.join(VECTORS_DIR).join(&name);
    let metadata = match fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(err) => {
            return Err(AdapterStoreError::new(format!(
                "cannot stat debris entry {}: {err}",
                path.display()
            )));
        }
    };
    if metadata.file_type().is_symlink() {
        return Err(AdapterStoreError::new(format!(
            "refusing to delete symlinked debris entry: {}",
            path.display()
        )));
    }
    if metadata.file_type().is_dir() {
        fs::remove_dir_all(&path)?;
    } else {
        fs::remove_file(&path)?;
    }
    sync_directory(&root.join(VECTORS_DIR))?;
    Ok(())
}

fn reject_empty_lazy_vector_artifacts(
    root: &Path,
    index_path: &str,
    allowed_prepared_generation: Option<&str>,
) -> Result<()> {
    let mut seen = HashSet::new();
    for relative in [index_path, INDEX_DIR] {
        let path = root.join(relative);
        if !seen.insert(path.clone()) {
            continue;
        }
        if path.try_exists()? && Some(relative) != allowed_prepared_generation {
            return Err(AdapterStoreError::new(format!(
                "empty_lazy adapter store must not contain {relative}"
            )));
        }
    }
    let vectors_root = root.join(VECTORS_DIR);
    if !vectors_root.try_exists()? {
        return Ok(());
    }
    let allowed_path = allowed_prepared_generation.and_then(|allowed| {
        let path = root.join(allowed);
        path.starts_with(&vectors_root).then_some(path)
    });
    for entry in fs::read_dir(&vectors_root)? {
        let entry = entry?;
        if Some(entry.path()) == allowed_path {
            continue;
        }
        // Crash debris (temp directories from an interrupted first save,
        // stray files) is tolerated: it is reclaimable, not a committed
        // generation. Anything that IS a committed generation name — or a
        // symlink, which no writer ever creates — still fails closed.
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if entry.file_type()?.is_symlink() || parse_generation_dir(&name).is_some() {
            return Err(AdapterStoreError::new(format!(
                "empty_lazy adapter store must not contain {VECTORS_DIR}"
            )));
        }
    }
    Ok(())
}

/// RAII guard for the adapter root's advisory writer lock, acquired via
/// [`acquire_writer_lock`].
///
/// Exclusion is enforced by two cooperating mechanisms:
///
/// 1. a non-blocking OS advisory file lock (`fs4::FileExt::try_lock`) on
///    `.ordinaldb.write.lock` inside the canonicalized root, which excludes
///    other processes, and
/// 2. a process-global registry of held lock paths, which excludes
///    re-acquisition from within the same process (OS advisory locks are
///    typically per-process and would otherwise be granted silently).
///    The registry entry is reserved atomically (check-and-insert under a
///    single registry lock scope) *before* the OS lock is attempted, and
///    rolled back if acquisition fails, so concurrent same-process
///    acquirers can never both succeed.
///
/// On acquisition the lock file is truncated and stamped with
/// `pid=<pid>` / `lock=advisory-v1`, then fsynced along with the root
/// directory. A symlinked or non-regular-file lock path is rejected.
/// Dropping the guard releases the OS lock and unregisters the path; the
/// lock file itself is left in place on disk.
pub struct WriterLockGuard {
    key: PathBuf,
    file: Option<File>,
}

impl WriterLockGuard {
    fn acquire(root: &Path) -> Result<Self> {
        let root = fs::canonicalize(root)?;
        let path = root.join(WRITE_LOCK_FILE);
        if let Ok(metadata) = fs::symlink_metadata(&path) {
            if metadata.file_type().is_symlink() {
                return Err(AdapterStoreError::new(format!(
                    "adapter writer lock must not be a symlink: {}",
                    path.display()
                )));
            }
            if !metadata.file_type().is_file() {
                return Err(AdapterStoreError::new(format!(
                    "adapter writer lock must be a file: {}",
                    path.display()
                )));
            }
        }
        let key = path.clone();
        reserve_writer_lock_registration(&key)?;
        match Self::lock_and_stamp(&root, &path) {
            Ok(file) => Ok(Self {
                key,
                file: Some(file),
            }),
            Err(err) => {
                release_writer_lock_registration(&key);
                Err(err)
            }
        }
    }

    /// Opens the lock file, takes the non-blocking OS advisory lock, and
    /// stamps + fsyncs it. The caller must already hold the registry
    /// reservation for this path (and must release it if this fails).
    fn lock_and_stamp(root: &Path, path: &Path) -> Result<File> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .map_err(AdapterStoreError::from)?;
        match FileExt::try_lock(&file) {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => {
                return Err(AdapterStoreError::new(format!(
                    "adapter writer lock already held: {}",
                    path.display()
                )));
            }
            Err(TryLockError::Error(err)) => return Err(AdapterStoreError::from(err)),
        }
        file.set_len(0)?;
        writeln!(file, "pid={}\nlock=advisory-v1", std::process::id())?;
        file.sync_all()?;
        sync_directory(root)?;
        Ok(file)
    }

    fn release(&mut self) {
        if let Some(file) = self.file.take() {
            let _ = FileExt::unlock(&file);
            release_writer_lock_registration(&self.key);
        }
    }
}

impl Drop for WriterLockGuard {
    fn drop(&mut self) {
        self.release();
    }
}

fn require_writer_lock_held_by_current_process(root: &Path) -> Result<()> {
    let root = fs::canonicalize(root)?;
    let path = root.join(WRITE_LOCK_FILE);
    let metadata = fs::symlink_metadata(&path)?;
    if metadata.file_type().is_symlink() {
        return Err(AdapterStoreError::new(format!(
            "adapter writer lock must not be a symlink: {}",
            path.display()
        )));
    }
    if !metadata.file_type().is_file() {
        return Err(AdapterStoreError::new(format!(
            "adapter writer lock must be a file: {}",
            path.display()
        )));
    }
    let held = held_writer_locks().lock().map_err(lock_registry_poisoned)?;
    if !held.contains(&path) {
        return Err(AdapterStoreError::new(format!(
            "adapter writer lock is not held by this process through the storage engine: {}",
            path.display()
        )));
    }
    Ok(())
}

/// Atomically reserves `key` in the process-global held-lock registry,
/// failing when it is already registered.
///
/// The check and the insert happen under a single registry lock scope. With
/// a check-then-insert split, two threads could both pass the check and
/// both be granted the OS lock on platforms/filesystems where advisory
/// locks are per-process rather than per-open-file-description (e.g.
/// `flock` emulated via `fcntl` on NFS), breaking writer exclusivity.
/// Reserving before the lock file is even opened also avoids the `fcntl`
/// footgun where closing a losing second descriptor would drop the
/// process's existing advisory lock.
fn reserve_writer_lock_registration(key: &Path) -> Result<()> {
    let mut held = held_writer_locks().lock().map_err(lock_registry_poisoned)?;
    if !held.insert(key.to_path_buf()) {
        return Err(AdapterStoreError::new(format!(
            "adapter writer lock already held: {}",
            key.display()
        )));
    }
    Ok(())
}

/// Removes `key` from the process-global held-lock registry. Used both on
/// guard release and to roll back a reservation whose OS-lock acquisition
/// failed.
fn release_writer_lock_registration(key: &Path) {
    if let Ok(mut held) = held_writer_locks().lock() {
        held.remove(key);
    }
}

fn held_writer_locks() -> &'static Mutex<HashSet<PathBuf>> {
    static HELD: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();
    HELD.get_or_init(|| Mutex::new(HashSet::new()))
}

fn lock_registry_poisoned<T>(_: std::sync::PoisonError<T>) -> AdapterStoreError {
    AdapterStoreError::new("adapter writer lock registry is poisoned")
}

#[cfg(unix)]
fn sync_file(path: &Path) -> Result<()> {
    fs::File::open(path)?.sync_all()?;
    Ok(())
}

#[cfg(windows)]
fn sync_file(path: &Path) -> Result<()> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)?
        .sync_all()?;
    Ok(())
}

#[cfg(all(not(unix), not(windows)))]
fn sync_file(path: &Path) -> Result<()> {
    fs::File::open(path)?.sync_all()?;
    Ok(())
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<()> {
    fs::File::open(path)?.sync_all()?;
    Ok(())
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> Result<()> {
    Ok(())
}

fn usize_to_u64(value: usize, name: &str) -> Result<u64> {
    u64::try_from(value).map_err(|_| AdapterStoreError::new(format!("{name} exceeds u64")))
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}
