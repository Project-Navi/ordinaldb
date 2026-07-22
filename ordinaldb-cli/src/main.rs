#[cfg(feature = "experimental-ltr")]
use std::collections::HashMap;
use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::fs::{File, OpenOptions};
#[cfg(feature = "experimental-ltr")]
use std::io::BufRead;
use std::io::BufReader;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use clap::{Parser, Subcommand};
#[cfg(feature = "experimental-ltr")]
use ordinaldb::artifacts::LTR_MODEL_AUX_NAME;
#[cfg(feature = "experimental-ltr")]
use ordinaldb::hybrid::{
    Bm25MmapIndex, LtrLoadOptions, TreeEnsembleReranker, DEFAULT_SPARSE_AUX_NAME,
};
#[cfg(feature = "experimental-ltr")]
use ordinaldb::manifest::{set_auxiliary_size_limit, VerifyOptions as OrdinalVerifyOptions};
#[cfg(test)]
use ordinaldb_adapter_store::write_legacy_snapshot;
use ordinaldb_adapter_store::{
    acquire_writer_lock, generation_gc_events, open_verified as open_verified_adapter_store,
    record_generation_gc_with_existing_lock, remove_generation_debris, scan_generation_directory,
    write_legacy_snapshot_with_origin, GenerationGcUpdate, LegacyPayloads, StoreOrigin,
    StoreRevision, VerifiedAdapterStore, ADAPTER_STORE_FILE, ADAPTER_STORE_SCHEMA_VERSION,
};
#[cfg(feature = "experimental-ltr")]
use ordinaldb_ltr::{
    normalize_bundle_relative_dir, read_verified_feature_cache_bundle_auxiliary,
    write_feature_cache_bundle_auxiliary, BundleFeatureCacheOptions, FeatureCacheData,
    FeatureCacheManifest, QueryCacheRecord, DEFAULT_LTR_FEATURE_CACHE_AUX_NAME,
};
use ordvec_manifest::{sha256_file, verify_for_load, ManifestIndexParams, VerifyOptions};
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Number, Value};

const ADAPTER_SCHEMA_VERSION: &str = "ordinaldb.adapter.v1";
const ID_MAP_SCHEMA_VERSION: &str = "ordinaldb.adapter.id_map.v1";
const DOCUMENTS_SCHEMA_VERSION: &str = "ordinaldb.adapter.documents.v1";
const METADATA_SCHEMA_VERSION: &str = "ordinaldb.adapter.metadata.v1";

const ADAPTER_FILE: &str = "adapter.json";
const ID_MAP_FILE: &str = "id_map.json";
const DOCUMENTS_FILE: &str = "documents.json";
const METADATA_FILE: &str = "metadata.json";
const ADAPTER_REDB_REVISION_FILE: &str = "adapter.redb.revision.json";
const INDEX_DIR: &str = "index.odb";
const VECTORS_DIR: &str = "vectors";
const MANIFEST_FILE: &str = "manifest.json";
const ROW_IDENTITY_KIND: &str = "row_id_identity";
const GENERATION_PIN_FILE: &str = ".ordinaldb.pin";
const MAX_GENERATION_MANIFEST_BYTES: u64 = 1024 * 1024;
static EXPORT_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Parser)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Inspect {
        path: PathBuf,
        #[arg(long)]
        json: bool,
    },
    Verify {
        path: PathBuf,
        #[arg(long)]
        json: bool,
    },
    Stats {
        path: PathBuf,
        #[arg(long)]
        json: bool,
    },
    Adapter {
        #[command(subcommand)]
        command: AdapterCommand,
    },
    Ltr {
        #[command(subcommand)]
        command: LtrCommand,
    },
}

#[derive(Subcommand)]
enum AdapterCommand {
    /// Export derived JSON sidecars from the authoritative adapter.redb.
    ExportJson { path: PathBuf },
    /// Import a legacy JSON adapter directory into a redb-backed target.
    ImportLegacy {
        source: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Delete reclaimable adapter generations after recording GC state in redb.
    Gc {
        path: PathBuf,
        #[arg(long, default_value_t = 1)]
        retain: usize,
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
enum LtrCommand {
    /// Export an experimental grouped LTR feature cache from a verified bundle.
    Features {
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long)]
        queries: PathBuf,
        #[arg(long)]
        qrels: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 100)]
        top_k: usize,
        #[arg(long, default_value_t = 2 * 1024 * 1024 * 1024u64)]
        max_auxiliary_artifact_bytes: u64,
        #[arg(long)]
        json: bool,
    },
    /// Train a local experimental LTR model from a feature cache.
    Train {
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long, default_value = "ordinaldb.ltr_features")]
        aux_name: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Attach a verified LTR model sidecar to a bundle.
    Attach {
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long)]
        model: PathBuf,
    },
    /// Inspect the LTR state for a bundle.
    Inspect {
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug)]
struct CliError(String);

impl Display for CliError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for CliError {}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum PathKind {
    CoreBundle,
    AdapterDirectory,
    Unknown,
}

#[derive(Debug, Serialize)]
struct InspectReport {
    kind: PathKind,
    path: String,
    schema_version: Option<String>,
    adapter: Option<String>,
    bits: Option<u8>,
    dim: Option<usize>,
    vector_count: usize,
    empty_lazy: bool,
    sidecar_count: usize,
    active_generation_id: Option<u64>,
    active_generation_path: Option<String>,
    active_generation_manifest_sha256: Option<String>,
    active_generation_manifest_size_bytes: Option<u64>,
    #[serde(flatten)]
    adapter_generations: Option<AdapterGenerationsReport>,
}

/// Adapter-directory-only generation bookkeeping. Plain core bundles have no
/// generation concept, so this is `None` for them and the fields are omitted
/// entirely from JSON output (via `#[serde(flatten)]` on the `Option`)
/// instead of being emitted as misleading zeros.
#[derive(Debug, Serialize)]
struct AdapterGenerationsReport {
    generation_count: usize,
    active_generation_count: usize,
    completed_generation_count: usize,
    retained_generation_count: usize,
    partial_generation_count: usize,
    reclaimable_generation_count: usize,
    orphan_generation_count: usize,
    orphan_generation_paths: Vec<String>,
    retained_generation_paths: Vec<String>,
    partial_generation_paths: Vec<String>,
    reclaimable_generation_paths: Vec<String>,
}

impl AdapterGenerationsReport {
    fn from_summary(summary: GenerationDirectorySummary) -> Self {
        Self {
            generation_count: summary.generation_count,
            active_generation_count: summary.active_generation_count,
            completed_generation_count: summary.completed_generation_count,
            retained_generation_count: summary.retained_generation_paths.len(),
            partial_generation_count: summary.partial_generation_paths.len(),
            reclaimable_generation_count: summary.reclaimable_generation_paths.len(),
            orphan_generation_count: summary.orphan_generation_paths.len(),
            orphan_generation_paths: summary.orphan_generation_paths,
            retained_generation_paths: summary.retained_generation_paths,
            partial_generation_paths: summary.partial_generation_paths,
            reclaimable_generation_paths: summary.reclaimable_generation_paths,
        }
    }
}

#[derive(Debug)]
struct GenerationDirectorySummary {
    generation_count: usize,
    active_generation_count: usize,
    completed_generation_count: usize,
    retained_generation_paths: Vec<String>,
    partial_generation_paths: Vec<String>,
    reclaimable_generation_paths: Vec<String>,
    orphan_generation_paths: Vec<String>,
    /// Human-readable warnings, one per reclaimable debris entry found
    /// under `vectors/` (see [`ordinaldb_adapter_store::scan_generation_directory`]).
    /// Debris is never fatal, so these are surfaced as warnings rather than
    /// errors — currently only consumed by `verify`'s report.
    debris_warnings: Vec<String>,
}

#[derive(Debug, Serialize)]
struct VerifyReport {
    kind: PathKind,
    path: String,
    valid: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    /// Human-readable warnings for non-fatal conditions (for example,
    /// reclaimable crash debris under `vectors/`) that do not affect
    /// `valid`. Omitted from JSON output entirely when empty.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    warnings: Vec<String>,
}

#[derive(Debug, Serialize)]
struct StatsReport {
    kind: PathKind,
    path: String,
    vector_count: usize,
    active_id_count: usize,
    #[serde(flatten)]
    adapter_generations: Option<AdapterGenerationsReport>,
    bytes_by_component: BTreeMap<String, u64>,
}

#[derive(Debug, Serialize)]
struct AdapterGcReport {
    path: String,
    dry_run: bool,
    retain: usize,
    active_generation_path: Option<String>,
    retained_generation_paths: Vec<String>,
    reclaimable_generation_paths: Vec<String>,
    deleted_generation_paths: Vec<String>,
    pinned_generation_paths: Vec<String>,
    redb_commit_sequence: Option<u64>,
}

#[cfg(feature = "experimental-ltr")]
#[derive(Debug, Serialize)]
struct LtrFeaturesReport {
    path: String,
    feature_names: Vec<String>,
    query_count: usize,
    row_count: usize,
    group_count: usize,
    dense_features_present: bool,
    bundle_manifest_sha256: Option<String>,
    qrels_source_sha256: Option<String>,
}

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(err) => {
            eprintln!("error: {err}");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<bool, Box<dyn Error>> {
    let cli = Cli::parse();
    match cli.command {
        Command::Inspect { path, json } => {
            let report = inspect_path(&path)?;
            emit_inspect(report, json)?;
            Ok(true)
        }
        Command::Verify { path, json } => {
            let report = verify_path(&path)?;
            let valid = report.valid;
            emit_verify(&report, json)?;
            Ok(valid)
        }
        Command::Stats { path, json } => {
            let report = stats_path(&path)?;
            emit_stats(&report, json)?;
            Ok(true)
        }
        Command::Adapter { command } => run_adapter_command(command),
        Command::Ltr { command } => run_ltr_command(command),
    }
}

fn run_adapter_command(command: AdapterCommand) -> Result<bool, Box<dyn Error>> {
    match command {
        AdapterCommand::ExportJson { path } => {
            export_adapter_json(&path)?;
            println!("exported derived adapter JSON sidecars: {}", path.display());
            Ok(true)
        }
        AdapterCommand::ImportLegacy { source, output } => {
            import_legacy_adapter(&source, &output)?;
            println!(
                "imported legacy adapter JSON into redb store: {}",
                output.display()
            );
            Ok(true)
        }
        AdapterCommand::Gc {
            path,
            retain,
            dry_run,
            json,
        } => {
            let report = gc_adapter_generations(&path, retain, dry_run)?;
            emit_adapter_gc(&report, json)?;
            Ok(true)
        }
    }
}

fn export_adapter_json(path: &Path) -> Result<(), Box<dyn Error>> {
    let store = open_verified_adapter_store(path, None)?;
    write_adapter_payload_exports(path, &store.payloads)?;
    write_json_pretty(
        &path.join(ADAPTER_REDB_REVISION_FILE),
        &adapter_revision_export(&store.manifest),
    )?;
    Ok(())
}

fn import_legacy_adapter(source: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    if output.exists() {
        return Err(Box::new(CliError(format!(
            "output path already exists: {}",
            output.display()
        ))));
    }
    if !source.is_dir() {
        return Err(Box::new(CliError(format!(
            "legacy source must be a directory: {}",
            source.display()
        ))));
    }
    if source.join(ADAPTER_STORE_FILE).exists() {
        return Err(Box::new(CliError(format!(
            "legacy source already contains {ADAPTER_STORE_FILE}: {}",
            source.display()
        ))));
    }
    copy_dir_recursive_no_symlinks(source, output)?;
    let mut output_guard = ImportOutputGuard::new(output.to_path_buf());
    let payloads = LegacyPayloads {
        adapter_json: read_legacy_payload(output, ADAPTER_FILE)?,
        id_map_json: read_legacy_payload(output, ID_MAP_FILE)?,
        documents_json: read_legacy_payload(output, DOCUMENTS_FILE)?,
        metadata_json: read_legacy_payload(output, METADATA_FILE)?,
    };
    let store =
        write_legacy_snapshot_with_origin(output, None, payloads, StoreOrigin::ImportedLegacyJson)?;
    write_json_pretty(
        &output.join(ADAPTER_REDB_REVISION_FILE),
        &adapter_revision_export(&store.manifest),
    )?;
    output_guard.disarm();
    Ok(())
}

fn gc_adapter_generations(
    path: &Path,
    retain: usize,
    dry_run: bool,
) -> Result<AdapterGcReport, Box<dyn Error>> {
    let store = open_verified_adapter_store(path, None)?;
    let active_generation_path = store
        .active_generation_path()
        .filter(|path| !path.is_empty())
        .map(str::to_string);
    let summary = generation_directory_summary(path, active_generation_path.as_deref())?;

    let mut pinned_generation_paths = Vec::new();
    let mut retained_generation_paths = Vec::new();
    let mut reclaimable_generation_paths = Vec::new();

    let mut retained_budget = retain;
    for generation_path in summary.retained_generation_paths.iter().rev() {
        if generation_is_pinned(path, generation_path)? {
            pinned_generation_paths.push(generation_path.clone());
            retained_generation_paths.push(generation_path.clone());
        } else if retained_budget > 0 {
            retained_budget -= 1;
            retained_generation_paths.push(generation_path.clone());
        } else {
            reclaimable_generation_paths.push(generation_path.clone());
        }
    }
    for generation_path in &summary.partial_generation_paths {
        if generation_is_pinned(path, generation_path)? {
            pinned_generation_paths.push(generation_path.clone());
        } else {
            reclaimable_generation_paths.push(generation_path.clone());
        }
    }
    retained_generation_paths.sort();
    pinned_generation_paths.sort();
    reclaimable_generation_paths.sort();

    if dry_run {
        return Ok(AdapterGcReport {
            path: path.display().to_string(),
            dry_run,
            retain,
            active_generation_path,
            retained_generation_paths,
            reclaimable_generation_paths,
            deleted_generation_paths: Vec::new(),
            pinned_generation_paths,
            redb_commit_sequence: store.manifest["commit_sequence"].as_u64(),
        });
    }

    let writer_lock = acquire_writer_lock(path)?;
    let (mut current_store, mut deleted_generation_paths) =
        recover_interrupted_adapter_gc(path, active_generation_path.as_deref(), store)?;

    if reclaimable_generation_paths.is_empty() {
        let redb_commit_sequence = current_store.manifest["commit_sequence"].as_u64();
        drop(writer_lock);
        return Ok(AdapterGcReport {
            path: path.display().to_string(),
            dry_run,
            retain,
            active_generation_path,
            retained_generation_paths,
            reclaimable_generation_paths,
            deleted_generation_paths,
            pinned_generation_paths,
            redb_commit_sequence,
        });
    }

    let expected = StoreRevision::from_manifest(&current_store.manifest)?;
    let reclaimable_updates = reclaimable_generation_paths
        .iter()
        .map(|generation_path| gc_update_for_path(generation_path, "reclaimable", "adapter_gc"))
        .collect::<Result<Vec<_>, _>>()?;
    let queued =
        record_generation_gc_with_existing_lock(path, Some(expected), &reclaimable_updates)?;

    let expected = StoreRevision::from_manifest(&queued.manifest)?;
    let deleting_updates = reclaimable_generation_paths
        .iter()
        .map(|generation_path| gc_update_for_path(generation_path, "deleting", "adapter_gc"))
        .collect::<Result<Vec<_>, _>>()?;
    let deleting =
        record_generation_gc_with_existing_lock(path, Some(expected), &deleting_updates)?;

    for generation_path in &reclaimable_generation_paths {
        delete_generation_dir(path, generation_path)?;
        deleted_generation_paths.push(generation_path.clone());
    }

    let final_store = if deleted_generation_paths.is_empty() {
        deleting
    } else {
        let expected = StoreRevision::from_manifest(&deleting.manifest)?;
        let deleted_updates = deleted_generation_paths
            .iter()
            .map(|generation_path| gc_update_for_path(generation_path, "deleted", "adapter_gc"))
            .collect::<Result<Vec<_>, _>>()?;
        record_generation_gc_with_existing_lock(path, Some(expected), &deleted_updates)?
    };
    current_store = final_store;
    drop(writer_lock);

    Ok(AdapterGcReport {
        path: path.display().to_string(),
        dry_run,
        retain,
        active_generation_path,
        retained_generation_paths,
        reclaimable_generation_paths,
        deleted_generation_paths,
        pinned_generation_paths,
        redb_commit_sequence: current_store.manifest["commit_sequence"].as_u64(),
    })
}

fn recover_interrupted_adapter_gc(
    root: &Path,
    active_generation_path: Option<&str>,
    store: VerifiedAdapterStore,
) -> Result<(VerifiedAdapterStore, Vec<String>), Box<dyn Error>> {
    let mut recovered_deleted_paths = Vec::new();
    for (generation_path, state) in latest_gc_states_by_path(root)? {
        if state != "deleting" || Some(generation_path.as_str()) == active_generation_path {
            continue;
        }
        let path = root.join(&generation_path);
        match std::fs::symlink_metadata(&path) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() {
                    return Err(Box::new(CliError(format!(
                        "generation directory must not be a symlink: {}",
                        path.display()
                    ))));
                }
                if !metadata.file_type().is_dir() {
                    return Err(Box::new(CliError(format!(
                        "generation path must be a directory: {}",
                        path.display()
                    ))));
                }
                if generation_is_pinned(root, &generation_path)? {
                    continue;
                }
                delete_generation_dir(root, &generation_path)?;
                recovered_deleted_paths.push(generation_path);
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                recovered_deleted_paths.push(generation_path);
            }
            Err(err) => {
                return Err(Box::new(CliError(format!(
                    "cannot stat interrupted GC generation {}: {err}",
                    path.display()
                ))));
            }
        }
    }

    if recovered_deleted_paths.is_empty() {
        return Ok((store, recovered_deleted_paths));
    }

    let expected = StoreRevision::from_manifest(&store.manifest)?;
    let deleted_updates = recovered_deleted_paths
        .iter()
        .map(|generation_path| {
            gc_update_for_path(generation_path, "deleted", "adapter_gc_recovery")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let recovered =
        record_generation_gc_with_existing_lock(root, Some(expected), &deleted_updates)?;
    Ok((recovered, recovered_deleted_paths))
}

fn latest_gc_states_by_path(root: &Path) -> Result<BTreeMap<String, String>, Box<dyn Error>> {
    let mut states = BTreeMap::new();
    for event in generation_gc_events(root)? {
        states.insert(event.path, event.state);
    }
    Ok(states)
}

fn gc_update_for_path(
    generation_path: &str,
    state: &str,
    reason: &str,
) -> Result<GenerationGcUpdate, Box<dyn Error>> {
    Ok(GenerationGcUpdate {
        generation_id: generation_id_from_gc_path(generation_path)?,
        path: generation_path.to_string(),
        state: state.to_string(),
        reason: reason.to_string(),
    })
}

fn generation_id_from_gc_path(generation_path: &str) -> Result<Option<u64>, Box<dyn Error>> {
    validate_relative_path(generation_path)?;
    let mut components = Path::new(generation_path).components();
    let Some(Component::Normal(first)) = components.next() else {
        return Err(Box::new(invalid_index_path(generation_path)));
    };
    let Some(Component::Normal(second)) = components.next() else {
        return Err(Box::new(invalid_index_path(generation_path)));
    };
    if components.next().is_some() || first != VECTORS_DIR {
        return Err(Box::new(invalid_index_path(generation_path)));
    }
    let name = second.to_string_lossy();
    if let Some(generation_id) = parse_generation_dir(&name) {
        return Ok(Some(generation_id));
    }
    if let Some(generation_id) = parse_partial_generation_dir(&name) {
        return Ok(Some(generation_id));
    }
    // Neither a canonical generation name nor a recognizable
    // interrupted-replacement scratch directory: this is crash debris that
    // adapter-store's own GC ledger and `remove_generation_debris` accept
    // (they classify by structure, not by parseable generation id), so this
    // must not be a hard error here either — it just has no recoverable
    // generation id to attribute the ledger entry to.
    Ok(None)
}

fn generation_is_pinned(root: &Path, generation_path: &str) -> Result<bool, Box<dyn Error>> {
    let pin_path = root.join(generation_path).join(GENERATION_PIN_FILE);
    match std::fs::symlink_metadata(&pin_path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(Box::new(CliError(format!(
            "generation pin must not be a symlink: {}",
            pin_path.display()
        )))),
        Ok(metadata) if !metadata.file_type().is_file() => Err(Box::new(CliError(format!(
            "generation pin must be a file: {}",
            pin_path.display()
        )))),
        Ok(_) => Ok(true),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(false),
        // The entry itself is a file (stray debris), so it cannot contain a
        // pin file; treat it as unpinned so gc can reclaim it.
        Err(err) if err.kind() == std::io::ErrorKind::NotADirectory => Ok(false),
        Err(err) => Err(Box::new(CliError(format!(
            "cannot stat generation pin {}: {err}",
            pin_path.display()
        )))),
    }
}

fn delete_generation_dir(root: &Path, generation_path: &str) -> Result<(), Box<dyn Error>> {
    let _ = generation_id_from_gc_path(generation_path)?;
    let mut components = Path::new(generation_path).components();
    let _ = components.next();
    let Some(Component::Normal(name)) = components.next() else {
        return Err(Box::new(invalid_index_path(generation_path)));
    };
    if parse_generation_dir(&name.to_string_lossy()).is_none() {
        // Not a canonical generation name: this is reclaimable crash debris
        // (an interrupted-replacement scratch directory, a stray file, or
        // any other non-canonical entry). `remove_generation_debris` handles
        // all of those uniformly — including stray files, which the
        // directory-only check below would reject — and centrally guards
        // the writer lock, the `vectors/`-only scope, and refuses canonical
        // generation names and symlinks.
        remove_generation_debris(root, generation_path)?;
        return Ok(());
    }

    let path = root.join(generation_path);
    let metadata = std::fs::symlink_metadata(&path).map_err(|err| {
        CliError(format!(
            "cannot stat generation directory {}: {err}",
            path.display()
        ))
    })?;
    if metadata.file_type().is_symlink() {
        return Err(Box::new(CliError(format!(
            "generation directory must not be a symlink: {}",
            path.display()
        ))));
    }
    if !metadata.file_type().is_dir() {
        return Err(Box::new(CliError(format!(
            "generation path must be a directory: {}",
            path.display()
        ))));
    }
    std::fs::remove_dir_all(&path)?;
    if let Some(parent) = path.parent() {
        sync_directory(parent)?;
    }
    Ok(())
}

fn read_legacy_payload(root: &Path, name: &str) -> Result<String, Box<dyn Error>> {
    let path = root.join(name);
    std::fs::read_to_string(&path).map_err(|err| {
        Box::new(CliError(format!(
            "cannot read legacy adapter payload {}: {err}",
            path.display()
        ))) as Box<dyn Error>
    })
}

fn write_adapter_payload_exports(
    path: &Path,
    payloads: &LegacyPayloads,
) -> Result<(), Box<dyn Error>> {
    write_file_if_changed(&path.join(ADAPTER_FILE), payloads.adapter_json.as_bytes())?;
    write_file_if_changed(&path.join(ID_MAP_FILE), payloads.id_map_json.as_bytes())?;
    write_file_if_changed(
        &path.join(DOCUMENTS_FILE),
        payloads.documents_json.as_bytes(),
    )?;
    write_file_if_changed(&path.join(METADATA_FILE), payloads.metadata_json.as_bytes())?;
    Ok(())
}

fn adapter_revision_export(manifest: &Value) -> Value {
    json_object([
        ("schema_version", manifest["schema_version"].clone()),
        ("store_uuid", manifest["store_uuid"].clone()),
        ("commit_sequence", manifest["commit_sequence"].clone()),
        (
            "active_generation_id",
            manifest["active_generation_id"].clone(),
        ),
        (
            "active_generation_path",
            manifest["active_generation_path"].clone(),
        ),
        (
            "active_generation_manifest_sha256",
            manifest["active_generation_manifest_sha256"].clone(),
        ),
        ("active_id_count", manifest["active_id_count"].clone()),
    ])
}

fn write_json_pretty(path: &Path, value: &Value) -> Result<(), Box<dyn Error>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    write_file_if_changed(path, &bytes)?;
    Ok(())
}

fn write_file_if_changed(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            return Err(Box::new(CliError(format!(
                "export target must not be a symlink: {}",
                path.display()
            ))));
        }
        Ok(metadata) if !metadata.is_file() => {
            return Err(Box::new(CliError(format!(
                "export target must be a file: {}",
                path.display()
            ))));
        }
        Ok(_) => {
            if std::fs::read(path)? == bytes {
                return Ok(());
            }
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => return Err(Box::new(err)),
    }
    let temp_path = export_temp_path(path);
    {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    std::fs::rename(&temp_path, path)?;
    Ok(())
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<(), Box<dyn Error>> {
    File::open(path)?.sync_all()?;
    Ok(())
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> Result<(), Box<dyn Error>> {
    Ok(())
}

fn export_temp_path(path: &Path) -> PathBuf {
    let mut temp_path = path.to_path_buf();
    let counter = EXPORT_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    temp_path.set_file_name(format!(
        ".{}.tmp-{}-{counter}",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("ordinaldb-export"),
        std::process::id()
    ));
    temp_path
}

struct ImportOutputGuard {
    path: PathBuf,
    armed: bool,
}

impl ImportOutputGuard {
    fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for ImportOutputGuard {
    fn drop(&mut self) {
        if self.armed {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

fn copy_dir_recursive_no_symlinks(source: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    let metadata = std::fs::symlink_metadata(source)?;
    if metadata.file_type().is_symlink() {
        return Err(Box::new(CliError(format!(
            "legacy source must not contain symlinks: {}",
            source.display()
        ))));
    }
    if !metadata.is_dir() {
        return Err(Box::new(CliError(format!(
            "legacy source must be a directory: {}",
            source.display()
        ))));
    }
    std::fs::create_dir(output)?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let entry_path = entry.path();
        let target_path = output.join(entry.file_name());
        let metadata = std::fs::symlink_metadata(&entry_path)?;
        if metadata.file_type().is_symlink() {
            return Err(Box::new(CliError(format!(
                "legacy source must not contain symlinks: {}",
                entry_path.display()
            ))));
        }
        if metadata.is_dir() {
            copy_dir_recursive_no_symlinks(&entry_path, &target_path)?;
        } else if metadata.is_file() {
            std::fs::copy(&entry_path, &target_path)?;
        } else {
            return Err(Box::new(CliError(format!(
                "legacy source contains unsupported file type: {}",
                entry_path.display()
            ))));
        }
    }
    Ok(())
}

fn json_object<const N: usize>(pairs: [(&str, Value); N]) -> Value {
    let mut object = Map::new();
    for (key, value) in pairs {
        object.insert(key.to_string(), value);
    }
    Value::Object(object)
}

#[cfg(feature = "experimental-ltr")]
fn run_ltr_command(command: LtrCommand) -> Result<bool, Box<dyn Error>> {
    match command {
        LtrCommand::Features {
            bundle,
            queries,
            qrels,
            out,
            top_k,
            max_auxiliary_artifact_bytes,
            json,
        } => {
            let report = export_ltr_features(
                &bundle,
                &queries,
                &qrels,
                &out,
                top_k,
                max_auxiliary_artifact_bytes,
            )?;
            emit_ltr_features(&report, json)?;
            Ok(true)
        }
        LtrCommand::Train { .. } => Err(Box::new(CliError(
            "ordinaldb ltr train is not implemented yet; future training will consume a manifest-verified bundle auxiliary, not a standalone feature path".to_string(),
        ))),
        LtrCommand::Attach { .. } => Err(Box::new(CliError(
            "ordinaldb ltr attach is not implemented yet; model sidecar attach/manifest update is the next product pass".to_string(),
        ))),
        LtrCommand::Inspect { .. } => Err(Box::new(CliError(
            "ordinaldb ltr inspect is not implemented yet; model sidecar inspect is the next product pass".to_string(),
        ))),
    }
}

#[cfg(not(feature = "experimental-ltr"))]
fn run_ltr_command(command: LtrCommand) -> Result<bool, Box<dyn Error>> {
    match command {
        LtrCommand::Features { .. }
        | LtrCommand::Train { .. }
        | LtrCommand::Attach { .. }
        | LtrCommand::Inspect { .. } => Err(Box::new(CliError(
            "LTR support was not compiled; rebuild ordinaldb-cli with --features experimental-ltr"
                .to_string(),
        ))),
    }
}

#[cfg(feature = "experimental-ltr")]
fn export_ltr_features(
    bundle: &Path,
    queries_path: &Path,
    qrels_path: &Path,
    out: &Path,
    top_k: usize,
    max_auxiliary_artifact_bytes: u64,
) -> Result<LtrFeaturesReport, Box<dyn Error>> {
    if top_k == 0 {
        return Err(Box::new(CliError("top_k must be positive".to_string())));
    }
    let out_relative = normalize_bundle_relative_dir(out)?;
    match classify_path(bundle)? {
        PathKind::CoreBundle => {}
        PathKind::AdapterDirectory | PathKind::Unknown => {
            return Err(Box::new(CliError(format!(
                "{} is not a core bundle with manifest.json",
                bundle.display()
            ))))
        }
    }

    let manifest_path = bundle.join(MANIFEST_FILE);
    let mut sparse_verify_options = OrdinalVerifyOptions::default();
    set_auxiliary_size_limit(&mut sparse_verify_options, max_auxiliary_artifact_bytes);
    let sparse = Bm25MmapIndex::open_verified_sidecar(
        &manifest_path,
        DEFAULT_SPARSE_AUX_NAME,
        sparse_verify_options,
    )?;

    let queries = read_ltr_queries(queries_path)?;
    let qrels = read_ltr_qrels(qrels_path)?;
    let feature_names = vec![
        "bm25_score".to_string(),
        "bm25_rank".to_string(),
        "query_len_chars".to_string(),
    ];

    let mut groups = Vec::with_capacity(queries.len());
    let mut row_ids = Vec::new();
    let mut labels = Vec::new();
    let mut features = Vec::new();
    let mut cache_queries = Vec::with_capacity(queries.len());
    for query in &queries {
        let hits = sparse.search(&query.query, top_k)?;
        let group_len = u32::try_from(hits.len()).map_err(|_| {
            CliError(format!(
                "query {} produced more than u32::MAX LTR candidates",
                query.query_id
            ))
        })?;
        groups.push(group_len);
        cache_queries.push(QueryCacheRecord {
            query_id: query.query_id.clone(),
            query: query.query.clone(),
        });

        let query_len = query.query.chars().count() as f32;
        for (rank_idx, hit) in hits.iter().enumerate() {
            row_ids.push(hit.row_id);
            labels.push(
                qrels
                    .get(&(query.query_id.clone(), hit.row_id))
                    .copied()
                    .unwrap_or(0.0),
            );
            features.push(hit.score);
            features.push((rank_idx + 1) as f32);
            features.push(query_len);
        }
    }

    let qrels_source_sha256 = Some(ordinaldb_ltr::sha256_path(qrels_path)?);
    let data = FeatureCacheData {
        feature_names,
        groups,
        row_ids,
        labels,
        features,
        queries: cache_queries,
        split: serde_json::json!({
            "kind": "none",
            "note": "feature export only; training split is chosen by ordinaldb ltr train"
        }),
        provenance: serde_json::json!({
            "producer": "ordinaldb-cli",
            "command": "ltr features",
            "feature_mode": "bm25_only",
            "bundle": bundle.display().to_string(),
            "queries": queries_path.display().to_string(),
            "qrels": qrels_path.display().to_string(),
            "top_k": top_k,
            "sparse_aux_name": DEFAULT_SPARSE_AUX_NAME,
            "bundle_binding": "outer ordvec-manifest auxiliary artifacts",
        }),
        label_kind: "qrels".to_string(),
        qrels_source_sha256,
        bundle_manifest_sha256: None,
        forbidden_features_present: Vec::new(),
        dense_features_present: false,
    };
    let bundle_report = write_feature_cache_bundle_auxiliary(
        bundle,
        &out_relative,
        &data,
        BundleFeatureCacheOptions {
            max_auxiliary_artifact_bytes,
            ..BundleFeatureCacheOptions::default()
        },
    )?;

    Ok(report_from_ltr_manifest(
        &bundle_report.cache_root,
        bundle_report.manifest,
    ))
}

#[cfg(feature = "experimental-ltr")]
#[derive(Debug)]
struct LtrQueryInput {
    query_id: String,
    query: String,
}

#[cfg(feature = "experimental-ltr")]
fn read_ltr_queries(path: &Path) -> Result<Vec<LtrQueryInput>, Box<dyn Error>> {
    #[derive(Deserialize)]
    struct RawQuery {
        query_id: Option<String>,
        id: Option<String>,
        query: Option<String>,
        text: Option<String>,
    }

    let file = File::open(path)?;
    let mut queries = Vec::new();
    let mut seen = HashSet::new();
    for (line_idx, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let raw: RawQuery = serde_json::from_str(&line).map_err(|error| {
            CliError(format!(
                "failed to parse query JSONL line {} in {}: {error}",
                line_idx + 1,
                path.display()
            ))
        })?;
        let query_id = raw.query_id.or(raw.id).ok_or_else(|| {
            CliError(format!(
                "query JSONL line {} in {} must contain query_id or id",
                line_idx + 1,
                path.display()
            ))
        })?;
        let query = raw.query.or(raw.text).ok_or_else(|| {
            CliError(format!(
                "query JSONL line {} in {} must contain query or text",
                line_idx + 1,
                path.display()
            ))
        })?;
        if query_id.trim().is_empty() || query.trim().is_empty() {
            return Err(Box::new(CliError(format!(
                "query JSONL line {} in {} has empty query_id or query",
                line_idx + 1,
                path.display()
            ))));
        }
        if !seen.insert(query_id.clone()) {
            return Err(Box::new(CliError(format!(
                "duplicate query id {query_id:?} in {}",
                path.display()
            ))));
        }
        queries.push(LtrQueryInput { query_id, query });
    }
    if queries.is_empty() {
        return Err(Box::new(CliError(format!(
            "{} contains no queries",
            path.display()
        ))));
    }
    Ok(queries)
}

#[cfg(feature = "experimental-ltr")]
fn read_ltr_qrels(path: &Path) -> Result<HashMap<(String, u64), f32>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut qrels = HashMap::new();
    for (line_idx, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parts = trimmed.split_whitespace().collect::<Vec<_>>();
        let (query_id, row_id_raw, label_raw) = match parts.as_slice() {
            [query_id, row_id, label] => (*query_id, *row_id, *label),
            [query_id, _iter, row_id, label] => (*query_id, *row_id, *label),
            _ => {
                return Err(Box::new(CliError(format!(
                    "qrels line {} in {} must be either '<query_id> <row_id> <label>' or TREC '<query_id> Q0 <row_id> <label>'",
                    line_idx + 1,
                    path.display()
                ))))
            }
        };
        let row_id = row_id_raw.parse::<u64>().map_err(|error| {
            CliError(format!(
                "qrels line {} in {} has invalid u64 row_id {row_id_raw:?}: {error}",
                line_idx + 1,
                path.display()
            ))
        })?;
        let label = label_raw.parse::<f32>().map_err(|error| {
            CliError(format!(
                "qrels line {} in {} has invalid f32 label {label_raw:?}: {error}",
                line_idx + 1,
                path.display()
            ))
        })?;
        if !label.is_finite() {
            return Err(Box::new(CliError(format!(
                "qrels line {} in {} has non-finite label",
                line_idx + 1,
                path.display()
            ))));
        }
        qrels.insert((query_id.to_string(), row_id), label);
    }
    Ok(qrels)
}

#[cfg(feature = "experimental-ltr")]
fn report_from_ltr_manifest(path: &Path, manifest: FeatureCacheManifest) -> LtrFeaturesReport {
    LtrFeaturesReport {
        path: path.display().to_string(),
        feature_names: manifest.feature_names,
        query_count: manifest.query_count,
        row_count: manifest.row_count,
        group_count: manifest.group_count,
        dense_features_present: manifest.dense_features_present,
        bundle_manifest_sha256: manifest.bundle_manifest_sha256,
        qrels_source_sha256: manifest.qrels_source_sha256,
    }
}

#[cfg(feature = "experimental-ltr")]
fn emit_ltr_features(report: &LtrFeaturesReport, as_json: bool) -> Result<(), Box<dyn Error>> {
    if as_json {
        println!("{}", serde_json::to_string_pretty(report)?);
    } else {
        println!("wrote LTR feature cache: {}", report.path);
        println!("queries: {}", report.query_count);
        println!("rows: {}", report.row_count);
        println!("groups: {}", report.group_count);
        println!("features: {}", report.feature_names.join(", "));
        println!("dense_features_present: {}", report.dense_features_present);
    }
    Ok(())
}

fn inspect_path(path: &Path) -> Result<InspectReport, Box<dyn Error>> {
    match classify_path(path)? {
        PathKind::CoreBundle => inspect_core_bundle(path),
        PathKind::AdapterDirectory => inspect_adapter_directory(path),
        PathKind::Unknown => unreachable!("classify_path never returns unknown"),
    }
}

fn verify_path(path: &Path) -> Result<VerifyReport, Box<dyn Error>> {
    let path_display = path.display().to_string();
    let kind = match classify_path(path) {
        Ok(kind) => kind,
        Err(err) => {
            return Ok(VerifyReport {
                kind: PathKind::Unknown,
                path: path_display,
                valid: false,
                error: Some(err.to_string()),
                warnings: Vec::new(),
            })
        }
    };
    let result = match kind {
        PathKind::CoreBundle => verify_core_bundle(path),
        PathKind::AdapterDirectory => verify_adapter_directory(path),
        PathKind::Unknown => unreachable!("classify_path never returns unknown"),
    };
    Ok(match result {
        Ok(warnings) => VerifyReport {
            kind,
            path: path_display,
            valid: true,
            error: None,
            warnings,
        },
        Err(err) => VerifyReport {
            kind,
            path: path_display,
            valid: false,
            error: Some(err.to_string()),
            warnings: Vec::new(),
        },
    })
}

fn stats_path(path: &Path) -> Result<StatsReport, Box<dyn Error>> {
    match classify_path(path)? {
        PathKind::CoreBundle => stats_core_bundle(path),
        PathKind::AdapterDirectory => stats_adapter_directory(path),
        PathKind::Unknown => unreachable!("classify_path never returns unknown"),
    }
}

fn classify_path(path: &Path) -> Result<PathKind, Box<dyn Error>> {
    if !path.exists() {
        return Err(Box::new(CliError(format!(
            "path {} does not exist",
            path.display()
        ))));
    }
    if path.join(MANIFEST_FILE).is_file() {
        return Ok(PathKind::CoreBundle);
    }
    if path.join(ADAPTER_STORE_FILE).is_file() || valid_legacy_adapter_marker(path)? {
        return Ok(PathKind::AdapterDirectory);
    }
    Err(Box::new(CliError(format!(
        "{} is neither a core .odb bundle nor an adapter directory",
        path.display()
    ))))
}

fn valid_legacy_adapter_marker(path: &Path) -> Result<bool, Box<dyn Error>> {
    let adapter_path = path.join(ADAPTER_FILE);
    match std::fs::symlink_metadata(&adapter_path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            return Err(Box::new(CliError(format!(
                "{ADAPTER_FILE} must not be a symlink: {}",
                adapter_path.display()
            ))));
        }
        Ok(metadata) if !metadata.is_file() => {
            return Err(Box::new(CliError(format!(
                "{ADAPTER_FILE} must be a file: {}",
                adapter_path.display()
            ))));
        }
        Ok(_) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(err) => return Err(Box::new(err)),
    }

    let adapter = read_json(&adapter_path)?;
    require_legacy_adapter_marker_shape(&adapter)?;
    Ok(true)
}

fn require_legacy_adapter_marker_shape(adapter: &Value) -> Result<(), Box<dyn Error>> {
    require_exact_object(
        adapter,
        &[
            "schema_version",
            "adapter",
            "bits",
            "dim",
            "empty_lazy",
            "index_path",
            "sidecars",
        ],
        ADAPTER_FILE,
    )?;
    require_schema(adapter, ADAPTER_SCHEMA_VERSION, ADAPTER_FILE)?;
    required_str(adapter, "adapter")?;
    required_bits(adapter)?;
    optional_dim(adapter)?;
    required_bool(adapter, "empty_lazy")?;
    validate_index_path(required_str(adapter, "index_path")?)?;
    let sidecars = required_object(adapter, "sidecars")?;
    for name in [ID_MAP_FILE, DOCUMENTS_FILE, METADATA_FILE] {
        let expected = sidecars
            .get(name)
            .ok_or_else(|| CliError(format!("adapter sidecars missing {name}")))?;
        validate_sidecar_descriptor(expected, name)?;
    }
    if sidecars.len() != 3 {
        return Err(Box::new(CliError(
            "adapter sidecars must contain exactly id_map, documents, and metadata".to_string(),
        )));
    }
    Ok(())
}

fn stats_core_bundle(path: &Path) -> Result<StatsReport, Box<dyn Error>> {
    let inspect = inspect_core_bundle(path)?;
    let mut bytes_by_component = BTreeMap::new();
    bytes_by_component.insert("total".to_string(), path_size_bytes(path)?);
    bytes_by_component.insert(
        "manifest".to_string(),
        optional_path_size_bytes(&path.join(MANIFEST_FILE))?,
    );
    Ok(StatsReport {
        kind: PathKind::CoreBundle,
        path: path.display().to_string(),
        vector_count: inspect.vector_count,
        active_id_count: inspect.vector_count,
        adapter_generations: inspect.adapter_generations,
        bytes_by_component,
    })
}

fn stats_adapter_directory(path: &Path) -> Result<StatsReport, Box<dyn Error>> {
    let inspect = inspect_adapter_directory(path)?;
    let active_id_count = if path.join(ADAPTER_STORE_FILE).is_file() {
        let store = open_verified_adapter_store(path, None)?;
        let id_map = parse_redb_payload(&store.payloads.id_map_json, ID_MAP_FILE)?;
        active_id_count_from_id_map(&id_map)?
    } else {
        let id_map = read_json(path.join(ID_MAP_FILE))?;
        active_id_count_from_id_map(&id_map)?
    };

    let mut bytes_by_component = BTreeMap::new();
    bytes_by_component.insert("total".to_string(), path_size_bytes(path)?);
    bytes_by_component.insert(
        "adapter_state".to_string(),
        optional_path_size_bytes(&path.join(ADAPTER_STORE_FILE))?,
    );
    bytes_by_component.insert(
        "compatibility_exports".to_string(),
        optional_path_size_bytes(&path.join(ADAPTER_FILE))?
            + optional_path_size_bytes(&path.join(ID_MAP_FILE))?
            + optional_path_size_bytes(&path.join(DOCUMENTS_FILE))?
            + optional_path_size_bytes(&path.join(METADATA_FILE))?,
    );
    bytes_by_component.insert(
        "vectors".to_string(),
        optional_path_size_bytes(&path.join(VECTORS_DIR))?
            + optional_path_size_bytes(&path.join(INDEX_DIR))?,
    );

    Ok(StatsReport {
        kind: PathKind::AdapterDirectory,
        path: path.display().to_string(),
        vector_count: inspect.vector_count,
        active_id_count,
        adapter_generations: inspect.adapter_generations,
        bytes_by_component,
    })
}

fn inspect_core_bundle(path: &Path) -> Result<InspectReport, Box<dyn Error>> {
    let plan = verify_for_load(path.join(MANIFEST_FILE), VerifyOptions::default())?;
    let metadata = plan.metadata();
    let bits = match metadata.params {
        ManifestIndexParams::RankQuant { bits } => Some(bits),
        _ => None,
    };
    Ok(InspectReport {
        kind: PathKind::CoreBundle,
        path: path.display().to_string(),
        schema_version: None,
        adapter: None,
        bits,
        dim: Some(metadata.dim),
        vector_count: metadata.vector_count,
        empty_lazy: false,
        sidecar_count: plan.auxiliary_artifacts().len(),
        active_generation_id: None,
        active_generation_path: None,
        active_generation_manifest_sha256: None,
        active_generation_manifest_size_bytes: None,
        adapter_generations: None,
    })
}

fn inspect_adapter_directory(path: &Path) -> Result<InspectReport, Box<dyn Error>> {
    if path.join(ADAPTER_STORE_FILE).is_file() {
        let store = open_verified_adapter_store(path, None)?;
        let active_generation_path = store
            .active_generation_path()
            .filter(|path| !path.is_empty())
            .map(str::to_string);
        let generation_summary =
            generation_directory_summary(path, active_generation_path.as_deref())?;
        return Ok(InspectReport {
            kind: PathKind::AdapterDirectory,
            path: path.display().to_string(),
            schema_version: Some(ADAPTER_STORE_SCHEMA_VERSION.to_string()),
            adapter: store.adapter_name().map(str::to_string),
            bits: store.bits(),
            dim: store.dim(),
            vector_count: store.vector_count().unwrap_or(0),
            empty_lazy: store.empty_lazy(),
            sidecar_count: 1,
            active_generation_id: store.active_generation_id(),
            active_generation_path,
            active_generation_manifest_sha256: store
                .active_generation_manifest_sha256()
                .map(str::to_string),
            active_generation_manifest_size_bytes: store.active_generation_manifest_size_bytes(),
            adapter_generations: Some(AdapterGenerationsReport::from_summary(generation_summary)),
        });
    }

    let adapter = read_json(path.join(ADAPTER_FILE))?;
    require_exact_object(
        &adapter,
        &[
            "schema_version",
            "adapter",
            "bits",
            "dim",
            "empty_lazy",
            "index_path",
            "sidecars",
        ],
        ADAPTER_FILE,
    )?;
    require_schema(&adapter, ADAPTER_SCHEMA_VERSION, ADAPTER_FILE)?;
    let empty_lazy = required_bool(&adapter, "empty_lazy")?;
    let adapter_name = required_str(&adapter, "adapter")?.to_string();
    let bits = required_bits(&adapter)?;
    let dim = optional_dim(&adapter)?;
    let sidecar_count = required_object(&adapter, "sidecars")?.len();
    let index_path = required_str(&adapter, "index_path")?;
    validate_index_path(index_path)?;
    let active_generation_id = if empty_lazy {
        Some(0)
    } else {
        Some(generation_id_from_index_path(index_path)?)
    };
    let active_generation_path = if empty_lazy {
        None
    } else {
        Some(index_path.to_string())
    };
    let generation_summary = generation_directory_summary(path, active_generation_path.as_deref())?;
    let vector_count = if empty_lazy {
        0
    } else {
        let manifest_path = active_generation_manifest_path(path, index_path)?;
        validate_generation_manifest_file(&manifest_path)?;
        let plan = verify_for_load(manifest_path, VerifyOptions::default())?;
        plan.metadata().vector_count
    };
    Ok(InspectReport {
        kind: PathKind::AdapterDirectory,
        path: path.display().to_string(),
        schema_version: Some(ADAPTER_SCHEMA_VERSION.to_string()),
        adapter: Some(adapter_name),
        bits: Some(bits),
        dim,
        vector_count,
        empty_lazy,
        sidecar_count,
        active_generation_id,
        active_generation_path,
        active_generation_manifest_sha256: None,
        active_generation_manifest_size_bytes: None,
        adapter_generations: Some(AdapterGenerationsReport::from_summary(generation_summary)),
    })
}

fn verify_core_bundle(path: &Path) -> Result<Vec<String>, Box<dyn Error>> {
    let plan = verify_for_load(path.join(MANIFEST_FILE), VerifyOptions::default())?;
    require_ordinaldb_manifest_shape(
        plan.metadata().params,
        plan.row_identity().kind(),
        "core bundle",
    )?;
    verify_recognized_auxiliary_sidecars(path, &plan)
}

/// Structurally validate each recognized hybrid/LTR auxiliary family through
/// its domain loader, on top of the manifest's sha256/size verification. A
/// sidecar whose bytes were tampered with *and* whose manifest hash was
/// re-patched passes checksum verification but must still fail here.
#[cfg(feature = "experimental-ltr")]
fn verify_recognized_auxiliary_sidecars(
    path: &Path,
    plan: &ordvec_manifest::VerifiedLoadPlan,
) -> Result<Vec<String>, Box<dyn Error>> {
    if plan.auxiliary_by_name(DEFAULT_SPARSE_AUX_NAME).is_some() {
        Bm25MmapIndex::open_from_verified_plan_unchecked_freshness(plan, DEFAULT_SPARSE_AUX_NAME)?;
    }
    if plan.auxiliary_by_name(LTR_MODEL_AUX_NAME).is_some() {
        TreeEnsembleReranker::load_from_verified_plan_unchecked_freshness(
            plan,
            LTR_MODEL_AUX_NAME,
            LtrLoadOptions::default(),
        )?;
    }
    if plan
        .auxiliary_by_name(DEFAULT_LTR_FEATURE_CACHE_AUX_NAME)
        .is_some()
    {
        read_verified_feature_cache_bundle_auxiliary(path, &BundleFeatureCacheOptions::default())?;
    }
    Ok(Vec::new())
}

/// Without `--features experimental-ltr` the domain loaders are not compiled
/// in, so recognized hybrid/LTR sidecars pass only the manifest checksum
/// layer. Disclose that gap as a warning instead of silently reporting the
/// sidecars as fully validated.
#[cfg(not(feature = "experimental-ltr"))]
fn verify_recognized_auxiliary_sidecars(
    _path: &Path,
    plan: &ordvec_manifest::VerifiedLoadPlan,
) -> Result<Vec<String>, Box<dyn Error>> {
    // Keep in sync with ordinaldb_hybrid::DEFAULT_SPARSE_AUX_NAME,
    // ordinaldb_hybrid::DEFAULT_LTR_MODEL_AUX_NAME, and
    // ordinaldb_ltr::DEFAULT_LTR_FEATURE_CACHE_AUX_NAME. The typed constants
    // live behind the experimental-ltr feature, so this build spells them out.
    const RECOGNIZED_AUX_NAMES: [&str; 3] = [
        "ordinaldb.sparse_bm25",
        "ordinaldb.ltr_model",
        "ordinaldb.ltr_features",
    ];
    let mut warnings = Vec::new();
    for name in RECOGNIZED_AUX_NAMES {
        if plan.auxiliary_by_name(name).is_some() {
            warnings.push(format!(
                "auxiliary {name:?} passed manifest checksum verification but was \
                 not structurally validated (ordinaldb-cli was built without \
                 --features experimental-ltr)"
            ));
        }
    }
    Ok(warnings)
}

fn verify_adapter_directory(path: &Path) -> Result<Vec<String>, Box<dyn Error>> {
    if path.join(ADAPTER_STORE_FILE).is_file() {
        let store = open_verified_adapter_store(path, None)?;
        let summary = generation_directory_summary(
            path,
            store
                .active_generation_path()
                .filter(|generation_path| !generation_path.is_empty()),
        )?;
        return Ok(summary.debris_warnings);
    }

    let adapter = read_json(path.join(ADAPTER_FILE))?;
    require_exact_object(
        &adapter,
        &[
            "schema_version",
            "adapter",
            "bits",
            "dim",
            "empty_lazy",
            "index_path",
            "sidecars",
        ],
        ADAPTER_FILE,
    )?;
    require_schema(&adapter, ADAPTER_SCHEMA_VERSION, ADAPTER_FILE)?;
    let bits = required_bits(&adapter)?;
    let dim = optional_dim(&adapter)?;
    let empty_lazy = required_bool(&adapter, "empty_lazy")?;
    let index_path = required_str(&adapter, "index_path")?;
    validate_index_path(index_path)?;
    let active_generation_path = if empty_lazy { None } else { Some(index_path) };
    let sidecars = required_object(&adapter, "sidecars")?;
    for name in [ID_MAP_FILE, DOCUMENTS_FILE, METADATA_FILE] {
        let expected = sidecars
            .get(name)
            .ok_or_else(|| CliError(format!("adapter sidecars missing {name}")))?;
        verify_sidecar(path.join(name), expected, name)?;
    }
    if sidecars.len() != 3 {
        return Err(Box::new(CliError(
            "adapter sidecars must contain exactly id_map, documents, and metadata".to_string(),
        )));
    }

    let id_map = read_json(path.join(ID_MAP_FILE))?;
    let documents = read_json(path.join(DOCUMENTS_FILE))?;
    let metadata = read_json(path.join(METADATA_FILE))?;
    require_exact_object(
        &id_map,
        &[
            "schema_version",
            "next_u64_id",
            "string_to_u64",
            "u64_to_slot",
        ],
        ID_MAP_FILE,
    )?;
    require_exact_object(&documents, &["schema_version", "documents"], DOCUMENTS_FILE)?;
    require_exact_object(&metadata, &["schema_version", "metadata"], METADATA_FILE)?;
    require_schema(&id_map, ID_MAP_SCHEMA_VERSION, ID_MAP_FILE)?;
    require_schema(&documents, DOCUMENTS_SCHEMA_VERSION, DOCUMENTS_FILE)?;
    require_schema(&metadata, METADATA_SCHEMA_VERSION, METADATA_FILE)?;

    let string_to_u64 = required_object(&id_map, "string_to_u64")?;
    let u64_to_slot = required_object(&id_map, "u64_to_slot")?;
    let next_u64_id = required_u64(&id_map, "next_u64_id")?;
    let document_map = required_object(&documents, "documents")?;
    let metadata_map = required_object(&metadata, "metadata")?;

    if empty_lazy {
        if dim.is_some() {
            return Err(Box::new(CliError(
                "empty_lazy adapter must have dim=null".to_string(),
            )));
        }
        reject_empty_lazy_vector_artifacts(path, index_path)?;
    }
    let summary = generation_directory_summary(path, active_generation_path)?;

    // Empty-lazy adapters are validated above and intentionally have no core index yet.
    let mut index_len = 0usize;
    if !empty_lazy {
        let manifest_path = active_generation_manifest_path(path, index_path)?;
        validate_generation_manifest_file(&manifest_path)?;
        let plan = verify_for_load(manifest_path, VerifyOptions::default())?;
        let index_metadata = plan.metadata();
        let index_bits = require_ordinaldb_manifest_shape(
            index_metadata.params,
            plan.row_identity().kind(),
            "adapter index manifest",
        )?;
        if index_bits != bits {
            return Err(Box::new(CliError(
                "adapter bits do not match index bits".to_string(),
            )));
        }
        let dim = dim.ok_or_else(|| {
            Box::new(CliError(
                "non-empty adapter must have a non-null dim".to_string(),
            )) as Box<dyn Error>
        })?;
        if index_metadata.dim != dim {
            return Err(Box::new(CliError(
                "adapter dim does not match index dim".to_string(),
            )));
        }
        index_len = index_metadata.vector_count;
    }

    verify_adapter_maps(
        string_to_u64,
        u64_to_slot,
        document_map,
        metadata_map,
        next_u64_id,
        index_len,
    )?;
    Ok(summary.debris_warnings)
}

fn parse_redb_payload(payload: &str, name: &str) -> Result<Value, Box<dyn Error>> {
    serde_json::from_str(payload).map_err(|err| {
        Box::new(CliError(format!(
            "failed to parse adapter.redb payload {name}: {err}"
        ))) as Box<dyn Error>
    })
}

fn active_id_count_from_id_map(id_map: &Value) -> Result<usize, Box<dyn Error>> {
    Ok(required_object(id_map, "string_to_u64")?.len())
}

fn optional_path_size_bytes(path: &Path) -> Result<u64, Box<dyn Error>> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => path_size_bytes(path),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(0),
        Err(err) => Err(Box::new(CliError(format!(
            "failed to stat {}: {err}",
            path.display()
        )))),
    }
}

fn path_size_bytes(path: &Path) -> Result<u64, Box<dyn Error>> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|err| CliError(format!("failed to stat {}: {err}", path.display())))?;
    if metadata.file_type().is_symlink() {
        return Err(Box::new(CliError(format!(
            "stats path must not contain a symlink: {}",
            path.display()
        ))));
    }
    if metadata.file_type().is_file() {
        return Ok(metadata.len());
    }
    if metadata.file_type().is_dir() {
        let mut total = 0;
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            total += path_size_bytes(&entry.path())?;
        }
        return Ok(total);
    }
    Ok(0)
}

fn validate_index_path(value: &str) -> Result<(), Box<dyn Error>> {
    validate_relative_path(value)?;
    if value == INDEX_DIR {
        return Ok(());
    }

    let mut components = Path::new(value).components();
    let Some(Component::Normal(first)) = components.next() else {
        return Err(Box::new(invalid_index_path(value)));
    };
    let Some(Component::Normal(second)) = components.next() else {
        return Err(Box::new(invalid_index_path(value)));
    };
    if components.next().is_some() {
        return Err(Box::new(invalid_index_path(value)));
    }
    if first != VECTORS_DIR || parse_generation_dir(second.to_string_lossy().as_ref()).is_none() {
        return Err(Box::new(invalid_index_path(value)));
    }
    Ok(())
}

fn generation_id_from_index_path(value: &str) -> Result<u64, Box<dyn Error>> {
    validate_index_path(value)?;
    if value == INDEX_DIR {
        return Ok(1);
    }
    let mut components = Path::new(value).components();
    let _ = components.next();
    let Some(Component::Normal(generation_dir)) = components.next() else {
        return Err(Box::new(invalid_index_path(value)));
    };
    parse_generation_dir(generation_dir.to_string_lossy().as_ref())
        .ok_or_else(|| Box::new(invalid_index_path(value)) as Box<dyn Error>)
}

fn generation_directory_summary(
    root: &Path,
    active_generation_path: Option<&str>,
) -> Result<GenerationDirectorySummary, Box<dyn Error>> {
    // `scan_generation_directory` classifies every `vectors/` entry as a
    // committed generation bundle or reclaimable debris (crash-interrupted
    // scratch directories, stray files, anything else non-canonical). Debris
    // is never fatal there — only symlinked entries and non-directory
    // entries that carry a *canonical* generation name still fail closed.
    let scan = scan_generation_directory(root)?;

    let mut completed_generation_paths = scan.generation_paths;
    completed_generation_paths.sort();
    let mut partial_generation_paths: Vec<String> =
        scan.debris.iter().map(|entry| entry.path.clone()).collect();
    partial_generation_paths.sort();
    let debris_warnings: Vec<String> = scan.debris.into_iter().map(|entry| entry.warning).collect();

    let active_generation_count = active_generation_path
        .filter(|active| completed_generation_paths.iter().any(|path| path == active))
        .map_or(0, |_| 1);
    let retained_generation_paths = completed_generation_paths
        .iter()
        .filter(|path| Some(path.as_str()) != active_generation_path)
        .cloned()
        .collect::<Vec<_>>();
    let reclaimable_generation_paths = partial_generation_paths.clone();
    let mut orphan_generation_paths = retained_generation_paths.clone();
    orphan_generation_paths.extend(partial_generation_paths.clone());
    orphan_generation_paths.sort();
    Ok(GenerationDirectorySummary {
        generation_count: completed_generation_paths.len() + partial_generation_paths.len(),
        active_generation_count,
        completed_generation_count: completed_generation_paths.len(),
        retained_generation_paths,
        partial_generation_paths,
        reclaimable_generation_paths,
        orphan_generation_paths,
        debris_warnings,
    })
}

fn validate_generation_manifest_file(path: &Path) -> Result<(), Box<dyn Error>> {
    let metadata = validate_regular_file(path, "generation manifest")?;
    if metadata.len() > MAX_GENERATION_MANIFEST_BYTES {
        return Err(Box::new(CliError(format!(
            "generation manifest too large: {} bytes exceeds {}",
            metadata.len(),
            MAX_GENERATION_MANIFEST_BYTES
        ))));
    }
    Ok(())
}

fn active_generation_manifest_path(
    root: &Path,
    index_path: &str,
) -> Result<PathBuf, Box<dyn Error>> {
    validate_index_path(index_path)?;
    let mut current = root.to_path_buf();
    for component in Path::new(index_path).components() {
        let Component::Normal(part) = component else {
            return Err(Box::new(invalid_index_path(index_path)));
        };
        current.push(part);
        let metadata = std::fs::symlink_metadata(&current)?;
        if metadata.file_type().is_symlink() {
            return Err(Box::new(CliError(format!(
                "active generation path must not contain a symlink: {}",
                current.display()
            ))));
        }
        if !metadata.file_type().is_dir() {
            return Err(Box::new(CliError(format!(
                "active generation path component must be a directory: {}",
                current.display()
            ))));
        }
    }
    Ok(current.join(MANIFEST_FILE))
}

fn validate_relative_path(value: &str) -> Result<(), Box<dyn Error>> {
    if value.is_empty() {
        return Err(Box::new(CliError(
            "active generation path must not be empty".to_string(),
        )));
    }
    if value.contains('\\') || value.contains("//") || value.ends_with('/') {
        return Err(Box::new(CliError(
            "active generation path must use normalized forward-slash components".to_string(),
        )));
    }
    let path = Path::new(value);
    if path.is_absolute() {
        return Err(Box::new(CliError(
            "active generation path must be relative".to_string(),
        )));
    }
    for component in path.components() {
        match component {
            Component::Normal(_) => {}
            _ => {
                return Err(Box::new(CliError(
                    "active generation path must not contain parent or special components"
                        .to_string(),
                )))
            }
        }
    }
    Ok(())
}

fn invalid_index_path(value: &str) -> CliError {
    CliError(format!(
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

fn parse_partial_generation_dir(name: &str) -> Option<u64> {
    let rest = name.strip_prefix('.')?;
    let (generation, suffix) = rest.split_once(".tmp-")?;
    if suffix.is_empty() {
        return None;
    }
    parse_generation_dir(generation)
}

fn reject_empty_lazy_vector_artifacts(root: &Path, index_path: &str) -> Result<(), Box<dyn Error>> {
    let mut seen = HashSet::new();
    for relative in [index_path, INDEX_DIR, VECTORS_DIR] {
        let path = root.join(relative);
        if !seen.insert(path.clone()) {
            continue;
        }
        match path.try_exists() {
            Ok(true) => {
                return Err(Box::new(CliError(format!(
                    "empty_lazy adapter must not contain {relative}"
                ))));
            }
            Ok(false) => {}
            Err(err) => {
                return Err(Box::new(CliError(format!(
                    "failed to stat {}: {err}",
                    path.display()
                ))));
            }
        }
    }
    Ok(())
}

fn require_ordinaldb_manifest_shape(
    params: ManifestIndexParams,
    row_identity_kind: &str,
    context: &str,
) -> Result<u8, Box<dyn Error>> {
    let ManifestIndexParams::RankQuant { bits } = params else {
        return Err(Box::new(CliError(format!(
            "{context} must describe a RankQuant index"
        ))));
    };
    if row_identity_kind != ROW_IDENTITY_KIND {
        return Err(Box::new(CliError(format!(
            "{context} must use {ROW_IDENTITY_KIND} row identity; got {row_identity_kind:?}"
        ))));
    }
    Ok(bits)
}

fn verify_adapter_maps(
    string_to_u64: &serde_json::Map<String, Value>,
    u64_to_slot: &serde_json::Map<String, Value>,
    documents: &serde_json::Map<String, Value>,
    metadata: &serde_json::Map<String, Value>,
    next_u64_id: u64,
    index_len: usize,
) -> Result<(), Box<dyn Error>> {
    if !same_keys(documents, string_to_u64) {
        return Err(Box::new(CliError(
            "documents keys do not match string_to_u64 keys".to_string(),
        )));
    }
    if !same_keys(metadata, string_to_u64) {
        return Err(Box::new(CliError(
            "metadata keys do not match string_to_u64 keys".to_string(),
        )));
    }
    if string_to_u64.len() != index_len || u64_to_slot.len() != index_len {
        return Err(Box::new(CliError(format!(
            "id_map count does not match index len {index_len}"
        ))));
    }

    let mut seen_u64 = HashSet::with_capacity(string_to_u64.len());
    for (string_id, value) in string_to_u64 {
        require_non_empty_string_id(string_id)?;
        let u64_id = value
            .as_u64()
            .ok_or_else(|| CliError("string_to_u64 values must be u64".to_string()))?;
        if !seen_u64.insert(u64_id) {
            return Err(Box::new(CliError(format!("duplicate u64 id {u64_id}"))));
        }
        if u64_id >= next_u64_id {
            return Err(Box::new(CliError(
                "next_u64_id must be greater than all allocated IDs".to_string(),
            )));
        }
    }
    for (string_id, document) in documents {
        require_non_empty_string_id(string_id)?;
        if !document.is_string() {
            return Err(Box::new(CliError(format!(
                "document for string ID {string_id:?} must be a string"
            ))));
        }
    }
    for (string_id, metadata_value) in metadata {
        require_non_empty_string_id(string_id)?;
        if !metadata_value.is_object() {
            return Err(Box::new(CliError(format!(
                "metadata for string ID {string_id:?} must be an object"
            ))));
        }
    }

    let mut seen_slots = HashSet::with_capacity(u64_to_slot.len());
    let mut actual_u64 = HashSet::with_capacity(u64_to_slot.len());
    for (key, value) in u64_to_slot {
        let u64_id = key
            .parse::<u64>()
            .map_err(|_| CliError(format!("u64_to_slot key {key:?} is not a u64")))?;
        let slot_u64 = value
            .as_u64()
            .ok_or_else(|| CliError("u64_to_slot values must be slots".to_string()))?;
        let slot = usize::try_from(slot_u64).map_err(|_| {
            CliError(format!(
                "u64 id {u64_id} points at oversized slot {slot_u64}"
            ))
        })?;
        if slot >= index_len {
            return Err(Box::new(CliError(format!(
                "u64 id {u64_id} points at stale slot {slot}"
            ))));
        }
        if !seen_slots.insert(slot) {
            return Err(Box::new(CliError(format!("duplicate vector slot {slot}"))));
        }
        actual_u64.insert(u64_id);
    }
    if actual_u64 != seen_u64 {
        return Err(Box::new(CliError(
            "u64_to_slot keys do not match string_to_u64 values".to_string(),
        )));
    }
    Ok(())
}

fn read_json(path: impl AsRef<Path>) -> Result<Value, Box<dyn Error>> {
    let path = path.as_ref();
    validate_regular_file(path, "JSON sidecar")?;
    let file = File::open(path)
        .map_err(|err| CliError(format!("failed to open {}: {err}", path.display())))?;
    let reader = BufReader::new(file);
    let mut deserializer = serde_json::Deserializer::from_reader(reader);
    let value = NoDuplicateValue::deserialize(&mut deserializer)
        .map_err(|err| CliError(format!("failed to parse JSON in {}: {err}", path.display())))?
        .0;
    deserializer
        .end()
        .map_err(|err| CliError(format!("trailing data in {}: {err}", path.display())))?;
    Ok(value)
}

fn require_schema(value: &Value, expected: &str, name: &str) -> Result<(), Box<dyn Error>> {
    let schema = required_str(value, "schema_version")?;
    if schema != expected {
        return Err(Box::new(CliError(format!(
            "{name} has unsupported schema {schema:?}"
        ))));
    }
    Ok(())
}

fn same_keys(left: &Map<String, Value>, right: &Map<String, Value>) -> bool {
    left.len() == right.len() && left.keys().all(|key| right.contains_key(key))
}

fn require_exact_object<'a>(
    value: &'a Value,
    expected: &[&str],
    name: &str,
) -> Result<&'a Map<String, Value>, Box<dyn Error>> {
    let object = value
        .as_object()
        .ok_or_else(|| CliError(format!("{name} must be a JSON object")))?;
    if object.len() != expected.len() || expected.iter().any(|key| !object.contains_key(*key)) {
        let mut actual = object.keys().map(String::as_str).collect::<Vec<_>>();
        actual.sort_unstable();
        return Err(Box::new(CliError(format!(
            "{name} has invalid keys: expected exactly [{}], got [{}]",
            expected.join(", "),
            actual.join(", ")
        ))));
    }
    Ok(object)
}

fn require_non_empty_string_id(value: &str) -> Result<(), Box<dyn Error>> {
    if value.is_empty() {
        return Err(Box::new(CliError(
            "string IDs must be non-empty".to_string(),
        )));
    }
    Ok(())
}

fn verify_sidecar(path: PathBuf, expected: &Value, name: &str) -> Result<(), Box<dyn Error>> {
    let (expected_sha, expected_size) = validate_sidecar_descriptor(expected, name)?;
    validate_regular_file(&path, name)?;
    let actual = sha256_file(&path).map_err(|err| {
        CliError(format!(
            "failed to read sidecar file {name} at {}: {err}",
            path.display()
        ))
    })?;
    if actual.sha256 != expected_sha || actual.size_bytes != expected_size {
        return Err(Box::new(CliError(format!(
            "sidecar integrity check failed for {name}"
        ))));
    }
    Ok(())
}

fn validate_sidecar_descriptor<'a>(
    expected: &'a Value,
    name: &str,
) -> Result<(&'a str, u64), Box<dyn Error>> {
    let expected = expected
        .as_object()
        .ok_or_else(|| CliError(format!("sidecar descriptor for {name} must be an object")))?;
    let expected_sha = expected
        .get("sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| CliError(format!("sidecar descriptor for {name} missing sha256")))?;
    if expected_sha.len() != 64
        || !expected_sha
            .bytes()
            .all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
    {
        return Err(Box::new(CliError(format!(
            "sidecar descriptor for {name} has invalid sha256"
        ))));
    }
    let expected_size = expected
        .get("file_size_bytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            CliError(format!(
                "sidecar descriptor for {name} missing file_size_bytes"
            ))
        })?;
    if expected.len() != 2 {
        return Err(Box::new(CliError(format!(
            "sidecar descriptor for {name} has unexpected keys"
        ))));
    }
    Ok((expected_sha, expected_size))
}

fn validate_regular_file(path: &Path, name: &str) -> Result<std::fs::Metadata, Box<dyn Error>> {
    let metadata = std::fs::symlink_metadata(path).map_err(|err| {
        CliError(format!(
            "failed to stat {name} at {}: {err}",
            path.display()
        ))
    })?;
    if metadata.file_type().is_symlink() {
        return Err(Box::new(CliError(format!(
            "{name} must not be a symlink: {}",
            path.display()
        ))));
    }
    if !metadata.file_type().is_file() {
        return Err(Box::new(CliError(format!(
            "{name} must be a file: {}",
            path.display()
        ))));
    }
    Ok(metadata)
}

fn required_object<'a>(
    value: &'a Value,
    key: &str,
) -> Result<&'a serde_json::Map<String, Value>, Box<dyn Error>> {
    value
        .get(key)
        .and_then(Value::as_object)
        .ok_or_else(|| Box::new(CliError(format!("{key} must be an object"))) as Box<dyn Error>)
}

fn required_str<'a>(value: &'a Value, key: &str) -> Result<&'a str, Box<dyn Error>> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| Box::new(CliError(format!("{key} must be a string"))) as Box<dyn Error>)
}

fn required_bool(value: &Value, key: &str) -> Result<bool, Box<dyn Error>> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| Box::new(CliError(format!("{key} must be a boolean"))) as Box<dyn Error>)
}

fn required_bits(value: &Value) -> Result<u8, Box<dyn Error>> {
    let bits = required_u64(value, "bits")?;
    match bits {
        1 | 2 | 4 => Ok(bits as u8),
        _ => Err(Box::new(CliError(
            "bits must be one of 1, 2, or 4".to_string(),
        ))),
    }
}

fn optional_dim(value: &Value) -> Result<Option<usize>, Box<dyn Error>> {
    let Some(dim) = optional_u64(value, "dim")? else {
        return Ok(None);
    };
    if !(2..=u16::MAX as u64).contains(&dim) {
        return Err(Box::new(CliError(format!(
            "dimension {dim} is out of range: dim must be >= 2 and representable as u16"
        ))));
    }
    Ok(Some(dim as usize))
}

fn required_u64(value: &Value, key: &str) -> Result<u64, Box<dyn Error>> {
    value.get(key).and_then(Value::as_u64).ok_or_else(|| {
        Box::new(CliError(format!("{key} must be an unsigned integer"))) as Box<dyn Error>
    })
}

fn optional_u64(value: &Value, key: &str) -> Result<Option<u64>, Box<dyn Error>> {
    match value.get(key) {
        Some(Value::Null) | None => Ok(None),
        Some(value) => value.as_u64().map(Some).ok_or_else(|| {
            Box::new(CliError(format!(
                "{key} must be null or an unsigned integer"
            ))) as Box<dyn Error>
        }),
    }
}

struct NoDuplicateValue(Value);

impl<'de> Deserialize<'de> for NoDuplicateValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
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

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(Value::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(Value::Number(Number::from(value)))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(Value::Number(Number::from(value)))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(Value::String(value.to_string()))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(Value::String(value))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(Value::Null)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(Value::Null)
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        NoDuplicateValue::deserialize(deserializer).map(|value| value.0)
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = seq.next_element::<NoDuplicateValue>()? {
            values.push(value.0);
        }
        Ok(Value::Array(values))
    }

    fn visit_map<A>(self, mut access: A) -> Result<Self::Value, A::Error>
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

fn emit_inspect(report: InspectReport, as_json: bool) -> Result<(), Box<dyn Error>> {
    if as_json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print!("{}", format_inspect_text(&report));
    }
    Ok(())
}

/// Renders `inspect` human-readable output. Adapter-directory-only
/// generation bookkeeping lines are only appended when `report` actually has
/// them (i.e. never for plain core bundles), so a core bundle's text output
/// never shows misleading all-zero generation lines.
fn format_inspect_text(report: &InspectReport) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(out, "kind: {:?}", report.kind);
    let _ = writeln!(out, "path: {}", report.path);
    if let Some(adapter) = &report.adapter {
        let _ = writeln!(out, "adapter: {adapter}");
    }
    if let Some(bits) = report.bits {
        let _ = writeln!(out, "bits: {bits}");
    }
    if let Some(dim) = report.dim {
        let _ = writeln!(out, "dim: {dim}");
    }
    if let Some(generation_id) = report.active_generation_id {
        let _ = writeln!(out, "active_generation_id: {generation_id}");
    }
    if let Some(path) = &report.active_generation_path {
        let _ = writeln!(out, "active_generation_path: {path}");
    }
    if let Some(digest) = &report.active_generation_manifest_sha256 {
        let _ = writeln!(out, "active_generation_manifest_sha256: {digest}");
    }
    if let Some(size) = report.active_generation_manifest_size_bytes {
        let _ = writeln!(out, "active_generation_manifest_size_bytes: {size}");
    }
    let _ = writeln!(out, "vectors: {}", report.vector_count);
    if let Some(adapter_generations) = &report.adapter_generations {
        write_adapter_generations_text(&mut out, adapter_generations);
    }
    let _ = writeln!(out, "empty_lazy: {}", report.empty_lazy);
    let _ = writeln!(out, "sidecars: {}", report.sidecar_count);
    out
}

/// Appends the adapter-directory-only generation bookkeeping lines shared by
/// `inspect` and `stats` text output. Callers only invoke this when the
/// report actually has adapter generations (i.e. never for plain core
/// bundles), so the lines never appear as misleading zeros.
fn write_adapter_generations_text(
    out: &mut String,
    adapter_generations: &AdapterGenerationsReport,
) {
    use std::fmt::Write as _;
    let _ = writeln!(out, "generations: {}", adapter_generations.generation_count);
    let _ = writeln!(
        out,
        "active_generations: {}",
        adapter_generations.active_generation_count
    );
    let _ = writeln!(
        out,
        "completed_generations: {}",
        adapter_generations.completed_generation_count
    );
    let _ = writeln!(
        out,
        "retained_generations: {}",
        adapter_generations.retained_generation_count
    );
    let _ = writeln!(
        out,
        "partial_generations: {}",
        adapter_generations.partial_generation_count
    );
    let _ = writeln!(
        out,
        "reclaimable_generations: {}",
        adapter_generations.reclaimable_generation_count
    );
    for path in &adapter_generations.retained_generation_paths {
        let _ = writeln!(out, "retained_generation_path: {path}");
    }
    for path in &adapter_generations.partial_generation_paths {
        let _ = writeln!(out, "partial_generation_path: {path}");
    }
    for path in &adapter_generations.reclaimable_generation_paths {
        let _ = writeln!(out, "reclaimable_generation_path: {path}");
    }
    let _ = writeln!(
        out,
        "orphan_generations: {}",
        adapter_generations.orphan_generation_count
    );
    for path in &adapter_generations.orphan_generation_paths {
        let _ = writeln!(out, "orphan_generation_path: {path}");
    }
}

fn emit_verify(report: &VerifyReport, as_json: bool) -> Result<(), Box<dyn Error>> {
    if as_json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        let kind = match report.kind {
            PathKind::CoreBundle => "core bundle",
            PathKind::AdapterDirectory => "adapter directory",
            PathKind::Unknown => "unknown path",
        };
        if report.valid {
            println!("OK: {kind} {}", report.path);
        } else {
            eprintln!(
                "FAILED: {kind} {}: {}",
                report.path,
                report.error.as_deref().unwrap_or("verification failed")
            );
        }
        for warning in &report.warnings {
            println!("WARN: {warning}");
        }
    }
    Ok(())
}

fn emit_stats(report: &StatsReport, as_json: bool) -> Result<(), Box<dyn Error>> {
    if as_json {
        println!("{}", serde_json::to_string_pretty(report)?);
    } else {
        print!("{}", format_stats_text(report));
    }
    Ok(())
}

/// Renders `stats` human-readable output. See [`format_inspect_text`] for
/// why the adapter-directory-only generation lines are conditional.
fn format_stats_text(report: &StatsReport) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(out, "kind: {:?}", report.kind);
    let _ = writeln!(out, "path: {}", report.path);
    let _ = writeln!(out, "vectors: {}", report.vector_count);
    let _ = writeln!(out, "active_ids: {}", report.active_id_count);
    if let Some(adapter_generations) = &report.adapter_generations {
        write_adapter_generations_text(&mut out, adapter_generations);
    }
    for (component, bytes) in &report.bytes_by_component {
        let _ = writeln!(out, "bytes_{component}: {bytes}");
    }
    out
}

fn emit_adapter_gc(report: &AdapterGcReport, as_json: bool) -> Result<(), Box<dyn Error>> {
    if as_json {
        println!("{}", serde_json::to_string_pretty(report)?);
    } else {
        println!("adapter_gc: {}", report.path);
        println!("dry_run: {}", report.dry_run);
        println!("retain: {}", report.retain);
        if let Some(active) = &report.active_generation_path {
            println!("active_generation_path: {active}");
        }
        println!(
            "retained_generations: {}",
            report.retained_generation_paths.len()
        );
        for path in &report.retained_generation_paths {
            println!("retained_generation_path: {path}");
        }
        println!(
            "reclaimable_generations: {}",
            report.reclaimable_generation_paths.len()
        );
        for path in &report.reclaimable_generation_paths {
            println!("reclaimable_generation_path: {path}");
        }
        println!(
            "deleted_generations: {}",
            report.deleted_generation_paths.len()
        );
        for path in &report.deleted_generation_paths {
            println!("deleted_generation_path: {path}");
        }
        println!(
            "pinned_generations: {}",
            report.pinned_generation_paths.len()
        );
        for path in &report.pinned_generation_paths {
            println!("pinned_generation_path: {path}");
        }
        if let Some(sequence) = report.redb_commit_sequence {
            println!("redb_commit_sequence: {sequence}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::{SystemTime, UNIX_EPOCH};

    const VECTORS: &[f32] = &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    const GENERATION_INDEX_PATH: &str = "vectors/g000000000001.odb";
    const SECOND_GENERATION_INDEX_PATH: &str = "vectors/g000000000002.odb";
    const THIRD_GENERATION_INDEX_PATH: &str = "vectors/g000000000003.odb";
    const FOURTH_GENERATION_INDEX_PATH: &str = "vectors/g000000000004.odb";
    const PARTIAL_GENERATION_PATH: &str = "vectors/.g000000000005.odb.tmp-123";
    /// The exact double-suffixed crash-debris name left when a SIGKILL
    /// mid-generation-replacement stacked a second `.tmp-*` decoration onto
    /// an already-decorated scratch directory. Regression fixture for that
    /// crash shape.
    const DOUBLE_SUFFIXED_DEBRIS_PATH: &str =
        "vectors/..g000000000005.odb.tmp-211848-1783014389442708489.tmp-211848-1783014389442721724";

    /// Unwraps the adapter-directory-only generation block, panicking with a
    /// clear message if a test expected it to be present (e.g. for an
    /// adapter directory report) but it was `None`.
    fn adapter_generations(
        adapter_generations: &Option<AdapterGenerationsReport>,
    ) -> &AdapterGenerationsReport {
        adapter_generations
            .as_ref()
            .expect("report must include adapter generations")
    }

    fn temp_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after UNIX_EPOCH")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "ordinaldb-cli-test-{name}-{}-{nanos}",
            std::process::id()
        ))
    }

    struct TempFileGuard(PathBuf);

    impl Drop for TempFileGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    struct TempDirGuard(PathBuf);

    impl Drop for TempDirGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn write_json(path: &Path, value: &Value) {
        let bytes = serde_json::to_vec(value).expect("serialize fixture");
        std::fs::write(path, bytes).expect("write fixture");
    }

    fn write_payload_exports(path: &Path, payloads: &LegacyPayloads) {
        std::fs::create_dir_all(path).expect("create adapter fixture");
        std::fs::write(path.join(ADAPTER_FILE), &payloads.adapter_json)
            .expect("write adapter export");
        std::fs::write(path.join(ID_MAP_FILE), &payloads.id_map_json).expect("write id map export");
        std::fs::write(path.join(DOCUMENTS_FILE), &payloads.documents_json)
            .expect("write documents export");
        std::fs::write(path.join(METADATA_FILE), &payloads.metadata_json)
            .expect("write metadata export");
    }

    fn sidecar_descriptor(path: &Path) -> Value {
        let artifact = sha256_file(path).expect("hash fixture");
        json!({
            "sha256": artifact.sha256,
            "file_size_bytes": artifact.size_bytes,
        })
    }

    fn write_empty_lazy_adapter(path: &Path) {
        std::fs::create_dir_all(path).expect("create adapter fixture");

        let id_map_path = path.join(ID_MAP_FILE);
        write_json(
            &id_map_path,
            &json!({
                "schema_version": ID_MAP_SCHEMA_VERSION,
                "next_u64_id": 1,
                "string_to_u64": {},
                "u64_to_slot": {},
            }),
        );

        let documents_path = path.join(DOCUMENTS_FILE);
        write_json(
            &documents_path,
            &json!({
                "schema_version": DOCUMENTS_SCHEMA_VERSION,
                "documents": {},
            }),
        );

        let metadata_path = path.join(METADATA_FILE);
        write_json(
            &metadata_path,
            &json!({
                "schema_version": METADATA_SCHEMA_VERSION,
                "metadata": {},
            }),
        );

        let mut sidecars = Map::new();
        sidecars.insert(ID_MAP_FILE.to_string(), sidecar_descriptor(&id_map_path));
        sidecars.insert(
            DOCUMENTS_FILE.to_string(),
            sidecar_descriptor(&documents_path),
        );
        sidecars.insert(
            METADATA_FILE.to_string(),
            sidecar_descriptor(&metadata_path),
        );

        write_json(
            &path.join(ADAPTER_FILE),
            &json!({
                "schema_version": ADAPTER_SCHEMA_VERSION,
                "adapter": "test",
                "bits": 2,
                "dim": null,
                "empty_lazy": true,
                "index_path": GENERATION_INDEX_PATH,
                "sidecars": sidecars,
            }),
        );
    }

    fn empty_lazy_payloads() -> LegacyPayloads {
        LegacyPayloads {
            adapter_json: serde_json::to_string(&json!({
                "schema_version": ADAPTER_SCHEMA_VERSION,
                "adapter": "test",
                "bits": 2,
                "dim": null,
                "empty_lazy": true,
                "index_path": GENERATION_INDEX_PATH,
                "sidecars": {
                    ID_MAP_FILE: {"sha256": "", "file_size_bytes": 0},
                    DOCUMENTS_FILE: {"sha256": "", "file_size_bytes": 0},
                    METADATA_FILE: {"sha256": "", "file_size_bytes": 0},
                },
            }))
            .expect("serialize adapter payload"),
            id_map_json: serde_json::to_string(&json!({
                "schema_version": ID_MAP_SCHEMA_VERSION,
                "next_u64_id": 1,
                "string_to_u64": {},
                "u64_to_slot": {},
            }))
            .expect("serialize id map payload"),
            documents_json: serde_json::to_string(&json!({
                "schema_version": DOCUMENTS_SCHEMA_VERSION,
                "documents": {},
            }))
            .expect("serialize documents payload"),
            metadata_json: serde_json::to_string(&json!({
                "schema_version": METADATA_SCHEMA_VERSION,
                "metadata": {},
            }))
            .expect("serialize metadata payload"),
        }
    }

    fn non_empty_payloads(index_path: &str) -> LegacyPayloads {
        LegacyPayloads {
            adapter_json: serde_json::to_string(&json!({
                "schema_version": ADAPTER_SCHEMA_VERSION,
                "adapter": "test",
                "bits": 2,
                "dim": 4,
                "empty_lazy": false,
                "index_path": index_path,
                "sidecars": {
                    ID_MAP_FILE: {"sha256": "", "file_size_bytes": 0},
                    DOCUMENTS_FILE: {"sha256": "", "file_size_bytes": 0},
                    METADATA_FILE: {"sha256": "", "file_size_bytes": 0},
                },
            }))
            .expect("serialize adapter payload"),
            id_map_json: serde_json::to_string(&json!({
                "schema_version": ID_MAP_SCHEMA_VERSION,
                "next_u64_id": 3,
                "string_to_u64": {"a": 1, "b": 2},
                "u64_to_slot": {"1": 0, "2": 1},
            }))
            .expect("serialize id map payload"),
            documents_json: serde_json::to_string(&json!({
                "schema_version": DOCUMENTS_SCHEMA_VERSION,
                "documents": {"a": "alpha", "b": "beta"},
            }))
            .expect("serialize documents payload"),
            metadata_json: serde_json::to_string(&json!({
                "schema_version": METADATA_SCHEMA_VERSION,
                "metadata": {"a": {}, "b": {}},
            }))
            .expect("serialize metadata payload"),
        }
    }

    fn write_index_at(root: &Path, index_path: &str) {
        let mut index = ordinaldb::OrdinalIndex::new(4, 2).unwrap();
        index.add_2d(VECTORS, 4).unwrap();
        let index_root = root.join(index_path);
        std::fs::create_dir_all(index_root.parent().unwrap()).unwrap();
        index.write(index_root).unwrap();
    }

    /// Writes a plain core `.odb` bundle (manifest.json directly at `root`,
    /// no adapter concepts at all) via the `ordinaldb` crate API, mirroring
    /// what a downstream tool builds directly on top of the core crate.
    fn write_core_bundle(root: &Path) {
        let mut index = ordinaldb::OrdinalIndex::new(4, 2).unwrap();
        index.add_2d(VECTORS, 4).unwrap();
        index.write(root).unwrap();
    }

    /// Row IDs shared by the dense and sparse sides of the hybrid fixture.
    #[cfg(feature = "experimental-ltr")]
    const HYBRID_FIXTURE_ROW_IDS: [u64; 3] = [10_000, 10_001, 10_002];

    /// Writes a core bundle carrying both a BM25 sparse sidecar and an LTR
    /// model sidecar as manifest-verified auxiliary artifacts, mirroring the
    /// bundle shape a hybrid + LTR consumer produces.
    #[cfg(feature = "experimental-ltr")]
    fn write_hybrid_ltr_core_bundle(bundle: &Path, scratch: &Path) {
        use ordinaldb::hybrid::{SparseIndexBuilder, TokenizerKind};
        use ordinaldb::manifest::{AuxiliaryArtifactDeclaration, CreateManifestOptions};
        use ordinaldb::{BuildOptions, OrdinalIndexBuilder, SignPolicy};

        std::fs::create_dir_all(scratch).unwrap();
        let sparse_source = scratch.join("sparse.bm25");
        let mut sparse = SparseIndexBuilder::new(TokenizerKind::Simple);
        let texts = ["alpha beta", "beta gamma", "gamma delta"];
        for (idx, &row_id) in HYBRID_FIXTURE_ROW_IDS.iter().enumerate() {
            sparse.add_text(row_id, texts[idx]).unwrap();
        }
        sparse.write_mmap(&sparse_source).unwrap();

        let model_source = scratch.join("ltr_model.json");
        let model = json!({
            "schema_version": "ordinaldb.ltr.tree_ensemble.v1",
            "model_id": "cli-verify-fixture",
            "model_family": "ordinaldb_tree_ensemble",
            "training_objective": "rank:pairwise",
            "booster": "gbtree",
            "base_score": 0.0,
            "learning_rate": 1.0,
            "feature_schema": {
                "schema_version": "ordinaldb.ltr.features.v1",
                "feature_names": ["bm25_score"],
            },
            "trees": [{"nodes": [{"leaf_value": 0.5}]}],
        });
        std::fs::write(&model_source, serde_json::to_vec_pretty(&model).unwrap()).unwrap();

        let dim = 8usize;
        let mut dense = OrdinalIndexBuilder::new(
            dim,
            2,
            BuildOptions {
                sign: SignPolicy::Disabled,
            },
        )
        .unwrap();
        for (idx, &row_id) in HYBRID_FIXTURE_ROW_IDS.iter().enumerate() {
            let vector: Vec<f32> = (0..dim)
                .map(|col| ((idx + 1) * (col + 2)) as f32 / 10.0)
                .collect();
            dense.add(row_id, &vector).unwrap();
        }
        dense
            .write_verified_bundle(
                bundle,
                CreateManifestOptions::default(),
                vec![
                    AuxiliaryArtifactDeclaration::required(
                        ordinaldb::artifacts::SPARSE_BM25_AUX_NAME,
                        &sparse_source,
                        "sparse.bm25",
                    ),
                    AuxiliaryArtifactDeclaration::required(
                        ordinaldb::artifacts::LTR_MODEL_AUX_NAME,
                        &model_source,
                        "ltr_model.json",
                    ),
                ],
            )
            .unwrap();
    }

    /// Recomputes the named auxiliary artifact's sha256/size in
    /// `manifest.json` after the artifact has been tampered with, simulating
    /// an attacker (or corruption plus a well-meaning re-hash) that keeps the
    /// manifest checksum layer green.
    #[cfg(feature = "experimental-ltr")]
    fn patch_manifest_auxiliary_hash(bundle: &Path, aux_name: &str) {
        let manifest_path = bundle.join(MANIFEST_FILE);
        let mut manifest: Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        let artifacts = manifest["auxiliary_artifacts"].as_array_mut().unwrap();
        let entry = artifacts
            .iter_mut()
            .find(|artifact| artifact["name"] == aux_name)
            .unwrap_or_else(|| panic!("manifest has no auxiliary named {aux_name}"));
        let relative_path = entry["path"].as_str().unwrap().to_string();
        let hash = sha256_file(bundle.join(&relative_path)).unwrap();
        entry["sha256"] = json!(hash.sha256);
        entry["file_size_bytes"] = json!(hash.size_bytes);
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
    }

    #[cfg(feature = "experimental-ltr")]
    #[test]
    fn verify_accepts_core_bundle_with_intact_hybrid_and_ltr_sidecars() {
        let bundle = temp_path("hybrid-intact");
        let _bundle_guard = TempDirGuard(bundle.clone());
        let scratch = temp_path("hybrid-intact-src");
        let _scratch_guard = TempDirGuard(scratch.clone());
        write_hybrid_ltr_core_bundle(&bundle, &scratch);

        let report = verify_path(&bundle).expect("verification runs");
        assert!(
            report.valid,
            "intact bundle must verify: {:?}",
            report.error
        );
    }

    #[cfg(feature = "experimental-ltr")]
    #[test]
    fn verify_rejects_bit_flipped_sparse_sidecar_with_patched_manifest_hash() {
        let bundle = temp_path("hybrid-sparse-tamper");
        let _bundle_guard = TempDirGuard(bundle.clone());
        let scratch = temp_path("hybrid-sparse-tamper-src");
        let _scratch_guard = TempDirGuard(scratch.clone());
        write_hybrid_ltr_core_bundle(&bundle, &scratch);

        // Flip one bit in the first stored term ("alpha" -> "Alpha"), which
        // Bm25MmapIndex::open rejects as unnormalized term bytes.
        let sparse_path = bundle.join("sparse.bm25");
        let mut bytes = std::fs::read(&sparse_path).unwrap();
        let term_bytes_offset = u64::from_le_bytes(bytes[96..104].try_into().unwrap()) as usize;
        bytes[term_bytes_offset] ^= 0x20;
        std::fs::write(&sparse_path, bytes).unwrap();
        patch_manifest_auxiliary_hash(&bundle, DEFAULT_SPARSE_AUX_NAME);

        let report = verify_path(&bundle).expect("verification runs");
        assert!(
            !report.valid,
            "bit-flipped sparse sidecar with a patched manifest hash must not verify"
        );
        let error = report.error.expect("invalid report carries an error");
        assert!(error.contains("not normalized"), "{error}");
    }

    #[cfg(feature = "experimental-ltr")]
    #[test]
    fn verify_rejects_tampered_ltr_model_sidecar_with_patched_manifest_hash() {
        let bundle = temp_path("hybrid-model-tamper");
        let _bundle_guard = TempDirGuard(bundle.clone());
        let scratch = temp_path("hybrid-model-tamper-src");
        let _scratch_guard = TempDirGuard(scratch.clone());
        write_hybrid_ltr_core_bundle(&bundle, &scratch);

        // Single-byte corruption that keeps the JSON parseable but violates
        // the model header contract TreeEnsembleReranker enforces.
        let model_path = bundle.join("ltr_model.json");
        let contents = std::fs::read_to_string(&model_path).unwrap();
        let tampered = contents.replace("rank:pairwise", "rank:pairwisf");
        assert_ne!(contents, tampered, "fixture must contain the objective");
        std::fs::write(&model_path, tampered).unwrap();
        patch_manifest_auxiliary_hash(&bundle, ordinaldb::artifacts::LTR_MODEL_AUX_NAME);

        let report = verify_path(&bundle).expect("verification runs");
        assert!(
            !report.valid,
            "tampered LTR model sidecar with a patched manifest hash must not verify"
        );
        let error = report.error.expect("invalid report carries an error");
        assert!(error.contains("training_objective"), "{error}");
    }

    #[cfg(not(feature = "experimental-ltr"))]
    #[test]
    fn verify_warns_when_recognized_sidecars_cannot_be_structurally_validated() {
        let bundle = temp_path("hybrid-feature-off");
        let _bundle_guard = TempDirGuard(bundle.clone());
        write_core_bundle(&bundle);

        // Attach an arbitrary payload under the recognized sparse auxiliary
        // name with a correct checksum: without the experimental-ltr feature
        // the CLI cannot open it structurally and must say so.
        let sidecar_path = bundle.join("sparse.bm25");
        std::fs::write(&sidecar_path, b"not actually a sparse index").unwrap();
        let hash = sha256_file(&sidecar_path).unwrap();
        let manifest_path = bundle.join(MANIFEST_FILE);
        let mut manifest: Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        if manifest.get("auxiliary_artifacts").is_none() {
            manifest["auxiliary_artifacts"] = json!([]);
        }
        manifest["auxiliary_artifacts"]
            .as_array_mut()
            .unwrap()
            .push(json!({
                "name": "ordinaldb.sparse_bm25",
                "path": "sparse.bm25",
                "sha256": hash.sha256,
                "file_size_bytes": hash.size_bytes,
            }));
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();

        let report = verify_path(&bundle).expect("verification runs");
        assert!(report.valid, "{:?}", report.error);
        assert!(
            report
                .warnings
                .iter()
                .any(|warning| warning.contains("ordinaldb.sparse_bm25")
                    && warning.contains("not structurally validated")),
            "feature-off verify must disclose the unvalidated sidecar: {:?}",
            report.warnings
        );
    }

    #[test]
    fn adapter_import_legacy_creates_redb_store_and_revision_export() {
        let source = temp_path("legacy-source");
        let source_guard = TempDirGuard(source.clone());
        write_empty_lazy_adapter(&source);
        let output = temp_path("legacy-output");
        let output_guard = TempDirGuard(output.clone());

        import_legacy_adapter(&source, &output).expect("import legacy adapter");

        assert!(output.join(ADAPTER_STORE_FILE).is_file());
        assert!(output.join(ADAPTER_REDB_REVISION_FILE).is_file());
        let revision = read_json(output.join(ADAPTER_REDB_REVISION_FILE)).unwrap();
        assert_eq!(
            revision["schema_version"],
            json!(ADAPTER_STORE_SCHEMA_VERSION)
        );
        assert_eq!(revision["commit_sequence"], json!(1));
        assert!(revision["store_uuid"].as_str().is_some());
        let imported =
            open_verified_adapter_store(&output, Some("test")).expect("open imported redb store");
        assert_eq!(imported.manifest["origin"], json!("imported_legacy_json"));
        assert_eq!(
            imported.manifest["migrated_from_json_sidecars"],
            json!(false)
        );
        verify_adapter_directory(&output).expect("imported redb store verifies");

        drop(output_guard);
        drop(source_guard);
    }

    #[test]
    fn adapter_import_legacy_rejects_redb_source_without_creating_output() {
        let source = temp_path("legacy-source-with-redb");
        let _source_guard = TempDirGuard(source.clone());
        let payloads = empty_lazy_payloads();
        write_payload_exports(&source, &payloads);
        write_legacy_snapshot(&source, None, payloads).expect("write redb source");
        let output = temp_path("legacy-redb-output");

        let err = import_legacy_adapter(&source, &output).expect_err("redb source must fail");

        assert!(
            err.to_string().contains("already contains adapter.redb"),
            "{err}"
        );
        assert!(!output.exists());
    }

    #[test]
    fn adapter_import_legacy_removes_output_after_failed_import() {
        let source = temp_path("legacy-source-missing-sidecars");
        let _source_guard = TempDirGuard(source.clone());
        std::fs::create_dir_all(&source).expect("create partial legacy source");
        std::fs::write(source.join(ADAPTER_FILE), "{}").expect("write incomplete adapter file");
        let output = temp_path("legacy-failed-output");

        let err = import_legacy_adapter(&source, &output).expect_err("missing sidecar must fail");

        assert!(err.to_string().contains(ID_MAP_FILE), "{err}");
        assert!(!output.exists());
    }

    #[test]
    fn adapter_export_json_recreates_derived_sidecars_from_redb() {
        let source = temp_path("export-source");
        let source_guard = TempDirGuard(source.clone());
        write_empty_lazy_adapter(&source);
        let output = temp_path("export-output");
        let output_guard = TempDirGuard(output.clone());
        import_legacy_adapter(&source, &output).expect("import legacy adapter");

        std::fs::write(
            output.join(DOCUMENTS_FILE),
            r#"{"schema_version":"ordinaldb.adapter.documents.v1","documents":{"drift":"value"}}"#,
        )
        .expect("write drifted export");

        export_adapter_json(&output).expect("export redb payloads");

        let documents = read_json(output.join(DOCUMENTS_FILE)).unwrap();
        assert_eq!(documents["documents"], json!({}));
        let revision = read_json(output.join(ADAPTER_REDB_REVISION_FILE)).unwrap();
        assert_eq!(revision["commit_sequence"], json!(1));
        assert!(revision["store_uuid"].as_str().is_some());

        drop(output_guard);
        drop(source_guard);
    }

    #[test]
    fn adapter_gc_deletes_reclaimable_generations_and_records_redb_state() {
        let path = temp_path("adapter-gc");
        let _guard = TempDirGuard(path.clone());

        write_index_at(&path, GENERATION_INDEX_PATH);
        let first = write_legacy_snapshot(&path, None, non_empty_payloads(GENERATION_INDEX_PATH))
            .expect("write first generation");
        write_index_at(&path, SECOND_GENERATION_INDEX_PATH);
        let second = write_legacy_snapshot(
            &path,
            Some(StoreRevision::from_manifest(&first.manifest).unwrap()),
            non_empty_payloads(SECOND_GENERATION_INDEX_PATH),
        )
        .expect("write second generation");
        write_index_at(&path, THIRD_GENERATION_INDEX_PATH);
        let third = write_legacy_snapshot(
            &path,
            Some(StoreRevision::from_manifest(&second.manifest).unwrap()),
            non_empty_payloads(THIRD_GENERATION_INDEX_PATH),
        )
        .expect("write third generation");
        write_index_at(&path, FOURTH_GENERATION_INDEX_PATH);
        write_legacy_snapshot(
            &path,
            Some(StoreRevision::from_manifest(&third.manifest).unwrap()),
            non_empty_payloads(FOURTH_GENERATION_INDEX_PATH),
        )
        .expect("write fourth generation");
        std::fs::write(
            path.join(SECOND_GENERATION_INDEX_PATH)
                .join(GENERATION_PIN_FILE),
            b"reader-pin\n",
        )
        .expect("write pin");
        std::fs::create_dir_all(path.join(PARTIAL_GENERATION_PATH))
            .expect("create partial generation");

        let report = gc_adapter_generations(&path, 1, false).expect("run adapter gc");

        assert_eq!(
            report.active_generation_path.as_deref(),
            Some(FOURTH_GENERATION_INDEX_PATH)
        );
        assert_eq!(
            report.retained_generation_paths,
            vec![
                SECOND_GENERATION_INDEX_PATH.to_string(),
                THIRD_GENERATION_INDEX_PATH.to_string(),
            ]
        );
        assert_eq!(
            report.pinned_generation_paths,
            vec![SECOND_GENERATION_INDEX_PATH.to_string()]
        );
        assert_eq!(
            report.deleted_generation_paths,
            vec![
                PARTIAL_GENERATION_PATH.to_string(),
                GENERATION_INDEX_PATH.to_string(),
            ]
        );
        assert!(!path.join(GENERATION_INDEX_PATH).exists());
        assert!(path.join(SECOND_GENERATION_INDEX_PATH).exists());
        assert!(path.join(THIRD_GENERATION_INDEX_PATH).exists());
        assert!(path.join(FOURTH_GENERATION_INDEX_PATH).exists());
        assert!(!path.join(PARTIAL_GENERATION_PATH).exists());
        assert_eq!(report.redb_commit_sequence, Some(7));
        verify_adapter_directory(&path).expect("adapter verifies after gc");

        let stats = stats_adapter_directory(&path).expect("stats after gc");
        let stats_generations = adapter_generations(&stats.adapter_generations);
        assert_eq!(stats_generations.active_generation_count, 1);
        assert_eq!(stats_generations.completed_generation_count, 3);
        assert_eq!(stats_generations.retained_generation_count, 2);
        assert_eq!(stats_generations.partial_generation_count, 0);
        assert_eq!(stats_generations.reclaimable_generation_count, 0);
    }

    #[test]
    fn adapter_gc_recovers_generation_left_deleting_after_crash() {
        let path = temp_path("adapter-gc-recover-deleting");
        let _guard = TempDirGuard(path.clone());

        write_index_at(&path, GENERATION_INDEX_PATH);
        let first = write_legacy_snapshot(&path, None, non_empty_payloads(GENERATION_INDEX_PATH))
            .expect("write first generation");
        write_index_at(&path, SECOND_GENERATION_INDEX_PATH);
        let second = write_legacy_snapshot(
            &path,
            Some(StoreRevision::from_manifest(&first.manifest).unwrap()),
            non_empty_payloads(SECOND_GENERATION_INDEX_PATH),
        )
        .expect("write second generation");

        let writer_lock = acquire_writer_lock(&path).expect("acquire writer lock");
        let queued = record_generation_gc_with_existing_lock(
            &path,
            Some(StoreRevision::from_manifest(&second.manifest).unwrap()),
            &[gc_update_for_path(GENERATION_INDEX_PATH, "reclaimable", "test").unwrap()],
        )
        .expect("record reclaimable");
        record_generation_gc_with_existing_lock(
            &path,
            Some(StoreRevision::from_manifest(&queued.manifest).unwrap()),
            &[gc_update_for_path(GENERATION_INDEX_PATH, "deleting", "test").unwrap()],
        )
        .expect("record deleting");
        drop(writer_lock);

        std::fs::remove_dir_all(path.join(GENERATION_INDEX_PATH))
            .expect("simulate crash after filesystem delete");

        let report =
            gc_adapter_generations(&path, 1, false).expect("recover interrupted adapter gc");

        assert_eq!(
            report.deleted_generation_paths,
            vec![GENERATION_INDEX_PATH.to_string()]
        );
        assert_eq!(report.redb_commit_sequence, Some(5));
        let latest = generation_gc_events(&path)
            .expect("read gc events")
            .into_iter()
            .rev()
            .find(|event| event.path == GENERATION_INDEX_PATH)
            .expect("generation gc event");
        assert_eq!(latest.state, "deleted");
        assert_eq!(latest.reason, "adapter_gc_recovery");
        verify_adapter_directory(&path).expect("adapter verifies after interrupted gc recovery");
    }

    /// Regression test for double-suffixed crash debris after an interrupted
    /// generation replacement: such a scratch directory under `vectors/` used
    /// to trip the fatal "malformed generation directory entry" branch of
    /// `generation_directory_summary`, permanently bricking `verify`. It
    /// must now be reported as a non-fatal warning instead.
    #[test]
    fn verify_path_reports_double_suffixed_debris_as_warning_not_failure() {
        let path = temp_path("verify-debris-warning");
        let _guard = TempDirGuard(path.clone());

        write_index_at(&path, GENERATION_INDEX_PATH);
        write_legacy_snapshot(&path, None, non_empty_payloads(GENERATION_INDEX_PATH))
            .expect("write first generation");
        std::fs::create_dir_all(path.join(DOUBLE_SUFFIXED_DEBRIS_PATH))
            .expect("create double-suffixed crash debris directory");

        let report = verify_path(&path).expect("verification runs");

        assert!(
            report.valid,
            "debris-only warnings must not fail verify: {report:?}"
        );
        assert!(report.error.is_none(), "{report:?}");
        assert_eq!(report.warnings.len(), 1, "{:?}", report.warnings);
        assert!(
            report.warnings[0].contains("reclaimable"),
            "{:?}",
            report.warnings
        );
    }

    /// Companion to the verify regression above: `adapter gc` must classify
    /// the same double-suffixed crash debris as reclaimable and delete it,
    /// rather than failing to even parse its generation id.
    #[test]
    fn adapter_gc_reclaims_double_suffixed_crash_debris() {
        let path = temp_path("gc-debris-reclaim");
        let _guard = TempDirGuard(path.clone());

        write_index_at(&path, GENERATION_INDEX_PATH);
        write_legacy_snapshot(&path, None, non_empty_payloads(GENERATION_INDEX_PATH))
            .expect("write first generation");
        std::fs::create_dir_all(path.join(DOUBLE_SUFFIXED_DEBRIS_PATH))
            .expect("create double-suffixed crash debris directory");

        let report = gc_adapter_generations(&path, 2, false).expect("gc reclaims debris");

        assert_eq!(
            report.reclaimable_generation_paths,
            vec![DOUBLE_SUFFIXED_DEBRIS_PATH.to_string()]
        );
        assert_eq!(
            report.deleted_generation_paths,
            vec![DOUBLE_SUFFIXED_DEBRIS_PATH.to_string()]
        );
        assert!(!path.join(DOUBLE_SUFFIXED_DEBRIS_PATH).exists());

        let verify_report = verify_path(&path).expect("verification runs after gc");
        assert!(verify_report.valid, "{verify_report:?}");
        assert!(
            verify_report.warnings.is_empty(),
            "{:?}",
            verify_report.warnings
        );
    }

    #[test]
    fn adapter_gc_reclaims_stray_file_debris() {
        // A stray FILE under vectors/ is classified as reclaimable debris by
        // scan_generation_directory, but the pin check used to stat
        // `<file>/.ordinaldb.pin`, fail with NotADirectory, and abort the
        // whole gc run before remove_generation_debris could reclaim it.
        let path = temp_path("gc-file-debris-reclaim");
        let _guard = TempDirGuard(path.clone());

        write_index_at(&path, GENERATION_INDEX_PATH);
        write_legacy_snapshot(&path, None, non_empty_payloads(GENERATION_INDEX_PATH))
            .expect("write first generation");
        let stray = "vectors/stray-debris.bin";
        std::fs::write(path.join(stray), b"leftover bytes").expect("create stray file debris");

        let report = gc_adapter_generations(&path, 2, false).expect("gc reclaims file debris");

        assert_eq!(report.reclaimable_generation_paths, vec![stray.to_string()]);
        assert_eq!(report.deleted_generation_paths, vec![stray.to_string()]);
        assert!(!path.join(stray).exists());

        let verify_report = verify_path(&path).expect("verification runs after gc");
        assert!(verify_report.valid, "{verify_report:?}");
        assert!(
            verify_report.warnings.is_empty(),
            "{:?}",
            verify_report.warnings
        );
    }

    #[test]
    fn read_json_rejects_duplicate_keys() {
        let path = temp_path("duplicate-keys.json");
        let _guard = TempFileGuard(path.clone());
        std::fs::write(&path, br#"{"a":1,"nested":{"b":2,"b":3}}"#).expect("write fixture");

        let err = read_json(&path).expect_err("duplicate keys must fail");

        let path_display = path.display().to_string();
        assert!(err.to_string().contains(&path_display));
        assert!(err.to_string().contains("duplicate JSON key"));
    }

    #[test]
    fn read_json_parse_errors_include_path() {
        let path = temp_path("invalid-json.json");
        let _guard = TempFileGuard(path.clone());
        std::fs::write(&path, br#"{"a":"#).expect("write fixture");

        let err = read_json(&path).expect_err("invalid JSON must fail");

        let path_display = path.display().to_string();
        assert!(err.to_string().contains(&path_display));
        assert!(err.to_string().contains("failed to parse JSON"));
    }

    #[test]
    fn bits_validation_rejects_out_of_range_values() {
        for bits in [1, 2, 4] {
            assert_eq!(required_bits(&json!({ "bits": bits })).unwrap(), bits);
        }

        assert!(required_bits(&json!({ "bits": 0 })).is_err());
        assert!(required_bits(&json!({ "bits": 257 })).is_err());
    }

    #[test]
    fn dim_validation_rejects_lossy_or_core_invalid_values() {
        assert_eq!(optional_dim(&json!({ "dim": null })).unwrap(), None);
        assert_eq!(optional_dim(&json!({ "dim": 64 })).unwrap(), Some(64));

        assert!(optional_dim(&json!({ "dim": 1 })).is_err());
        assert!(optional_dim(&json!({ "dim": 65_536 })).is_err());
        assert!(optional_dim(&json!({ "dim": u64::MAX })).is_err());
    }

    #[test]
    fn ordinaldb_manifest_shape_requires_rankquant_row_id_identity() {
        assert_eq!(
            require_ordinaldb_manifest_shape(
                ManifestIndexParams::RankQuant { bits: 2 },
                ROW_IDENTITY_KIND,
                "test manifest",
            )
            .unwrap(),
            2
        );

        let err = require_ordinaldb_manifest_shape(
            ManifestIndexParams::SignBitmap,
            ROW_IDENTITY_KIND,
            "test manifest",
        )
        .expect_err("non-RankQuant manifest must fail");
        assert!(err.to_string().contains("RankQuant"));

        let err = require_ordinaldb_manifest_shape(
            ManifestIndexParams::RankQuant { bits: 2 },
            "jsonl",
            "test manifest",
        )
        .expect_err("non-row-id identity must fail");
        assert!(err.to_string().contains(ROW_IDENTITY_KIND));
        assert!(err.to_string().contains("jsonl"));
    }

    #[test]
    fn adapter_index_path_validation_accepts_legacy_and_generation_paths() {
        validate_index_path(INDEX_DIR).expect("legacy index path remains readable");
        validate_index_path(GENERATION_INDEX_PATH).expect("generation path must verify");
        assert_eq!(generation_id_from_index_path(INDEX_DIR).unwrap(), 1);
        assert_eq!(
            generation_id_from_index_path("vectors/g000000000042.odb").unwrap(),
            42
        );

        for invalid in [
            "",
            "/tmp/index.odb",
            "../index.odb",
            "vectors/index.odb",
            "vectors//g000000000001.odb",
            "vectors/g000000000001.odb/",
            r"vectors\g000000000001.odb",
            "vectors/g000000000000.odb",
            "vectors/g1.odb",
        ] {
            assert!(
                validate_index_path(invalid).is_err(),
                "{invalid:?} must fail"
            );
        }
    }

    #[test]
    fn adapter_maps_reject_bad_document_and_metadata_values() {
        let string_to_u64 = json!({ "doc-1": 1 }).as_object().unwrap().clone();
        let u64_to_slot = json!({ "1": 0 }).as_object().unwrap().clone();
        let bad_documents = json!({ "doc-1": 42 }).as_object().unwrap().clone();
        let metadata = json!({ "doc-1": { "source": "test" } })
            .as_object()
            .unwrap()
            .clone();

        let err = verify_adapter_maps(
            &string_to_u64,
            &u64_to_slot,
            &bad_documents,
            &metadata,
            2,
            1,
        )
        .expect_err("non-string document must fail");
        assert!(err.to_string().contains("must be a string"));

        let documents = json!({ "doc-1": "text" }).as_object().unwrap().clone();
        let bad_metadata = json!({ "doc-1": "meta" }).as_object().unwrap().clone();
        let err = verify_adapter_maps(
            &string_to_u64,
            &u64_to_slot,
            &documents,
            &bad_metadata,
            2,
            1,
        )
        .expect_err("non-object metadata must fail");
        assert!(err.to_string().contains("must be an object"));
    }

    #[test]
    fn verify_adapter_rejects_empty_lazy_with_index_dir() {
        let path = temp_path("empty-lazy-stale-index");
        let _guard = TempDirGuard(path.clone());
        write_empty_lazy_adapter(&path);
        std::fs::create_dir(path.join(INDEX_DIR)).expect("create stale index dir");

        let err = verify_adapter_directory(&path)
            .expect_err("empty_lazy adapters must not contain index.odb");

        assert!(err
            .to_string()
            .contains("empty_lazy adapter must not contain index.odb"));
    }

    #[test]
    fn verify_adapter_rejects_empty_lazy_with_vectors_dir() {
        let path = temp_path("empty-lazy-stale-vectors");
        let _guard = TempDirGuard(path.clone());
        write_empty_lazy_adapter(&path);
        std::fs::create_dir_all(path.join(GENERATION_INDEX_PATH))
            .expect("create stale generation dir");

        let err = verify_adapter_directory(&path)
            .expect_err("empty_lazy adapters must not contain vectors/");

        assert!(err
            .to_string()
            .contains("empty_lazy adapter must not contain vectors"));
    }

    #[test]
    fn inspect_and_verify_accept_redb_adapter_store() {
        let path = temp_path("empty-lazy-redb");
        let _guard = TempDirGuard(path.clone());
        let payloads = empty_lazy_payloads();
        write_payload_exports(&path, &payloads);
        write_legacy_snapshot(&path, None, payloads).expect("write redb fixture");

        verify_adapter_directory(&path).expect("redb adapter store must verify");
        let report = inspect_adapter_directory(&path).expect("redb adapter store must inspect");

        assert!(matches!(report.kind, PathKind::AdapterDirectory));
        assert_eq!(
            report.schema_version.as_deref(),
            Some(ADAPTER_STORE_SCHEMA_VERSION)
        );
        assert_eq!(report.adapter.as_deref(), Some("test"));
        assert_eq!(report.bits, Some(2));
        assert_eq!(report.dim, None);
        assert_eq!(report.vector_count, 0);
        assert!(report.empty_lazy);
        assert_eq!(report.sidecar_count, 1);
        assert_eq!(report.active_generation_id, Some(0));
        assert_eq!(report.active_generation_path, None);
        assert_eq!(report.active_generation_manifest_sha256, None);
        assert_eq!(report.active_generation_manifest_size_bytes, None);
        let generations = adapter_generations(&report.adapter_generations);
        assert_eq!(generations.generation_count, 0);
        assert_eq!(generations.active_generation_count, 0);
        assert_eq!(generations.completed_generation_count, 0);
        assert_eq!(generations.retained_generation_count, 0);
        assert_eq!(generations.partial_generation_count, 0);
        assert_eq!(generations.reclaimable_generation_count, 0);
        assert_eq!(generations.orphan_generation_count, 0);
        assert!(generations.orphan_generation_paths.is_empty());
        assert!(generations.retained_generation_paths.is_empty());
        assert!(generations.partial_generation_paths.is_empty());
        assert!(generations.reclaimable_generation_paths.is_empty());
    }

    #[test]
    fn stats_reports_redb_adapter_counts_and_bytes() {
        let path = temp_path("stats-empty-lazy-redb");
        let _guard = TempDirGuard(path.clone());
        let payloads = empty_lazy_payloads();
        write_payload_exports(&path, &payloads);
        write_legacy_snapshot(&path, None, payloads).expect("write redb fixture");

        let report = stats_adapter_directory(&path).expect("redb adapter store stats");

        assert!(matches!(report.kind, PathKind::AdapterDirectory));
        assert_eq!(report.vector_count, 0);
        assert_eq!(report.active_id_count, 0);
        let generations = adapter_generations(&report.adapter_generations);
        assert_eq!(generations.generation_count, 0);
        assert_eq!(generations.active_generation_count, 0);
        assert_eq!(generations.completed_generation_count, 0);
        assert_eq!(generations.retained_generation_count, 0);
        assert_eq!(generations.partial_generation_count, 0);
        assert_eq!(generations.reclaimable_generation_count, 0);
        assert_eq!(generations.orphan_generation_count, 0);
        assert!(generations.orphan_generation_paths.is_empty());
        assert!(generations.retained_generation_paths.is_empty());
        assert!(generations.partial_generation_paths.is_empty());
        assert!(generations.reclaimable_generation_paths.is_empty());
        assert!(report.bytes_by_component["total"] > 0);
        assert!(report.bytes_by_component["adapter_state"] > 0);
        assert!(report.bytes_by_component["compatibility_exports"] > 0);
        assert_eq!(report.bytes_by_component["vectors"], 0);
    }

    #[test]
    fn stats_reports_legacy_adapter_counts_and_bytes() {
        let path = temp_path("stats-empty-lazy-legacy");
        let _guard = TempDirGuard(path.clone());
        write_empty_lazy_adapter(&path);

        let report = stats_path(&path).expect("legacy adapter stats");

        assert!(matches!(report.kind, PathKind::AdapterDirectory));
        assert_eq!(report.vector_count, 0);
        assert_eq!(report.active_id_count, 0);
        let generations = adapter_generations(&report.adapter_generations);
        assert_eq!(generations.generation_count, 0);
        assert_eq!(generations.active_generation_count, 0);
        assert_eq!(generations.completed_generation_count, 0);
        assert_eq!(generations.retained_generation_count, 0);
        assert_eq!(generations.partial_generation_count, 0);
        assert_eq!(generations.reclaimable_generation_count, 0);
        assert_eq!(generations.orphan_generation_count, 0);
        assert!(generations.orphan_generation_paths.is_empty());
        assert!(generations.retained_generation_paths.is_empty());
        assert!(generations.partial_generation_paths.is_empty());
        assert!(generations.reclaimable_generation_paths.is_empty());
        assert!(report.bytes_by_component["total"] > 0);
        assert_eq!(report.bytes_by_component["adapter_state"], 0);
        assert!(report.bytes_by_component["compatibility_exports"] > 0);
        assert_eq!(report.bytes_by_component["vectors"], 0);
    }

    /// JSON keys that only make sense for adapter directories (generation
    /// bookkeeping). A plain core bundle has no generation concept, so these
    /// must be entirely absent from its `inspect --json` / `stats --json`
    /// output rather than present-but-zero.
    const ADAPTER_GENERATION_JSON_KEYS: &[&str] = &[
        "generation_count",
        "active_generation_count",
        "completed_generation_count",
        "retained_generation_count",
        "partial_generation_count",
        "reclaimable_generation_count",
        "orphan_generation_count",
        "orphan_generation_paths",
        "retained_generation_paths",
        "partial_generation_paths",
        "reclaimable_generation_paths",
    ];

    #[test]
    fn inspect_core_bundle_omits_adapter_generation_fields() {
        let path = temp_path("core-bundle-inspect");
        let _guard = TempDirGuard(path.clone());
        write_core_bundle(&path);

        let report = inspect_path(&path).expect("inspect a freshly written core bundle");

        assert!(matches!(report.kind, PathKind::CoreBundle));
        assert!(
            report.adapter_generations.is_none(),
            "plain core bundles must not carry adapter generation bookkeeping"
        );

        let text = format_inspect_text(&report);
        assert!(
            !text.contains("generations:"),
            "core bundle inspect text must not print generation lines:\n{text}"
        );
        assert!(
            !text.contains("_generation_path:"),
            "core bundle inspect text must not print generation path lines:\n{text}"
        );

        let json = serde_json::to_value(&report).expect("serialize inspect report");
        let object = json.as_object().expect("inspect JSON must be an object");
        for key in ADAPTER_GENERATION_JSON_KEYS {
            assert!(
                !object.contains_key(*key),
                "core bundle inspect JSON must omit {key}, got: {json}"
            );
        }
    }

    #[test]
    fn stats_core_bundle_omits_adapter_generation_fields() {
        let path = temp_path("core-bundle-stats");
        let _guard = TempDirGuard(path.clone());
        write_core_bundle(&path);

        let report = stats_path(&path).expect("stats a freshly written core bundle");

        assert!(matches!(report.kind, PathKind::CoreBundle));
        assert!(
            report.adapter_generations.is_none(),
            "plain core bundles must not carry adapter generation bookkeeping"
        );

        let text = format_stats_text(&report);
        assert!(
            !text.contains("generations:"),
            "core bundle stats text must not print generation lines:\n{text}"
        );
        assert!(
            !text.contains("_generation_path:"),
            "core bundle stats text must not print generation path lines:\n{text}"
        );

        let json = serde_json::to_value(&report).expect("serialize stats report");
        let object = json.as_object().expect("stats JSON must be an object");
        for key in ADAPTER_GENERATION_JSON_KEYS {
            assert!(
                !object.contains_key(*key),
                "core bundle stats JSON must omit {key}, got: {json}"
            );
        }
    }

    #[test]
    fn inspect_adapter_directory_retains_adapter_generation_fields() {
        let path = temp_path("adapter-inspect-retains-generations");
        let _guard = TempDirGuard(path.clone());
        write_empty_lazy_adapter(&path);

        let report = inspect_path(&path).expect("inspect adapter directory");

        assert!(matches!(report.kind, PathKind::AdapterDirectory));
        assert!(
            report.adapter_generations.is_some(),
            "adapter directories must keep adapter generation bookkeeping"
        );

        let text = format_inspect_text(&report);
        assert!(text.contains("generations: 0"), "{text}");
        assert!(text.contains("active_generations: 0"), "{text}");
        assert!(text.contains("orphan_generations: 0"), "{text}");

        let json = serde_json::to_value(&report).expect("serialize inspect report");
        let object = json.as_object().expect("inspect JSON must be an object");
        for key in ADAPTER_GENERATION_JSON_KEYS {
            assert!(
                object.contains_key(*key),
                "adapter directory inspect JSON must contain {key}, got: {json}"
            );
        }
    }

    #[test]
    fn stats_adapter_directory_retains_adapter_generation_fields() {
        let path = temp_path("adapter-stats-retains-generations");
        let _guard = TempDirGuard(path.clone());
        write_empty_lazy_adapter(&path);

        let report = stats_path(&path).expect("stats adapter directory");

        assert!(matches!(report.kind, PathKind::AdapterDirectory));
        assert!(
            report.adapter_generations.is_some(),
            "adapter directories must keep adapter generation bookkeeping"
        );

        let text = format_stats_text(&report);
        assert!(text.contains("generations: 0"), "{text}");
        assert!(text.contains("active_generations: 0"), "{text}");
        assert!(text.contains("orphan_generations: 0"), "{text}");

        let json = serde_json::to_value(&report).expect("serialize stats report");
        let object = json.as_object().expect("stats JSON must be an object");
        for key in ADAPTER_GENERATION_JSON_KEYS {
            assert!(
                object.contains_key(*key),
                "adapter directory stats JSON must contain {key}, got: {json}"
            );
        }
    }

    #[test]
    fn active_id_count_reads_id_map_records() {
        let count = active_id_count_from_id_map(&json!({
            "schema_version": ID_MAP_SCHEMA_VERSION,
            "next_u64_id": 3,
            "string_to_u64": {"doc-a": 1, "doc-b": 2},
            "u64_to_slot": {"1": 0, "2": 1},
        }))
        .expect("active id count");

        assert_eq!(count, 2);
    }

    #[cfg(unix)]
    #[test]
    fn stats_byte_accounting_rejects_symlinked_components() {
        let path = temp_path("stats-symlink");
        let _guard = TempDirGuard(path.clone());
        std::fs::create_dir_all(&path).expect("create stats root");
        let outside = temp_path("stats-outside");
        let _outside_guard = TempDirGuard(outside.clone());
        std::fs::create_dir_all(&outside).expect("create outside root");
        std::fs::write(outside.join("payload"), b"payload").expect("write outside payload");
        std::os::unix::fs::symlink(outside.join("payload"), path.join("payload"))
            .expect("create payload symlink");

        let err = path_size_bytes(&path).expect_err("symlinked components must fail closed");

        assert!(
            err.to_string()
                .contains("stats path must not contain a symlink"),
            "{err}"
        );
    }

    #[test]
    fn generation_directory_summary_reports_orphans_and_classifies_malformed_as_debris() {
        let path = temp_path("generation-summary");
        let _guard = TempDirGuard(path.clone());
        std::fs::create_dir_all(path.join("vectors/g000000000001.odb"))
            .expect("create active generation");
        std::fs::create_dir_all(path.join("vectors/g000000000002.odb"))
            .expect("create orphan generation");
        std::fs::create_dir_all(path.join("vectors/.g000000000003.odb.tmp-123"))
            .expect("create partial temp generation");

        let summary =
            generation_directory_summary(&path, Some("vectors/g000000000001.odb")).unwrap();
        assert_eq!(summary.generation_count, 3);
        assert_eq!(summary.active_generation_count, 1);
        assert_eq!(summary.completed_generation_count, 2);
        assert_eq!(
            summary.retained_generation_paths,
            vec!["vectors/g000000000002.odb".to_string()]
        );
        assert_eq!(
            summary.partial_generation_paths,
            vec!["vectors/.g000000000003.odb.tmp-123".to_string()]
        );
        assert_eq!(
            summary.reclaimable_generation_paths,
            vec!["vectors/.g000000000003.odb.tmp-123".to_string()]
        );
        assert_eq!(
            summary.orphan_generation_paths,
            vec![
                "vectors/.g000000000003.odb.tmp-123".to_string(),
                "vectors/g000000000002.odb".to_string(),
            ]
        );
        assert_eq!(
            summary.debris_warnings.len(),
            1,
            "{:?}",
            summary.debris_warnings
        );
        assert!(
            summary.debris_warnings[0].contains("reclaimable"),
            "{:?}",
            summary.debris_warnings
        );

        // A directory whose name parses neither as a canonical generation
        // nor as a recognizable interrupted-replacement scratch directory
        // used to be a fatal "malformed generation directory entry". It is
        // now reclaimable debris with a structured warning, never fatal.
        std::fs::create_dir_all(path.join("vectors/gbad.odb"))
            .expect("create malformed generation");
        let summary = generation_directory_summary(&path, Some("vectors/g000000000001.odb"))
            .expect("malformed generation entries are reclaimable debris, not fatal");
        assert_eq!(summary.generation_count, 4);
        assert_eq!(summary.completed_generation_count, 2);
        assert_eq!(
            summary.partial_generation_paths,
            vec![
                "vectors/.g000000000003.odb.tmp-123".to_string(),
                "vectors/gbad.odb".to_string(),
            ]
        );
        assert_eq!(
            summary.debris_warnings.len(),
            2,
            "{:?}",
            summary.debris_warnings
        );
        assert!(
            summary
                .debris_warnings
                .iter()
                .any(|warning| warning.contains("gbad.odb")),
            "{:?}",
            summary.debris_warnings
        );
        for warning in &summary.debris_warnings {
            assert!(warning.contains("reclaimable"), "{warning}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn generation_directory_summary_rejects_symlink_generation() {
        let path = temp_path("generation-symlink");
        let _guard = TempDirGuard(path.clone());
        std::fs::create_dir_all(path.join("vectors/g000000000001.odb"))
            .expect("create active generation");
        std::os::unix::fs::symlink(
            path.join("vectors/g000000000001.odb"),
            path.join("vectors/g000000000002.odb"),
        )
        .expect("create symlink generation");

        let err = generation_directory_summary(&path, Some("vectors/g000000000001.odb"))
            .expect_err("symlink generation entries must fail closed");
        assert!(err.to_string().contains("symlink"), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn generation_directory_summary_rejects_symlink_vectors_directory() {
        let path = temp_path("vectors-symlink");
        let _guard = TempDirGuard(path.clone());
        std::fs::create_dir_all(&path).expect("create root");
        let outside = temp_path("outside-vectors");
        let _outside_guard = TempDirGuard(outside.clone());
        std::fs::create_dir_all(&outside).expect("create outside");
        std::os::unix::fs::symlink(&outside, path.join("vectors")).expect("create vectors symlink");

        let err = generation_directory_summary(&path, None)
            .expect_err("vectors symlink must fail closed");
        assert!(err.to_string().contains("must not be a symlink"), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn active_generation_manifest_path_rejects_symlink_parent() {
        let path = temp_path("manifest-parent-symlink");
        let _guard = TempDirGuard(path.clone());
        std::fs::create_dir_all(&path).expect("create root");
        let outside = temp_path("outside-manifest-parent");
        let _outside_guard = TempDirGuard(outside.clone());
        std::fs::create_dir_all(&outside).expect("create outside");
        std::os::unix::fs::symlink(&outside, path.join("vectors")).expect("create vectors symlink");

        let err = active_generation_manifest_path(&path, GENERATION_INDEX_PATH)
            .expect_err("generation parent symlink must fail closed");
        assert!(
            err.to_string()
                .contains("active generation path must not contain a symlink"),
            "{err}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn verify_adapter_rejects_symlink_sidecar() {
        let path = temp_path("symlink-sidecar");
        let _guard = TempDirGuard(path.clone());
        write_empty_lazy_adapter(&path);
        let documents_path = path.join(DOCUMENTS_FILE);
        let outside = path.join("outside-documents.json");
        std::fs::rename(&documents_path, &outside).expect("move sidecar");
        std::os::unix::fs::symlink(&outside, &documents_path).expect("create sidecar symlink");

        let err = verify_adapter_directory(&path).expect_err("sidecar symlink must fail closed");
        assert!(err.to_string().contains("must not be a symlink"), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn verify_adapter_rejects_symlink_adapter_store() {
        let path = temp_path("symlink-redb-store");
        let _guard = TempDirGuard(path.clone());
        let payloads = empty_lazy_payloads();
        write_payload_exports(&path, &payloads);
        write_legacy_snapshot(&path, None, payloads).expect("write redb fixture");
        let outside = path.with_extension("outside-redb");
        std::fs::rename(path.join(ADAPTER_STORE_FILE), &outside).expect("move redb store");
        std::os::unix::fs::symlink(&outside, path.join(ADAPTER_STORE_FILE))
            .expect("create redb symlink");

        let err = verify_adapter_directory(&path).expect_err("redb symlink must fail closed");

        assert!(err.to_string().contains("must not be a symlink"), "{err}");
        let _ = std::fs::remove_file(outside);
    }

    #[test]
    fn generation_manifest_validation_rejects_oversized_file() {
        let path = temp_path("oversized-generation-manifest");
        let _guard = TempDirGuard(path.clone());
        std::fs::create_dir_all(&path).expect("create fixture dir");
        let manifest_path = path.join(MANIFEST_FILE);
        std::fs::write(
            &manifest_path,
            vec![b' '; (MAX_GENERATION_MANIFEST_BYTES + 1) as usize],
        )
        .expect("write oversized manifest");

        let err = validate_generation_manifest_file(&manifest_path)
            .expect_err("oversized generation manifest must fail closed");
        assert!(err.to_string().contains("manifest too large"), "{err}");
    }

    #[test]
    fn generation_manifest_validation_rejects_non_file() {
        let path = temp_path("non-file-generation-manifest");
        let _guard = TempDirGuard(path.clone());
        std::fs::create_dir_all(path.join(MANIFEST_FILE)).expect("create manifest directory");

        let err = validate_generation_manifest_file(&path.join(MANIFEST_FILE))
            .expect_err("generation manifest directory must fail closed");
        assert!(err.to_string().contains("must be a file"), "{err}");
    }

    #[test]
    fn verify_accepts_redb_adapter_store_with_stale_json_exports() {
        let path = temp_path("empty-lazy-redb-drift");
        let _guard = TempDirGuard(path.clone());
        let payloads = empty_lazy_payloads();
        write_payload_exports(&path, &payloads);
        write_legacy_snapshot(&path, None, payloads).expect("write redb fixture");
        std::fs::write(
            path.join(DOCUMENTS_FILE),
            r#"{"schema_version":"ordinaldb.adapter.documents.v1","documents":{"stale":"value"}}"#,
        )
        .expect("mutate export");

        verify_adapter_directory(&path).expect("adapter.redb is authoritative");
    }

    #[test]
    fn verify_path_returns_invalid_report_for_unknown_paths() {
        let path = temp_path("missing");

        let report = verify_path(&path).expect("verification failures are reports");

        assert!(matches!(report.kind, PathKind::Unknown));
        assert!(!report.valid);
        assert!(report.error.unwrap().contains("does not exist"));
    }

    #[test]
    fn verify_path_reports_vectors_only_directory_as_unknown() {
        let path = temp_path("vectors-only");
        let _guard = TempDirGuard(path.clone());
        std::fs::create_dir_all(path.join(GENERATION_INDEX_PATH)).expect("create debris");

        let report = verify_path(&path).expect("verification failures are reports");

        assert!(matches!(report.kind, PathKind::Unknown));
        assert!(!report.valid);
        assert!(report.error.unwrap().contains("neither a core .odb bundle"));
    }

    #[test]
    fn verify_path_reports_index_manifest_without_adapter_as_unknown() {
        let path = temp_path("index-manifest-only");
        let _guard = TempDirGuard(path.clone());
        std::fs::create_dir_all(path.join(INDEX_DIR)).expect("create debris");
        std::fs::write(path.join(INDEX_DIR).join(MANIFEST_FILE), "{}\n")
            .expect("write debris manifest");

        let report = verify_path(&path).expect("verification failures are reports");

        assert!(matches!(report.kind, PathKind::Unknown));
        assert!(!report.valid);
        assert!(report.error.unwrap().contains("neither a core .odb bundle"));
    }

    #[test]
    fn verify_path_reports_malformed_adapter_json_as_unknown() {
        let path = temp_path("bad-adapter-json-marker");
        let _guard = TempDirGuard(path.clone());
        std::fs::create_dir_all(&path).expect("create fixture");
        std::fs::write(path.join(ADAPTER_FILE), "{}\n").expect("write bad adapter marker");

        let report = verify_path(&path).expect("verification failures are reports");

        assert!(matches!(report.kind, PathKind::Unknown));
        assert!(!report.valid);
        assert!(report
            .error
            .unwrap()
            .contains("adapter.json has invalid keys"));
    }
}
