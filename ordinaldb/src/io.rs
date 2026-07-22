//! Bundle I/O for OrdinalDB indexes.
//!
//! Handles atomic write/replace of on-disk bundle directories (write to a
//! temp directory, verify, fsync, rename into place), crash recovery from
//! `.{bundle}.bak-{pid}-{nanos}` backup directories, and the binary row-ID
//! sidecar format. The public constants below name the artifacts inside a
//! bundle directory and are re-exported from [`crate::artifacts`].

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
#[cfg(windows)]
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::manifest::AuxiliaryArtifactDeclaration;
use ordvec::{RankQuant, SignBitmap};
use ordvec_manifest::{
    create_manifest_for_index_with_options, verify_for_load, write_manifest_file,
    CreateAuxiliaryArtifact, CreateManifestOptions, CreateRowIdentity, ManifestIndexKind,
    ManifestIndexParams, VerifyOptions,
};

/// File name of the verified manifest inside an OrdinalDB bundle directory.
pub const MANIFEST_FILE: &str = "manifest.json";
/// File name of the primary RankQuant index artifact inside a bundle directory.
pub const INDEX_FILE: &str = "index.ovrq";
/// File name of the optional sign bitmap sidecar inside a bundle directory.
pub const SIGN_FILE: &str = "sign.ovsb";
/// File name of the row-ID sidecar inside an ID-mapped bundle directory.
pub const IDS_FILE: &str = "ids.bin";

/// Manifest auxiliary-artifact name under which the sign bitmap sidecar is
/// registered (marked required when present).
pub const SIGN_AUX_NAME: &str = "ordinaldb.sign";
/// Manifest auxiliary-artifact name under which the row-ID sidecar is
/// registered (marked required when present).
pub const IDS_AUX_NAME: &str = "ordinaldb.ids";
/// Embedding-model identifier stamped into manifests written by OrdinalDB.
pub const EMBEDDING_MODEL: &str = "ordinaldb.local";

const IDS_MAGIC: &[u8; 8] = b"ODBIDS1\0";

pub(crate) struct LoadedOrdinalArtifacts {
    pub rankquant: RankQuant,
    pub sign: Option<SignBitmap>,
}

pub(crate) struct LoadedIdMapArtifacts {
    pub rankquant: RankQuant,
    pub sign: Option<SignBitmap>,
    pub ids: Vec<u64>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct BundleWriteOptions {
    pub manifest_options: CreateManifestOptions,
    pub auxiliary_artifacts: Vec<AuxiliaryArtifactDeclaration>,
}

pub(crate) fn write_ordinal_bundle(
    path: impl AsRef<Path>,
    rankquant: &RankQuant,
    sign: Option<&SignBitmap>,
) -> io::Result<()> {
    write_bundle(
        path.as_ref(),
        rankquant,
        sign,
        None,
        BundleWriteOptions::default(),
    )
}

pub(crate) fn write_ordinal_bundle_with_options(
    path: impl AsRef<Path>,
    rankquant: &RankQuant,
    sign: Option<&SignBitmap>,
    options: BundleWriteOptions,
) -> io::Result<()> {
    write_bundle(path.as_ref(), rankquant, sign, None, options)
}

pub(crate) fn write_id_map_bundle(
    path: impl AsRef<Path>,
    rankquant: &RankQuant,
    sign: Option<&SignBitmap>,
    ids: &[u64],
) -> io::Result<()> {
    if ids.len() != rankquant.len() {
        return Err(invalid_input(format!(
            "id count {} does not match index len {}",
            ids.len(),
            rankquant.len()
        )));
    }
    write_bundle(
        path.as_ref(),
        rankquant,
        sign,
        Some(ids),
        BundleWriteOptions::default(),
    )
}

pub(crate) fn write_id_map_bundle_with_options(
    path: impl AsRef<Path>,
    rankquant: &RankQuant,
    sign: Option<&SignBitmap>,
    ids: &[u64],
    options: BundleWriteOptions,
) -> io::Result<()> {
    if ids.len() != rankquant.len() {
        return Err(invalid_input(format!(
            "id count {} does not match index len {}",
            ids.len(),
            rankquant.len()
        )));
    }
    write_bundle(path.as_ref(), rankquant, sign, Some(ids), options)
}

pub(crate) fn load_ordinal_bundle(path: impl AsRef<Path>) -> io::Result<LoadedOrdinalArtifacts> {
    let loaded = load_bundle(path.as_ref())?;
    if loaded.ids_path.is_some() {
        return Err(invalid_data(
            "bundle contains OrdinalDB IDs; load it with IdMapIndex::load",
        ));
    }
    Ok(LoadedOrdinalArtifacts {
        rankquant: loaded.rankquant,
        sign: loaded.sign,
    })
}

pub(crate) fn load_id_map_bundle(path: impl AsRef<Path>) -> io::Result<LoadedIdMapArtifacts> {
    let loaded = load_bundle(path.as_ref())?;
    let ids_path = loaded
        .ids_path
        .ok_or_else(|| invalid_data("bundle is missing required OrdinalDB ID sidecar"))?;
    let ids = read_ids_file(&ids_path, loaded.rankquant.len())?;
    Ok(LoadedIdMapArtifacts {
        rankquant: loaded.rankquant,
        sign: loaded.sign,
        ids,
    })
}

fn write_bundle(
    path: &Path,
    rankquant: &RankQuant,
    sign: Option<&SignBitmap>,
    ids: Option<&[u64]>,
    options: BundleWriteOptions,
) -> io::Result<()> {
    let temp = temp_bundle_path(path)?;
    if temp.exists() {
        fs::remove_dir_all(&temp)?;
    }
    fs::create_dir_all(&temp)?;
    sync_parent_directory(&temp)?;

    let verify_options = verify_options_from_create(&options.manifest_options);
    let write_result = write_bundle_contents(&temp, rankquant, sign, ids, options)
        .and_then(|()| verify_written_bundle(&temp, verify_options))
        .and_then(|()| sync_bundle_tree(&temp));
    if let Err(err) = write_result {
        let _ = fs::remove_dir_all(&temp);
        return Err(err);
    }

    replace_bundle(path, &temp)
}

fn write_bundle_contents(
    bundle: &Path,
    rankquant: &RankQuant,
    sign: Option<&SignBitmap>,
    ids: Option<&[u64]>,
    options: BundleWriteOptions,
) -> io::Result<()> {
    let index_path = bundle.join(INDEX_FILE);
    let manifest_path = bundle.join(MANIFEST_FILE);
    rankquant.write(&index_path)?;

    let sign_path = if let Some(sign) = sign {
        let path = bundle.join(SIGN_FILE);
        sign.write(&path)?;
        Some(path)
    } else {
        None
    };

    let ids_path = if let Some(ids) = ids {
        let path = bundle.join(IDS_FILE);
        write_ids_file(&path, ids)?;
        Some(path)
    } else {
        None
    };

    let mut manifest_options = options.manifest_options;
    if !manifest_options.auxiliary_artifacts.is_empty() {
        return Err(invalid_input(
            "pass OrdinalDB bundle sidecars with AuxiliaryArtifactDeclaration, not CreateManifestOptions::auxiliary_artifacts",
        ));
    }
    if let Some(path) = sign_path.as_ref() {
        manifest_options
            .auxiliary_artifacts
            .push(CreateAuxiliaryArtifact {
                name: SIGN_AUX_NAME.to_string(),
                path: path.clone(),
                required: true,
            });
    }
    if let Some(path) = ids_path.as_ref() {
        manifest_options
            .auxiliary_artifacts
            .push(CreateAuxiliaryArtifact {
                name: IDS_AUX_NAME.to_string(),
                path: path.clone(),
                required: true,
            });
    }
    copy_auxiliary_artifacts(bundle, &mut manifest_options, &options.auxiliary_artifacts)?;

    let manifest = create_manifest_for_index_with_options(
        &index_path,
        CreateRowIdentity::RowIdIdentity,
        EMBEDDING_MODEL,
        &manifest_path,
        manifest_options,
    )
    .map_err(io_other)?;

    write_manifest_file(&manifest, &manifest_path).map_err(io_other)
}

fn copy_auxiliary_artifacts(
    bundle: &Path,
    manifest_options: &mut CreateManifestOptions,
    artifacts: &[AuxiliaryArtifactDeclaration],
) -> io::Result<()> {
    let mut seen_names = manifest_options
        .auxiliary_artifacts
        .iter()
        .map(|artifact| artifact.name.trim().to_string())
        .collect::<HashSet<_>>();
    let mut seen_paths = HashSet::<PathBuf>::new();
    for artifact in artifacts {
        let name = artifact.name.trim();
        if name.is_empty() {
            return Err(invalid_input("auxiliary artifact name must be non-empty"));
        }
        if !seen_names.insert(name.to_string()) {
            return Err(invalid_input(format!(
                "auxiliary artifact name {name:?} is duplicated or reserved"
            )));
        }
        let relative_path = validate_bundle_relative_path(&artifact.bundle_path)?;
        if relative_path == Path::new(MANIFEST_FILE)
            || relative_path == Path::new(INDEX_FILE)
            || relative_path == Path::new(SIGN_FILE)
            || relative_path == Path::new(IDS_FILE)
        {
            return Err(invalid_input(format!(
                "auxiliary artifact path {} conflicts with a reserved OrdinalDB artifact",
                relative_path.display()
            )));
        }
        if !seen_paths.insert(relative_path.to_path_buf()) {
            return Err(invalid_input(format!(
                "auxiliary artifact path {} is duplicated",
                relative_path.display()
            )));
        }
        let target = bundle.join(relative_path);
        if target.exists() {
            return Err(invalid_input(format!(
                "auxiliary artifact path {} would overwrite an existing bundle artifact",
                relative_path.display()
            )));
        }
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&artifact.source_path, &target)?;
        manifest_options
            .auxiliary_artifacts
            .push(CreateAuxiliaryArtifact {
                name: name.to_string(),
                path: target,
                required: artifact.required,
            });
    }
    Ok(())
}

fn validate_bundle_relative_path(path: &Path) -> io::Result<&Path> {
    if path.is_absolute() || path.as_os_str().is_empty() {
        return Err(invalid_input(format!(
            "bundle auxiliary path {} must be a non-empty relative path",
            path.display()
        )));
    }
    for component in path.components() {
        use std::path::Component;
        if !matches!(component, Component::Normal(_)) {
            return Err(invalid_input(format!(
                "bundle auxiliary path {} must not contain path escapes",
                path.display()
            )));
        }
    }
    Ok(path)
}

fn verify_options_from_create(options: &CreateManifestOptions) -> VerifyOptions {
    VerifyOptions {
        allow_absolute_paths: options.allow_absolute_paths,
        allow_path_escape: options.allow_path_escape,
        limits: options.limits.clone(),
        ..VerifyOptions::default()
    }
}

fn verify_written_bundle(bundle: &Path, options: VerifyOptions) -> io::Result<()> {
    verify_for_load(bundle.join(MANIFEST_FILE), options)
        .map(|_| ())
        .map_err(invalid_data_err)
}

fn replace_bundle(path: &Path, temp: &Path) -> io::Result<()> {
    let mut backup = None;
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() {
                let _ = fs::remove_dir_all(temp);
                return Err(invalid_input(format!(
                    "cannot replace symlink path {} with an OrdinalDB bundle",
                    path.display()
                )));
            }
            if !metadata.file_type().is_dir() {
                let _ = fs::remove_dir_all(temp);
                return Err(invalid_input(format!(
                    "cannot replace non-directory path {} with an OrdinalDB bundle",
                    path.display()
                )));
            }
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => {}
        Err(err) => return Err(err),
    }
    if path.exists() {
        let backup_path = unique_backup_bundle_path(path)?;
        rename_path(path, &backup_path)?;
        sync_parent_directory(path)?;
        backup = Some(backup_path);
    }
    match rename_path(temp, path) {
        Ok(()) => {
            sync_parent_directory(path)?;
            if let Some(backup_path) = backup {
                let _ = fs::remove_dir_all(&backup_path);
                let _ = sync_parent_directory(&backup_path);
            }
            Ok(())
        }
        Err(err) => {
            if let Some(backup_path) = backup {
                if !path.exists() {
                    let _ = rename_path(&backup_path, path);
                    let _ = sync_parent_directory(path);
                }
            }
            Err(err)
        }
    }
}

pub(crate) fn recover_bundle_if_missing(path: &Path) -> io::Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() {
                return Err(invalid_data(format!(
                    "OrdinalDB bundle path must not be a symlink: {}",
                    path.display()
                )));
            }
            return Ok(());
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => {}
        Err(err) => return Err(err),
    }

    for backup in backup_bundle_candidates(path)? {
        if verify_for_load(backup.join(MANIFEST_FILE), VerifyOptions::default()).is_err() {
            continue;
        }
        rename_path(&backup, path)?;
        sync_parent_directory(path)?;
        return Ok(());
    }
    Ok(())
}

#[cfg(windows)]
fn rename_path(from: &Path, to: &Path) -> io::Result<()> {
    let mut last_error = None;
    for attempt in 0..10 {
        match fs::rename(from, to) {
            Ok(()) => return Ok(()),
            Err(err) if err.kind() == io::ErrorKind::PermissionDenied => {
                last_error = Some(err);
                std::thread::sleep(Duration::from_millis(25 * (attempt + 1)));
            }
            Err(err) => return Err(err),
        }
    }
    Err(last_error.unwrap_or_else(|| io::Error::new(io::ErrorKind::Other, "rename retry failed")))
}

#[cfg(not(windows))]
fn rename_path(from: &Path, to: &Path) -> io::Result<()> {
    fs::rename(from, to)
}

fn backup_bundle_candidates(path: &Path) -> io::Result<Vec<PathBuf>> {
    let file_name = path.file_name().ok_or_else(|| {
        invalid_input(format!(
            "bundle path {} must name a directory",
            path.display()
        ))
    })?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let prefix = format!(".{}.bak-", file_name.to_string_lossy());
    // Sort key: (nanos, pid, name). Backup names embed unpadded integers, so
    // lexicographic order is NOT chronological order across differing digit
    // widths (pid 999 vs 1000, or nanos with fewer digits). Parse the
    // "{pid}-{nanos}" suffix numerically; for names that do not parse (for
    // example backups produced by older builds), fall back to the directory
    // modification time so those backups still rank by recency. The name is
    // kept as a final tiebreak so ordering stays deterministic.
    let mut candidates: Vec<(u128, u128, String, PathBuf)> = Vec::new();
    match fs::read_dir(parent) {
        Ok(entries) => {
            for entry in entries {
                let entry = entry?;
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if name.starts_with(&prefix) {
                    let metadata = fs::symlink_metadata(entry.path())?;
                    if metadata.file_type().is_symlink() {
                        return Err(invalid_data(format!(
                            "backup bundle candidate must not be a symlink: {}",
                            entry.path().display()
                        )));
                    }
                    if metadata.file_type().is_dir() {
                        let (nanos, pid) = parse_backup_suffix(&name[prefix.len()..])
                            .unwrap_or_else(|| (modified_nanos(&metadata), 0));
                        candidates.push((nanos, pid, name.into_owned(), entry.path()));
                    }
                }
            }
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => {}
        Err(err) => return Err(err),
    }
    candidates.sort_by(|a, b| {
        b.0.cmp(&a.0)
            .then_with(|| b.1.cmp(&a.1))
            .then_with(|| b.2.cmp(&a.2))
    });
    Ok(candidates.into_iter().map(|(_, _, _, path)| path).collect())
}

/// Parses the `{pid}-{nanos}` suffix of a backup bundle name produced by
/// [`unique_backup_bundle_path`], returning `(nanos, pid)` ordered so the
/// creation timestamp is the primary sort key. Returns `None` for names that
/// do not match the scheme exactly (both parts must be non-empty ASCII digit
/// runs that fit in a `u128`).
fn parse_backup_suffix(suffix: &str) -> Option<(u128, u128)> {
    let (pid, nanos) = suffix.split_once('-')?;
    if pid.is_empty()
        || nanos.is_empty()
        || !pid.bytes().all(|byte| byte.is_ascii_digit())
        || !nanos.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    Some((nanos.parse().ok()?, pid.parse().ok()?))
}

/// Best-effort modification time of a backup candidate as nanoseconds since
/// the Unix epoch, used to rank backups whose names carry no parseable
/// timestamp. Returns 0 when the platform reports no usable mtime, which
/// ranks such candidates last (newest-first ordering).
fn modified_nanos(metadata: &fs::Metadata) -> u128 {
    metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
        .unwrap_or(0)
}

fn sync_bundle_tree(path: &Path) -> io::Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() {
        return Err(invalid_input(format!(
            "bundle path must not contain a symlink: {}",
            path.display()
        )));
    }
    if metadata.file_type().is_file() {
        sync_file(path)?;
        return Ok(());
    }
    if metadata.file_type().is_dir() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            sync_bundle_tree(&entry.path())?;
        }
        sync_directory(path)?;
    }
    Ok(())
}

#[cfg(windows)]
fn sync_file(path: &Path) -> io::Result<()> {
    fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)?
        .sync_all()
}

#[cfg(not(windows))]
fn sync_file(path: &Path) -> io::Result<()> {
    File::open(path)?.sync_all()
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> io::Result<()> {
    File::open(path)?.sync_all()
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> io::Result<()> {
    Ok(())
}

/// Syncs the directory containing `path`. A bare relative name (for example
/// `"docs.odb"`) has `parent() == Some("")`, and opening the empty path fails
/// with `NotFound`; treat that case as the current directory instead.
fn sync_parent_directory(path: &Path) -> io::Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    sync_directory(parent)
}

struct LoadedBundle {
    rankquant: RankQuant,
    sign: Option<SignBitmap>,
    ids_path: Option<PathBuf>,
}

fn load_bundle(path: &Path) -> io::Result<LoadedBundle> {
    recover_bundle_if_missing(path)?;
    let plan = verify_for_load(path.join(MANIFEST_FILE), VerifyOptions::default())
        .map_err(invalid_data_err)?;
    let metadata = plan.metadata();
    if metadata.kind != ManifestIndexKind::RankQuant {
        return Err(invalid_data(format!(
            "OrdinalDB bundles require a RankQuant primary artifact; got {:?}",
            metadata.kind
        )));
    }
    let ManifestIndexParams::RankQuant {
        bits: metadata_bits,
    } = metadata.params
    else {
        return Err(invalid_data(
            "OrdinalDB bundle primary artifact has non-RankQuant params",
        ));
    };
    if plan.row_identity().kind() != "row_id_identity" {
        return Err(invalid_data(format!(
            "OrdinalDB bundles require row_id_identity row identity; got {:?}",
            plan.row_identity().kind()
        )));
    }

    let rankquant = RankQuant::load(plan.artifact_path())?;
    if metadata.dim != rankquant.dim()
        || metadata.vector_count != rankquant.len()
        || metadata_bits != rankquant.bits()
        || plan.row_identity().row_count() != rankquant.len()
    {
        return Err(invalid_data(
            "verified manifest metadata does not match loaded RankQuant",
        ));
    }

    // This backs the convenience `load()` entry points, which take no
    // `DenseLoadOptions`; they inherit the same default sign policy as
    // `open_verified` (`SignLoadPolicy::RequireIfSupported`).
    let sign = match auxiliary_path(&plan, SIGN_AUX_NAME)? {
        Some(path) => Some(SignBitmap::load(path)?),
        None if crate::ordinal::sign_compatible(metadata.dim, metadata_bits) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                crate::DenseError::MissingSignSidecar,
            ));
        }
        None => None,
    };
    let ids_path = auxiliary_path(&plan, IDS_AUX_NAME)?;

    Ok(LoadedBundle {
        rankquant,
        sign,
        ids_path,
    })
}

pub(crate) fn auxiliary_path(
    plan: &ordvec_manifest::VerifiedLoadPlan,
    name: &str,
) -> io::Result<Option<PathBuf>> {
    let mut found = None;
    for artifact in plan.auxiliary_artifacts() {
        if artifact.name() == name {
            if found.is_some() {
                return Err(invalid_data(format!(
                    "auxiliary artifact {name:?} is duplicated"
                )));
            }
            let path = artifact.path().ok_or_else(|| {
                invalid_data(format!("auxiliary artifact {name:?} has no verified path"))
            })?;
            found = Some(path.to_path_buf());
        }
    }
    Ok(found)
}

fn write_ids_file(path: &Path, ids: &[u64]) -> io::Result<()> {
    let mut file = File::create(path)?;
    file.write_all(IDS_MAGIC)?;
    file.write_all(&(ids.len() as u64).to_le_bytes())?;
    for id in ids {
        file.write_all(&id.to_le_bytes())?;
    }
    file.flush()
}

pub(crate) fn read_ids_file(path: &Path, expected_len: usize) -> io::Result<Vec<u64>> {
    let mut file = File::open(path)?;
    let expected = u64::try_from(expected_len).map_err(|_| {
        invalid_data(format!(
            "ID sidecar expected row count {expected_len} exceeds u64::MAX"
        ))
    })?;
    let expected_file_size = expected
        .checked_mul(8)
        .and_then(|bytes| bytes.checked_add(16))
        .ok_or_else(|| invalid_data("ID sidecar expected file size overflow"))?;
    let observed_file_size = file.metadata()?.len();
    if observed_file_size < expected_file_size {
        return Err(invalid_data("truncated OrdinalDB ID sidecar"));
    }
    if observed_file_size > expected_file_size {
        return Err(invalid_data("trailing bytes in OrdinalDB ID sidecar"));
    }

    let mut magic = [0u8; 8];
    read_exact_invalid(&mut file, &mut magic)?;
    if &magic != IDS_MAGIC {
        return Err(invalid_data("invalid OrdinalDB ID sidecar magic"));
    }

    let count = read_u64(&mut file)?;
    if count != expected {
        return Err(invalid_data(format!(
            "ID sidecar count {count} does not match index len {expected_len}"
        )));
    }

    let mut ids = Vec::with_capacity(expected_len);
    let mut seen = HashSet::with_capacity(expected_len);
    for _ in 0..expected_len {
        let id = read_u64(&mut file)?;
        if !seen.insert(id) {
            return Err(invalid_data(format!(
                "duplicate ID {id} in persisted ID sidecar"
            )));
        }
        ids.push(id);
    }

    let mut trailing = [0u8; 1];
    if file.read(&mut trailing)? != 0 {
        return Err(invalid_data("trailing bytes in OrdinalDB ID sidecar"));
    }
    Ok(ids)
}

fn read_u64(reader: &mut impl Read) -> io::Result<u64> {
    let mut bytes = [0u8; 8];
    read_exact_invalid(reader, &mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_exact_invalid(reader: &mut impl Read, buf: &mut [u8]) -> io::Result<()> {
    reader.read_exact(buf).map_err(|err| {
        if err.kind() == io::ErrorKind::UnexpectedEof {
            invalid_data("truncated OrdinalDB ID sidecar")
        } else {
            err
        }
    })
}

fn temp_bundle_path(path: &Path) -> io::Result<PathBuf> {
    let file_name = path.file_name().ok_or_else(|| {
        invalid_input(format!(
            "bundle path {} must name a directory",
            path.display()
        ))
    })?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(io_other)?
        .as_nanos();
    let file_name = file_name.to_string_lossy();
    let temp_name = format!(
        ".{}.tmp-{}-{now}",
        canonical_bundle_name(&file_name),
        std::process::id()
    );
    Ok(parent.join(temp_name))
}

/// Strips the ".{name}.tmp-{pid}-{nanos}" scratch decoration (applied one or
/// more times) from a bundle file name, returning the canonical bundle name.
///
/// Callers such as adapter generation replacement write bundles into their
/// own scratch paths that already carry this decoration. Deriving a fresh
/// scratch name from the decorated name would stack suffixes
/// ("..g….odb.tmp-….tmp-…"), leaving crash debris that generation-directory
/// tooling cannot attribute to a generation. Names that do not match the
/// scratch shape are returned unchanged.
fn canonical_bundle_name(file_name: &str) -> &str {
    let trimmed = file_name.trim_start_matches('.');
    if trimmed.len() == file_name.len() {
        // No leading dot: not a scratch-decorated name.
        return file_name;
    }
    match trimmed.split_once(".tmp-") {
        Some((canonical, suffix)) if !canonical.is_empty() && !suffix.is_empty() => canonical,
        _ => file_name,
    }
}

fn unique_backup_bundle_path(path: &Path) -> io::Result<PathBuf> {
    let file_name = path.file_name().ok_or_else(|| {
        invalid_input(format!(
            "bundle path {} must name a directory",
            path.display()
        ))
    })?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(io_other)?
        .as_nanos();
    let backup_name = format!(
        ".{}.bak-{}-{now}",
        file_name.to_string_lossy(),
        std::process::id()
    );
    Ok(parent.join(backup_name))
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn invalid_data_err(err: impl std::fmt::Display) -> io::Error {
    invalid_data(err.to_string())
}

fn io_other(err: impl std::fmt::Display) -> io::Error {
    io::Error::other(err.to_string())
}

#[cfg(test)]
mod tests {
    use super::{
        load_ordinal_bundle, read_ids_file, replace_bundle, temp_bundle_path,
        unique_backup_bundle_path, write_ordinal_bundle, IDS_MAGIC,
    };
    use ordvec::RankQuant;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    const VECTORS: &[f32] = &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];

    #[test]
    fn replace_bundle_restores_existing_target_when_temp_rename_fails() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("ordinaldb-replace-bundle-{stamp}"));
        let target = root.join("target.odb");
        let missing_temp = root.join("missing-temp.odb");
        fs::create_dir_all(&target).unwrap();
        fs::write(target.join("marker"), b"still here").unwrap();

        let err = replace_bundle(&target, &missing_temp).unwrap_err();
        assert!(err.kind() == std::io::ErrorKind::NotFound, "{err}");
        assert_eq!(fs::read(target.join("marker")).unwrap(), b"still here");

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn load_bundle_recovers_verified_backup_when_target_is_missing() {
        let root = temp_root("recover-backup");
        let target = root.join("target.odb");
        write_valid_bundle(&target);
        let backup = unique_backup_bundle_path(&target).unwrap();
        fs::rename(&target, &backup).unwrap();

        let loaded = load_ordinal_bundle(&target).unwrap();

        assert_eq!(loaded.rankquant.len(), 2);
        assert!(target.join("manifest.json").is_file());
        assert!(!backup.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn load_bundle_rejects_symlink_backup_candidate() {
        let root = temp_root("reject-symlink-backup");
        let target = root.join("target.odb");
        let real_backup = root.join("real-backup.odb");
        write_valid_bundle(&real_backup);
        let backup = root.join(".target.odb.bak-symlink");
        std::os::unix::fs::symlink(&real_backup, &backup).unwrap();

        let err = match load_ordinal_bundle(&target) {
            Ok(_) => panic!("symlink backup candidate must fail"),
            Err(err) => err,
        };

        assert!(
            err.to_string()
                .contains("backup bundle candidate must not be a symlink"),
            "{err}"
        );
        assert!(!target.exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn replace_bundle_does_not_remove_preexisting_backup_before_publish() {
        let root = temp_root("preserve-old-backup");
        let target = root.join("target.odb");
        let missing_temp = root.join("missing-temp.odb");
        let old_backup = root.join(".target.odb.bak-old");
        fs::create_dir_all(&target).unwrap();
        fs::write(target.join("marker"), b"still here").unwrap();
        fs::create_dir_all(&old_backup).unwrap();
        fs::write(old_backup.join("old-marker"), b"old backup").unwrap();

        let err = replace_bundle(&target, &missing_temp).unwrap_err();

        assert!(err.kind() == std::io::ErrorKind::NotFound, "{err}");
        assert_eq!(fs::read(target.join("marker")).unwrap(), b"still here");
        assert_eq!(
            fs::read(old_backup.join("old-marker")).unwrap(),
            b"old backup"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn replace_bundle_rejects_symlink_target_without_mutating_temp() {
        let root = temp_root("symlink-target");
        let real = root.join("real.odb");
        let target = root.join("target.odb");
        let temp = root.join("temp.odb");
        fs::create_dir_all(&real).unwrap();
        fs::create_dir_all(&temp).unwrap();
        std::os::unix::fs::symlink(&real, &target).unwrap();

        let err = replace_bundle(&target, &temp).unwrap_err();

        assert!(err.to_string().contains("symlink"), "{err}");
        assert!(!temp.exists());
        assert!(target.is_symlink());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn recover_bundle_prefers_newest_backup_when_pid_widths_differ() {
        let root = temp_root("newest-backup-pid-width");
        let target = root.join("target.odb");

        // Newest backup (largest nanos) is written FIRST so its directory
        // mtime is older; a recovery order keyed on mtime alone would pick
        // the stale backup and fail this test.
        let newest = root.join(".target.odb.bak-1000-2222");
        write_bundle_with_rows(&newest, 3);
        // Stale backup: pid 999, smaller nanos. Lexicographically
        // ".target.odb.bak-999-..." sorts AFTER ".target.odb.bak-1000-..."
        // because '9' > '1', so a lexicographic sort restores the stale copy.
        let stale = root.join(".target.odb.bak-999-1111");
        write_bundle_with_rows(&stale, 2);

        let loaded = load_ordinal_bundle(&target).unwrap();

        assert_eq!(
            loaded.rankquant.len(),
            3,
            "recovery must restore the chronologically newest backup (nanos 2222), \
             not the lexicographically largest name"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn recover_bundle_prefers_newest_backup_when_nanos_widths_differ() {
        let root = temp_root("newest-backup-nanos-width");
        let target = root.join("target.odb");

        // Same pid for both backups; only the nanos digit count differs.
        // Newest written first so mtime cannot mask a wrong ordering.
        let newest = root.join(".target.odb.bak-42-100");
        write_bundle_with_rows(&newest, 3);
        // Stale backup: nanos 99. "99" sorts after "100" lexicographically.
        let stale = root.join(".target.odb.bak-42-99");
        write_bundle_with_rows(&stale, 2);

        let loaded = load_ordinal_bundle(&target).unwrap();

        assert_eq!(
            loaded.rankquant.len(),
            3,
            "recovery must restore the chronologically newest backup (nanos 100), \
             not the lexicographically largest name"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn recover_bundle_restores_backup_with_unparseable_suffix() {
        let root = temp_root("legacy-backup-name");
        let target = root.join("target.odb");
        // A backup whose suffix does not parse as "{pid}-{nanos}" must still
        // be restorable (fallback path for backups from older builds).
        let legacy = root.join(".target.odb.bak-legacy");
        write_bundle_with_rows(&legacy, 2);

        let loaded = load_ordinal_bundle(&target).unwrap();

        assert_eq!(loaded.rankquant.len(), 2);
        assert!(target.join("manifest.json").is_file());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn temp_bundle_path_derives_single_suffix_from_canonical_generation_name() {
        // Adapter generation replacement writes bundles into caller-provided
        // scratch paths shaped ".g000000000005.odb.tmp-{pid}-{nanos}". The
        // internal scratch name must be re-derived from the CANONICAL bundle
        // name, never by re-tempifying the already-decorated name — a SIGKILL
        // mid-replacement used to leave "..g…odb.tmp-…tmp-…" debris that gc
        // and verify could not parse.
        let root = temp_root("single-suffix-temp");
        fs::create_dir_all(root.join("vectors")).unwrap();

        for decorated in [
            ".g000000000005.odb.tmp-211848-1783014389442708489",
            "..g000000000005.odb.tmp-1-2.tmp-3-4",
        ] {
            let temp = temp_bundle_path(&root.join("vectors").join(decorated)).unwrap();
            let name = temp.file_name().unwrap().to_string_lossy().into_owned();
            assert!(
                name.starts_with(".g000000000005.odb.tmp-"),
                "temp name for {decorated:?} must be derived from the canonical \
                 generation name, got {name:?}"
            );
            assert_eq!(
                name.matches(".tmp-").count(),
                1,
                "temp name must carry exactly one temp suffix, got {name:?}"
            );
        }

        // Canonical (undecorated) targets keep the existing naming scheme.
        let temp = temp_bundle_path(&root.join("vectors").join("g000000000005.odb")).unwrap();
        let name = temp.file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.starts_with(".g000000000005.odb.tmp-"), "{name}");
        assert_eq!(name.matches(".tmp-").count(), 1, "{name}");

        // Dot-prefixed names that are NOT scratch-decorated are left alone.
        let temp = temp_bundle_path(&root.join("vectors").join(".hidden.odb")).unwrap();
        let name = temp.file_name().unwrap().to_string_lossy().into_owned();
        assert!(name.starts_with("..hidden.odb.tmp-"), "{name}");

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn write_bundle_to_decorated_scratch_target_round_trips() {
        // End-to-end over the replacement flow's first hop: writing a bundle
        // to an already-decorated scratch target must publish it there and
        // leave no other entries behind in the parent directory.
        let root = temp_root("decorated-scratch-write");
        let scratch = root
            .join("vectors")
            .join(".g000000000005.odb.tmp-4242-1234567890");
        write_valid_bundle(&scratch);

        let loaded = load_ordinal_bundle(&scratch).unwrap();
        assert_eq!(loaded.rankquant.len(), 2);
        let leftovers: Vec<String> = fs::read_dir(root.join("vectors"))
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name != ".g000000000005.odb.tmp-4242-1234567890")
            .collect();
        assert!(
            leftovers.is_empty(),
            "no scratch residue may remain: {leftovers:?}"
        );

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn read_ids_file_rejects_truncated_file_before_large_allocation() {
        let root = temp_root("truncated-ids");
        let path = root.join("ids.bin");
        fs::create_dir_all(&root).unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(IDS_MAGIC);
        bytes.extend_from_slice(&1_000_000u64.to_le_bytes());
        fs::write(&path, bytes).unwrap();

        let err = read_ids_file(&path, 1_000_000).unwrap_err();
        assert!(err.to_string().contains("truncated"), "{err}");

        let _ = fs::remove_dir_all(root);
    }

    fn write_valid_bundle(path: &std::path::Path) {
        let mut rankquant = RankQuant::new(4, 2);
        rankquant.add(VECTORS);
        write_ordinal_bundle(path, &rankquant, None).unwrap();
    }

    fn write_bundle_with_rows(path: &std::path::Path, rows: usize) {
        let mut rankquant = RankQuant::new(4, 2);
        let vectors = (0..rows * 4).map(|value| value as f32).collect::<Vec<_>>();
        rankquant.add(&vectors);
        write_ordinal_bundle(path, &rankquant, None).unwrap();
    }

    fn temp_root(name: &str) -> std::path::PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("ordinaldb-{name}-{stamp}"));
        fs::create_dir_all(&root).unwrap();
        root
    }
}
