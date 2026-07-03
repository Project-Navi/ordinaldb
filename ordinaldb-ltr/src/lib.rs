#![warn(missing_docs)]

//! **EXPERIMENTAL.** Offline learning-to-rank (LTR) feature-cache tooling
//! for OrdinalDB.
//!
//! This crate builds, attaches, and verifies *feature caches* — per-query,
//! per-candidate-row feature matrices and relevance labels, plus the
//! queries/split/provenance metadata needed to train an LTR reranker — as
//! a family of checksummed auxiliary artifacts on an OrdinalDB `.odb`
//! bundle's `manifest.json`. It is the offline data-preparation half of
//! OrdinalDB's LTR story; the online half (scoring candidates with a
//! trained model at query time) lives behind `ordinaldb-hybrid`'s
//! `experimental-ltr` Cargo feature.
//!
//! This crate, its on-disk schema (see [`FEATURE_CACHE_SCHEMA_VERSION`]),
//! and its public API are all pre-1.0 and may change or be removed in any
//! `0.x` release without notice. Depend on it explicitly and pin a version
//! if you use it — do not treat it as a stable, load-bearing dependency.
//!
//! # Where the LTR boundary is today
//!
//! Serving-side LTR is implemented and tested: `TreeEnsembleReranker`,
//! `LtrFeatureBatch`, and `rerank_fused_batch` (in `ordinaldb-hybrid`
//! behind its `ltr` feature, re-exported at `ordinaldb::hybrid` behind
//! `experimental-ltr`), plus this crate's feature-cache write/read paths.
//! Everything on the training side is not:
//!
//! - **No training path exists anywhere in OrdinalDB.** The CLI's
//!   `ordinaldb ltr train`, `ltr attach`, and `ltr inspect` subcommands are
//!   stubs that return a "not implemented yet" error. You bring an external
//!   trainer and convert its output to `LtrTreeEnsembleRecord`'s JSON
//!   format yourself; the model header currently requires
//!   `training_objective` `"rank:pairwise"` and `booster` `"gbtree"`.
//! - **`LtrFeatureBatch::from_inputs` requires an explicit score in every
//!   configured source for every fused row.** A row found by only one
//!   retrieval mode (normal in hybrid search) has no entry in the other
//!   mode's `RankedBatch`, and feature building errors on the first such
//!   gap instead of substituting a sentinel. The current workaround is
//!   score backfill: run each side with `top_k` large enough (up to the
//!   corpus size) that every fused candidate has a score from every
//!   configured source, separate from the smaller `top_k` used for the
//!   user-facing results.
//! - **`ordinaldb ltr features` exports exactly the feature triple
//!   `[bm25_score, bm25_rank, query_len_chars]`** — no dense or fused
//!   features yet.
//!
//! For the working downstream walkthrough of the hybrid (non-LTR) surface
//! this crate builds on, see `examples/downstream-smoke/src/main.rs` in the
//! repository.

use std::error::Error;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Schema-version tag stamped into `feature_schema.json` and
/// `manifest.json`. [`write_feature_cache`] always writes this value, and
/// the verified readers reject any cache whose manifest declares a
/// different one.
pub const FEATURE_CACHE_SCHEMA_VERSION: &str = "ordinaldb.ltr.features.v1";
/// Default manifest auxiliary-artifact name prefix used to attach a
/// feature cache to an OrdinalDB bundle; see
/// [`BundleFeatureCacheOptions::aux_name`].
pub const DEFAULT_LTR_FEATURE_CACHE_AUX_NAME: &str = "ordinaldb.ltr_features";
/// Default per-file size ceiling (2 GiB) enforced when verifying a
/// feature-cache auxiliary artifact; see
/// [`BundleFeatureCacheOptions::max_auxiliary_artifact_bytes`].
pub const DEFAULT_MAX_FEATURE_AUXILIARY_BYTES: u64 = 2 * 1024 * 1024 * 1024;
/// File name of the feature-cache directory's schema document (JSON; see
/// [`FeatureSchemaRecord`]).
pub const FEATURE_SCHEMA_FILE: &str = "feature_schema.json";
/// File name of the per-query candidate-row-count file: `u32` values,
/// little-endian, one per query, in query order.
pub const GROUPS_FILE: &str = "groups.u32";
/// File name of the per-row external row-ID file: `u64` values,
/// little-endian, concatenated in group (query) order.
pub const ROW_IDS_FILE: &str = "row_ids.u64";
/// File name of the per-row relevance-label file: `f32` values,
/// little-endian, parallel to [`ROW_IDS_FILE`].
pub const LABELS_FILE: &str = "labels.f32";
/// File name of the row-major feature matrix: `f32` values, little-endian,
/// `row_count * feature_names.len()` entries, parallel to
/// [`ROW_IDS_FILE`].
pub const FEATURES_FILE: &str = "features.f32";
/// File name of the per-query metadata file: JSON Lines, one
/// [`QueryCacheRecord`] per line, in group order.
pub const QUERIES_FILE: &str = "queries.jsonl";
/// File name of the (caller-defined) train/val/test split-assignment
/// document.
pub const SPLIT_FILE: &str = "split.json";
/// File name of the (caller-defined) provenance document describing how
/// the cache was generated.
pub const PROVENANCE_FILE: &str = "provenance.json";
/// File name of the feature-cache directory's own manifest (see
/// [`FeatureCacheManifest`]) — distinct from the OrdinalDB bundle-level
/// `manifest.json` that a feature cache is attached to.
pub const MANIFEST_FILE: &str = "manifest.json";

/// This crate's `Result` alias, using [`LtrError`].
pub type Result<T> = std::result::Result<T, LtrError>;

/// Errors returned by this crate's feature-cache read/write/verify
/// functions.
#[derive(Debug)]
pub enum LtrError {
    /// A filesystem operation failed.
    Io(io::Error),
    /// A JSON side-file (schema, split, provenance, or a `queries.jsonl`
    /// line) failed to (de)serialize.
    Json(serde_json::Error),
    /// The feature-cache data or manifest failed a shape/content
    /// validation — for example, mismatched group/row/feature counts, a
    /// non-finite feature or label value, a duplicate feature name, an
    /// unexpected schema version, or a rejected symlink/path-escape
    /// attempt.
    Invalid(String),
    /// Loading, verifying, or rewriting the *OrdinalDB bundle's*
    /// `manifest.json` failed while attaching or reading a feature-cache
    /// auxiliary.
    Manifest(String),
}

impl LtrError {
    /// Construct an [`LtrError::Invalid`] with the given message.
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }

    /// Construct an [`LtrError::Manifest`] with the given message.
    pub fn manifest(message: impl Into<String>) -> Self {
        Self::Manifest(message.into())
    }
}

impl fmt::Display for LtrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "io error: {error}"),
            Self::Json(error) => write!(f, "json error: {error}"),
            Self::Invalid(message) => write!(f, "invalid LTR feature cache: {message}"),
            Self::Manifest(message) => write!(f, "manifest verification failed: {message}"),
        }
    }
}

impl Error for LtrError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::Invalid(_) | Self::Manifest(_) => None,
        }
    }
}

impl From<io::Error> for LtrError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for LtrError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

/// One row of `queries.jsonl`: a query's cache-local identifier (as
/// referenced by [`FeatureCacheData::groups`] ordering) and its raw text.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct QueryCacheRecord {
    /// Cache-local query identifier, as referenced by
    /// [`FeatureCacheData::groups`] ordering.
    pub query_id: String,
    /// The query's raw text.
    pub query: String,
}

/// Contents of `feature_schema.json`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FeatureSchemaRecord {
    /// See [`FEATURE_CACHE_SCHEMA_VERSION`].
    pub schema_version: String,
    /// Ordered feature-column names, matching the columns of
    /// `features.f32`.
    pub feature_names: Vec<String>,
    /// Feature names flagged as unsafe for certain training uses (e.g.
    /// label-leaking or dense-embedding-derived features), surfaced for
    /// downstream training tooling to exclude.
    pub forbidden_features: Vec<String>,
    /// Whether dense (embedding-derived) features are present among
    /// `feature_names`.
    pub dense_features_present: bool,
}

/// Top-level manifest for a feature-cache directory (`manifest.json`
/// inside the cache directory, not the OrdinalDB bundle's own manifest).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FeatureCacheManifest {
    /// See [`FEATURE_CACHE_SCHEMA_VERSION`].
    pub schema_version: String,
    /// Mirrors [`FeatureSchemaRecord::feature_names`].
    pub feature_names: Vec<String>,
    /// Number of queries in the cache.
    pub query_count: usize,
    /// Total number of candidate rows across all queries.
    pub row_count: usize,
    /// Number of groups in `groups.u32`; must equal `query_count`.
    pub group_count: usize,
    /// Free-form description of where labels came from (e.g. `"qrels"`).
    pub label_kind: String,
    /// SHA-256 of the labels/qrels source file this cache was built from,
    /// if tracked.
    pub qrels_source_sha256: Option<String>,
    /// SHA-256 of the OrdinalDB bundle `manifest.json` this cache was
    /// generated against, for provenance/staleness checks.
    pub bundle_manifest_sha256: Option<String>,
    /// Mirrors [`FeatureSchemaRecord::forbidden_features`].
    pub forbidden_features_present: Vec<String>,
    /// Mirrors [`FeatureSchemaRecord::dense_features_present`].
    pub dense_features_present: bool,
    /// Per-file integrity descriptors for every cache file.
    pub files: FeatureCacheFiles,
}

/// One [`ArtifactDescriptor`] per file in a feature-cache directory.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FeatureCacheFiles {
    /// Descriptor for [`FEATURE_SCHEMA_FILE`].
    pub feature_schema: ArtifactDescriptor,
    /// Descriptor for [`GROUPS_FILE`].
    pub groups_u32: ArtifactDescriptor,
    /// Descriptor for [`ROW_IDS_FILE`].
    pub row_ids_u64: ArtifactDescriptor,
    /// Descriptor for [`LABELS_FILE`].
    pub labels_f32: ArtifactDescriptor,
    /// Descriptor for [`FEATURES_FILE`].
    pub features_f32: ArtifactDescriptor,
    /// Descriptor for [`QUERIES_FILE`].
    pub queries_jsonl: ArtifactDescriptor,
    /// Descriptor for [`SPLIT_FILE`].
    pub split_json: ArtifactDescriptor,
    /// Descriptor for [`PROVENANCE_FILE`].
    pub provenance_json: ArtifactDescriptor,
}

/// Integrity descriptor for a single on-disk feature-cache file, verified
/// byte-for-byte (size and SHA-256) on read.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ArtifactDescriptor {
    /// Path relative to the feature-cache directory.
    pub path: String,
    /// Hex-encoded SHA-256 of the file's contents.
    pub sha256: String,
    /// File size in bytes.
    pub file_size_bytes: u64,
}

/// In-memory form of everything [`write_feature_cache`] needs to write a
/// feature-cache directory.
///
/// `groups`, `row_ids`, `labels`, and `features` share a common row order:
/// `groups[i]` is the number of candidate rows for query `i`; `row_ids`
/// and `labels` have one entry per candidate row, concatenated in group
/// (query) order; `features` is the same rows laid out row-major with
/// `feature_names.len()` columns each.
#[derive(Clone, Debug)]
pub struct FeatureCacheData {
    /// Ordered feature-column names.
    pub feature_names: Vec<String>,
    /// Per-query candidate-row counts; `groups.len()` must equal
    /// `queries.len()`.
    pub groups: Vec<u32>,
    /// External row IDs, one per candidate row, in group order.
    pub row_ids: Vec<u64>,
    /// Relevance labels, parallel to `row_ids`. Every value must be
    /// finite.
    pub labels: Vec<f32>,
    /// Row-major feature matrix, `row_ids.len() * feature_names.len()`
    /// values. Every value must be finite.
    pub features: Vec<f32>,
    /// Per-query metadata, one entry per group, in the same order as
    /// `groups`.
    pub queries: Vec<QueryCacheRecord>,
    /// Caller-defined train/val/test split assignment.
    pub split: Value,
    /// Caller-defined provenance describing how the cache was generated.
    pub provenance: Value,
    /// Free-form description of where `labels` came from (e.g.
    /// `"qrels"`).
    pub label_kind: String,
    /// SHA-256 of the labels/qrels source file, if tracked.
    pub qrels_source_sha256: Option<String>,
    /// SHA-256 of the OrdinalDB bundle manifest this cache was generated
    /// against, if tracked.
    pub bundle_manifest_sha256: Option<String>,
    /// Feature names flagged as unsafe for certain training uses.
    pub forbidden_features_present: Vec<String>,
    /// Whether dense (embedding-derived) features are present.
    pub dense_features_present: bool,
}

/// Options controlling how a feature cache is attached to, or read from,
/// an OrdinalDB bundle's `manifest.json` as a family of named auxiliary
/// artifacts.
#[derive(Clone, Debug)]
pub struct BundleFeatureCacheOptions {
    /// Base name used to derive the manifest auxiliary-artifact names for
    /// the cache's manifest and each of its 8 files (see
    /// [`feature_cache_auxiliary_paths`]).
    pub aux_name: String,
    /// Maximum size, in bytes, allowed for any single feature-cache
    /// auxiliary file.
    pub max_auxiliary_artifact_bytes: u64,
}

impl Default for BundleFeatureCacheOptions {
    fn default() -> Self {
        Self {
            aux_name: DEFAULT_LTR_FEATURE_CACHE_AUX_NAME.to_string(),
            max_auxiliary_artifact_bytes: DEFAULT_MAX_FEATURE_AUXILIARY_BYTES,
        }
    }
}

/// Report returned by [`write_feature_cache_bundle_auxiliary`] describing
/// what was written and where.
#[derive(Clone, Debug)]
pub struct FeatureCacheBundleReport {
    /// Absolute path to the feature-cache directory that was written
    /// (`bundle.join(cache_relative_dir)`).
    pub cache_root: PathBuf,
    /// The (normalized) bundle-relative directory the cache was written
    /// under.
    pub cache_relative_dir: PathBuf,
    /// The feature-cache directory's own manifest.
    pub manifest: FeatureCacheManifest,
    /// Path to the (re-)verified OrdinalDB bundle `manifest.json`.
    pub verified_manifest_path: PathBuf,
    /// The `aux_name` the cache was registered under.
    pub aux_name: String,
}

/// Result of [`read_verified_feature_cache_bundle_auxiliary`]: the parsed,
/// validated cache manifest plus the resolved and verified path to each
/// cache file.
#[derive(Clone, Debug)]
pub struct VerifiedFeatureCache {
    /// The parsed, shape-validated feature-cache manifest.
    pub manifest: FeatureCacheManifest,
    /// Path to the feature cache's own (verified) `manifest.json`.
    pub manifest_path: PathBuf,
    /// Resolved, verified path to each cache file.
    pub paths: VerifiedFeatureCachePaths,
}

/// Resolved, verified absolute paths for each of a feature cache's 8
/// files.
#[derive(Clone, Debug)]
pub struct VerifiedFeatureCachePaths {
    /// Verified path to [`FEATURE_SCHEMA_FILE`].
    pub feature_schema: PathBuf,
    /// Verified path to [`GROUPS_FILE`].
    pub groups_u32: PathBuf,
    /// Verified path to [`ROW_IDS_FILE`].
    pub row_ids_u64: PathBuf,
    /// Verified path to [`LABELS_FILE`].
    pub labels_f32: PathBuf,
    /// Verified path to [`FEATURES_FILE`].
    pub features_f32: PathBuf,
    /// Verified path to [`QUERIES_FILE`].
    pub queries_jsonl: PathBuf,
    /// Verified path to [`SPLIT_FILE`].
    pub split_json: PathBuf,
    /// Verified path to [`PROVENANCE_FILE`].
    pub provenance_json: PathBuf,
}

/// Validate `data` and write it as a feature-cache directory at `root`
/// (created if missing), including its own `manifest.json`.
///
/// # Errors
/// Returns [`LtrError::Invalid`] if `data` fails validation (empty or
/// duplicate feature names, `groups.len() != queries.len()`, row-count
/// mismatches among `row_ids`/`labels`/`features`, or a non-finite
/// feature/label value), if `root` is currently a symlink, or if any
/// individual cache file already exists — this function never overwrites
/// or truncates an existing file; the caller must clear a stale cache
/// directory first. Returns [`LtrError::Io`] for other filesystem errors.
pub fn write_feature_cache(
    root: impl AsRef<Path>,
    data: &FeatureCacheData,
) -> Result<FeatureCacheManifest> {
    let root = root.as_ref();
    validate_data(data)?;
    fs::create_dir_all(root)?;
    reject_existing_symlink_path(root)?;

    let schema = FeatureSchemaRecord {
        schema_version: FEATURE_CACHE_SCHEMA_VERSION.to_string(),
        feature_names: data.feature_names.clone(),
        forbidden_features: data.forbidden_features_present.clone(),
        dense_features_present: data.dense_features_present,
    };
    write_json(root.join(FEATURE_SCHEMA_FILE), &schema)?;
    write_u32_file(root.join(GROUPS_FILE), &data.groups)?;
    write_u64_file(root.join(ROW_IDS_FILE), &data.row_ids)?;
    write_f32_file(root.join(LABELS_FILE), &data.labels)?;
    write_f32_file(root.join(FEATURES_FILE), &data.features)?;
    write_queries(root.join(QUERIES_FILE), &data.queries)?;
    write_json(root.join(SPLIT_FILE), &data.split)?;
    write_json(root.join(PROVENANCE_FILE), &data.provenance)?;

    let manifest = FeatureCacheManifest {
        schema_version: FEATURE_CACHE_SCHEMA_VERSION.to_string(),
        feature_names: data.feature_names.clone(),
        query_count: data.queries.len(),
        row_count: data.row_ids.len(),
        group_count: data.groups.len(),
        label_kind: data.label_kind.clone(),
        qrels_source_sha256: data.qrels_source_sha256.clone(),
        bundle_manifest_sha256: data.bundle_manifest_sha256.clone(),
        forbidden_features_present: data.forbidden_features_present.clone(),
        dense_features_present: data.dense_features_present,
        files: FeatureCacheFiles {
            feature_schema: describe(root, FEATURE_SCHEMA_FILE)?,
            groups_u32: describe(root, GROUPS_FILE)?,
            row_ids_u64: describe(root, ROW_IDS_FILE)?,
            labels_f32: describe(root, LABELS_FILE)?,
            features_f32: describe(root, FEATURES_FILE)?,
            queries_jsonl: describe(root, QUERIES_FILE)?,
            split_json: describe(root, SPLIT_FILE)?,
            provenance_json: describe(root, PROVENANCE_FILE)?,
        },
    };
    write_json(root.join(MANIFEST_FILE), &manifest)?;
    Ok(manifest)
}

/// Write a feature cache under `bundle/cache_relative_dir` and register
/// its 9 files (the cache's own manifest plus its 8 side-files) as named
/// auxiliary artifacts on the OrdinalDB bundle's `manifest.json` at
/// `bundle`.
///
/// # Errors
/// Returns [`LtrError::Invalid`] if `cache_relative_dir` is not a valid
/// bundle-relative path (see [`normalize_bundle_relative_dir`]) or a
/// symlink is encountered along it; otherwise as [`write_feature_cache`]
/// and [`attach_feature_cache_auxiliaries`].
pub fn write_feature_cache_bundle_auxiliary(
    bundle: impl AsRef<Path>,
    cache_relative_dir: impl AsRef<Path>,
    data: &FeatureCacheData,
    options: BundleFeatureCacheOptions,
) -> Result<FeatureCacheBundleReport> {
    let bundle = bundle.as_ref();
    let cache_relative_dir = normalize_bundle_relative_dir(cache_relative_dir)?;
    reject_bundle_relative_symlinks(bundle, &cache_relative_dir)?;
    let cache_root = bundle.join(&cache_relative_dir);
    let manifest = write_feature_cache(&cache_root, data)?;
    attach_feature_cache_auxiliaries(
        bundle.join(ordinaldb::artifacts::MANIFEST_FILE),
        &cache_relative_dir,
        &options,
    )?;
    let verified_manifest_path = verify_feature_cache_bundle_auxiliary(bundle, &options)?;
    Ok(FeatureCacheBundleReport {
        cache_root,
        cache_relative_dir,
        manifest,
        verified_manifest_path,
        aux_name: options.aux_name,
    })
}

/// Register an already-written feature-cache directory's 9 files as named
/// auxiliary artifacts on the OrdinalDB bundle manifest at `manifest_path`.
///
/// Re-attaching under the same `options.aux_name` is idempotent: any
/// previously-registered auxiliaries sharing that name are replaced, not
/// duplicated. Each file is checked to be a real (non-symlink) file within
/// `options.max_auxiliary_artifact_bytes` before its SHA-256 is computed
/// and written into the manifest.
///
/// # Errors
/// Returns [`LtrError::Manifest`] if the bundle manifest cannot be
/// loaded or rewritten, or [`LtrError::Invalid`] if `options.aux_name` is
/// empty, a cache file is missing/not a regular file/too large, or a
/// symlink is encountered.
pub fn attach_feature_cache_auxiliaries(
    manifest_path: impl AsRef<Path>,
    cache_relative_dir: impl AsRef<Path>,
    options: &BundleFeatureCacheOptions,
) -> Result<()> {
    validate_aux_name(&options.aux_name)?;
    let cache_relative_dir = normalize_bundle_relative_dir(cache_relative_dir)?;
    let manifest_path = manifest_path.as_ref();
    let mut verify_options = ordinaldb::manifest::VerifyOptions::default();
    ordinaldb::manifest::set_auxiliary_size_limit(
        &mut verify_options,
        options.max_auxiliary_artifact_bytes,
    );
    let mut document =
        ordinaldb::manifest::load_manifest_file_with_options(manifest_path, &verify_options)
            .map_err(|error| {
                LtrError::manifest(format!(
                    "failed to load bundle manifest {}: {error}",
                    manifest_path.display()
                ))
            })?;
    let base_dir = manifest_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));

    document
        .manifest
        .auxiliary_artifacts
        .retain(|artifact| !is_feature_cache_auxiliary_name(&artifact.name, &options.aux_name));

    for (name, rel_path) in feature_cache_auxiliary_paths(&cache_relative_dir, &options.aux_name)? {
        let full_path = base_dir.join(&rel_path);
        reject_bundle_relative_symlinks(base_dir, &rel_path)?;
        let metadata = fs::symlink_metadata(&full_path)?;
        if !metadata.is_file() {
            return Err(LtrError::invalid(format!(
                "LTR feature auxiliary {} is not a regular file",
                full_path.display()
            )));
        }
        if metadata.len() > options.max_auxiliary_artifact_bytes {
            return Err(LtrError::invalid(format!(
                "LTR feature auxiliary {} is {} bytes, exceeding max_auxiliary_artifact_bytes={}",
                full_path.display(),
                metadata.len(),
                options.max_auxiliary_artifact_bytes
            )));
        }
        document
            .manifest
            .auxiliary_artifacts
            .push(ordinaldb::manifest::AuxiliaryArtifact {
                name,
                path: manifest_relative_path_string(&rel_path)?,
                sha256: sha256_path(&full_path)?,
                file_size_bytes: metadata.len(),
                required: true,
            });
    }
    ordinaldb::manifest::write_manifest_file(&document.manifest, manifest_path).map_err(
        |error| {
            LtrError::manifest(format!(
                "failed to write bundle manifest {}: {error}",
                manifest_path.display()
            ))
        },
    )?;
    Ok(())
}

/// Verify `bundle`'s manifest and every registered feature-cache auxiliary
/// file under `options.aux_name`, returning the parsed cache manifest and
/// resolved paths.
///
/// This is the read-side counterpart to
/// [`write_feature_cache_bundle_auxiliary`] intended for untrusted or
/// previously-persisted bundles: every path is checked to stay inside the
/// canonicalized bundle root and to be a real (non-symlink) file, and each
/// file's size and SHA-256 are checked against the cache's own manifest
/// descriptors — defending against symlink swaps or tampering between
/// write and read.
///
/// # Errors
/// Returns [`LtrError::Manifest`] if the bundle manifest fails
/// verification or is missing the requested auxiliary, or
/// [`LtrError::Invalid`] if any cache file, path, or descriptor fails
/// validation.
pub fn read_verified_feature_cache_bundle_auxiliary(
    bundle: impl AsRef<Path>,
    options: &BundleFeatureCacheOptions,
) -> Result<VerifiedFeatureCache> {
    validate_aux_name(&options.aux_name)?;
    let bundle = bundle.as_ref();
    let bundle_root = canonical_bundle_root(bundle)?;
    let manifest_path = bundle.join(ordinaldb::artifacts::MANIFEST_FILE);
    let mut verify_options = ordinaldb::manifest::VerifyOptions::default();
    ordinaldb::manifest::set_auxiliary_size_limit(
        &mut verify_options,
        options.max_auxiliary_artifact_bytes,
    );
    let plan =
        ordinaldb::manifest::verify_for_load(&manifest_path, verify_options).map_err(|error| {
            LtrError::manifest(format!(
                "failed to verify bundle manifest {}: {error}",
                manifest_path.display()
            ))
        })?;
    let manifest_artifact = plan.require_auxiliary(&options.aux_name).map_err(|error| {
        LtrError::manifest(format!(
            "verified bundle is missing required LTR feature auxiliary {:?}: {error}",
            options.aux_name
        ))
    })?;
    validate_verified_bundle_path(&bundle_root, manifest_artifact)?;
    let manifest = read_feature_cache_manifest_file(manifest_artifact)?;
    validate_feature_cache_manifest_shape(&manifest)?;

    let paths = VerifiedFeatureCachePaths {
        feature_schema: require_feature_auxiliary(
            &bundle_root,
            &plan,
            &options.aux_name,
            "feature_schema",
        )?,
        groups_u32: require_feature_auxiliary(&bundle_root, &plan, &options.aux_name, "groups")?,
        row_ids_u64: require_feature_auxiliary(&bundle_root, &plan, &options.aux_name, "row_ids")?,
        labels_f32: require_feature_auxiliary(&bundle_root, &plan, &options.aux_name, "labels")?,
        features_f32: require_feature_auxiliary(
            &bundle_root,
            &plan,
            &options.aux_name,
            "features",
        )?,
        queries_jsonl: require_feature_auxiliary(
            &bundle_root,
            &plan,
            &options.aux_name,
            "queries",
        )?,
        split_json: require_feature_auxiliary(&bundle_root, &plan, &options.aux_name, "split")?,
        provenance_json: require_feature_auxiliary(
            &bundle_root,
            &plan,
            &options.aux_name,
            "provenance",
        )?,
    };
    validate_verified_feature_descriptors(&manifest, &paths)?;
    Ok(VerifiedFeatureCache {
        manifest,
        manifest_path: manifest_artifact.to_path_buf(),
        paths,
    })
}

/// Cheaper existence/verification-only form of
/// [`read_verified_feature_cache_bundle_auxiliary`]: runs the same full
/// verification but returns only the path to the cache's own
/// `manifest.json` auxiliary artifact.
///
/// # Errors
/// See [`read_verified_feature_cache_bundle_auxiliary`].
pub fn verify_feature_cache_bundle_auxiliary(
    bundle: impl AsRef<Path>,
    options: &BundleFeatureCacheOptions,
) -> Result<PathBuf> {
    validate_aux_name(&options.aux_name)?;
    let manifest_path = bundle.as_ref().join(ordinaldb::artifacts::MANIFEST_FILE);
    let mut verify_options = ordinaldb::manifest::VerifyOptions::default();
    ordinaldb::manifest::set_auxiliary_size_limit(
        &mut verify_options,
        options.max_auxiliary_artifact_bytes,
    );
    let plan =
        ordinaldb::manifest::verify_for_load(&manifest_path, verify_options).map_err(|error| {
            LtrError::manifest(format!(
                "failed to verify bundle manifest {}: {error}",
                manifest_path.display()
            ))
        })?;
    let path = plan.require_auxiliary(&options.aux_name).map_err(|error| {
        LtrError::manifest(format!(
            "verified bundle is missing required LTR feature auxiliary {:?}: {error}",
            options.aux_name
        ))
    })?;
    read_verified_feature_cache_bundle_auxiliary(bundle, options)?;
    Ok(path.to_path_buf())
}

fn read_feature_cache_manifest_file(path: impl AsRef<Path>) -> Result<FeatureCacheManifest> {
    let file = File::open(path)?;
    Ok(serde_json::from_reader(file)?)
}

/// Validate and normalize a path meant to be relative to a bundle root:
/// rejects empty or absolute paths and any `..`/root/prefix component,
/// and collapses `.` components.
///
/// # Errors
/// Returns [`LtrError::Invalid`] if `path` is empty, absolute, escapes the
/// bundle root, or normalizes to nothing.
pub fn normalize_bundle_relative_dir(path: impl AsRef<Path>) -> Result<PathBuf> {
    let path = path.as_ref();
    if path.as_os_str().is_empty() || path.is_absolute() {
        return Err(LtrError::invalid(
            "feature cache path must be a non-empty bundle-relative path",
        ));
    }
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::Normal(part) => out.push(part),
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir
            | std::path::Component::RootDir
            | std::path::Component::Prefix(_) => {
                return Err(LtrError::invalid(format!(
                    "feature cache path {} must stay inside the bundle",
                    path.display()
                )));
            }
        }
    }
    if out.as_os_str().is_empty() {
        return Err(LtrError::invalid(
            "feature cache path must contain at least one normal component",
        ));
    }
    Ok(out)
}

/// The 9 `(auxiliary_name, bundle_relative_path)` pairs that
/// [`attach_feature_cache_auxiliaries`] registers for a cache at
/// `cache_relative_dir` under `aux_name`: `aux_name` itself (the cache's
/// manifest) plus `{aux_name}.feature_schema`, `.groups`, `.row_ids`,
/// `.labels`, `.features`, `.queries`, `.split`, and `.provenance`.
///
/// # Errors
/// Returns [`LtrError::Invalid`] if `aux_name` is empty or
/// `cache_relative_dir` is not a valid bundle-relative path.
pub fn feature_cache_auxiliary_paths(
    cache_relative_dir: impl AsRef<Path>,
    aux_name: &str,
) -> Result<Vec<(String, PathBuf)>> {
    validate_aux_name(aux_name)?;
    let cache_relative_dir = normalize_bundle_relative_dir(cache_relative_dir)?;
    Ok([
        (aux_name.to_string(), MANIFEST_FILE),
        (format!("{aux_name}.feature_schema"), FEATURE_SCHEMA_FILE),
        (format!("{aux_name}.groups"), GROUPS_FILE),
        (format!("{aux_name}.row_ids"), ROW_IDS_FILE),
        (format!("{aux_name}.labels"), LABELS_FILE),
        (format!("{aux_name}.features"), FEATURES_FILE),
        (format!("{aux_name}.queries"), QUERIES_FILE),
        (format!("{aux_name}.split"), SPLIT_FILE),
        (format!("{aux_name}.provenance"), PROVENANCE_FILE),
    ]
    .into_iter()
    .map(|(name, file)| (name, cache_relative_dir.join(file)))
    .collect())
}

/// Streaming SHA-256 hex digest of a file's contents (64 KiB read
/// buffer). Used for every integrity descriptor in this crate.
///
/// # Errors
/// Returns [`LtrError::Io`] if `path` cannot be opened or read.
pub fn sha256_path(path: impl AsRef<Path>) -> Result<String> {
    let path = path.as_ref();
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = io::Read::read(&mut file, &mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn validate_data(data: &FeatureCacheData) -> Result<()> {
    if data.feature_names.is_empty() {
        return Err(LtrError::invalid("feature_names must not be empty"));
    }
    let mut seen = std::collections::HashSet::with_capacity(data.feature_names.len());
    for name in &data.feature_names {
        if name.trim().is_empty() {
            return Err(LtrError::invalid("feature names must not be empty"));
        }
        if !seen.insert(name) {
            return Err(LtrError::invalid(format!(
                "duplicate feature name {name:?}"
            )));
        }
    }
    if data.groups.len() != data.queries.len() {
        return Err(LtrError::invalid(format!(
            "group count {} does not match query count {}",
            data.groups.len(),
            data.queries.len()
        )));
    }
    let row_count = data
        .groups
        .iter()
        .try_fold(0usize, |sum, &group| sum.checked_add(group as usize))
        .ok_or_else(|| LtrError::invalid("group row count overflows usize"))?;
    if row_count != data.row_ids.len() || row_count != data.labels.len() {
        return Err(LtrError::invalid(format!(
            "group row count {row_count} does not match row_ids {} and labels {}",
            data.row_ids.len(),
            data.labels.len()
        )));
    }
    let expected_features = row_count
        .checked_mul(data.feature_names.len())
        .ok_or_else(|| LtrError::invalid("feature matrix length overflows usize"))?;
    if data.features.len() != expected_features {
        return Err(LtrError::invalid(format!(
            "feature value count {} does not match rows * features {expected_features}",
            data.features.len()
        )));
    }
    for (idx, value) in data.features.iter().enumerate() {
        if !value.is_finite() {
            return Err(LtrError::invalid(format!(
                "feature value {idx} is not finite"
            )));
        }
    }
    for (idx, label) in data.labels.iter().enumerate() {
        if !label.is_finite() {
            return Err(LtrError::invalid(format!("label {idx} is not finite")));
        }
    }
    Ok(())
}

fn validate_feature_cache_manifest_shape(manifest: &FeatureCacheManifest) -> Result<()> {
    if manifest.schema_version != FEATURE_CACHE_SCHEMA_VERSION {
        return Err(LtrError::invalid(format!(
            "feature cache schema_version {:?} is not supported",
            manifest.schema_version
        )));
    }
    if manifest.feature_names.is_empty() {
        return Err(LtrError::invalid(
            "feature cache manifest feature_names must not be empty",
        ));
    }
    let mut seen = std::collections::HashSet::with_capacity(manifest.feature_names.len());
    for name in &manifest.feature_names {
        if name.trim().is_empty() {
            return Err(LtrError::invalid(
                "feature cache manifest feature names must not be empty",
            ));
        }
        if !seen.insert(name) {
            return Err(LtrError::invalid(format!(
                "duplicate feature cache manifest feature name {name:?}"
            )));
        }
    }
    if manifest.query_count != manifest.group_count {
        return Err(LtrError::invalid(format!(
            "feature cache manifest query_count {} does not match group_count {}",
            manifest.query_count, manifest.group_count
        )));
    }
    let expected_feature_values = manifest
        .row_count
        .checked_mul(manifest.feature_names.len())
        .ok_or_else(|| LtrError::invalid("feature cache manifest value count overflows usize"))?;
    checked_bytes(
        "groups.u32",
        manifest.group_count,
        std::mem::size_of::<u32>(),
        manifest.files.groups_u32.file_size_bytes,
    )?;
    checked_bytes(
        "row_ids.u64",
        manifest.row_count,
        std::mem::size_of::<u64>(),
        manifest.files.row_ids_u64.file_size_bytes,
    )?;
    checked_bytes(
        "labels.f32",
        manifest.row_count,
        std::mem::size_of::<f32>(),
        manifest.files.labels_f32.file_size_bytes,
    )?;
    checked_bytes(
        "features.f32",
        expected_feature_values,
        std::mem::size_of::<f32>(),
        manifest.files.features_f32.file_size_bytes,
    )?;
    Ok(())
}

fn checked_bytes(name: &str, count: usize, width: usize, actual: u64) -> Result<()> {
    let expected = count
        .checked_mul(width)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| LtrError::invalid(format!("{name} byte count overflows u64")))?;
    if actual != expected {
        return Err(LtrError::invalid(format!(
            "{name} size {actual} does not match expected {expected}"
        )));
    }
    Ok(())
}

fn validate_verified_feature_descriptors(
    manifest: &FeatureCacheManifest,
    paths: &VerifiedFeatureCachePaths,
) -> Result<()> {
    validate_descriptor(
        &manifest.files.feature_schema,
        FEATURE_SCHEMA_FILE,
        &paths.feature_schema,
    )?;
    validate_descriptor(&manifest.files.groups_u32, GROUPS_FILE, &paths.groups_u32)?;
    validate_descriptor(
        &manifest.files.row_ids_u64,
        ROW_IDS_FILE,
        &paths.row_ids_u64,
    )?;
    validate_descriptor(&manifest.files.labels_f32, LABELS_FILE, &paths.labels_f32)?;
    validate_descriptor(
        &manifest.files.features_f32,
        FEATURES_FILE,
        &paths.features_f32,
    )?;
    validate_descriptor(
        &manifest.files.queries_jsonl,
        QUERIES_FILE,
        &paths.queries_jsonl,
    )?;
    validate_descriptor(&manifest.files.split_json, SPLIT_FILE, &paths.split_json)?;
    validate_descriptor(
        &manifest.files.provenance_json,
        PROVENANCE_FILE,
        &paths.provenance_json,
    )?;
    Ok(())
}

fn validate_descriptor(
    descriptor: &ArtifactDescriptor,
    expected_name: &str,
    verified_path: &Path,
) -> Result<()> {
    if descriptor.path != expected_name {
        return Err(LtrError::invalid(format!(
            "feature cache descriptor path {:?} must be {:?}",
            descriptor.path, expected_name
        )));
    }
    let metadata = fs::symlink_metadata(verified_path)?;
    if !metadata.is_file() {
        return Err(LtrError::invalid(format!(
            "verified feature cache artifact {} is not a regular file",
            verified_path.display()
        )));
    }
    if descriptor.file_size_bytes != metadata.len() {
        return Err(LtrError::invalid(format!(
            "feature cache descriptor {:?} size {} does not match verified artifact size {}",
            expected_name,
            descriptor.file_size_bytes,
            metadata.len()
        )));
    }
    let sha256 = sha256_path(verified_path)?;
    if descriptor.sha256 != sha256 {
        return Err(LtrError::invalid(format!(
            "feature cache descriptor {:?} sha256 does not match verified artifact",
            expected_name
        )));
    }
    Ok(())
}

fn validate_aux_name(aux_name: &str) -> Result<()> {
    if aux_name.trim().is_empty() || aux_name != aux_name.trim() {
        return Err(LtrError::invalid("LTR feature auxiliary name is empty"));
    }
    Ok(())
}

fn is_feature_cache_auxiliary_name(name: &str, aux_name: &str) -> bool {
    name == aux_name
        || name
            .strip_prefix(aux_name)
            .is_some_and(|suffix| suffix.starts_with('.'))
}

fn require_feature_auxiliary(
    bundle_root: &Path,
    plan: &ordinaldb::manifest::VerifiedLoadReport,
    aux_name: &str,
    suffix: &str,
) -> Result<PathBuf> {
    let name = format!("{aux_name}.{suffix}");
    let path = plan.require_auxiliary(&name).map_err(|error| {
        LtrError::manifest(format!(
            "verified bundle is missing required LTR feature auxiliary {name:?}: {error}"
        ))
    })?;
    validate_verified_bundle_path(bundle_root, path)?;
    Ok(path.to_path_buf())
}

fn canonical_bundle_root(bundle: &Path) -> Result<PathBuf> {
    fs::canonicalize(bundle).map_err(|error| {
        LtrError::manifest(format!(
            "failed to canonicalize bundle root {}: {error}",
            bundle.display()
        ))
    })
}

fn validate_verified_bundle_path(bundle_root: &Path, path: &Path) -> Result<()> {
    let relative_path = path.strip_prefix(bundle_root).map_err(|_| {
        LtrError::manifest(format!(
            "verified LTR feature auxiliary {} is outside bundle {}",
            path.display(),
            bundle_root.display()
        ))
    })?;
    reject_bundle_relative_symlinks(bundle_root, relative_path)
}

fn reject_bundle_relative_symlinks(bundle: &Path, relative_path: &Path) -> Result<()> {
    let relative_path = normalize_bundle_relative_dir(relative_path)?;
    reject_existing_symlink_path(bundle)?;

    let mut cursor = bundle.to_path_buf();
    for component in relative_path.components() {
        let std::path::Component::Normal(part) = component else {
            continue;
        };
        cursor.push(part);
        reject_existing_symlink_path(&cursor)?;
    }
    Ok(())
}

fn reject_existing_symlink_path(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(LtrError::invalid(format!(
            "feature cache path {} must not be a symlink",
            path.display()
        ))),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(LtrError::Io(error)),
    }
}

fn manifest_relative_path_string(path: &Path) -> Result<String> {
    let value = path.to_str().ok_or_else(|| {
        LtrError::invalid(format!(
            "manifest auxiliary path {} is not valid UTF-8",
            path.display()
        ))
    })?;
    Ok(value.replace('\\', "/"))
}

fn write_json(path: PathBuf, value: &impl Serialize) -> Result<()> {
    let file = create_feature_cache_file(&path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn write_queries(path: PathBuf, queries: &[QueryCacheRecord]) -> Result<()> {
    let file = create_feature_cache_file(&path)?;
    let mut writer = BufWriter::new(file);
    for query in queries {
        serde_json::to_writer(&mut writer, query)?;
        writer.write_all(b"\n")?;
    }
    Ok(())
}

fn write_u32_file(path: PathBuf, values: &[u32]) -> Result<()> {
    let mut writer = BufWriter::new(create_feature_cache_file(&path)?);
    for &value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn write_u64_file(path: PathBuf, values: &[u64]) -> Result<()> {
    let mut writer = BufWriter::new(create_feature_cache_file(&path)?);
    for &value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn write_f32_file(path: PathBuf, values: &[f32]) -> Result<()> {
    let mut writer = BufWriter::new(create_feature_cache_file(&path)?);
    for &value in values {
        writer.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn create_feature_cache_file(path: &Path) -> Result<File> {
    create_new_feature_cache_file(path).map_err(|error| {
        if error.kind() == io::ErrorKind::AlreadyExists {
            LtrError::invalid(format!(
                "feature cache path {} already exists; refusing to overwrite",
                path.display()
            ))
        } else {
            LtrError::Io(error)
        }
    })
}

#[cfg(unix)]
fn create_new_feature_cache_file(path: &Path) -> io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;

    OpenOptions::new()
        .write(true)
        .create_new(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
}

#[cfg(not(unix))]
fn create_new_feature_cache_file(path: &Path) -> io::Result<File> {
    reject_existing_symlink_path(path).map_err(|error| match error {
        LtrError::Io(error) => error,
        other => io::Error::new(io::ErrorKind::InvalidInput, other),
    })?;
    OpenOptions::new().write(true).create_new(true).open(path)
}

fn describe(root: &Path, name: &str) -> Result<ArtifactDescriptor> {
    let path = root.join(name);
    reject_existing_symlink_path(&path)?;
    let metadata = fs::symlink_metadata(&path)?;
    Ok(ArtifactDescriptor {
        path: name.to_string(),
        sha256: sha256_path(&path)?,
        file_size_bytes: metadata.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> FeatureCacheData {
        FeatureCacheData {
            feature_names: vec!["bm25_score".to_string(), "bm25_rank".to_string()],
            groups: vec![2],
            row_ids: vec![10, 20],
            labels: vec![1.0, 0.0],
            features: vec![3.0, 1.0, 2.0, 2.0],
            queries: vec![QueryCacheRecord {
                query_id: "q1".to_string(),
                query: "alpha".to_string(),
            }],
            split: serde_json::json!({"kind": "none"}),
            provenance: serde_json::json!({"producer": "test"}),
            label_kind: "qrels".to_string(),
            qrels_source_sha256: Some("qrels".to_string()),
            bundle_manifest_sha256: Some("bundle".to_string()),
            forbidden_features_present: Vec::new(),
            dense_features_present: false,
        }
    }

    #[test]
    fn feature_cache_roundtrip_writes_manifest_and_binary_files() {
        let root =
            std::env::temp_dir().join(format!("ordinaldb-ltr-cache-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let data = sample_data();

        let manifest = write_feature_cache(&root, &data).unwrap();

        assert_eq!(manifest.query_count, 1);
        assert_eq!(manifest.row_count, 2);
        assert_eq!(manifest.files.features_f32.file_size_bytes, 16);
        assert!(root.join(MANIFEST_FILE).is_file());
        let _ = fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn bundle_writer_rejects_symlinked_cache_directory() {
        let root =
            std::env::temp_dir().join(format!("ordinaldb-ltr-symlink-test-{}", std::process::id()));
        let bundle = root.join("index.odb");
        let outside = root.join("outside");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&bundle).unwrap();
        fs::create_dir_all(&outside).unwrap();
        std::os::unix::fs::symlink(&outside, bundle.join("ltr")).unwrap();

        let err = write_feature_cache_bundle_auxiliary(
            &bundle,
            "ltr",
            &sample_data(),
            BundleFeatureCacheOptions::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("symlink"), "{err}");
        assert!(!outside.join(MANIFEST_FILE).exists());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn feature_cache_writer_refuses_existing_artifact_without_truncating() {
        let root = std::env::temp_dir().join(format!(
            "ordinaldb-ltr-existing-file-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let artifact = root.join(FEATURE_SCHEMA_FILE);
        fs::write(&artifact, b"keep").unwrap();

        let err = write_feature_cache(&root, &sample_data()).unwrap_err();
        assert!(err.to_string().contains("already exists"), "{err}");
        assert_eq!(fs::read(&artifact).unwrap(), b"keep");
        let _ = fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn feature_cache_writer_rejects_final_symlink_artifact() {
        let root = std::env::temp_dir().join(format!(
            "ordinaldb-ltr-final-symlink-test-{}",
            std::process::id()
        ));
        let outside = root.join("outside.json");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(&outside, b"outside").unwrap();
        std::os::unix::fs::symlink(&outside, root.join(FEATURE_SCHEMA_FILE)).unwrap();

        let err = write_feature_cache(&root, &sample_data()).unwrap_err();
        assert!(err.to_string().contains("already exists"), "{err}");
        assert_eq!(fs::read(&outside).unwrap(), b"outside");
        let _ = fs::remove_dir_all(&root);
    }

    #[cfg(unix)]
    #[test]
    fn verified_reader_rejects_symlinked_cache_directory() {
        let root = std::env::temp_dir().join(format!(
            "ordinaldb-ltr-read-symlink-test-{}",
            std::process::id()
        ));
        let bundle = root.join("index.odb");
        let outside = root.join("outside-ltr");
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();

        let mut index = ordinaldb::OrdinalIndex::new(64, 2).unwrap();
        index.add(&vec![0.25; 64]);
        index
            .write_verified_bundle(
                &bundle,
                ordinaldb::manifest::CreateManifestOptions::default(),
                Vec::new(),
            )
            .unwrap();
        write_feature_cache_bundle_auxiliary(
            &bundle,
            "ltr",
            &sample_data(),
            BundleFeatureCacheOptions::default(),
        )
        .unwrap();

        fs::rename(bundle.join("ltr"), &outside).unwrap();
        std::os::unix::fs::symlink(&outside, bundle.join("ltr")).unwrap();

        let err = read_verified_feature_cache_bundle_auxiliary(
            &bundle,
            &BundleFeatureCacheOptions::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("symlink"), "{err}");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn bundle_relative_paths_reject_escape_and_absolute_paths() {
        let normalized = normalize_bundle_relative_dir("ltr/features").unwrap();
        assert_eq!(
            normalized.components().collect::<Vec<_>>(),
            Path::new("ltr")
                .join("features")
                .components()
                .collect::<Vec<_>>()
        );
        assert!(normalize_bundle_relative_dir("../features").is_err());
        assert!(normalize_bundle_relative_dir("/tmp/features").is_err());
        assert!(normalize_bundle_relative_dir(".").is_err());
    }

    #[test]
    fn feature_auxiliary_name_matching_is_prefix_component_aware() {
        assert!(is_feature_cache_auxiliary_name(
            "ordinaldb.ltr_features",
            DEFAULT_LTR_FEATURE_CACHE_AUX_NAME
        ));
        assert!(is_feature_cache_auxiliary_name(
            "ordinaldb.ltr_features.features",
            DEFAULT_LTR_FEATURE_CACHE_AUX_NAME
        ));
        assert!(!is_feature_cache_auxiliary_name(
            "ordinaldb.ltr_features_v2",
            DEFAULT_LTR_FEATURE_CACHE_AUX_NAME
        ));
    }
}
