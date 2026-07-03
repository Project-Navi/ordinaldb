//! Bundle manifest creation and verification.
//!
//! `manifest.json` is the integrity anchor of an OrdinalDB `.odb` bundle: it
//! records the primary artifact's shape (dim, bits, row count), its
//! row-identity kind, and a list of auxiliary artifacts (sign sidecar, ID
//! sidecar, and any caller-supplied extras) each with a SHA-256 checksum
//! and size. This module re-exports the manifest primitives from the
//! `ordvec-manifest` crate and adds a small amount of OrdinalDB-specific
//! glue ([`crate::manifest::AuxiliaryArtifactDeclaration`] and the
//! size-limit helpers below).

use std::path::{Path, PathBuf};

pub use ordvec_manifest::{
    load_manifest_file_with_options, write_manifest_file, AuxiliaryArtifact,
    CreateAuxiliaryArtifact, CreateManifestOptions, CreateRowIdentity, IndexManifest,
    ManifestError, ManifestIndexKind, ManifestIndexParams, RequireAuxiliaryError, ResourceLimits,
    VerificationReport, VerifyOptions,
};

/// Successful outcome of [`verify_for_load`]: a verified plan describing
/// where to load the bundle's primary and auxiliary artifacts from. See
/// `ordvec_manifest::VerifiedLoadPlan` for its methods (`artifact_path`,
/// `metadata`, `row_identity`, `auxiliary_artifacts`, `require_auxiliary`).
pub type VerifiedLoadReport = ordvec_manifest::VerifiedLoadPlan;
/// A single verified auxiliary artifact within a [`VerifiedLoadReport`].
pub type VerifiedAuxiliaryArtifactReport = ordvec_manifest::VerifiedAuxiliaryArtifactPlan;
/// The verified row-identity declaration within a [`VerifiedLoadReport`].
pub type VerifiedRowIdentityReport = ordvec_manifest::VerifiedRowIdentityPlan;
/// Error type returned by [`verify_for_load`] when a manifest fails
/// structural verification (checksum mismatch, size-limit violation,
/// unsafe path, unsupported row-identity kind, ...).
pub type VerificationError = ordvec_manifest::VerifiedLoadPlanError;

/// Declares an extra file to copy into a bundle and register as a named,
/// checksummed auxiliary artifact when calling `write_verified_bundle` on
/// [`crate::OrdinalIndex`] or [`crate::IdMapIndex`].
///
/// `bundle_path` must be a bundle-relative path with no absolute or `..`
/// components, and must not collide with one of the bundle's reserved
/// files (`manifest.json`, `index.ovrq`, `sign.ovsb`, `ids.bin`) or with
/// another declared artifact's name or path — violating either is an
/// error at write time.
#[derive(Clone, Debug)]
pub struct AuxiliaryArtifactDeclaration {
    /// Unique name under which the artifact is registered in
    /// `manifest.json`.
    pub name: String,
    /// Existing file to copy from.
    pub source_path: PathBuf,
    /// Destination path, relative to the bundle root, that the file is
    /// copied to.
    pub bundle_path: PathBuf,
    /// Whether `verify_for_load` must treat a missing or invalid copy of
    /// this artifact as a hard failure.
    pub required: bool,
}

impl AuxiliaryArtifactDeclaration {
    /// Declare a required auxiliary artifact: verification fails if it is
    /// missing or invalid.
    pub fn required(
        name: impl Into<String>,
        source_path: impl Into<PathBuf>,
        bundle_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            name: name.into(),
            source_path: source_path.into(),
            bundle_path: bundle_path.into(),
            required: true,
        }
    }

    /// Declare an optional auxiliary artifact: verification tolerates it
    /// being missing.
    pub fn optional(
        name: impl Into<String>,
        source_path: impl Into<PathBuf>,
        bundle_path: impl Into<PathBuf>,
    ) -> Self {
        Self {
            name: name.into(),
            source_path: source_path.into(),
            bundle_path: bundle_path.into(),
            required: false,
        }
    }
}

/// Verify a bundle's `manifest.json` at `manifest_path` (checksums, size
/// limits, and path safety, per `options`), returning a plan describing
/// where to load each artifact from.
///
/// Thin wrapper over `ordvec_manifest::verify_for_load`.
///
/// # Errors
/// Returns [`VerificationError`] if the manifest is malformed, an artifact
/// is missing or fails its checksum, or a size/path-safety limit in
/// `options` is violated.
pub fn verify_for_load(
    manifest_path: impl AsRef<Path>,
    options: VerifyOptions,
) -> Result<VerifiedLoadReport, VerificationError> {
    ordvec_manifest::verify_for_load(manifest_path, options)
}

/// Return a copy of `options` with `limits.max_auxiliary_artifact_bytes`
/// set to `bytes` (the maximum size `verify_for_load` will accept for any
/// single auxiliary artifact).
pub fn with_auxiliary_size_limit(mut options: VerifyOptions, bytes: u64) -> VerifyOptions {
    options.limits.max_auxiliary_artifact_bytes = bytes;
    options
}

/// In-place equivalent of [`with_auxiliary_size_limit`].
pub fn set_auxiliary_size_limit(options: &mut VerifyOptions, bytes: u64) {
    options.limits.max_auxiliary_artifact_bytes = bytes;
}
