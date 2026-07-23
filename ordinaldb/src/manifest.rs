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

use sha2::{Digest, Sha256};

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

/// Reading extensions for a verified auxiliary artifact
/// ([`VerifiedAuxiliaryArtifactReport`]).
///
/// [`verify_for_load`] returns a *snapshot*: the plan records each
/// artifact's path and checksum but pins no bytes, holds no descriptors,
/// and takes no locks. A caller that later does a bare `std::fs::read` of
/// [`VerifiedAuxiliaryArtifactReport::path`] therefore parses whatever the
/// file holds *now*, which another actor may have changed since
/// verification (a time-of-check/time-of-use gap). This trait provides the
/// read that re-checks.
pub trait VerifiedAuxiliaryArtifactExt {
    /// Read the artifact's bytes from its verified path and re-check them
    /// against the length and SHA-256 the manifest recorded, returning the
    /// bytes only on an exact match.
    ///
    /// The bytes returned are the very bytes that were hashed — the hash is
    /// computed in memory over the single read — so the check cannot be
    /// defeated by a mutation racing between a separate hash and read. This
    /// makes the safe path the shortest path, closing the TOCTOU a bare
    /// `std::fs::read` of the path leaves open.
    ///
    /// # Errors
    /// Returns a [`VerificationError`] if the artifact has no verified path
    /// or recorded digest, if the file cannot be read, or if its current
    /// bytes do not match the recorded length or SHA-256.
    fn read_verified(&self) -> Result<Vec<u8>, VerificationError>;
}

impl VerifiedAuxiliaryArtifactExt for VerifiedAuxiliaryArtifactReport {
    fn read_verified(&self) -> Result<Vec<u8>, VerificationError> {
        let path = self.path().ok_or_else(|| {
            VerificationError::from(ManifestError::invalid(format!(
                "auxiliary artifact {:?} has no verified path to read",
                self.name()
            )))
        })?;
        let expected_sha = self.sha256().ok_or_else(|| {
            VerificationError::from(ManifestError::invalid(format!(
                "auxiliary artifact {:?} has no recorded sha256 to verify against",
                self.name()
            )))
        })?;
        let bytes =
            std::fs::read(path).map_err(|err| VerificationError::from(ManifestError::from(err)))?;
        // Length is the cheap first discriminator; SHA-256 is authoritative.
        if self.size_bytes().is_some_and(|len| len != bytes.len() as u64) {
            return Err(sha256_reverification_failed(self.name()));
        }
        let digest = hex::encode(Sha256::digest(&bytes));
        if !digest.eq_ignore_ascii_case(expected_sha) {
            return Err(sha256_reverification_failed(self.name()));
        }
        Ok(bytes)
    }
}

fn sha256_reverification_failed(name: &str) -> VerificationError {
    VerificationError::from(ManifestError::invalid(format!(
        "auxiliary artifact {name:?} failed sha256 re-verification: \
         on-disk bytes changed since the manifest was verified"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IdMapIndex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_TEMP: AtomicUsize = AtomicUsize::new(0);

    fn temp_bundle(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "ordinaldb-manifest-{name}-{}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn write_idmap_bundle(path: &std::path::Path) {
        let dim = 64;
        let mut index = IdMapIndex::new(dim, 2).unwrap();
        let vectors = vec![0.125f32; 3 * dim];
        index.add_with_ids(&vectors, &[10, 20, 30]).unwrap();
        index.write(path).unwrap();
    }

    #[test]
    fn read_verified_returns_bytes_then_detects_on_disk_mutation() {
        let bundle = temp_bundle("read-verified");
        let _ = std::fs::remove_dir_all(&bundle);
        write_idmap_bundle(&bundle);

        let plan =
            verify_for_load(bundle.join("manifest.json"), VerifyOptions::default()).unwrap();
        let ids_aux = plan
            .auxiliary_by_name(crate::artifacts::IDS_AUX_NAME)
            .expect("ID sidecar is a declared auxiliary artifact");

        // A fresh verified read returns exactly the on-disk bytes.
        let bytes = ids_aux.read_verified().expect("fresh bytes must verify");
        let on_disk = std::fs::read(ids_aux.path().unwrap()).unwrap();
        assert_eq!(bytes, on_disk);
        assert!(!bytes.is_empty());

        // Mutating the file (same length) must fail the SHA-256 re-check —
        // this is the TOCTOU a bare `std::fs::read` of the path would miss.
        let ids_path = ids_aux.path().unwrap().to_path_buf();
        let mut mutated = on_disk.clone();
        let last = mutated.len() - 1;
        mutated[last] ^= 0xFF;
        std::fs::write(&ids_path, &mutated).unwrap();
        let err = ids_aux
            .read_verified()
            .expect_err("mutated bytes must fail re-verification");
        let message = err.to_string().to_lowercase();
        assert!(
            message.contains("sha") || message.contains("verif"),
            "unexpected error: {err}"
        );

        let _ = std::fs::remove_dir_all(&bundle);
    }
}
