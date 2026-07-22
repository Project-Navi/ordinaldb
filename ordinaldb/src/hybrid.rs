//! Hybrid (dense + sparse) search support, re-exported from
//! `ordinaldb-hybrid`, plus [`crate::hybrid::HybridBundle`]: a one-call
//! open for the dense-plus-sparse pair stored in a single verified `.odb`
//! bundle.
//!
//! Only compiled with the `hybrid` Cargo feature; the learning-to-rank
//! reranking items additionally require `experimental-ltr` (see the
//! `ordinaldb-hybrid` crate's docs for its stability status).
//!
//! For a complete downstream walkthrough (build a bundle with a BM25
//! sidecar, dense search, sparse search, RRF fusion), see
//! `examples/downstream-smoke/src/main.rs` in the repository.

use std::error::Error;
use std::fmt;
use std::path::Path;

pub use ordinaldb_hybrid::{
    rrf_fuse_batch, Bm25MmapIndex, FusedBatch, FusedRow, HybridError, PreparedAllowlist,
    RankedBatch, Result, RrfConfig, ScoredRow, SparseBuildReport, SparseIndexBuilder,
    SparseInspectReport, TokenizerKind, DEFAULT_SPARSE_AUX_NAME,
};

#[cfg(feature = "experimental-ltr")]
pub use ordinaldb_hybrid::{
    rerank_fused_batch, LtrCandidateFeatures, LtrDocFeatureLookup, LtrFeatureBatch,
    LtrFeatureInputs, LtrFeatureSchema, LtrLoadOptions, LtrModelInfo, LtrRerankConfig, LtrReranker,
    LtrTree, LtrTreeEnsembleRecord, LtrTreeNode, TreeEnsembleReranker, DEFAULT_LTR_MODEL_AUX_NAME,
    LTR_FEATURE_SCHEMA_V1, LTR_MODEL_SCHEMA_V1,
};

use crate::error::DenseError;
use crate::manifest::VerifyOptions;
use crate::ordinal::DenseLoadOptions;
use crate::IdMapIndex;

/// Error from [`HybridBundle::open_verified`], reporting which side of the
/// bundle failed to open.
#[derive(Debug)]
pub enum HybridBundleError {
    /// The dense (`IdMapIndex`) side failed manifest verification or load.
    Dense(DenseError),
    /// The sparse (BM25 sidecar) side failed manifest verification or load.
    Sparse(HybridError),
}

impl fmt::Display for HybridBundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dense(error) => write!(f, "hybrid bundle dense side: {error}"),
            Self::Sparse(error) => write!(f, "hybrid bundle sparse side: {error}"),
        }
    }
}

impl Error for HybridBundleError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Dense(error) => Some(error),
            Self::Sparse(error) => Some(error),
        }
    }
}

impl From<DenseError> for HybridBundleError {
    fn from(error: DenseError) -> Self {
        Self::Dense(error)
    }
}

impl From<HybridError> for HybridBundleError {
    fn from(error: HybridError) -> Self {
        Self::Sparse(error)
    }
}

/// The dense and sparse halves of one verified `.odb` bundle, opened
/// together from its `manifest.json`.
///
/// Both fields are plain public handles: use them exactly as you would use
/// an [`IdMapIndex`] and a [`Bm25MmapIndex`] opened by hand, then combine
/// per-query results with [`rrf_fuse_batch`].
pub struct HybridBundle {
    /// The dense vector index, with stable `u64` row IDs.
    pub dense: IdMapIndex,
    /// The BM25 sparse index sidecar, keyed by the same row IDs.
    pub sparse: Bm25MmapIndex,
}

impl HybridBundle {
    /// Open the dense index and the named sparse sidecar of one bundle in a
    /// single call, verifying the manifest for both sides.
    ///
    /// This is the convenience form of the two-step open every hybrid
    /// consumer otherwise writes by hand: [`IdMapIndex::open_verified`] with
    /// `dense_load_options`, then [`Bm25MmapIndex::open_verified_sidecar`]
    /// with `sparse_aux_name` (usually [`DEFAULT_SPARSE_AUX_NAME`]).
    ///
    /// # Errors
    /// Returns [`HybridBundleError::Dense`] if the dense side fails
    /// verification or load, and [`HybridBundleError::Sparse`] if the sparse
    /// sidecar is missing, fails verification, or is malformed.
    ///
    /// # Example
    /// ```
    /// use ordinaldb::artifacts::{MANIFEST_FILE, SPARSE_BM25_AUX_NAME};
    /// use ordinaldb::hybrid::{
    ///     rrf_fuse_batch, HybridBundle, RrfConfig, SparseIndexBuilder, TokenizerKind,
    /// };
    /// use ordinaldb::manifest::{AuxiliaryArtifactDeclaration, CreateManifestOptions, VerifyOptions};
    /// use ordinaldb::{BuildOptions, DenseLoadOptions, OrdinalIndexBuilder, SignPolicy};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let dir = std::env::temp_dir().join(format!(
    /// #     "ordinaldb-hybrid-bundle-doctest-{}-{}",
    /// #     std::process::id(),
    /// #     std::time::SystemTime::now()
    /// #         .duration_since(std::time::UNIX_EPOCH)?
    /// #         .as_nanos(),
    /// # ));
    /// # std::fs::create_dir_all(&dir)?;
    /// let sparse_path = dir.join("sparse.bm25");
    /// let bundle_path = dir.join("docs.odb");
    ///
    /// // One row id space shared by both sides.
    /// let ids = [7_u64, 8, 9];
    /// let texts = ["users need access", "rotate the certificate", "pool exhausted"];
    ///
    /// let mut sparse = SparseIndexBuilder::new(TokenizerKind::Simple);
    /// for (&row_id, text) in ids.iter().zip(texts) {
    ///     sparse.add_text(row_id, text)?;
    /// }
    /// sparse.write_mmap(&sparse_path)?;
    ///
    /// let mut dense = OrdinalIndexBuilder::new(8, 2, BuildOptions { sign: SignPolicy::Disabled })?;
    /// for (slot, &row_id) in ids.iter().enumerate() {
    ///     let vector: Vec<f32> = (0..8).map(|col| ((slot + 1) * (col + 2)) as f32).collect();
    ///     dense.add(row_id, &vector)?;
    /// }
    /// dense.write_verified_bundle(
    ///     &bundle_path,
    ///     CreateManifestOptions::default(),
    ///     vec![AuxiliaryArtifactDeclaration::required(
    ///         SPARSE_BM25_AUX_NAME,
    ///         &sparse_path,
    ///         "sparse.bm25",
    ///     )],
    /// )?;
    ///
    /// // The one-call reopen this method exists for.
    /// let bundle = HybridBundle::open_verified(
    ///     bundle_path.join(MANIFEST_FILE),
    ///     VerifyOptions::default(),
    ///     DenseLoadOptions::default(),
    ///     SPARSE_BM25_AUX_NAME,
    /// )?;
    ///
    /// let query: Vec<f32> = (0..8).map(|col| (col + 2) as f32).collect();
    /// let dense_rows = bundle.dense.search_batch_rows(&query, 3)?;
    /// let sparse_rows = bundle.sparse.search_batch(&["access"], 3)?;
    /// let fused = rrf_fuse_batch(&dense_rows, &sparse_rows, RrfConfig::default())?;
    /// assert!(fused.hits().iter().all(|hit| ids.contains(&hit.row_id)));
    /// # std::fs::remove_dir_all(&dir)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn open_verified(
        manifest_path: impl AsRef<Path>,
        verify_options: VerifyOptions,
        dense_load_options: DenseLoadOptions,
        sparse_aux_name: &str,
    ) -> std::result::Result<Self, HybridBundleError> {
        let manifest_path = manifest_path.as_ref();
        let dense =
            IdMapIndex::open_verified(manifest_path, verify_options.clone(), dense_load_options)?;
        let sparse =
            Bm25MmapIndex::open_verified_sidecar(manifest_path, sparse_aux_name, verify_options)?;
        Ok(Self { dense, sparse })
    }
}
