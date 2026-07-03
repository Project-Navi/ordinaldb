//! Hybrid retrieval primitives for OrdinalDB.
//!
//! This crate owns the reusable sparse side of hybrid retrieval: BM25 mmap
//! search, stable row-id results, manifest-verified sidecar loading, and
//! deterministic reciprocal rank fusion. The core `ordinaldb` crate remains
//! vector-only by default; its `hybrid` Cargo feature re-exports this API at
//! `ordinaldb::hybrid` (including the `HybridBundle` one-call open for a
//! bundle's dense-plus-sparse pair).
//!
//! For the full downstream walkthrough against the `ordinaldb` crate —
//! build dense + sparse, write one verified bundle, reopen both sides, and
//! RRF-fuse the results — see `examples/downstream-smoke/src/main.rs` in
//! the repository.
//!
//! # Example: verified sparse sidecar + RRF fusion
//!
//! Build a BM25 index, register it as a checksummed auxiliary artifact of a
//! manifest-verified bundle, reopen it through manifest verification, and
//! fuse it with dense results:
//!
//! ```
//! use ordinaldb_hybrid::{
//!     rrf_fuse_batch, Bm25MmapIndex, RankedBatch, RrfConfig, ScoredRow, SparseIndexBuilder,
//!     TokenizerKind, DEFAULT_SPARSE_AUX_NAME,
//! };
//! use ordvec_manifest::{
//!     create_manifest_for_index_with_options, write_manifest_file, CreateAuxiliaryArtifact,
//!     CreateManifestOptions, CreateRowIdentity, VerifyOptions,
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let dir = tempfile::TempDir::new()?;
//! let sparse_path = dir.path().join("sparse.bm25");
//! let index_path = dir.path().join("index.ovrq");
//! let manifest_path = dir.path().join("manifest.json");
//!
//! // 1. Build the sparse side. With `CreateRowIdentity::RowIdIdentity`,
//! //    sparse row ids must be the dense index's row ordinals `0..n`.
//! let mut builder = SparseIndexBuilder::new(TokenizerKind::Simple);
//! builder.add_text(0, "users need access to the bucket")?;
//! builder.add_text(1, "rotate the client certificate")?;
//! builder.add_text(2, "connection pool exhausted")?;
//! builder.write_mmap(&sparse_path)?;
//!
//! // 2. Write the dense primary artifact and a manifest registering the
//! //    sparse index as a required auxiliary. (Applications building on the
//! //    `ordinaldb` crate do this via `write_verified_bundle` instead.)
//! let mut rankquant = ordvec::RankQuant::new(4, 2);
//! rankquant.add(&[
//!     1.0, 0.0, 0.0, 0.0, //
//!     0.0, 1.0, 0.0, 0.0, //
//!     0.0, 0.0, 1.0, 0.0,
//! ]);
//! rankquant.write(&index_path)?;
//! let mut options = CreateManifestOptions::default();
//! options.auxiliary_artifacts.push(CreateAuxiliaryArtifact {
//!     name: DEFAULT_SPARSE_AUX_NAME.to_string(),
//!     path: sparse_path.clone(),
//!     required: true,
//! });
//! let manifest = create_manifest_for_index_with_options(
//!     &index_path,
//!     CreateRowIdentity::RowIdIdentity,
//!     "example-model",
//!     &manifest_path,
//!     options,
//! )?;
//! write_manifest_file(&manifest, &manifest_path)?;
//!
//! // 3. Reopen the sparse sidecar through manifest verification.
//! let sparse = Bm25MmapIndex::open_verified_sidecar(
//!     &manifest_path,
//!     DEFAULT_SPARSE_AUX_NAME,
//!     VerifyOptions::default(),
//! )?;
//! let sparse_rows = sparse.search_batch(&["access"], 3)?;
//!
//! // 4. Fuse with a dense ranked batch (hand-written here; in a real
//! //    application these scores come from `ordinaldb` dense search).
//! let dense_rows = RankedBatch::from_ranked_lists(vec![vec![
//!     ScoredRow { row_id: 2, score: 0.9 },
//!     ScoredRow { row_id: 0, score: 0.7 },
//! ]])?;
//! let fused = rrf_fuse_batch(&dense_rows, &sparse_rows, RrfConfig::default())?;
//! // Row 0 is found by both sides, so RRF ranks it first.
//! assert_eq!(fused.hits_for_query(0).unwrap()[0].row_id, 0);
//! # Ok(())
//! # }
//! ```

mod error;
mod fusion;
#[cfg(feature = "ltr")]
mod ltr;
#[cfg(feature = "ltr")]
mod ltr_features;
#[cfg(feature = "ltr")]
mod ltr_manifest;
#[cfg(feature = "ltr")]
mod ltr_model;
mod sparse;

pub use error::{HybridError, Result};
pub use fusion::{rrf_fuse_batch, FusedBatch, FusedRow, RankedBatch, RrfConfig, ScoredRow};
#[cfg(feature = "ltr")]
pub use ltr::{
    rerank_fused_batch, LtrCandidateFeatures, LtrDocFeatureLookup, LtrFeatureBatch,
    LtrFeatureInputs, LtrFeatureSchema, LtrLoadOptions, LtrModelInfo, LtrRerankConfig, LtrReranker,
    LtrTree, LtrTreeEnsembleRecord, LtrTreeNode, TreeEnsembleReranker, DEFAULT_LTR_MODEL_AUX_NAME,
    LTR_FEATURE_SCHEMA_V1, LTR_MODEL_SCHEMA_V1,
};
pub use sparse::{
    Bm25MmapIndex, PreparedAllowlist, SparseBuildReport, SparseIndexBuilder, SparseInspectReport,
    TokenizerKind, DEFAULT_SPARSE_AUX_NAME,
};

#[cfg(test)]
mod manifest_api_tests {
    use ordvec_manifest::{verify_for_load, VerifiedLoadPlan, VerifyOptions};

    #[test]
    fn ordvec_manifest_exposes_verified_auxiliary_api() {
        fn assert_api(
            manifest_path: &std::path::Path,
            options: VerifyOptions,
        ) -> std::result::Result<(), ordvec_manifest::VerifiedLoadPlanError> {
            let plan: VerifiedLoadPlan = verify_for_load(manifest_path, options)?;
            let _ = plan.auxiliary_by_name("ordinaldb.sparse_bm25");
            let _ = plan.require_auxiliary("ordinaldb.sparse_bm25");
            Ok(())
        }

        let _ = assert_api;
    }
}
