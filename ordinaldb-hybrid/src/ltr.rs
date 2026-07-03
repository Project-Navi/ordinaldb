//! Experimental learning-to-rank inference support.
//!
//! This module is serving-only. It scores already-trained, OrdinalDB-owned tree
//! ensemble model artifacts over explicit hybrid feature batches. Training and
//! XGBoost model conversion stay outside this crate.

use crate::{RankedBatch, Result};

pub use crate::ltr_features::{
    rerank_fused_batch, LtrCandidateFeatures, LtrDocFeatureLookup, LtrFeatureBatch,
    LtrFeatureInputs,
};
pub use crate::ltr_manifest::DEFAULT_LTR_MODEL_AUX_NAME;
pub use crate::ltr_model::{
    LtrFeatureSchema, LtrLoadOptions, LtrModelInfo, LtrRerankConfig, LtrTree,
    LtrTreeEnsembleRecord, LtrTreeNode, TreeEnsembleReranker, LTR_FEATURE_SCHEMA_V1,
    LTR_MODEL_SCHEMA_V1,
};

/// Inference-only reranker over an already-built LTR feature batch.
pub trait LtrReranker {
    fn model_info(&self) -> &LtrModelInfo;

    fn score_features(&self, features: &[f32]) -> Result<f32>;

    fn rerank_batch(
        &self,
        features: &LtrFeatureBatch,
        config: LtrRerankConfig,
    ) -> Result<RankedBatch>;
}
