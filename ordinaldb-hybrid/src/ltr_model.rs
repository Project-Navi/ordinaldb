use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ltr::LtrReranker;
use crate::ltr_features::{LtrFeatureBatch, LtrFeatureInputs};
use crate::{HybridError, RankedBatch, Result, ScoredRow};

pub const LTR_MODEL_SCHEMA_V1: &str = "ordinaldb.ltr.tree_ensemble.v1";
pub const LTR_FEATURE_SCHEMA_V1: &str = "ordinaldb.ltr.features.v1";
const DOC_CAT_MATCH: &str = "doc_cat_match";

/// Runtime guardrails for loading an OrdinalDB LTR model artifact.
#[derive(Clone, Debug)]
pub struct LtrLoadOptions {
    pub max_model_bytes: u64,
    pub max_trees: usize,
    pub max_nodes_per_tree: usize,
    pub max_depth: usize,
    pub max_features: usize,
    pub allow_leakage_features_for_research: bool,
}

impl Default for LtrLoadOptions {
    fn default() -> Self {
        Self {
            max_model_bytes: 64 * 1024 * 1024,
            max_trees: 4096,
            max_nodes_per_tree: 4096,
            max_depth: 32,
            max_features: 256,
            allow_leakage_features_for_research: false,
        }
    }
}

/// Feature schema embedded in an OrdinalDB LTR model artifact.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LtrFeatureSchema {
    pub schema_version: String,
    pub feature_names: Vec<String>,
    #[serde(default)]
    pub forbidden_features: Vec<String>,
    #[serde(default)]
    pub dense_features_required: bool,
}

/// Public model metadata, excluding tree nodes.
#[derive(Clone, Debug, PartialEq)]
pub struct LtrModelInfo {
    pub schema_version: String,
    pub model_id: String,
    pub model_family: String,
    pub training_objective: String,
    pub booster: String,
    pub base_score: f32,
    pub learning_rate: f32,
    pub feature_schema: LtrFeatureSchema,
    pub training_provenance: BTreeMap<String, Value>,
}

/// OrdinalDB-owned tree ensemble format. This is the runtime format, not raw
/// XGBoost JSON.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LtrTreeEnsembleRecord {
    pub schema_version: String,
    pub model_id: String,
    pub model_family: String,
    pub training_objective: String,
    pub booster: String,
    pub base_score: f32,
    #[serde(default = "default_learning_rate")]
    pub learning_rate: f32,
    pub feature_schema: LtrFeatureSchema,
    #[serde(default)]
    pub training_provenance: BTreeMap<String, Value>,
    pub trees: Vec<LtrTree>,
}

/// One decision tree in the normalized OrdinalDB LTR model format.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LtrTree {
    pub nodes: Vec<LtrTreeNode>,
}

/// A leaf or numerical split node.
///
/// Split semantics are `feature < threshold => left`, otherwise `right`.
/// `default_left` is metadata for exporter parity; P0 scoring rejects
/// non-finite feature values instead of treating them as missing.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LtrTreeNode {
    #[serde(default)]
    pub split_feature: Option<u16>,
    #[serde(default)]
    pub threshold: Option<f32>,
    #[serde(default)]
    pub default_left: bool,
    #[serde(default)]
    pub left: Option<u32>,
    #[serde(default)]
    pub right: Option<u32>,
    #[serde(default)]
    pub leaf_value: Option<f32>,
}

/// Pure-Rust scorer for an OrdinalDB tree ensemble LTR artifact.
#[derive(Clone, Debug)]
pub struct TreeEnsembleReranker {
    info: LtrModelInfo,
    trees: Vec<CompiledTree>,
}

#[derive(Clone, Debug)]
struct CompiledTree {
    nodes: Vec<CompiledNode>,
}

#[derive(Clone, Copy, Debug)]
enum CompiledNode {
    Leaf(f32),
    Split {
        feature: usize,
        threshold: f32,
        left: usize,
        right: usize,
    },
}

/// Reranking configuration for a scored LTR feature batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LtrRerankConfig {
    pub top_k: usize,
}

impl Default for LtrRerankConfig {
    fn default() -> Self {
        Self { top_k: usize::MAX }
    }
}

impl TreeEnsembleReranker {
    pub fn from_record(record: LtrTreeEnsembleRecord) -> Result<Self> {
        Self::from_record_with_options(record, LtrLoadOptions::default())
    }

    pub fn from_record_with_options(
        record: LtrTreeEnsembleRecord,
        options: LtrLoadOptions,
    ) -> Result<Self> {
        validate_options(&options)?;
        validate_feature_schema(&record.feature_schema, &options)?;
        validate_model_header(&record)?;
        if record.trees.len() > options.max_trees {
            return Err(HybridError::limit(format!(
                "LTR model has {} trees, limit is {}",
                record.trees.len(),
                options.max_trees
            )));
        }
        if !record.base_score.is_finite() {
            return Err(HybridError::ltr("LTR base_score must be finite"));
        }
        if !record.learning_rate.is_finite() {
            return Err(HybridError::ltr("LTR learning_rate must be finite"));
        }

        let feature_count = record.feature_schema.feature_names.len();
        let trees = record
            .trees
            .iter()
            .enumerate()
            .map(|(idx, tree)| compile_tree(idx, tree, feature_count, &options))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            info: LtrModelInfo {
                schema_version: record.schema_version,
                model_id: record.model_id,
                model_family: record.model_family,
                training_objective: record.training_objective,
                booster: record.booster,
                base_score: record.base_score,
                learning_rate: record.learning_rate,
                feature_schema: record.feature_schema,
                training_provenance: record.training_provenance,
            },
            trees,
        })
    }

    /// Load an OrdinalDB LTR model directly from a file without manifest
    /// verification. Prefer `load_verified_sidecar` for product loading.
    pub fn load_unverified(path: impl AsRef<Path>, options: LtrLoadOptions) -> Result<Self> {
        let bytes = read_regular_file(path.as_ref(), options.max_model_bytes)?;
        Self::from_slice(&bytes, options)
    }

    pub fn from_slice(bytes: &[u8], options: LtrLoadOptions) -> Result<Self> {
        if bytes.len() as u64 > options.max_model_bytes {
            return Err(HybridError::limit(format!(
                "LTR model is {} bytes, limit is {}",
                bytes.len(),
                options.max_model_bytes
            )));
        }
        let record: LtrTreeEnsembleRecord = serde_json::from_slice(bytes)
            .map_err(|error| HybridError::ltr(format!("invalid LTR model JSON: {error}")))?;
        Self::from_record_with_options(record, options)
    }

    pub fn rerank(
        &self,
        inputs: &LtrFeatureInputs<'_>,
        config: LtrRerankConfig,
    ) -> Result<RankedBatch> {
        let features = LtrFeatureBatch::from_inputs(inputs, &self.info.feature_schema)?;
        self.rerank_batch(&features, config)
    }
}

impl LtrReranker for TreeEnsembleReranker {
    fn model_info(&self) -> &LtrModelInfo {
        &self.info
    }

    fn score_features(&self, features: &[f32]) -> Result<f32> {
        if features.len() != self.info.feature_schema.feature_names.len() {
            return Err(HybridError::ltr(format!(
                "feature vector length {} does not match model feature count {}",
                features.len(),
                self.info.feature_schema.feature_names.len()
            )));
        }
        for (idx, value) in features.iter().enumerate() {
            if !value.is_finite() {
                return Err(HybridError::ltr(format!(
                    "feature {} ({}) is not finite",
                    idx, self.info.feature_schema.feature_names[idx]
                )));
            }
        }

        let mut score = self.info.base_score;
        for tree in &self.trees {
            score += self.info.learning_rate * score_tree(tree, features)?;
        }
        if !score.is_finite() {
            return Err(HybridError::ltr("LTR score is not finite"));
        }
        Ok(score)
    }

    fn rerank_batch(
        &self,
        features: &LtrFeatureBatch,
        config: LtrRerankConfig,
    ) -> Result<RankedBatch> {
        if features.feature_names() != self.info.feature_schema.feature_names.as_slice() {
            return Err(HybridError::ltr(
                "feature batch schema does not match model feature schema",
            ));
        }

        let mut offsets = Vec::with_capacity(features.query_count() + 1);
        let mut hits = Vec::new();
        offsets.push(0);
        for query_idx in 0..features.query_count() {
            let mut query_hits = features
                .rows_for_query(query_idx)
                .unwrap_or(&[])
                .iter()
                .map(|candidate| {
                    Ok(ScoredRow {
                        row_id: candidate.row_id,
                        score: self.score_features(&candidate.features)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            query_hits.sort_by(|a, b| {
                b.score
                    .total_cmp(&a.score)
                    .then_with(|| a.row_id.cmp(&b.row_id))
            });
            query_hits.truncate(config.top_k);
            hits.extend(query_hits);
            offsets.push(hits.len());
        }
        RankedBatch::from_sorted_offsets_hits(offsets, hits)
    }
}

fn validate_options(options: &LtrLoadOptions) -> Result<()> {
    if options.max_model_bytes == 0 {
        return Err(HybridError::limit("LTR max_model_bytes must be positive"));
    }
    if options.max_trees == 0 {
        return Err(HybridError::limit("LTR max_trees must be positive"));
    }
    if options.max_nodes_per_tree == 0 {
        return Err(HybridError::limit(
            "LTR max_nodes_per_tree must be positive",
        ));
    }
    if options.max_depth == 0 {
        return Err(HybridError::limit("LTR max_depth must be positive"));
    }
    if options.max_features == 0 || options.max_features > u16::MAX as usize {
        return Err(HybridError::limit(
            "LTR max_features must be in 1..=u16::MAX",
        ));
    }
    Ok(())
}

fn validate_model_header(record: &LtrTreeEnsembleRecord) -> Result<()> {
    if record.schema_version != LTR_MODEL_SCHEMA_V1 {
        return Err(HybridError::ltr(format!(
            "unsupported LTR model schema {:?}",
            record.schema_version
        )));
    }
    if record.model_id.trim().is_empty() {
        return Err(HybridError::ltr("LTR model_id must not be empty"));
    }
    if record.model_family != "xgboost_gbtree" && record.model_family != "ordinaldb_tree_ensemble" {
        return Err(HybridError::ltr(format!(
            "unsupported LTR model_family {:?}",
            record.model_family
        )));
    }
    if record.training_objective != "rank:pairwise" {
        return Err(HybridError::ltr(format!(
            "unsupported LTR training_objective {:?}",
            record.training_objective
        )));
    }
    if record.booster != "gbtree" {
        return Err(HybridError::ltr(format!(
            "unsupported LTR booster {:?}",
            record.booster
        )));
    }
    Ok(())
}

fn validate_feature_schema(schema: &LtrFeatureSchema, options: &LtrLoadOptions) -> Result<()> {
    if schema.schema_version != LTR_FEATURE_SCHEMA_V1 {
        return Err(HybridError::ltr(format!(
            "unsupported LTR feature schema {:?}",
            schema.schema_version
        )));
    }
    if schema.feature_names.is_empty() {
        return Err(HybridError::ltr("LTR feature schema must not be empty"));
    }
    if schema.feature_names.len() > options.max_features {
        return Err(HybridError::limit(format!(
            "LTR feature schema has {} features, limit is {}",
            schema.feature_names.len(),
            options.max_features
        )));
    }
    let mut seen = std::collections::HashSet::with_capacity(schema.feature_names.len());
    for name in &schema.feature_names {
        if name.trim().is_empty() {
            return Err(HybridError::ltr("LTR feature names must not be empty"));
        }
        if !seen.insert(name) {
            return Err(HybridError::ltr(format!(
                "duplicate LTR feature name {name:?}"
            )));
        }
        let forbidden_by_default = name == DOC_CAT_MATCH;
        let listed_forbidden = schema.forbidden_features.iter().any(|item| item == name);
        if (forbidden_by_default || listed_forbidden)
            && !options.allow_leakage_features_for_research
        {
            return Err(HybridError::ltr(format!(
                "forbidden leakage feature {name:?} is present in LTR feature schema"
            )));
        }
    }
    Ok(())
}

fn compile_tree(
    tree_idx: usize,
    tree: &LtrTree,
    feature_count: usize,
    options: &LtrLoadOptions,
) -> Result<CompiledTree> {
    if tree.nodes.is_empty() {
        return Err(HybridError::ltr(format!(
            "LTR tree {tree_idx} has no nodes"
        )));
    }
    if tree.nodes.len() > options.max_nodes_per_tree {
        return Err(HybridError::limit(format!(
            "LTR tree {tree_idx} has {} nodes, limit is {}",
            tree.nodes.len(),
            options.max_nodes_per_tree
        )));
    }

    let mut compiled = Vec::with_capacity(tree.nodes.len());
    for (node_idx, node) in tree.nodes.iter().enumerate() {
        compiled.push(compile_node(
            tree_idx,
            node_idx,
            node,
            tree.nodes.len(),
            feature_count,
        )?);
    }

    let mut visited = vec![false; compiled.len()];
    let mut stack = vec![false; compiled.len()];
    validate_reachable_tree(
        tree_idx,
        0,
        1,
        options.max_depth,
        &compiled,
        &mut visited,
        &mut stack,
    )?;
    if let Some(unreachable) = visited.iter().position(|&item| !item) {
        return Err(HybridError::ltr(format!(
            "LTR tree {tree_idx} has unreachable node {unreachable}"
        )));
    }

    Ok(CompiledTree { nodes: compiled })
}

fn compile_node(
    tree_idx: usize,
    node_idx: usize,
    node: &LtrTreeNode,
    node_count: usize,
    feature_count: usize,
) -> Result<CompiledNode> {
    match (
        node.leaf_value,
        node.split_feature,
        node.threshold,
        node.left,
        node.right,
    ) {
        (Some(value), None, None, None, None) => {
            if !value.is_finite() {
                return Err(HybridError::ltr(format!(
                    "LTR tree {tree_idx} node {node_idx} leaf value is not finite"
                )));
            }
            Ok(CompiledNode::Leaf(value))
        }
        (None, Some(feature), Some(threshold), Some(left), Some(right)) => {
            if usize::from(feature) >= feature_count {
                return Err(HybridError::ltr(format!(
                    "LTR tree {tree_idx} node {node_idx} split feature {} exceeds feature count {feature_count}",
                    feature
                )));
            }
            if !threshold.is_finite() {
                return Err(HybridError::ltr(format!(
                    "LTR tree {tree_idx} node {node_idx} threshold is not finite"
                )));
            }
            let left = usize::try_from(left)
                .map_err(|_| HybridError::limit("LTR left child index overflows usize"))?;
            let right = usize::try_from(right)
                .map_err(|_| HybridError::limit("LTR right child index overflows usize"))?;
            if left >= node_count || right >= node_count {
                return Err(HybridError::ltr(format!(
                    "LTR tree {tree_idx} node {node_idx} child index is out of range"
                )));
            }
            Ok(CompiledNode::Split {
                feature: usize::from(feature),
                threshold,
                left,
                right,
            })
        }
        _ => Err(HybridError::ltr(format!(
            "LTR tree {tree_idx} node {node_idx} must be either a leaf or a complete numerical split"
        ))),
    }
}

fn validate_reachable_tree(
    tree_idx: usize,
    node_idx: usize,
    depth: usize,
    max_depth: usize,
    nodes: &[CompiledNode],
    visited: &mut [bool],
    stack: &mut [bool],
) -> Result<()> {
    if depth > max_depth {
        return Err(HybridError::limit(format!(
            "LTR tree {tree_idx} exceeds max depth {max_depth}"
        )));
    }
    if stack[node_idx] {
        return Err(HybridError::ltr(format!(
            "LTR tree {tree_idx} contains a cycle at node {node_idx}"
        )));
    }
    stack[node_idx] = true;
    visited[node_idx] = true;
    if let CompiledNode::Split { left, right, .. } = nodes[node_idx] {
        validate_reachable_tree(tree_idx, left, depth + 1, max_depth, nodes, visited, stack)?;
        validate_reachable_tree(tree_idx, right, depth + 1, max_depth, nodes, visited, stack)?;
    }
    stack[node_idx] = false;
    Ok(())
}

fn score_tree(tree: &CompiledTree, features: &[f32]) -> Result<f32> {
    let mut node_idx = 0usize;
    for _ in 0..=tree.nodes.len() {
        match tree.nodes[node_idx] {
            CompiledNode::Leaf(value) => return Ok(value),
            CompiledNode::Split {
                feature,
                threshold,
                left,
                right,
            } => {
                let value = features[feature];
                if !value.is_finite() {
                    return Err(HybridError::ltr(format!(
                        "feature {feature} is not finite during LTR scoring"
                    )));
                }
                node_idx = if value < threshold { left } else { right };
            }
        }
    }
    Err(HybridError::ltr("LTR tree traversal did not reach a leaf"))
}

fn read_regular_file(path: &Path, max_bytes: u64) -> Result<Vec<u8>> {
    let mut file = open_no_follow(path)?;
    let metadata = file.metadata()?;
    if !metadata.file_type().is_file() {
        return Err(HybridError::ltr(format!(
            "LTR model path is not a regular file: {}",
            path.display()
        )));
    }
    if metadata.len() > max_bytes {
        return Err(HybridError::limit(format!(
            "LTR model is {} bytes, limit is {max_bytes}",
            metadata.len()
        )));
    }

    let model_len = checked_model_file_len(metadata.len())?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(model_len)
        .map_err(|_| HybridError::limit("LTR model allocation too large"))?;
    file.read_to_end(&mut bytes)?;
    let bytes_len = u64::try_from(bytes.len())
        .map_err(|_| HybridError::limit("LTR model length exceeds u64::MAX"))?;
    if bytes_len > max_bytes {
        return Err(HybridError::limit(format!(
            "LTR model is {bytes_len} bytes, limit is {max_bytes}"
        )));
    }
    Ok(bytes)
}

fn checked_model_file_len(len: u64) -> Result<usize> {
    usize::try_from(len).map_err(|_| {
        HybridError::limit(format!(
            "LTR model is {len} bytes, which cannot fit in memory on this platform"
        ))
    })
}

#[cfg(unix)]
fn open_no_follow(path: &Path) -> Result<File> {
    use std::os::unix::fs::OpenOptionsExt;

    Ok(std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)?)
}

#[cfg(not(unix))]
fn open_no_follow(path: &Path) -> Result<File> {
    Ok(File::open(path)?)
}

fn default_learning_rate() -> f32 {
    1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        rerank_fused_batch, LtrDocFeatureLookup, LtrFeatureInputs, LtrReranker, RankedBatch,
        RrfConfig, ScoredRow,
    };
    use std::collections::HashMap;

    fn test_record() -> LtrTreeEnsembleRecord {
        LtrTreeEnsembleRecord {
            schema_version: LTR_MODEL_SCHEMA_V1.to_string(),
            model_id: "ltr:test:v1".to_string(),
            model_family: "xgboost_gbtree".to_string(),
            training_objective: "rank:pairwise".to_string(),
            booster: "gbtree".to_string(),
            base_score: 0.0,
            learning_rate: 1.0,
            feature_schema: LtrFeatureSchema {
                schema_version: LTR_FEATURE_SCHEMA_V1.to_string(),
                feature_names: vec![
                    "bm25_score".to_string(),
                    "bm25_rank".to_string(),
                    "rank_cos".to_string(),
                    "rank_cos_rank".to_string(),
                    "doc_len_chars".to_string(),
                    "query_len_chars".to_string(),
                    "rrf_score".to_string(),
                ],
                forbidden_features: vec!["doc_cat_match".to_string()],
                dense_features_required: false,
            },
            training_provenance: BTreeMap::new(),
            trees: vec![
                LtrTree {
                    nodes: vec![
                        LtrTreeNode {
                            split_feature: Some(1),
                            threshold: Some(2.5),
                            default_left: true,
                            left: Some(1),
                            right: Some(2),
                            leaf_value: None,
                        },
                        LtrTreeNode {
                            split_feature: None,
                            threshold: None,
                            default_left: false,
                            left: None,
                            right: None,
                            leaf_value: Some(0.5),
                        },
                        LtrTreeNode {
                            split_feature: None,
                            threshold: None,
                            default_left: false,
                            left: None,
                            right: None,
                            leaf_value: Some(-0.5),
                        },
                    ],
                },
                LtrTree {
                    nodes: vec![
                        LtrTreeNode {
                            split_feature: Some(6),
                            threshold: Some(0.03),
                            default_left: true,
                            left: Some(1),
                            right: Some(2),
                            leaf_value: None,
                        },
                        LtrTreeNode {
                            split_feature: None,
                            threshold: None,
                            default_left: false,
                            left: None,
                            right: None,
                            leaf_value: Some(-0.1),
                        },
                        LtrTreeNode {
                            split_feature: None,
                            threshold: None,
                            default_left: false,
                            left: None,
                            right: None,
                            leaf_value: Some(0.2),
                        },
                    ],
                },
            ],
        }
    }

    #[test]
    fn tree_ensemble_scores_and_reranks_deterministically() {
        let model = TreeEnsembleReranker::from_record(test_record()).unwrap();
        let score_a = model
            .score_features(&[3.0, 1.0, 0.9, 1.0, 100.0, 12.0, 0.04])
            .unwrap();
        let score_b = model
            .score_features(&[2.0, 3.0, 0.8, 2.0, 100.0, 12.0, 0.02])
            .unwrap();
        assert_eq!(score_a, 0.7);
        assert_eq!(score_b, -0.6);

        let features = LtrFeatureBatch::from_ranked_lists(
            model.model_info().feature_schema.feature_names.clone(),
            vec![vec![
                crate::LtrCandidateFeatures {
                    row_id: 20,
                    features: vec![2.0, 3.0, 0.8, 2.0, 100.0, 12.0, 0.02],
                },
                crate::LtrCandidateFeatures {
                    row_id: 10,
                    features: vec![3.0, 1.0, 0.9, 1.0, 100.0, 12.0, 0.04],
                },
            ]],
        )
        .unwrap();
        let reranked = model
            .rerank_batch(&features, LtrRerankConfig { top_k: 10 })
            .unwrap();
        assert_eq!(
            reranked.hits(),
            &[
                ScoredRow {
                    row_id: 10,
                    score: 0.7
                },
                ScoredRow {
                    row_id: 20,
                    score: -0.6
                },
            ]
        );
    }

    #[test]
    fn leakage_feature_is_rejected_by_default() {
        let mut record = test_record();
        record
            .feature_schema
            .feature_names
            .push("doc_cat_match".to_string());
        let err = TreeEnsembleReranker::from_record(record).unwrap_err();
        assert!(err.to_string().contains("forbidden leakage feature"));
    }

    #[test]
    fn unknown_objective_is_rejected() {
        let mut record = test_record();
        record.training_objective = "rank:ndcg".to_string();
        let err = TreeEnsembleReranker::from_record(record).unwrap_err();
        assert!(err
            .to_string()
            .contains("unsupported LTR training_objective"));
    }

    #[test]
    fn feature_order_mismatch_is_rejected() {
        let model = TreeEnsembleReranker::from_record(test_record()).unwrap();
        let features = LtrFeatureBatch::from_ranked_lists(
            vec![
                "bm25_rank".to_string(),
                "bm25_score".to_string(),
                "rank_cos".to_string(),
                "rank_cos_rank".to_string(),
                "doc_len_chars".to_string(),
                "query_len_chars".to_string(),
                "rrf_score".to_string(),
            ],
            vec![vec![crate::LtrCandidateFeatures {
                row_id: 1,
                features: vec![1.0; 7],
            }]],
        )
        .unwrap();
        let err = model
            .rerank_batch(&features, LtrRerankConfig::default())
            .unwrap_err();
        assert!(err.to_string().contains("feature batch schema"));
    }

    #[test]
    fn feature_builder_refuses_missing_dense_for_full_schema() {
        let mut record = test_record();
        record.feature_schema.feature_names = vec![
            "bm25_score".to_string(),
            "bm25_rank".to_string(),
            "dense_cos".to_string(),
            "dense_rank".to_string(),
            "rank_cos".to_string(),
            "rank_cos_rank".to_string(),
            "doc_len_chars".to_string(),
            "query_len_chars".to_string(),
            "rrf_score".to_string(),
        ];
        record.feature_schema.dense_features_required = true;
        let model = TreeEnsembleReranker::from_record(record).unwrap();

        let bm25 = RankedBatch::single(vec![ScoredRow {
            row_id: 10,
            score: 3.0,
        }])
        .unwrap();
        let rank_cos = RankedBatch::single(vec![ScoredRow {
            row_id: 10,
            score: 0.9,
        }])
        .unwrap();
        let fused = crate::rrf_fuse_batch(&rank_cos, &bm25, RrfConfig::default()).unwrap();
        let mut doc_lens: HashMap<u64, u32> = HashMap::new();
        doc_lens.insert(10, 100);
        let inputs = LtrFeatureInputs {
            fused: &fused,
            rank_cos_scores: Some(&rank_cos),
            dense_cos_scores: None,
            bm25_scores: Some(&bm25),
            doc_len_chars: Some(&doc_lens as &dyn LtrDocFeatureLookup),
            query_len_chars: Some(&[12]),
        };
        let err = rerank_fused_batch(&model, &inputs, LtrRerankConfig::default()).unwrap_err();
        assert!(err.to_string().contains("dense_cos"));
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn model_file_length_must_fit_usize() {
        let err = checked_model_file_len(u64::from(u32::MAX) + 1).unwrap_err();
        assert!(err.to_string().contains("cannot fit in memory"), "{err}");
    }
}
