use std::collections::{BTreeMap, HashMap};

use crate::ltr::LtrReranker;
use crate::{FusedBatch, FusedRow, HybridError, LtrFeatureSchema, RankedBatch, Result};

/// Lookup table for per-row structural features.
pub trait LtrDocFeatureLookup {
    fn doc_len_chars(&self, row_id: u64) -> Option<u32>;
}

impl LtrDocFeatureLookup for HashMap<u64, u32> {
    fn doc_len_chars(&self, row_id: u64) -> Option<u32> {
        self.get(&row_id).copied()
    }
}

impl LtrDocFeatureLookup for BTreeMap<u64, u32> {
    fn doc_len_chars(&self, row_id: u64) -> Option<u32> {
        self.get(&row_id).copied()
    }
}

/// Explicit sources used to build a model feature matrix.
///
/// `fused` supplies row ids and RRF scores only. BM25, rank-cosine, and dense
/// cosine are deliberately separate so a compact ordinal store cannot silently
/// satisfy a full dense model with the wrong signal.
pub struct LtrFeatureInputs<'a> {
    pub fused: &'a FusedBatch,
    pub rank_cos_scores: Option<&'a RankedBatch>,
    pub dense_cos_scores: Option<&'a RankedBatch>,
    pub bm25_scores: Option<&'a RankedBatch>,
    pub doc_len_chars: Option<&'a dyn LtrDocFeatureLookup>,
    pub query_len_chars: Option<&'a [u32]>,
}

/// One candidate row and its feature vector in model feature order.
#[derive(Clone, Debug, PartialEq)]
pub struct LtrCandidateFeatures {
    pub row_id: u64,
    pub features: Vec<f32>,
}

/// Batched LTR feature rows.
#[derive(Clone, Debug, PartialEq)]
pub struct LtrFeatureBatch {
    feature_names: Vec<String>,
    offsets: Vec<usize>,
    rows: Vec<LtrCandidateFeatures>,
}

impl LtrFeatureBatch {
    pub fn from_ranked_lists(
        feature_names: Vec<String>,
        lists: Vec<Vec<LtrCandidateFeatures>>,
    ) -> Result<Self> {
        validate_feature_names(&feature_names)?;
        let mut offsets = Vec::with_capacity(lists.len() + 1);
        let mut rows = Vec::new();
        offsets.push(0);
        for list in lists {
            for candidate in list {
                validate_candidate(&candidate, feature_names.len())?;
                rows.push(candidate);
            }
            offsets.push(rows.len());
        }
        Ok(Self {
            feature_names,
            offsets,
            rows,
        })
    }

    pub fn from_inputs(inputs: &LtrFeatureInputs<'_>, schema: &LtrFeatureSchema) -> Result<Self> {
        validate_feature_names(&schema.feature_names)?;
        validate_inputs(inputs)?;

        let mut offsets = Vec::with_capacity(inputs.fused.query_count() + 1);
        let mut rows = Vec::new();
        offsets.push(0);
        for query_idx in 0..inputs.fused.query_count() {
            let bm25 = source_map(inputs.bm25_scores, query_idx, "bm25_scores")?;
            let rank_cos = source_map(inputs.rank_cos_scores, query_idx, "rank_cos_scores")?;
            let dense_cos = source_map(inputs.dense_cos_scores, query_idx, "dense_cos_scores")?;
            for fused in inputs.fused.hits_for_query(query_idx).unwrap_or(&[]) {
                let features = schema
                    .feature_names
                    .iter()
                    .map(|name| {
                        feature_value(name, fused, query_idx, inputs, &bm25, &rank_cos, &dense_cos)
                    })
                    .collect::<Result<Vec<_>>>()?;
                rows.push(LtrCandidateFeatures {
                    row_id: fused.row_id,
                    features,
                });
            }
            offsets.push(rows.len());
        }

        Ok(Self {
            feature_names: schema.feature_names.clone(),
            offsets,
            rows,
        })
    }

    pub fn query_count(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub fn rows_for_query(&self, query_idx: usize) -> Option<&[LtrCandidateFeatures]> {
        let start = *self.offsets.get(query_idx)?;
        let end = *self.offsets.get(query_idx + 1)?;
        self.rows.get(start..end)
    }

    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    pub fn rows(&self) -> &[LtrCandidateFeatures] {
        &self.rows
    }

    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }
}

pub fn rerank_fused_batch(
    reranker: &impl LtrReranker,
    inputs: &LtrFeatureInputs<'_>,
    config: crate::LtrRerankConfig,
) -> Result<RankedBatch> {
    let features = LtrFeatureBatch::from_inputs(inputs, &reranker.model_info().feature_schema)?;
    reranker.rerank_batch(&features, config)
}

#[derive(Clone, Copy)]
struct SourceHit {
    rank: usize,
    score: f32,
}

fn source_map(
    batch: Option<&RankedBatch>,
    query_idx: usize,
    label: &str,
) -> Result<HashMap<u64, SourceHit>> {
    let Some(batch) = batch else {
        return Ok(HashMap::new());
    };
    let rows = batch
        .hits_for_query(query_idx)
        .ok_or_else(|| HybridError::ltr(format!("{label} is missing query index {query_idx}")))?;
    let mut map = HashMap::new();
    map.try_reserve(rows.len())
        .map_err(|_| HybridError::limit("LTR source map allocation too large"))?;
    for (rank_idx, row) in rows.iter().enumerate() {
        if !row.score.is_finite() {
            return Err(HybridError::ltr(format!(
                "{label} row {} has non-finite score",
                row.row_id
            )));
        }
        map.entry(row.row_id).or_insert(SourceHit {
            rank: rank_idx + 1,
            score: row.score,
        });
    }
    Ok(map)
}

fn feature_value(
    name: &str,
    fused: &FusedRow,
    query_idx: usize,
    inputs: &LtrFeatureInputs<'_>,
    bm25: &HashMap<u64, SourceHit>,
    rank_cos: &HashMap<u64, SourceHit>,
    dense_cos: &HashMap<u64, SourceHit>,
) -> Result<f32> {
    let value = match name {
        "rrf_score" => fused.fused_score,
        "bm25_score" => required_source(bm25, fused.row_id, "bm25_score")?.score,
        "bm25_rank" => required_source(bm25, fused.row_id, "bm25_rank")?.rank as f32,
        "rank_cos" => required_source(rank_cos, fused.row_id, "rank_cos")?.score,
        "rank_cos_rank" => required_source(rank_cos, fused.row_id, "rank_cos_rank")?.rank as f32,
        "dense_cos" => required_source(dense_cos, fused.row_id, "dense_cos")?.score,
        "dense_rank" => required_source(dense_cos, fused.row_id, "dense_rank")?.rank as f32,
        "doc_len_chars" => inputs
            .doc_len_chars
            .and_then(|lookup| lookup.doc_len_chars(fused.row_id))
            .ok_or_else(|| {
                HybridError::ltr(format!(
                    "doc_len_chars is missing for row_id {}",
                    fused.row_id
                ))
            })? as f32,
        "query_len_chars" => *inputs
            .query_len_chars
            .and_then(|lengths| lengths.get(query_idx))
            .ok_or_else(|| {
                HybridError::ltr(format!(
                    "query_len_chars is missing for query index {query_idx}"
                ))
            })? as f32,
        other => {
            return Err(HybridError::ltr(format!(
                "unsupported LTR feature {other:?}"
            )))
        }
    };
    if !value.is_finite() {
        return Err(HybridError::ltr(format!(
            "LTR feature {name:?} for row_id {} is not finite",
            fused.row_id
        )));
    }
    Ok(value)
}

fn required_source<'a>(
    map: &'a HashMap<u64, SourceHit>,
    row_id: u64,
    feature_name: &str,
) -> Result<&'a SourceHit> {
    map.get(&row_id).ok_or_else(|| {
        HybridError::ltr(format!(
            "{feature_name} requires an explicit source score for row_id {row_id}"
        ))
    })
}

fn validate_inputs(inputs: &LtrFeatureInputs<'_>) -> Result<()> {
    let query_count = inputs.fused.query_count();
    for (label, batch) in [
        ("rank_cos_scores", inputs.rank_cos_scores),
        ("dense_cos_scores", inputs.dense_cos_scores),
        ("bm25_scores", inputs.bm25_scores),
    ] {
        if let Some(batch) = batch {
            if batch.query_count() != query_count {
                return Err(HybridError::ltr(format!(
                    "{label} query count {} does not match fused query count {query_count}",
                    batch.query_count()
                )));
            }
        }
    }
    if let Some(lengths) = inputs.query_len_chars {
        if lengths.len() != query_count {
            return Err(HybridError::ltr(format!(
                "query_len_chars count {} does not match fused query count {query_count}",
                lengths.len()
            )));
        }
    }
    Ok(())
}

fn validate_feature_names(feature_names: &[String]) -> Result<()> {
    if feature_names.is_empty() {
        return Err(HybridError::ltr("LTR feature name list must not be empty"));
    }
    let mut seen = std::collections::HashSet::with_capacity(feature_names.len());
    for name in feature_names {
        if name.trim().is_empty() {
            return Err(HybridError::ltr("LTR feature names must not be empty"));
        }
        if !seen.insert(name) {
            return Err(HybridError::ltr(format!(
                "duplicate LTR feature name {name:?}"
            )));
        }
    }
    Ok(())
}

fn validate_candidate(candidate: &LtrCandidateFeatures, feature_count: usize) -> Result<()> {
    if candidate.features.len() != feature_count {
        return Err(HybridError::ltr(format!(
            "candidate row_id {} has {} features, expected {feature_count}",
            candidate.row_id,
            candidate.features.len()
        )));
    }
    for (idx, value) in candidate.features.iter().enumerate() {
        if !value.is_finite() {
            return Err(HybridError::ltr(format!(
                "candidate row_id {} feature {idx} is not finite",
                candidate.row_id
            )));
        }
    }
    Ok(())
}
