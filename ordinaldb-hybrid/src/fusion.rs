use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use crate::{HybridError, Result};

/// A ranked result keyed by stable OrdinalDB row id.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoredRow {
    pub row_id: u64,
    pub score: f32,
}

/// Batched ranked rows.
///
/// `offsets.len() == query_count + 1`, and hits for query `q` are in
/// `hits[offsets[q]..offsets[q + 1]]`.
#[derive(Clone, Debug, PartialEq)]
pub struct RankedBatch {
    offsets: Vec<usize>,
    hits: Vec<ScoredRow>,
}

impl RankedBatch {
    pub fn empty(query_count: usize) -> Self {
        Self {
            offsets: vec![0; query_count + 1],
            hits: Vec::new(),
        }
    }

    pub fn single(hits: Vec<ScoredRow>) -> Result<Self> {
        Self::from_ranked_lists(vec![hits])
    }

    pub fn from_offsets_hits(offsets: Vec<usize>, hits: Vec<ScoredRow>) -> Result<Self> {
        validate_offsets(&offsets, hits.len())?;
        let mut normalized_offsets = Vec::with_capacity(offsets.len());
        let mut normalized_hits = Vec::with_capacity(hits.len());
        normalized_offsets.push(0);
        for window in offsets.windows(2) {
            let mut list = hits[window[0]..window[1]].to_vec();
            normalize_ranked_list(&mut list)?;
            normalized_hits.extend(list);
            normalized_offsets.push(normalized_hits.len());
        }
        Ok(Self {
            offsets: normalized_offsets,
            hits: normalized_hits,
        })
    }

    pub fn from_sorted_offsets_hits(offsets: Vec<usize>, hits: Vec<ScoredRow>) -> Result<Self> {
        validate_offsets(&offsets, hits.len())?;
        let batch = Self { offsets, hits };
        for window in batch.offsets.windows(2) {
            validate_sorted_ranked_slice(&batch.hits[window[0]..window[1]])?;
        }
        Ok(batch)
    }

    pub fn from_ranked_lists(lists: Vec<Vec<ScoredRow>>) -> Result<Self> {
        let mut offsets = Vec::with_capacity(lists.len() + 1);
        let mut hits = Vec::new();
        offsets.push(0);
        for mut list in lists {
            normalize_ranked_list(&mut list)?;
            hits.extend(list);
            offsets.push(hits.len());
        }
        Ok(Self { offsets, hits })
    }

    pub fn from_flat_scores_ids(
        nq: usize,
        k: usize,
        scores: Vec<f32>,
        ids: Vec<u64>,
    ) -> Result<Self> {
        if scores.len() != ids.len() {
            return Err(HybridError::batch(format!(
                "score count {} does not match id count {}",
                scores.len(),
                ids.len()
            )));
        }
        let expected = nq
            .checked_mul(k)
            .ok_or_else(|| HybridError::limit("nq * k overflows usize"))?;
        if scores.len() != expected {
            return Err(HybridError::batch(format!(
                "flat result length {} does not match nq * k {}",
                scores.len(),
                expected
            )));
        }
        let mut offsets = Vec::with_capacity(nq + 1);
        let mut hits = Vec::with_capacity(expected);
        offsets.push(0);
        for query_idx in 0..nq {
            let start = query_idx * k;
            let end = start + k;
            hits.extend(
                scores[start..end]
                    .iter()
                    .zip(&ids[start..end])
                    .map(|(&score, &row_id)| ScoredRow { row_id, score }),
            );
            offsets.push(hits.len());
        }
        Self::from_offsets_hits(offsets, hits)
    }

    pub fn from_flat_scores_slots(
        nq: usize,
        k: usize,
        scores: Vec<f32>,
        slots: Vec<u32>,
    ) -> Result<Self> {
        let ids = slots.into_iter().map(u64::from).collect();
        Self::from_flat_scores_ids(nq, k, scores, ids)
    }

    /// Build from flat signed indices, treating negative values as sentinel
    /// padding and skipping them before ranking.
    pub fn from_flat_scores_i64_indices(
        nq: usize,
        k: usize,
        scores: Vec<f32>,
        indices: Vec<i64>,
    ) -> Result<Self> {
        if scores.len() != indices.len() {
            return Err(HybridError::batch(format!(
                "score count {} does not match index count {}",
                scores.len(),
                indices.len()
            )));
        }
        let expected = nq
            .checked_mul(k)
            .ok_or_else(|| HybridError::limit("nq * k overflows usize"))?;
        if scores.len() != expected {
            return Err(HybridError::batch(format!(
                "flat result length {} does not match nq * k {}",
                scores.len(),
                expected
            )));
        }

        let mut offsets = Vec::with_capacity(nq + 1);
        let mut hits = Vec::with_capacity(expected);
        offsets.push(0);
        for query_idx in 0..nq {
            let start = query_idx * k;
            let end = start + k;
            for (&score, &index) in scores[start..end].iter().zip(&indices[start..end]) {
                if index < 0 {
                    continue;
                }
                hits.push(ScoredRow {
                    row_id: index as u64,
                    score,
                });
            }
            offsets.push(hits.len());
        }
        Self::from_offsets_hits(offsets, hits)
    }

    pub fn query_count(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub fn hits_for_query(&self, query_idx: usize) -> Option<&[ScoredRow]> {
        let start = *self.offsets.get(query_idx)?;
        let end = *self.offsets.get(query_idx + 1)?;
        self.hits.get(start..end)
    }

    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    pub fn hits(&self) -> &[ScoredRow] {
        &self.hits
    }
}

/// Reciprocal rank fusion configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RrfConfig {
    pub rank_constant: f32,
    pub dense_window: usize,
    pub sparse_window: usize,
    pub top_k: usize,
}

impl Default for RrfConfig {
    fn default() -> Self {
        Self {
            rank_constant: 60.0,
            dense_window: usize::MAX,
            sparse_window: usize::MAX,
            top_k: usize::MAX,
        }
    }
}

impl RrfConfig {
    fn validate(self) -> Result<Self> {
        if !self.rank_constant.is_finite() || self.rank_constant <= 0.0 {
            return Err(HybridError::batch(
                "RRF rank_constant must be finite and positive",
            ));
        }
        Ok(self)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FusedRow {
    pub row_id: u64,
    pub fused_score: f32,
    pub dense_rank: Option<usize>,
    pub sparse_rank: Option<usize>,
    pub dense_score: Option<f32>,
    pub sparse_score: Option<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FusedBatch {
    offsets: Vec<usize>,
    hits: Vec<FusedRow>,
}

impl FusedBatch {
    pub fn query_count(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub fn hits_for_query(&self, query_idx: usize) -> Option<&[FusedRow]> {
        let start = *self.offsets.get(query_idx)?;
        let end = *self.offsets.get(query_idx + 1)?;
        self.hits.get(start..end)
    }

    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    pub fn hits(&self) -> &[FusedRow] {
        &self.hits
    }
}

pub fn rrf_fuse_batch(
    dense: &RankedBatch,
    sparse: &RankedBatch,
    config: RrfConfig,
) -> Result<FusedBatch> {
    let config = config.validate()?;
    if dense.query_count() != sparse.query_count() {
        return Err(HybridError::batch(format!(
            "dense query count {} does not match sparse query count {}",
            dense.query_count(),
            sparse.query_count()
        )));
    }

    let mut offsets = Vec::with_capacity(dense.query_count() + 1);
    let mut hits = Vec::new();
    offsets.push(0);
    for query_idx in 0..dense.query_count() {
        let mut accum = HashMap::<u64, Accumulator>::new();
        add_rrf_source(
            &mut accum,
            dense.hits_for_query(query_idx).unwrap_or(&[]),
            config.rank_constant,
            config.dense_window,
            Source::Dense,
        );
        add_rrf_source(
            &mut accum,
            sparse.hits_for_query(query_idx).unwrap_or(&[]),
            config.rank_constant,
            config.sparse_window,
            Source::Sparse,
        );
        let mut query_hits = accum
            .into_iter()
            .map(|(row_id, acc)| FusedRow {
                row_id,
                fused_score: acc.fused_score,
                dense_rank: acc.dense_rank,
                sparse_rank: acc.sparse_rank,
                dense_score: acc.dense_score,
                sparse_score: acc.sparse_score,
            })
            .collect::<Vec<_>>();
        query_hits.sort_by(|a, b| {
            b.fused_score
                .total_cmp(&a.fused_score)
                .then_with(|| a.row_id.cmp(&b.row_id))
        });
        query_hits.truncate(config.top_k);
        hits.extend(query_hits);
        offsets.push(hits.len());
    }
    Ok(FusedBatch { offsets, hits })
}

#[derive(Default)]
struct Accumulator {
    fused_score: f32,
    dense_rank: Option<usize>,
    sparse_rank: Option<usize>,
    dense_score: Option<f32>,
    sparse_score: Option<f32>,
}

#[derive(Clone, Copy)]
enum Source {
    Dense,
    Sparse,
}

fn add_rrf_source(
    accum: &mut HashMap<u64, Accumulator>,
    rows: &[ScoredRow],
    rank_constant: f32,
    window: usize,
    source: Source,
) {
    for (rank_idx, row) in rows.iter().take(window).enumerate() {
        let rank = rank_idx + 1;
        let contribution = 1.0 / (rank_constant + rank as f32);
        let entry = accum.entry(row.row_id).or_default();
        entry.fused_score += contribution;
        match source {
            Source::Dense => {
                entry.dense_rank.get_or_insert(rank);
                entry.dense_score.get_or_insert(row.score);
            }
            Source::Sparse => {
                entry.sparse_rank.get_or_insert(rank);
                entry.sparse_score.get_or_insert(row.score);
            }
        }
    }
}

pub(crate) fn normalize_ranked_list(rows: &mut Vec<ScoredRow>) -> Result<()> {
    sort_ranked_slice(rows.as_mut_slice())?;
    let mut seen = HashSet::with_capacity(rows.len());
    let mut write = 0usize;
    for read in 0..rows.len() {
        if !seen.insert(rows[read].row_id) {
            continue;
        }
        rows[write] = rows[read];
        write += 1;
    }
    rows.truncate(write);
    Ok(())
}

fn sort_ranked_slice(rows: &mut [ScoredRow]) -> Result<()> {
    for row in rows.iter() {
        if !row.score.is_finite() {
            return Err(HybridError::batch(format!(
                "row {} has non-finite score {}",
                row.row_id, row.score
            )));
        }
    }
    rows.sort_by(rank_row_order);
    Ok(())
}

fn rank_row_order(a: &ScoredRow, b: &ScoredRow) -> Ordering {
    b.score
        .total_cmp(&a.score)
        .then_with(|| a.row_id.cmp(&b.row_id))
}

fn validate_sorted_ranked_slice(rows: &[ScoredRow]) -> Result<()> {
    for row in rows {
        if !row.score.is_finite() {
            return Err(HybridError::batch(format!(
                "row {} has non-finite score {}",
                row.row_id, row.score
            )));
        }
    }
    for pair in rows.windows(2) {
        let prev = pair[0];
        let next = pair[1];
        if rank_row_order(&prev, &next) == Ordering::Greater {
            return Err(HybridError::batch(
                "ranked rows must be sorted by score desc, row_id asc",
            ));
        }
    }
    validate_no_duplicate_rows(rows)
}

fn validate_no_duplicate_rows(rows: &[ScoredRow]) -> Result<()> {
    let mut seen = HashSet::with_capacity(rows.len());
    for row in rows {
        if !seen.insert(row.row_id) {
            return Err(HybridError::batch(format!(
                "duplicate row_id {} in ranked rows",
                row.row_id
            )));
        }
    }
    Ok(())
}

fn validate_offsets(offsets: &[usize], hit_len: usize) -> Result<()> {
    if offsets.is_empty() {
        return Err(HybridError::batch(
            "offsets must contain at least one entry",
        ));
    }
    if offsets[0] != 0 {
        return Err(HybridError::batch("offsets must start at zero"));
    }
    for pair in offsets.windows(2) {
        if pair[0] > pair[1] {
            return Err(HybridError::batch("offsets must be monotonic"));
        }
    }
    if *offsets.last().unwrap() != hit_len {
        return Err(HybridError::batch(format!(
            "final offset {} does not match hits length {}",
            offsets.last().unwrap(),
            hit_len
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranked_batch_rejects_invalid_offsets_and_scores() {
        assert!(RankedBatch::from_offsets_hits(vec![1], Vec::new()).is_err());
        assert!(RankedBatch::from_offsets_hits(
            vec![0, 2],
            vec![ScoredRow {
                row_id: 1,
                score: 1.0
            }]
        )
        .is_err());
        assert!(RankedBatch::single(vec![ScoredRow {
            row_id: 1,
            score: f32::NAN,
        }])
        .is_err());
    }

    #[test]
    fn ranked_batch_sorts_and_collapses_duplicates() {
        let batch = RankedBatch::single(vec![
            ScoredRow {
                row_id: 2,
                score: 1.0,
            },
            ScoredRow {
                row_id: 1,
                score: 1.0,
            },
            ScoredRow {
                row_id: 2,
                score: 0.5,
            },
        ])
        .unwrap();
        assert_eq!(
            batch.hits(),
            vec![
                ScoredRow {
                    row_id: 1,
                    score: 1.0
                },
                ScoredRow {
                    row_id: 2,
                    score: 1.0
                },
            ]
        );
    }

    #[test]
    fn from_offsets_hits_collapses_duplicates() {
        let batch = RankedBatch::from_offsets_hits(
            vec![0, 4],
            vec![
                ScoredRow {
                    row_id: 2,
                    score: 0.7,
                },
                ScoredRow {
                    row_id: 1,
                    score: 0.9,
                },
                ScoredRow {
                    row_id: 2,
                    score: 0.6,
                },
                ScoredRow {
                    row_id: 3,
                    score: 0.5,
                },
            ],
        )
        .unwrap();

        assert_eq!(
            batch.hits_for_query(0).unwrap(),
            &[
                ScoredRow {
                    row_id: 1,
                    score: 0.9
                },
                ScoredRow {
                    row_id: 2,
                    score: 0.7
                },
                ScoredRow {
                    row_id: 3,
                    score: 0.5
                },
            ]
        );
    }

    #[test]
    fn flat_i64_indices_skip_negative_sentinels() {
        let batch = RankedBatch::from_flat_scores_i64_indices(
            2,
            3,
            vec![0.9, 0.1, f32::NAN, 0.2, 0.8, 0.7],
            vec![4, 4, -1, -1, 3, 2],
        )
        .unwrap();

        assert_eq!(
            batch.hits_for_query(0).unwrap(),
            &[ScoredRow {
                row_id: 4,
                score: 0.9
            }]
        );
        assert_eq!(
            batch.hits_for_query(1).unwrap(),
            &[
                ScoredRow {
                    row_id: 3,
                    score: 0.8
                },
                ScoredRow {
                    row_id: 2,
                    score: 0.7
                },
            ]
        );
    }

    #[test]
    fn sorted_batch_validation_uses_canonical_float_order() {
        let wrong_signed_zero_order = RankedBatch::from_sorted_offsets_hits(
            vec![0, 2],
            vec![
                ScoredRow {
                    row_id: 1,
                    score: -0.0,
                },
                ScoredRow {
                    row_id: 2,
                    score: 0.0,
                },
            ],
        );
        assert!(wrong_signed_zero_order.is_err());

        let canonical = RankedBatch::from_sorted_offsets_hits(
            vec![0, 2],
            vec![
                ScoredRow {
                    row_id: 2,
                    score: 0.0,
                },
                ScoredRow {
                    row_id: 1,
                    score: -0.0,
                },
            ],
        )
        .unwrap();
        assert_eq!(canonical.hits().len(), 2);
    }

    #[test]
    fn rrf_known_answer_and_tie_policy() {
        let dense = RankedBatch::single(vec![
            ScoredRow {
                row_id: 10,
                score: 0.9,
            },
            ScoredRow {
                row_id: 20,
                score: 0.8,
            },
        ])
        .unwrap();
        let sparse = RankedBatch::single(vec![
            ScoredRow {
                row_id: 20,
                score: 3.0,
            },
            ScoredRow {
                row_id: 10,
                score: 2.0,
            },
        ])
        .unwrap();
        let fused = rrf_fuse_batch(&dense, &sparse, RrfConfig::default()).unwrap();
        let rows = fused.hits_for_query(0).unwrap();
        assert_eq!(rows[0].row_id, 10);
        assert_eq!(rows[1].row_id, 20);
    }

    #[test]
    fn rrf_uses_rank_constant_and_collapses_duplicates_before_rank() {
        let dense = RankedBatch::single(vec![
            ScoredRow {
                row_id: 1,
                score: 9.0,
            },
            ScoredRow {
                row_id: 1,
                score: 8.0,
            },
            ScoredRow {
                row_id: 2,
                score: 7.0,
            },
        ])
        .unwrap();
        let sparse = RankedBatch::single(Vec::new()).unwrap();
        let fused = rrf_fuse_batch(
            &dense,
            &sparse,
            RrfConfig {
                rank_constant: 10.0,
                ..RrfConfig::default()
            },
        )
        .unwrap();
        let rows = fused.hits_for_query(0).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].row_id, 1);
        assert_eq!(rows[0].dense_rank, Some(1));
        assert!(rows[0].fused_score > rows[1].fused_score);
    }

    #[test]
    fn rrf_dense_sparse_cases() {
        let dense = RankedBatch::from_ranked_lists(vec![
            vec![
                ScoredRow {
                    row_id: 1,
                    score: 9.0,
                },
                ScoredRow {
                    row_id: 2,
                    score: 8.0,
                },
            ],
            vec![ScoredRow {
                row_id: 3,
                score: 7.0,
            }],
        ])
        .unwrap();
        let sparse = RankedBatch::from_ranked_lists(vec![
            vec![ScoredRow {
                row_id: 4,
                score: 3.0,
            }],
            vec![
                ScoredRow {
                    row_id: 3,
                    score: 4.0,
                },
                ScoredRow {
                    row_id: 5,
                    score: 2.0,
                },
            ],
        ])
        .unwrap();
        let fused = rrf_fuse_batch(&dense, &sparse, RrfConfig::default()).unwrap();
        assert_eq!(fused.query_count(), 2);
        assert_eq!(fused.hits_for_query(0).unwrap().len(), 3);

        let fused = rrf_fuse_batch(&dense, &sparse, RrfConfig::default()).unwrap();
        assert_eq!(fused.query_count(), 2);
        assert_eq!(fused.hits_for_query(0).unwrap().len(), 3);
        assert_eq!(fused.hits_for_query(1).unwrap().len(), 2);
    }
}
