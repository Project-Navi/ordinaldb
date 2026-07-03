//! Demo step 4: learning-to-rank reranking on top of the RRF-fused hybrid
//! results, using the *inference-only* types re-exported at
//! `ordinaldb::hybrid` under the `experimental-ltr` feature.
//!
//! There is no trainer to call: `ordinaldb-ltr`'s own docs say training and
//! XGBoost conversion are explicitly out of scope for that crate, and the
//! CLI's `ordinaldb ltr train` is a stub that returns "not implemented
//! yet". So the model below is a small, hand-authored, explainable
//! "boosted stumps" ensemble -- three independent single-split trees, each
//! nudging the score up or down based on one signal -- built directly
//! against the real `LtrTreeEnsembleRecord` wire format and scored through
//! the real `TreeEnsembleReranker`. It is not gradient-boosted and was not
//! trained on relevance judgments; it exists to prove the serving path
//! works end to end on genuine hybrid search output, not to demonstrate
//! ranking quality.

use std::collections::HashMap;
use std::path::Path;

use ordinaldb::artifacts::LTR_MODEL_AUX_NAME;
use ordinaldb::hybrid::{
    rerank_fused_batch, rrf_fuse_batch, LtrDocFeatureLookup, LtrFeatureSchema, LtrLoadOptions,
    LtrRerankConfig, LtrReranker, LtrTree, LtrTreeEnsembleRecord, LtrTreeNode, RankedBatch,
    RrfConfig, ScoredRow, TreeEnsembleReranker, LTR_FEATURE_SCHEMA_V1, LTR_MODEL_SCHEMA_V1,
};
use ordinaldb::manifest::VerifyOptions;

use crate::embedder::Embedder;
use crate::index::BuiltStore;
use crate::queries::demo_queries;

const RERANK_K: usize = 8;

fn feature_names() -> Vec<String> {
    [
        "rrf_score",
        "bm25_score",
        "bm25_rank",
        "rank_cos",
        "rank_cos_rank",
        "dense_cos",
        "dense_rank",
        "doc_len_chars",
        "query_len_chars",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

fn feature_index(name: &str) -> u16 {
    feature_names().iter().position(|n| n == name).unwrap() as u16
}

fn stump(feature: &str, threshold: f32, left_leaf: f32, right_leaf: f32) -> LtrTree {
    LtrTree {
        nodes: vec![
            LtrTreeNode {
                split_feature: Some(feature_index(feature)),
                threshold: Some(threshold),
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
                leaf_value: Some(left_leaf),
            },
            LtrTreeNode {
                split_feature: None,
                threshold: None,
                default_left: false,
                left: None,
                right: None,
                leaf_value: Some(right_leaf),
            },
        ],
    }
}

fn hand_authored_model() -> LtrTreeEnsembleRecord {
    let mut provenance = std::collections::BTreeMap::new();
    provenance.insert(
        "producer".to_string(),
        serde_json::json!("supportsearch cookbook demo"),
    );
    provenance.insert(
        "note".to_string(),
        serde_json::json!(
            "hand-authored heuristic stumps, NOT gradient-boosted and NOT trained on \
             relevance judgments -- ordinaldb-ltr has no trainer to consume yet \
             (`ordinaldb ltr train` returns \"not implemented yet\"). model_family/\
             training_objective/booster below are fabricated to satisfy \
             TreeEnsembleReranker::validate_model_header, which only accepts \
             training_objective==\"rank:pairwise\" and booster==\"gbtree\" -- there is \
             no metadata option in the wire format for \"hand-authored, not trained\"."
        ),
    );

    LtrTreeEnsembleRecord {
        schema_version: LTR_MODEL_SCHEMA_V1.to_string(),
        model_id: "supportsearch:hand-authored-heuristic:v1".to_string(),
        model_family: "ordinaldb_tree_ensemble".to_string(),
        training_objective: "rank:pairwise".to_string(),
        booster: "gbtree".to_string(),
        base_score: 0.0,
        learning_rate: 1.0,
        feature_schema: LtrFeatureSchema {
            schema_version: LTR_FEATURE_SCHEMA_V1.to_string(),
            feature_names: feature_names(),
            forbidden_features: Vec::new(),
            dense_features_required: true,
        },
        training_provenance: provenance,
        trees: vec![
            // Reward a strong exact-term (BM25) match.
            stump("bm25_score", 8.0, -0.3, 0.6),
            // Reward a strong semantic (true cosine) match.
            stump("dense_cos", 0.55, -0.2, 0.5),
            // Small nudge for candidates both signals already agree on.
            stump("rrf_score", 0.02, -0.1, 0.2),
        ],
    }
}

/// Write the hand-authored model as the on-disk JSON `TreeEnsembleReranker`
/// itself parses (`LtrTreeEnsembleRecord`'s serde form), so it can be
/// declared as a bundle auxiliary artifact before the bundle is written.
pub fn write_model_file(path: &Path) -> anyhow::Result<()> {
    let record = hand_authored_model();
    let bytes = serde_json::to_vec_pretty(&record)?;
    std::fs::write(path, bytes)?;
    Ok(())
}

pub fn run(store: &BuiltStore, embedder: &dyn Embedder) -> anyhow::Result<()> {
    println!("=== LTR reranking demo (--features experimental-ltr) ===\n");

    let reranker = TreeEnsembleReranker::load_verified_sidecar(
        &store.manifest_path,
        LTR_MODEL_AUX_NAME,
        VerifyOptions::default(),
        LtrLoadOptions::default(),
    )?;
    println!(
        "loaded verified LTR model sidecar: model_id={:?} features={:?}\n",
        reranker.model_info().model_id,
        reranker.model_info().feature_schema.feature_names
    );

    let titles: HashMap<u64, &str> = store.docs.iter().map(|d| (d.id, d.title)).collect();
    let doc_len_chars: HashMap<u64, u32> = store
        .docs
        .iter()
        .map(|d| (d.id, (d.title.len() + d.body.len()) as u32))
        .collect();
    let corpus_size = store.docs.len();

    for q in demo_queries() {
        println!("----------------------------------------------------------------");
        println!("{}", q.label);
        println!("query: {:?}", q.text);

        // Search wide enough (the whole corpus) on both sides so every row
        // that ends up in the RRF-fused candidate set also has an explicit
        // score in every per-signal source LtrFeatureInputs requires --
        // see the README's LTR findings section for why this is load-
        // bearing, not just generous headroom.
        let query_embedding = embedder.embed_one(q.text)?;
        let dense_full = store.dense.search_batch_rows(&query_embedding, corpus_size)?;
        let sparse_full = store.sparse.search_batch(&[q.text], corpus_size)?;
        let fused = rrf_fuse_batch(
            &dense_full,
            &sparse_full,
            RrfConfig {
                top_k: RERANK_K,
                ..RrfConfig::default()
            },
        )?;

        let dense_cos = true_cosine_batch(&query_embedding, store, &fused)?;
        let query_len_chars = [q.text.chars().count() as u32];

        let inputs = ordinaldb::hybrid::LtrFeatureInputs {
            fused: &fused,
            rank_cos_scores: Some(&dense_full),
            dense_cos_scores: Some(&dense_cos),
            bm25_scores: Some(&sparse_full),
            doc_len_chars: Some(&doc_len_chars as &dyn LtrDocFeatureLookup),
            query_len_chars: Some(&query_len_chars),
        };

        match rerank_fused_batch(&reranker, &inputs, LtrRerankConfig { top_k: RERANK_K }) {
            Ok(reranked) => {
                println!("  RRF fused top-{RERANK_K} -> LTR-reranked top-{RERANK_K}:");
                for (i, hit) in reranked.hits().iter().enumerate() {
                    let marker = if hit.row_id == q.gold_id { "  <-- GOLD" } else { "" };
                    println!(
                        "    {}. [{}] ltr_score={:.4}  {:?}{marker}",
                        i + 1,
                        hit.row_id,
                        hit.score,
                        titles.get(&hit.row_id).unwrap_or(&"?")
                    );
                }
            }
            Err(err) => {
                println!(
                    "  LTR rerank FAILED: {err}\n\
                     \x20 (this is the documented feature-coverage gap: a row with zero \
                     evidence in one signal -- e.g. zero BM25 term overlap for the \
                     paraphrase query -- has no entry in that signal's RankedBatch at \
                     all, and LtrFeatureInputs::from_inputs requires an explicit score \
                     for every row in the fused set, for every configured feature. See \
                     the README's LTR findings section.)"
                );
            }
        }
        println!();
    }

    Ok(())
}

/// True float cosine similarity between the query embedding and each
/// fused candidate's stored embedding -- deliberately distinct from OrdVec's
/// quantized `rank_cos` score, per `LtrFeatureInputs`'s own doc comment
/// ("BM25, rank-cosine, and dense cosine are deliberately separate").
fn true_cosine_batch(
    query_embedding: &[f32],
    store: &BuiltStore,
    fused: &ordinaldb::hybrid::FusedBatch,
) -> anyhow::Result<RankedBatch> {
    let mut rows = Vec::new();
    for hit in fused.hits_for_query(0).unwrap_or(&[]) {
        let Some(doc_embedding) = store.embeddings.get(&hit.row_id) else {
            continue;
        };
        rows.push(ScoredRow {
            row_id: hit.row_id,
            score: cosine(query_embedding, doc_embedding),
        });
    }
    Ok(RankedBatch::single(rows)?)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    //! Minimal, corpus-independent repro of a real API friction point found
    //! while building the demo above: `LtrFeatureBatch::from_inputs`
    //! requires an explicit score, in *every* per-signal source, for every
    //! row that appears in the RRF-fused candidate set -- with no notion of
    //! "this signal doesn't apply to this row". A row that is only found by
    //! one retrieval mode (the normal case whenever BM25 and dense
    //! disagree, which is the entire premise of hybrid search) has no
    //! entry in the other mode's `RankedBatch` at all, so building an LTR
    //! feature batch that uses that mode's score as a feature fails outright
    //! for that row rather than substituting a sentinel/missing value.
    use ordinaldb::hybrid::{
        rrf_fuse_batch, LtrFeatureBatch, LtrFeatureInputs, LtrFeatureSchema, RankedBatch,
        RrfConfig, ScoredRow,
    };

    #[test]
    fn ltr_features_require_every_fused_row_in_every_configured_source() {
        // Row 2 is found by dense search only -- exactly what happens for
        // this cookbook's own paraphrase query when a candidate has zero
        // BM25 term overlap with the query.
        let dense = RankedBatch::single(vec![
            ScoredRow { row_id: 1, score: 0.9 },
            ScoredRow { row_id: 2, score: 0.8 },
        ])
        .unwrap();
        let sparse = RankedBatch::single(vec![ScoredRow { row_id: 1, score: 5.0 }]).unwrap();
        let fused = rrf_fuse_batch(&dense, &sparse, RrfConfig::default()).unwrap();
        assert_eq!(fused.hits_for_query(0).unwrap().len(), 2, "row 2 is in the fused set");

        let schema = LtrFeatureSchema {
            schema_version: ordinaldb::hybrid::LTR_FEATURE_SCHEMA_V1.to_string(),
            feature_names: vec!["bm25_score".to_string()],
            forbidden_features: Vec::new(),
            dense_features_required: false,
        };
        let inputs = LtrFeatureInputs {
            fused: &fused,
            rank_cos_scores: None,
            dense_cos_scores: None,
            bm25_scores: Some(&sparse),
            doc_len_chars: None,
            query_len_chars: None,
        };

        let err = LtrFeatureBatch::from_inputs(&inputs, &schema).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("bm25_score") && message.contains("row_id 2"),
            "expected a missing-source error naming the uncovered row, got: {message}"
        );
    }
}
