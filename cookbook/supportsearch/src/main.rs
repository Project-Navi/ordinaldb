mod corpus;
mod corpus_filler;
mod embedder;
mod index;
mod queries;
#[cfg(feature = "experimental-ltr")]
mod ltr_demo;

use std::collections::HashMap;

use ordinaldb::hybrid::{rrf_fuse_batch, RankedBatch, RrfConfig, ScoredRow};

use embedder::{Embedder, FastEmbedEmbedder, HashEmbedder};

const TOP_K: usize = 5;

fn main() -> anyhow::Result<()> {
    println!("=== supportsearch: hybrid (BM25 + dense) search over a support KB ===\n");

    let use_hash_fallback = std::env::var("SUPPORTSEARCH_HASH_EMBEDDER").is_ok();
    let embedder: Box<dyn Embedder> = if use_hash_fallback {
        println!("SUPPORTSEARCH_HASH_EMBEDDER set: using the non-semantic hash fallback.");
        Box::new(HashEmbedder)
    } else {
        match FastEmbedEmbedder::try_new() {
            Ok(model) => Box::new(model),
            Err(err) => {
                eprintln!(
                    "fastembed init failed ({err}); falling back to the non-semantic \
                     hash embedder. Query class 2 (paraphrase) will NOT demonstrate \
                     dense search winning under this fallback -- see README."
                );
                Box::new(HashEmbedder)
            }
        }
    };

    let data_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    std::fs::create_dir_all(&data_dir)?;

    #[cfg(feature = "experimental-ltr")]
    let extra_auxiliaries = {
        let model_path = data_dir.join("kb.ltr_model.json.tmp");
        ltr_demo::write_model_file(&model_path)?;
        vec![ordinaldb::manifest::AuxiliaryArtifactDeclaration::required(
            ordinaldb::artifacts::LTR_MODEL_AUX_NAME,
            &model_path,
            "ltr_model.json",
        )]
    };
    #[cfg(not(feature = "experimental-ltr"))]
    let extra_auxiliaries = Vec::new();

    let store = index::build_and_persist(embedder.as_ref(), &data_dir, extra_auxiliaries)?;
    let titles: HashMap<u64, &str> = store.docs.iter().map(|d| (d.id, d.title)).collect();

    let mut categories: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for doc in &store.docs {
        *categories.entry(doc.category).or_default() += 1;
    }
    println!(
        "\ncorpus: {} documents across {} categories, dense dim={}, bits={}",
        store.docs.len(),
        categories.len(),
        embedder::DIM,
        index::BITS
    );
    println!("bundle: {}\n", store.bundle_path.display());

    for q in queries::demo_queries() {
        println!("----------------------------------------------------------------");
        println!("{}", q.label);
        println!("query: {:?}", q.text);
        println!("gold:  [{}] {:?}", q.gold_id, titles[&q.gold_id]);
        println!("       ({})", q.claim);

        let query_embedding = embedder.embed_one(q.text)?;
        let dense = store.dense.search_batch_rows(&query_embedding, TOP_K)?;
        let sparse = store.sparse.search_batch(&[q.text], TOP_K)?;
        let fused = rrf_fuse_batch(&dense, &sparse, RrfConfig {
            top_k: TOP_K,
            ..RrfConfig::default()
        })?;

        print_ranked("dense (OrdVec cosine-rank)", &dense, q.gold_id, &titles);
        print_ranked("sparse (BM25)", &sparse, q.gold_id, &titles);
        print_fused(&fused, q.gold_id, &titles);
        println!();
    }

    println!("----------------------------------------------------------------");
    #[cfg(feature = "experimental-ltr")]
    {
        ltr_demo::run(&store, embedder.as_ref())?;
    }
    #[cfg(not(feature = "experimental-ltr"))]
    {
        println!(
            "(LTR reranking demo skipped: rebuild with `--features experimental-ltr` to see it.)"
        );
    }

    println!("\ncheck the persisted store with the ops CLI:");
    println!(
        "  cargo run -p ordinaldb-cli -- inspect {}",
        store.bundle_path.display()
    );
    println!(
        "  cargo run -p ordinaldb-cli -- verify  {}",
        store.bundle_path.display()
    );

    Ok(())
}

fn rank_of(batch_hits: &[ScoredRow], gold_id: u64) -> Option<usize> {
    batch_hits.iter().position(|h| h.row_id == gold_id).map(|i| i + 1)
}

fn print_ranked(
    label: &str,
    batch: &RankedBatch,
    gold_id: u64,
    titles: &HashMap<u64, &str>,
) {
    let hits = batch.hits_for_query(0).unwrap_or(&[]);
    println!("\n  [{label}] top-{}:", hits.len());
    for (i, hit) in hits.iter().enumerate() {
        let marker = if hit.row_id == gold_id { "  <-- GOLD" } else { "" };
        println!(
            "    {}. [{}] score={:.4}  {:?}{marker}",
            i + 1,
            hit.row_id,
            hit.score,
            titles.get(&hit.row_id).unwrap_or(&"?")
        );
    }
    match rank_of(hits, gold_id) {
        Some(rank) => println!("  gold rank in {label}: {rank}"),
        None => println!("  gold rank in {label}: NOT in top-{TOP_K} (miss)"),
    }
}

fn print_fused(
    fused: &ordinaldb::hybrid::FusedBatch,
    gold_id: u64,
    titles: &HashMap<u64, &str>,
) {
    let hits = fused.hits_for_query(0).unwrap_or(&[]);
    println!("\n  [RRF fused] top-{}:", hits.len());
    for (i, hit) in hits.iter().enumerate() {
        let marker = if hit.row_id == gold_id { "  <-- GOLD" } else { "" };
        println!(
            "    {}. [{}] fused={:.5} dense_rank={:?} sparse_rank={:?}  {:?}{marker}",
            i + 1,
            hit.row_id,
            hit.fused_score,
            hit.dense_rank,
            hit.sparse_rank,
            titles.get(&hit.row_id).unwrap_or(&"?")
        );
    }
    let gold_rank = hits.iter().position(|h| h.row_id == gold_id).map(|i| i + 1);
    match gold_rank {
        Some(1) => println!("  gold rank in RRF fused: 1 (WINS)"),
        Some(rank) => println!("  gold rank in RRF fused: {rank}"),
        None => println!("  gold rank in RRF fused: NOT in top-{TOP_K} (miss)"),
    }
}
