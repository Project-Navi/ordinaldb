//! Builds the dense (OrdVec) + sparse (BM25) hybrid store for the KB
//! corpus and persists it as one manifest-verified `.odb` bundle, mirroring
//! `examples/downstream-smoke/src/main.rs`'s pattern (the only working,
//! docs/api.md-referenced hybrid walkthrough at the time this was written).

use std::path::{Path, PathBuf};

use ordinaldb::artifacts::{MANIFEST_FILE, SPARSE_BM25_AUX_NAME};
use ordinaldb::hybrid::{Bm25MmapIndex, SparseIndexBuilder, TokenizerKind};
use ordinaldb::manifest::{AuxiliaryArtifactDeclaration, CreateManifestOptions, VerifyOptions};
use ordinaldb::{BuildOptions, DenseLoadOptions, IdMapIndex, OrdinalIndexBuilder, SignPolicy};

use crate::corpus::{corpus, Doc};
use crate::embedder::{Embedder, DIM};

pub const BITS: u8 = 2;

pub struct BuiltStore {
    pub docs: Vec<Doc>,
    // Only read back by the optional `--features experimental-ltr` demo
    // (true dense-cosine features, and the verified LTR model sidecar
    // path), so both are unused dead code in the default build.
    #[allow(dead_code)]
    pub embeddings: std::collections::HashMap<u64, Vec<f32>>,
    pub dense: IdMapIndex,
    pub sparse: Bm25MmapIndex,
    pub bundle_path: PathBuf,
    #[allow(dead_code)]
    pub manifest_path: PathBuf,
}

/// Build the corpus's dense + sparse artifacts, write them as one
/// manifest-verified `.odb` bundle (BM25 registered as a required auxiliary
/// artifact, exactly like `downstream-smoke`), then immediately reopen both
/// through their verified-load entry points -- proving the persisted bundle
/// round-trips, not just the in-memory builders.
pub fn build_and_persist(
    embedder: &dyn Embedder,
    data_dir: &Path,
    extra_auxiliaries: Vec<AuxiliaryArtifactDeclaration>,
) -> anyhow::Result<BuiltStore> {
    std::fs::create_dir_all(data_dir)?;
    let bundle_path = data_dir.join("kb.odb");
    let sparse_source = data_dir.join("kb.sparse.bm25.tmp");
    // A stale bundle_path/sparse_source can be either shape (a leftover
    // directory or a leftover plain file) depending on how a prior run
    // died, so clean up both possibilities for both paths -- same pattern
    // as `examples/downstream-smoke`'s `cleanup()` -- instead of assuming
    // each path is only ever one shape and leaving an obstruction behind
    // that turns into a confusing IO error further down.
    remove_stale_path(&bundle_path);
    remove_stale_path(&sparse_source);

    let docs = corpus();
    println!(
        "embedding {} documents with {}...",
        docs.len(),
        embedder.name()
    );
    // Dense embeds the body only; BM25 (below) indexes title + body. This
    // is a deliberate, common RAG-pipeline asymmetry, not an oversight: a
    // short title can otherwise dominate a mean-pooled sentence embedding
    // of a much longer body, while lexical search benefits from titles
    // being searchable verbatim.
    let texts: Vec<&str> = docs.iter().map(|d| d.body.as_str()).collect();
    let started = std::time::Instant::now();
    let vectors = embedder.embed_batch(&texts)?;
    println!("  embedded in {:?}", started.elapsed());

    let mut embeddings = std::collections::HashMap::with_capacity(docs.len());
    for (doc, vector) in docs.iter().zip(&vectors) {
        embeddings.insert(doc.id, vector.clone());
    }

    // Sparse (BM25) side: index title + body under IdentifierSubtokens so
    // camelCase/identifier-shaped tokens (and the plain error-code run
    // itself) are both indexed.
    let mut sparse_builder = SparseIndexBuilder::new(TokenizerKind::IdentifierSubtokens);
    for doc in &docs {
        sparse_builder.add_text(doc.id, &format!("{} {}", doc.title, doc.body))?;
    }
    let sparse_report = sparse_builder.write_mmap(&sparse_source)?;
    println!(
        "  BM25 sidecar: {} terms, {} postings, tokenizer={:?}",
        sparse_report.term_count, sparse_report.postings_count, sparse_report.tokenizer
    );

    // Dense (OrdVec) side.
    // DIM=384 is sign-capable; Required fails fast instead of silently
    // writing a bundle the require_sign load below would reject.
    let mut dense_builder = OrdinalIndexBuilder::new(
        DIM,
        BITS,
        BuildOptions {
            sign: SignPolicy::Required,
        },
    )?;
    for doc in &docs {
        dense_builder.add(doc.id, &embeddings[&doc.id])?;
    }

    let mut auxiliaries = vec![AuxiliaryArtifactDeclaration::required(
        SPARSE_BM25_AUX_NAME,
        &sparse_source,
        "sparse.bm25",
    )];
    auxiliaries.extend(extra_auxiliaries);

    let report =
        dense_builder.write_verified_bundle(&bundle_path, CreateManifestOptions::default(), auxiliaries)?;
    let _ = std::fs::remove_file(&sparse_source);
    println!(
        "  wrote verified bundle: has_sign={} has_ids={}",
        report.has_sign, report.has_ids
    );

    let manifest_path = bundle_path.join(MANIFEST_FILE);
    // One call opens both sides of the bundle through their verified-load
    // entry points; HybridBundleError says which side failed.
    let hybrid = ordinaldb::hybrid::HybridBundle::open_verified(
        &manifest_path,
        VerifyOptions::default(),
        DenseLoadOptions {
            require_sign: true,
            expected_dim: Some(DIM),
            expected_bits: Some(BITS),
        },
        SPARSE_BM25_AUX_NAME,
    )?;
    let (dense, sparse) = (hybrid.dense, hybrid.sparse);

    Ok(BuiltStore {
        docs,
        embeddings,
        dense,
        sparse,
        bundle_path,
        manifest_path,
    })
}

/// Remove whatever currently occupies `path`, regardless of whether it's a
/// directory or a plain file. `.odb` bundles are directories and their
/// sparse sidecar sources are files, but a stray manual artifact or a run
/// that died at an unexpected point can leave either path as the "wrong"
/// shape; trying only the matching removal silently leaves it behind (both
/// calls below use `let _ =` because exactly one is expected to succeed).
fn remove_stale_path(path: &Path) {
    let _ = std::fs::remove_dir_all(path);
    let _ = std::fs::remove_file(path);
}
