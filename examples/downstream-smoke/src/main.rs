use ordinaldb::artifacts::{MANIFEST_FILE, SPARSE_BM25_AUX_NAME};
use ordinaldb::hybrid::{
    rrf_fuse_batch, Bm25MmapIndex, RrfConfig, SparseIndexBuilder, TokenizerKind,
};
use ordinaldb::manifest::{AuxiliaryArtifactDeclaration, CreateManifestOptions, VerifyOptions};
use ordinaldb::{BuildOptions, DenseLoadOptions, IdMapIndex, OrdinalIndexBuilder};
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let dim = 64;
    let row_count = 24;
    let data = vectors(row_count, dim);
    let query = vectors(1, dim);
    let ids = (0..row_count)
        .map(|idx| 10_000 + idx as u64)
        .collect::<Vec<_>>();

    let sparse_source = temp_path("ordinaldb-downstream-sparse.bm25");
    let bundle_path = temp_path("ordinaldb-downstream-boundary.odb");
    cleanup(&sparse_source);
    cleanup(&bundle_path);

    let mut sparse_builder = SparseIndexBuilder::new(TokenizerKind::IdentifierSubtokens);
    for (idx, &row_id) in ids.iter().enumerate() {
        sparse_builder
            .add_text(
                row_id,
                &format!(
                    "DocumentScore_{idx} alpha_terms beta_terms launchStory repeateds"
                ),
            )
            .expect("add sparse text");
    }
    sparse_builder
        .write_mmap(&sparse_source)
        .expect("write sparse mmap");

    let mut dense_builder =
        OrdinalIndexBuilder::new(dim, 2, BuildOptions { sign: true }).expect("dense builder");
    for (idx, &row_id) in ids.iter().enumerate() {
        dense_builder
            .add(row_id, &data[idx * dim..(idx + 1) * dim])
            .expect("add dense vector");
    }
    let report = dense_builder
        .write_verified_bundle(
            &bundle_path,
            CreateManifestOptions::default(),
            vec![AuxiliaryArtifactDeclaration::required(
                SPARSE_BM25_AUX_NAME,
                &sparse_source,
                "sparse.bm25",
            )],
        )
        .expect("write verified bundle");
    assert!(report.has_sign);
    assert!(report.has_ids);

    let manifest_path = bundle_path.join(MANIFEST_FILE);
    let loaded = IdMapIndex::open_verified(
        &manifest_path,
        VerifyOptions::default(),
        DenseLoadOptions {
            require_sign: true,
            expected_dim: Some(dim),
            expected_bits: Some(2),
        },
    )
    .expect("open verified dense bundle");
    let dense_rows = loaded
        .search_batch_rows(&query, 5)
        .expect("dense row search");
    assert_eq!(dense_rows.query_count(), 1);
    assert!(dense_rows.hits().iter().all(|hit| ids.contains(&hit.row_id)));

    let sparse = Bm25MmapIndex::open_verified_sidecar(
        &manifest_path,
        SPARSE_BM25_AUX_NAME,
        VerifyOptions::default(),
    )
    .expect("open verified sparse sidecar");
    let sparse_rows = sparse
        .search_batch(&["alpha_terms launchStory"], 5)
        .expect("sparse row search");
    assert_eq!(sparse_rows.query_count(), 1);
    assert!(sparse_rows.hits().iter().all(|hit| ids.contains(&hit.row_id)));

    let fused = rrf_fuse_batch(
        &dense_rows,
        &sparse_rows,
        RrfConfig {
            top_k: 5,
            ..RrfConfig::default()
        },
    )
    .expect("rrf fuse");
    assert_eq!(fused.query_count(), 1);
    assert!(!fused.hits().is_empty());
    assert!(fused.hits().iter().all(|hit| ids.contains(&hit.row_id)));

    cleanup(&sparse_source);
    cleanup(&bundle_path);
}

fn vectors(n: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0; n * dim];
    for row in 0..n {
        for col in 0..dim {
            let x = (((row + 5) * (col + 13) + row * 17 + col * 7) % 61) as f32 - 30.0;
            out[row * dim + col] = x / 31.0;
        }
    }
    out
}

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("{name}-{}", std::process::id()))
}

fn cleanup(path: &Path) {
    let _ = fs::remove_dir_all(path);
    let _ = fs::remove_file(path);
}
