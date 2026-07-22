#![cfg(feature = "hybrid")]

use ordinaldb::artifacts::{MANIFEST_FILE, SPARSE_BM25_AUX_NAME};
use ordinaldb::hybrid::{
    rrf_fuse_batch, Bm25MmapIndex, HybridBundle, HybridBundleError, RrfConfig, SparseIndexBuilder,
    TokenizerKind,
};
use ordinaldb::manifest::{AuxiliaryArtifactDeclaration, CreateManifestOptions, VerifyOptions};
use ordinaldb::{
    BuildOptions, DenseError, DenseLoadOptions, IdMapIndex, OrdinalIndex, OrdinalIndexBuilder,
    SignPolicy,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn verified_idmap_bundle_loads_dense_and_sparse_with_stable_row_ids() {
    let ids = [10, 20, 30, 40];
    let (bundle, sparse_source) = write_bundle_with_sparse(&ids, &ids, SignPolicy::Optional);
    let manifest_path = bundle.join(MANIFEST_FILE);

    let dense = IdMapIndex::open_verified(
        &manifest_path,
        VerifyOptions::default(),
        DenseLoadOptions {
            require_sign: true,
            expected_dim: Some(64),
            expected_bits: Some(2),
        },
    )
    .expect("verified dense load");
    let rows = dense.search_rows(&vectors(1, 64), 3).expect("row search");
    assert!(rows.iter().all(|row| ids.contains(&row.row_id)));

    let sparse = Bm25MmapIndex::open_verified_sidecar(
        &manifest_path,
        SPARSE_BM25_AUX_NAME,
        VerifyOptions::default(),
    )
    .expect("verified sparse load");
    let sparse_rows = sparse
        .search("alpha CamelHTTP terms", 3)
        .expect("sparse search");
    assert!(sparse_rows.iter().all(|row| ids.contains(&row.row_id)));

    cleanup(&bundle);
    cleanup(&sparse_source);
}

#[test]
fn verified_sparse_sidecar_rejects_wrong_stable_row_ids() {
    let ids = [10, 20, 30];
    let sparse_ids = [10, 999, 30];
    let (bundle, sparse_source) = write_bundle_with_sparse(&ids, &sparse_ids, SignPolicy::Optional);
    let manifest_path = bundle.join(MANIFEST_FILE);

    let err = match Bm25MmapIndex::open_verified_sidecar(
        &manifest_path,
        SPARSE_BM25_AUX_NAME,
        VerifyOptions::default(),
    ) {
        Ok(_) => panic!("mismatched sparse row IDs should be rejected"),
        Err(err) => err,
    };
    assert!(
        err.to_string()
            .contains("sparse row-id table does not match verified OrdinalDB ID sidecar"),
        "{err}"
    );

    cleanup(&bundle);
    cleanup(&sparse_source);
}

#[test]
fn verified_dense_load_can_require_sign_sidecar() {
    let ids = [0, 1, 2];
    let (bundle, sparse_source) = write_bundle_with_sparse(&ids, &ids, SignPolicy::Disabled);
    let manifest_path = bundle.join(MANIFEST_FILE);

    let err = match IdMapIndex::open_verified(
        &manifest_path,
        VerifyOptions::default(),
        DenseLoadOptions {
            require_sign: true,
            expected_dim: Some(64),
            expected_bits: Some(2),
        },
    ) {
        Ok(_) => panic!("missing required sign sidecar should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, DenseError::MissingSignSidecar), "{err}");

    cleanup(&bundle);
    cleanup(&sparse_source);
}

#[test]
fn verified_bundle_rejects_reserved_auxiliary_name() {
    let bundle = temp_path("ordinaldb-boundary-reserved-aux.odb");
    let aux = write_aux_source("ordinaldb-boundary-reserved-aux.bin", b"sidecar");
    cleanup(&bundle);

    let mut builder = OrdinalIndexBuilder::new(
        64,
        2,
        BuildOptions {
            sign: SignPolicy::Optional,
        },
    )
    .expect("dense builder");
    builder.add(1, &vectors(1, 64)).expect("add vector");
    let err = builder
        .write_verified_bundle(
            &bundle,
            CreateManifestOptions::default(),
            vec![AuxiliaryArtifactDeclaration::required(
                ordinaldb::artifacts::SIGN_AUX_NAME,
                &aux,
                "sidecars/custom-sign.bin",
            )],
        )
        .expect_err("reserved auxiliary name should be rejected");
    assert!(err.to_string().contains("duplicated or reserved"), "{err}");

    cleanup(&bundle);
    cleanup(&aux);
}

#[test]
fn verified_bundle_rejects_duplicate_auxiliary_bundle_path() {
    let bundle = temp_path("ordinaldb-boundary-duplicate-aux-path.odb");
    let aux_a = write_aux_source("ordinaldb-boundary-duplicate-aux-a.bin", b"a");
    let aux_b = write_aux_source("ordinaldb-boundary-duplicate-aux-b.bin", b"b");
    cleanup(&bundle);

    let mut builder = OrdinalIndexBuilder::new(
        64,
        2,
        BuildOptions {
            sign: SignPolicy::Disabled,
        },
    )
    .expect("dense builder");
    builder.add(1, &vectors(1, 64)).expect("add vector");
    let err = builder
        .write_verified_bundle(
            &bundle,
            CreateManifestOptions::default(),
            vec![
                AuxiliaryArtifactDeclaration::required(
                    "ordinaldb.test_a",
                    &aux_a,
                    "aux/shared.bin",
                ),
                AuxiliaryArtifactDeclaration::required(
                    "ordinaldb.test_b",
                    &aux_b,
                    "aux/shared.bin",
                ),
            ],
        )
        .expect_err("duplicate auxiliary path should be rejected");
    assert!(err.to_string().contains("duplicated"), "{err}");

    cleanup(&bundle);
    cleanup(&aux_a);
    cleanup(&aux_b);
}

#[test]
fn candidate_slot_validation_reports_slot_not_row_id() {
    let mut index = OrdinalIndex::new(64, 2).expect("index");
    index.add(&vectors(2, 64));
    let err = match index.search_checked_with_candidates(&vectors(1, 64), 1, &[2]) {
        Ok(_) => panic!("out-of-range candidate slot should fail"),
        Err(err) => err,
    };
    assert!(
        matches!(err, DenseError::CandidateSlotOutOfRange(2)),
        "{err}"
    );
    assert!(err.to_string().contains("candidate slot 2"), "{err}");
}

#[test]
fn declared_but_unloadable_sign_sidecar_is_not_reported_missing() {
    let bundle = temp_path("ordinaldb-boundary-unloadable-sign.odb");
    cleanup(&bundle);

    let mut builder = OrdinalIndexBuilder::new(
        64,
        2,
        BuildOptions {
            sign: SignPolicy::Optional,
        },
    )
    .expect("dense builder");
    builder.add(1, &vectors(1, 64)).expect("add vector");
    builder
        .write_verified_bundle(&bundle, CreateManifestOptions::default(), Vec::new())
        .expect("write bundle");
    fs::remove_file(bundle.join(ordinaldb::artifacts::SIGN_FILE)).expect("remove sign sidecar");

    let err = match IdMapIndex::open_verified(
        bundle.join(MANIFEST_FILE),
        VerifyOptions::default(),
        DenseLoadOptions {
            require_sign: true,
            expected_dim: Some(64),
            expected_bits: Some(2),
        },
    ) {
        Ok(_) => panic!("declared but missing sign sidecar should fail"),
        Err(err) => err,
    };
    assert!(!matches!(err, DenseError::MissingSignSidecar), "{err}");

    cleanup(&bundle);
}

#[test]
fn batch_dense_sparse_and_rrf_preserve_per_query_allowlists() {
    let ids = [10, 20, 30, 40];
    let (bundle, sparse_source) = write_bundle_with_sparse(&ids, &ids, SignPolicy::Optional);
    let manifest_path = bundle.join(MANIFEST_FILE);
    let dense = IdMapIndex::open_verified(
        &manifest_path,
        VerifyOptions::default(),
        DenseLoadOptions {
            require_sign: true,
            expected_dim: Some(64),
            expected_bits: Some(2),
        },
    )
    .expect("verified dense load");
    let sparse = Bm25MmapIndex::open_verified_sidecar(
        &manifest_path,
        SPARSE_BM25_AUX_NAME,
        VerifyOptions::default(),
    )
    .expect("verified sparse load");

    let queries = vectors(2, 64);
    let allow_0 = [10, 20];
    let allow_1 = [30, 40];
    let dense_rows = dense
        .search_batch_rows_with_allowlists(
            &queries,
            2,
            Some([allow_0.as_slice(), allow_1.as_slice()]),
        )
        .expect("dense batch allowlist search");
    assert_eq!(dense_rows.query_count(), 2);
    assert!(dense_rows
        .hits_for_query(0)
        .unwrap()
        .iter()
        .all(|hit| allow_0.contains(&hit.row_id)));
    assert!(dense_rows
        .hits_for_query(1)
        .unwrap()
        .iter()
        .all(|hit| allow_1.contains(&hit.row_id)));

    let sparse_rows = sparse
        .search_batch_with_allowlists(
            &["alpha doc_0", "alpha doc_3"],
            2,
            &[Some(allow_0.as_slice()), Some(allow_1.as_slice())],
        )
        .expect("sparse batch allowlist search");
    assert_eq!(sparse_rows.query_count(), 2);
    assert!(sparse_rows
        .hits_for_query(0)
        .unwrap()
        .iter()
        .all(|hit| allow_0.contains(&hit.row_id)));
    assert!(sparse_rows
        .hits_for_query(1)
        .unwrap()
        .iter()
        .all(|hit| allow_1.contains(&hit.row_id)));

    let fused = rrf_fuse_batch(
        &dense_rows,
        &sparse_rows,
        RrfConfig {
            top_k: 2,
            ..RrfConfig::default()
        },
    )
    .expect("rrf batch fuse");
    assert_eq!(fused.query_count(), 2);
    assert!(fused
        .hits_for_query(0)
        .unwrap()
        .iter()
        .all(|hit| allow_0.contains(&hit.row_id)));
    assert!(fused
        .hits_for_query(1)
        .unwrap()
        .iter()
        .all(|hit| allow_1.contains(&hit.row_id)));

    cleanup(&bundle);
    cleanup(&sparse_source);
}

#[test]
fn hybrid_bundle_open_verified_returns_dense_and_sparse_from_one_manifest() {
    let ids = [10, 20, 30, 40];
    let (bundle, sparse_source) = write_bundle_with_sparse(&ids, &ids, SignPolicy::Optional);
    let manifest_path = bundle.join(MANIFEST_FILE);

    let opened = HybridBundle::open_verified(
        &manifest_path,
        VerifyOptions::default(),
        DenseLoadOptions {
            require_sign: true,
            expected_dim: Some(64),
            expected_bits: Some(2),
        },
        SPARSE_BM25_AUX_NAME,
    )
    .expect("open dense + sparse from one manifest");

    let dense_rows = opened
        .dense
        .search_batch_rows(&vectors(1, 64), 3)
        .expect("dense row search");
    let sparse_rows = opened
        .sparse
        .search_batch(&["alpha doc_0"], 3)
        .expect("sparse row search");
    let fused = rrf_fuse_batch(
        &dense_rows,
        &sparse_rows,
        RrfConfig {
            top_k: 3,
            ..RrfConfig::default()
        },
    )
    .expect("rrf fuse");
    assert_eq!(fused.query_count(), 1);
    assert!(!fused.hits().is_empty());
    assert!(fused.hits().iter().all(|hit| ids.contains(&hit.row_id)));

    cleanup(&bundle);
    cleanup(&sparse_source);
}

#[test]
fn hybrid_bundle_open_verified_reports_dense_and_sparse_failures_distinctly() {
    let ids = [10, 20, 30];
    let (bundle, sparse_source) = write_bundle_with_sparse(&ids, &ids, SignPolicy::Optional);
    let manifest_path = bundle.join(MANIFEST_FILE);

    let dense_err = match HybridBundle::open_verified(
        &manifest_path,
        VerifyOptions::default(),
        DenseLoadOptions {
            require_sign: true,
            expected_dim: Some(32),
            expected_bits: Some(2),
        },
        SPARSE_BM25_AUX_NAME,
    ) {
        Ok(_) => panic!("wrong expected_dim must fail the dense side"),
        Err(err) => err,
    };
    assert!(
        matches!(dense_err, HybridBundleError::Dense(_)),
        "{dense_err}"
    );

    let sparse_err = match HybridBundle::open_verified(
        &manifest_path,
        VerifyOptions::default(),
        DenseLoadOptions {
            require_sign: true,
            expected_dim: Some(64),
            expected_bits: Some(2),
        },
        "ordinaldb.no_such_sidecar",
    ) {
        Ok(_) => panic!("unknown sparse auxiliary name must fail the sparse side"),
        Err(err) => err,
    };
    assert!(
        matches!(sparse_err, HybridBundleError::Sparse(_)),
        "{sparse_err}"
    );

    cleanup(&bundle);
    cleanup(&sparse_source);
}

fn write_bundle_with_sparse(
    dense_ids: &[u64],
    sparse_ids: &[u64],
    sign: SignPolicy,
) -> (PathBuf, PathBuf) {
    assert_eq!(dense_ids.len(), sparse_ids.len());
    let dim = 64;
    let data = vectors(dense_ids.len(), dim);
    let bundle = temp_path("ordinaldb-boundary-api.odb");
    let sparse_source = temp_path("ordinaldb-boundary-api-sparse.bm25");
    cleanup(&bundle);
    cleanup(&sparse_source);

    let mut sparse_builder = SparseIndexBuilder::new(TokenizerKind::IdentifierSubtokens);
    for (idx, &row_id) in sparse_ids.iter().enumerate() {
        sparse_builder
            .add_text(
                row_id,
                &format!("alpha_terms CamelHTTP repeateds doc_{idx}"),
            )
            .expect("add sparse text");
    }
    sparse_builder
        .write_mmap(&sparse_source)
        .expect("write sparse mmap");

    let mut dense_builder =
        OrdinalIndexBuilder::new(dim, 2, BuildOptions { sign }).expect("dense builder");
    for (idx, &row_id) in dense_ids.iter().enumerate() {
        dense_builder
            .add(row_id, &data[idx * dim..(idx + 1) * dim])
            .expect("add dense vector");
    }
    dense_builder
        .write_verified_bundle(
            &bundle,
            CreateManifestOptions::default(),
            vec![AuxiliaryArtifactDeclaration::required(
                SPARSE_BM25_AUX_NAME,
                &sparse_source,
                "sparse.bm25",
            )],
        )
        .expect("write verified bundle");

    (bundle, sparse_source)
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
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("{name}-{}-{stamp}", std::process::id()))
}

fn write_aux_source(name: &str, bytes: &[u8]) -> PathBuf {
    let path = temp_path(name);
    cleanup(&path);
    fs::write(&path, bytes).expect("write auxiliary source");
    path
}

fn cleanup(path: &Path) {
    let _ = fs::remove_dir_all(path);
    let _ = fs::remove_file(path);
}
