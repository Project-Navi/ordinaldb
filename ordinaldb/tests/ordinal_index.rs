use ordinaldb::{
    artifacts::{EMBEDDING_MODEL, SIGN_AUX_NAME},
    AddError, ConstructError, DenseError, DenseSearchExecution, DenseSearchOptions, OrdinalIndex,
    TwoStageOptions,
};
use ordvec::{RankQuant, SignBitmap};
use ordvec_manifest::{
    create_manifest_for_index_with_options, write_manifest_file, CreateAuxiliaryArtifact,
    CreateManifestOptions, CreateRowIdentity,
};
use std::fs;
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

const DIM: usize = 64;
static NEXT_TEMP: AtomicUsize = AtomicUsize::new(0);

fn vectors(n: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0; n * dim];
    for row in 0..n {
        for col in 0..dim {
            let x = (((row + 3) * (col + 5) + row * 17 + col * 11) % 37) as f32 - 18.0;
            out[row * dim + col] = x / 19.0;
        }
    }
    out
}

#[test]
fn add_search_matches_direct_ordvec_rankquant() {
    let data = vectors(24, DIM);
    let queries = vectors(3, DIM);

    let mut idx = OrdinalIndex::new(DIM, 2).unwrap();
    idx.add(&data);

    let mut direct = RankQuant::new(DIM, 2);
    direct.add(&data);

    let got = idx.search(&queries, 5);
    let expected = direct.search_asymmetric(&queries, 5);
    assert_eq!(got.nq, expected.nq);
    assert_eq!(got.k, expected.k);
    assert_eq!(got.indices, expected.indices);
    assert_eq!(got.scores, expected.scores);
}

#[test]
fn default_search_uses_signbitmap_rankquant_two_stage() {
    let data = vectors(400, DIM);
    let queries = vectors(2, DIM);

    let mut idx = OrdinalIndex::new(DIM, 2).unwrap();
    idx.add(&data);

    let mut sign = SignBitmap::new(DIM);
    sign.add(&data);
    let mut rankquant = RankQuant::new(DIM, 2);
    rankquant.add(&data);

    let got = idx.search(&queries, 3);
    assert_eq!(got.nq, 2);
    assert_eq!(got.k, 3);
    for query_index in 0..got.nq {
        let query = &queries[query_index * DIM..(query_index + 1) * DIM];
        let candidates = sign.top_m_candidates(query, 256);
        assert_eq!(candidates.len(), 256);
        let (scores, indices) = rankquant.search_asymmetric_subset(query, &candidates, 3);
        assert_eq!(got.scores_for_query(query_index), scores.as_slice());
        assert_eq!(got.indices_for_query(query_index), indices.as_slice());
    }
}

#[test]
fn sign_two_stage_options_can_match_fixed_native_candidate_count() {
    let data = vectors(400, DIM);
    let queries = vectors(2, DIM);

    let mut idx = OrdinalIndex::new(DIM, 2).unwrap();
    idx.add(&data);

    let mut sign = SignBitmap::new(DIM);
    sign.add(&data);
    let mut rankquant = RankQuant::new(DIM, 2);
    rankquant.add(&data);

    let options = DenseSearchOptions::sign_two_stage(TwoStageOptions::fixed_candidates(64));
    let got = idx.search_with_options(&queries, 3, options);
    assert_eq!(got.nq, 2);
    assert_eq!(got.k, 3);
    for query_index in 0..got.nq {
        let query = &queries[query_index * DIM..(query_index + 1) * DIM];
        let candidates = sign.top_m_candidates(query, 64);
        assert_eq!(candidates.len(), 64);
        let (scores, indices) = rankquant.search_asymmetric_subset(query, &candidates, 3);
        assert_eq!(got.scores_for_query(query_index), scores.as_slice());
        assert_eq!(got.indices_for_query(query_index), indices.as_slice());
    }
}

#[test]
fn sign_two_stage_options_clamp_output_k_to_candidate_count() {
    let data = vectors(40, DIM);
    let queries = vectors(2, DIM);
    let mut idx = OrdinalIndex::new(DIM, 2).unwrap();
    idx.add(&data);

    let options = DenseSearchOptions::sign_two_stage(TwoStageOptions::fixed_candidates(2));
    let report = idx.search_with_report(&queries, 5, options).unwrap();
    assert_eq!(report.plan.execution, DenseSearchExecution::SignTwoStage);
    assert_eq!(report.plan.candidate_count, Some(2));
    assert_eq!(report.plan.effective_k, 2);
    assert_eq!(report.results.k, 2);
    assert_eq!(report.results.indices.len(), 4);
    assert!(report.results.indices.iter().all(|idx| *idx >= 0));
}

#[test]
fn sign_two_stage_rejects_zero_candidate_options_for_nonempty_search() {
    let data = vectors(8, DIM);
    let queries = vectors(1, DIM);
    let mut idx = OrdinalIndex::new(DIM, 2).unwrap();
    idx.add(&data);

    let options = DenseSearchOptions::sign_two_stage(TwoStageOptions {
        min_candidates: 0,
        k_multiplier: 0,
        max_candidates: None,
    });
    let err = idx
        .search_checked_with_options(&queries, 1, options)
        .unwrap_err();
    assert!(matches!(err, DenseError::Limit(_)), "{err}");
    assert!(err.to_string().contains("candidate_count"), "{err}");

    let clamped_to_zero = DenseSearchOptions::sign_two_stage(TwoStageOptions {
        min_candidates: 1,
        k_multiplier: 0,
        max_candidates: Some(0),
    });
    let err = idx
        .search_checked_with_options(&queries, 1, clamped_to_zero)
        .unwrap_err();
    assert!(matches!(err, DenseError::Limit(_)), "{err}");
}

#[test]
fn exact_rankquant_mode_bypasses_sign_sidecar() {
    let data = vectors(80, DIM);
    let queries = vectors(3, DIM);
    let mut idx = OrdinalIndex::new(DIM, 2).unwrap();
    idx.add(&data);

    let mut direct = RankQuant::new(DIM, 2);
    direct.add(&data);

    let got = idx.search_with_options(&queries, 5, DenseSearchOptions::exact_rankquant());
    let expected = direct.search_asymmetric(&queries, 5);
    assert_eq!(got.nq, expected.nq);
    assert_eq!(got.k, expected.k);
    assert_eq!(got.indices, expected.indices);
    assert_eq!(got.scores, expected.scores);
}

#[test]
fn sign_two_stage_mode_requires_sign_sidecar() {
    let mut idx = OrdinalIndex::new(DIM, 4).unwrap();
    idx.add(&vectors(8, DIM));
    let err = match idx.search_checked_with_options(
        &vectors(1, DIM),
        3,
        DenseSearchOptions::sign_two_stage(TwoStageOptions::default()),
    ) {
        Ok(_) => panic!("missing sign sidecar should be rejected"),
        Err(err) => err,
    };
    assert!(matches!(err, DenseError::MissingSignSidecar), "{err}");
    assert_eq!(err.to_string(), "operation requires a sign sidecar");
}

#[test]
fn unchecked_sign_two_stage_without_sign_falls_back_to_exact() {
    let data = vectors(8, DIM);
    let queries = vectors(1, DIM);
    let mut idx = OrdinalIndex::new(DIM, 4).unwrap();
    idx.add(&data);

    let mut direct = RankQuant::new(DIM, 4);
    direct.add(&data);

    let got = idx.search_with_options(
        &queries,
        3,
        DenseSearchOptions::sign_two_stage(TwoStageOptions::default()),
    );
    let expected = direct.search_asymmetric(&queries, 3);
    assert_eq!(got.indices, expected.indices);
    assert_eq!(got.scores, expected.scores);
}

#[test]
fn dense_search_report_records_default_sign_plan() {
    let data = vectors(400, DIM);
    let queries = vectors(2, DIM);
    let mut idx = OrdinalIndex::new(DIM, 2).unwrap();
    idx.add(&data);

    let report = idx
        .search_with_report(&queries, 3, DenseSearchOptions::default())
        .unwrap();
    assert_eq!(report.plan.execution, DenseSearchExecution::SignTwoStage);
    assert_eq!(report.plan.query_count, 2);
    assert_eq!(report.plan.requested_k, 3);
    assert_eq!(report.plan.effective_k, 3);
    assert_eq!(report.plan.search_space, 400);
    assert_eq!(report.plan.candidate_count, Some(256));
    assert_eq!(report.results.nq, 2);
    assert_eq!(report.results.k, 3);
    assert!(report.timings.total >= report.timings.validation);
}

#[test]
fn lazy_dim_locks_on_first_non_empty_add() {
    let mut idx = OrdinalIndex::new_lazy(2).unwrap();
    assert_eq!(idx.dim_opt(), None);

    idx.add_2d(&[], DIM).unwrap();
    assert_eq!(idx.dim_opt(), None);

    idx.add_2d(&vectors(2, DIM), DIM).unwrap();
    assert_eq!(idx.dim_opt(), Some(DIM));
    assert_eq!(idx.dim(), DIM);
    assert_eq!(idx.len(), 2);
}

#[test]
fn wrong_dim_errors_without_mutation() {
    let mut idx = OrdinalIndex::new(DIM, 2).unwrap();
    idx.add_2d(&vectors(2, DIM), DIM).unwrap();
    let err = idx.add_2d(&vectors(1, 32), 32).unwrap_err();
    assert_eq!(
        err,
        AddError::DimMismatch {
            existing: DIM,
            got: 32
        }
    );
    assert_eq!(idx.len(), 2);
}

#[test]
fn bits_one_two_four_work_and_three_errors() {
    for bits in [1, 2, 4] {
        let mut idx = OrdinalIndex::new(DIM, bits).unwrap();
        idx.add(&vectors(4, DIM));
        assert_eq!(idx.bits(), bits);
        assert_eq!(idx.search(&vectors(1, DIM), 2).k, 2);
    }

    assert!(matches!(
        OrdinalIndex::new(DIM, 3),
        Err(ConstructError::UnsupportedBits(3))
    ));
}

#[test]
fn empty_index_search_is_safe_and_k_clamps() {
    let idx = OrdinalIndex::new(DIM, 2).unwrap();
    let res = idx.search(&vectors(2, DIM), usize::MAX);
    assert_eq!(res.nq, 2);
    assert_eq!(res.k, 0);
    assert!(res.scores.is_empty());
    assert!(res.indices.is_empty());

    let lazy = OrdinalIndex::new_lazy(2).unwrap();
    let res = lazy.search(&vectors(2, DIM), 10);
    assert_eq!(res.nq, 0);
    assert_eq!(res.k, 0);
}

#[test]
fn k_greater_than_len_clamps() {
    let mut idx = OrdinalIndex::new(DIM, 2).unwrap();
    idx.add(&vectors(3, DIM));
    let res = idx.search(&vectors(2, DIM), 99);
    assert_eq!(res.nq, 2);
    assert_eq!(res.k, 3);
    assert_eq!(res.indices.len(), 6);

    let checked = idx.search_checked(&vectors(2, DIM), usize::MAX).unwrap();
    assert_eq!(checked.nq, 2);
    assert_eq!(checked.k, 3);
    assert_eq!(checked.indices.len(), 6);

    let exact = idx
        .search_with_report(&vectors(2, DIM), 99, DenseSearchOptions::exact_rankquant())
        .unwrap();
    assert_eq!(exact.plan.execution, DenseSearchExecution::ExactRankQuant);
    assert_eq!(exact.plan.requested_k, 99);
    assert_eq!(exact.plan.effective_k, 3);
    assert_eq!(exact.results.k, 3);
    assert_eq!(exact.results.indices.len(), 6);
}

#[test]
fn mask_search_returns_only_allowed_slots_and_matches_subset() {
    let data = vectors(20, DIM);
    let queries = vectors(2, DIM);
    let mut idx = OrdinalIndex::new(DIM, 2).unwrap();
    idx.add(&data);

    let mut direct = RankQuant::new(DIM, 2);
    direct.add(&data);

    let mut mask = vec![false; 20];
    for slot in [1, 4, 7, 9, 15] {
        mask[slot] = true;
    }
    let candidates: Vec<u32> = mask
        .iter()
        .enumerate()
        .filter_map(|(idx, allowed)| allowed.then_some(idx as u32))
        .collect();

    let got = idx.search_with_mask(&queries, 3, Some(&mask));
    assert_eq!(got.nq, 2);
    assert_eq!(got.k, 3);
    for query_index in 0..got.nq {
        let query = &queries[query_index * DIM..(query_index + 1) * DIM];
        let (scores, indices) = direct.search_asymmetric_subset(query, &candidates, 3);
        assert_eq!(got.scores_for_query(query_index), scores.as_slice());
        assert_eq!(got.indices_for_query(query_index), indices.as_slice());
        assert!(got
            .indices_for_query(query_index)
            .iter()
            .all(|idx| mask[*idx as usize]));
    }
}

#[test]
fn empty_mask_returns_zero_results() {
    let mut idx = OrdinalIndex::new(DIM, 2).unwrap();
    idx.add(&vectors(4, DIM));
    let res = idx.search_with_mask(&vectors(2, DIM), 3, Some(&[false; 4]));
    assert_eq!(res.nq, 2);
    assert_eq!(res.k, 0);
    assert!(res.scores.is_empty());
    assert!(res.indices.is_empty());
}

#[test]
fn swap_remove_reduces_len_and_keeps_searchable() {
    let mut idx = OrdinalIndex::new(DIM, 2).unwrap();
    idx.add(&vectors(8, DIM));
    assert_eq!(idx.swap_remove(2), 7);
    assert_eq!(idx.len(), 7);
    let res = idx.search(&vectors(1, DIM), 7);
    assert_eq!(res.k, 7);
    assert!(res.indices.iter().all(|idx| (0..7).contains(idx)));
}

#[test]
fn add_2d_rejects_invalid_input_value_without_mutation() {
    let mut idx = OrdinalIndex::new(DIM, 2).unwrap();
    let mut data = vectors(1, DIM);
    data[9] = f32::NAN;
    let err = idx.add_2d(&data, DIM).unwrap_err();
    assert!(matches!(
        err,
        AddError::InvalidInputValue {
            vector_index: 0,
            coord_index: 9,
            ..
        }
    ));
    assert_eq!(idx.len(), 0);
}

#[test]
fn write_load_roundtrip_preserves_search_for_supported_bits() {
    for bits in [1, 2, 4] {
        let path = temp_bundle(&format!("ordinal-roundtrip-bits-{bits}"));
        cleanup(&path);

        let data = vectors(64, DIM);
        let queries = vectors(3, DIM);
        let mut idx = OrdinalIndex::new(DIM, bits).unwrap();
        idx.add(&data);
        let before = idx.search(&queries, 7);

        idx.write(&path).unwrap();
        assert!(path.join("manifest.json").exists());
        assert!(path.join("index.ovrq").exists());
        if bits == 2 {
            assert!(path.join("sign.ovsb").exists());
        }

        let loaded = OrdinalIndex::load(&path).unwrap();
        let after = loaded.search(&queries, 7);
        assert_eq!(loaded.dim(), DIM);
        assert_eq!(loaded.bits(), bits);
        assert_eq!(loaded.len(), idx.len());
        assert_eq!(after.scores, before.scores);
        assert_eq!(after.indices, before.indices);

        cleanup(&path);
    }
}

#[test]
fn lazy_persistence_rejects_before_dim_is_set() {
    let path = temp_bundle("ordinal-lazy-write");
    cleanup(&path);
    let idx = OrdinalIndex::new_lazy(2).unwrap();
    let err = idx.write(&path).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    cleanup(&path);
}

#[test]
fn loaded_signbitmap_index_delete_stays_searchable_without_stale_slots() {
    let path = temp_bundle("ordinal-loaded-delete");
    cleanup(&path);

    let data = vectors(400, DIM);
    let query = vectors(1, DIM);
    let mut idx = OrdinalIndex::new(DIM, 2).unwrap();
    idx.add(&data);
    idx.write(&path).unwrap();

    let mut loaded = OrdinalIndex::load(&path).unwrap();
    assert_eq!(loaded.swap_remove(12), 399);
    assert_eq!(loaded.len(), 399);

    let results = loaded.search(&query, 399);
    assert_eq!(results.k, 399);
    assert!(results.indices.iter().all(|slot| (0..399).contains(slot)));

    cleanup(&path);
}

#[test]
fn load_rejects_manifest_verified_primary_artifact_corruption() {
    let path = temp_bundle("ordinal-corrupt-index");
    cleanup(&path);

    let mut idx = OrdinalIndex::new(DIM, 2).unwrap();
    idx.add(&vectors(4, DIM));
    idx.write(&path).unwrap();
    fs::File::create(path.join("index.ovrq"))
        .unwrap()
        .write_all(b"not a rankquant file")
        .unwrap();

    let err = OrdinalIndex::load(&path).err().unwrap();
    assert_eq!(err.kind(), ErrorKind::InvalidData);

    cleanup(&path);
}

#[test]
fn load_accepts_legacy_manifest_artifact_filenames() {
    let path = temp_bundle("ordinal-legacy-artifact-names");
    cleanup(&path);
    fs::create_dir_all(&path).unwrap();

    let data = vectors(16, DIM);
    let queries = vectors(2, DIM);
    let mut rankquant = RankQuant::new(DIM, 2);
    rankquant.add(&data);
    let mut sign = SignBitmap::new(DIM);
    sign.add(&data);

    let index_path = path.join("index.tvrq");
    let sign_path = path.join("sign.tvsb");
    let manifest_path = path.join("manifest.json");
    rankquant.write(&index_path).unwrap();
    sign.write(&sign_path).unwrap();

    let mut options = CreateManifestOptions::default();
    options.auxiliary_artifacts.push(CreateAuxiliaryArtifact {
        name: SIGN_AUX_NAME.to_string(),
        path: sign_path,
        required: true,
    });
    let manifest = create_manifest_for_index_with_options(
        &index_path,
        CreateRowIdentity::RowIdIdentity,
        EMBEDDING_MODEL,
        &manifest_path,
        options,
    )
    .unwrap();
    write_manifest_file(&manifest, &manifest_path).unwrap();

    let loaded = OrdinalIndex::load(&path).unwrap();
    assert_eq!(
        loaded.search(&queries, 5).indices,
        rankquant.search_asymmetric(&queries, 5).indices
    );

    cleanup(&path);
}

fn temp_bundle(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "ordinaldb-{name}-{}-{}",
        std::process::id(),
        NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
    ))
}

fn cleanup(path: &Path) {
    let _ = fs::remove_dir_all(path);
    let _ = fs::remove_file(path);
}
