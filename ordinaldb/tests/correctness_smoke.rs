use ordinaldb::{IdMapIndex, OrdinalIndex};
use ordvec::RankQuant;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

const DIM: usize = 64;
static NEXT_TEMP: AtomicUsize = AtomicUsize::new(0);

#[test]
fn loaded_positional_search_matches_direct_ordvec_rankquant_when_candidates_cover_all_rows() {
    let path = temp_bundle("correctness-positional-load");
    cleanup(&path);

    let data = vectors(96, DIM);
    let queries = vectors(4, DIM);

    for bits in [1, 2, 4] {
        let mut idx = OrdinalIndex::new(DIM, bits).unwrap();
        idx.add(&data);
        idx.write(&path).unwrap();
        let loaded = OrdinalIndex::load(&path).unwrap();

        let mut direct = RankQuant::new(DIM, bits);
        direct.add(&data);
        let expected = direct.search_asymmetric(&queries, 10);
        let got = loaded.search(&queries, 10);

        assert_eq!(got.nq, expected.nq);
        assert_eq!(got.k, expected.k);
        assert_eq!(got.indices, expected.indices);
        assert_eq!(got.scores, expected.scores);

        cleanup(&path);
    }
}

#[test]
fn loaded_idmap_search_matches_loaded_positional_translation() {
    let positional_path = temp_bundle("correctness-positional-translate");
    let idmap_path = temp_bundle("correctness-idmap-translate");
    cleanup(&positional_path);
    cleanup(&idmap_path);

    let data = vectors(48, DIM);
    let queries = vectors(3, DIM);
    let ids: Vec<u64> = (0..48).map(|idx| 900_000 + idx as u64).collect();

    let mut positional = OrdinalIndex::new(DIM, 2).unwrap();
    positional.add(&data);
    positional.write(&positional_path).unwrap();
    let positional = OrdinalIndex::load(&positional_path).unwrap();

    let mut mapped = IdMapIndex::new(DIM, 2).unwrap();
    mapped.add_with_ids(&data, &ids).unwrap();
    mapped.write(&idmap_path).unwrap();
    let mapped = IdMapIndex::load(&idmap_path).unwrap();

    let positional_results = positional.search(&queries, 9);
    let (mapped_scores, mapped_ids) = mapped.search(&queries, 9);
    let expected_ids: Vec<u64> = positional_results
        .indices
        .iter()
        .map(|slot| ids[*slot as usize])
        .collect();

    assert_eq!(mapped_scores, positional_results.scores);
    assert_eq!(mapped_ids, expected_ids);

    cleanup(&positional_path);
    cleanup(&idmap_path);
}

#[test]
fn loaded_allowlist_matches_direct_rankquant_subset_translation() {
    let path = temp_bundle("correctness-allowlist-load");
    cleanup(&path);

    let data = vectors(64, DIM);
    let queries = vectors(2, DIM);
    let ids: Vec<u64> = (0..64).map(|idx| 70_000 + idx as u64).collect();
    let allowlist = [70_001, 70_007, 70_013, 70_055];
    let candidates = [1u32, 7, 13, 55];

    let mut direct = RankQuant::new(DIM, 2);
    direct.add(&data);

    let mut mapped = IdMapIndex::new(DIM, 2).unwrap();
    mapped.add_with_ids(&data, &ids).unwrap();
    mapped.write(&path).unwrap();
    let mapped = IdMapIndex::load(&path).unwrap();

    let (scores, found_ids) = mapped.search_with_allowlist(&queries, 3, Some(&allowlist));
    let mut expected_scores = Vec::new();
    let mut expected_ids = Vec::new();
    for query in queries.chunks_exact(DIM) {
        let (row_scores, row_indices) = direct.search_asymmetric_subset(query, &candidates, 3);
        expected_scores.extend(row_scores);
        expected_ids.extend(row_indices.iter().map(|slot| ids[*slot as usize]));
    }

    assert_eq!(scores, expected_scores);
    assert_eq!(found_ids, expected_ids);

    cleanup(&path);
}

fn vectors(n: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0; n * dim];
    for row in 0..n {
        for col in 0..dim {
            let x = (((row + 11) * (col + 7) + row * 23 + col * 3) % 53) as f32 - 26.0;
            out[row * dim + col] = x / 27.0;
        }
    }
    out
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
