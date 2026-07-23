use ordinaldb::{AddError, IdMapIndex, OrdinalIndex};
use ordvec_manifest::{load_manifest_file, sha256_file, write_manifest_file};
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
            let x = (((row + 7) * (col + 3) + row * 13 + col * 19) % 43) as f32 - 21.0;
            out[row * dim + col] = x / 23.0;
        }
    }
    out
}

#[test]
fn add_with_ids_maps_returned_ids() {
    let data = vectors(8, DIM);
    let ids: Vec<u64> = (100..108).collect();
    let mut idx = IdMapIndex::new(DIM, 2).unwrap();
    idx.add_with_ids(&data, &ids).unwrap();

    for (row, expected_id) in ids.iter().enumerate() {
        let query = &data[row * DIM..(row + 1) * DIM];
        let (_, got) = idx.search(query, 1);
        assert_eq!(got, vec![*expected_id]);
    }
}

#[test]
fn duplicate_ids_in_batch_rejected_without_partial_mutation() {
    let data = vectors(3, DIM);
    let mut idx = IdMapIndex::new(DIM, 2).unwrap();
    let err = idx.add_with_ids(&data, &[1, 2, 2]).unwrap_err();
    assert_eq!(err, AddError::IdAlreadyPresent(2));
    assert_eq!(idx.len(), 0);
    assert!(!idx.contains(1));
}

#[test]
fn existing_id_rejected_without_partial_mutation() {
    let data = vectors(4, DIM);
    let mut idx = IdMapIndex::new(DIM, 2).unwrap();
    idx.add_with_ids(&data[..2 * DIM], &[10, 11]).unwrap();
    let err = idx
        .add_with_ids(&data[2 * DIM..4 * DIM], &[12, 10])
        .unwrap_err();
    assert_eq!(err, AddError::IdAlreadyPresent(10));
    assert_eq!(idx.len(), 2);
    assert!(idx.contains(10));
    assert!(!idx.contains(12));
}

#[test]
fn remove_missing_returns_false() {
    let mut idx = IdMapIndex::new(DIM, 2).unwrap();
    idx.add_with_ids(&vectors(2, DIM), &[1, 2]).unwrap();
    assert!(!idx.remove(99));
    assert_eq!(idx.len(), 2);
}

#[test]
fn remove_present_updates_mapping_and_hides_deleted_id() {
    let data = vectors(10, DIM);
    let ids: Vec<u64> = (0..10).map(|i| 1000 + i).collect();
    let mut idx = IdMapIndex::new(DIM, 2).unwrap();
    idx.add_with_ids(&data, &ids).unwrap();

    assert!(idx.remove(1002));
    assert_eq!(idx.len(), 9);
    assert!(!idx.contains(1002));

    let (_, got) = idx.search(&data[2 * DIM..3 * DIM], 9);
    assert!(!got.contains(&1002));
    assert!(got.iter().all(|id| idx.contains(*id)));
}

#[test]
fn allowlist_returns_only_allowed_ids() {
    let data = vectors(12, DIM);
    let ids: Vec<u64> = (0..12).map(|i| 500 + i).collect();
    let allow = [501, 504, 509, 509];
    let mut idx = IdMapIndex::new(DIM, 2).unwrap();
    idx.add_with_ids(&data, &ids).unwrap();

    let (_, got) = idx.search_with_allowlist(&vectors(2, DIM), 10, Some(&allow));
    assert_eq!(got.len(), 2 * 3);
    assert!(got.iter().all(|id| [501, 504, 509].contains(id)));
}

#[test]
fn empty_allowlist_returns_zero_results() {
    let mut idx = IdMapIndex::new(DIM, 2).unwrap();
    idx.add_with_ids(&vectors(3, DIM), &[1, 2, 3]).unwrap();
    let (scores, ids) = idx.search_with_allowlist(&vectors(2, DIM), 5, Some(&[]));
    assert!(scores.is_empty());
    assert!(ids.is_empty());
}

#[test]
#[should_panic(expected = "allowlist is not present")]
fn allowlist_missing_id_panics_clearly() {
    let mut idx = IdMapIndex::new(DIM, 2).unwrap();
    idx.add_with_ids(&vectors(3, DIM), &[1, 2, 3]).unwrap();
    let _ = idx.search_with_allowlist(&vectors(1, DIM), 1, Some(&[9]));
}

#[test]
fn allowlist_after_delete_behaves_correctly() {
    let data = vectors(8, DIM);
    let ids: Vec<u64> = (20..28).collect();
    let mut idx = IdMapIndex::new(DIM, 2).unwrap();
    idx.add_with_ids(&data, &ids).unwrap();
    assert!(idx.remove(23));

    let (_, got) = idx.search_with_allowlist(&vectors(1, DIM), 5, Some(&[20, 24, 27]));
    assert!(got.iter().all(|id| [20, 24, 27].contains(id)));
    assert!(!got.contains(&23));
}

#[test]
fn idmap_search_matches_positional_translation() {
    let data = vectors(9, DIM);
    let query = vectors(1, DIM);
    let ids: Vec<u64> = (0..9).map(|i| 10_000 + i as u64).collect();

    let mut positional = OrdinalIndex::new(DIM, 2).unwrap();
    positional.add(&data);

    let mut mapped = IdMapIndex::new(DIM, 2).unwrap();
    mapped.add_with_ids(&data, &ids).unwrap();

    let positional_results = positional.search(&query, 6);
    let (mapped_scores, mapped_ids) = mapped.search(&query, 6);
    let expected_ids: Vec<u64> = positional_results
        .indices
        .iter()
        .map(|slot| ids[*slot as usize])
        .collect();

    assert_eq!(mapped_scores, positional_results.scores);
    assert_eq!(mapped_ids, expected_ids);
}

#[test]
fn write_load_roundtrip_preserves_id_search() {
    let path = temp_bundle("idmap-roundtrip");
    cleanup(&path);

    let data = vectors(32, DIM);
    let query = vectors(2, DIM);
    let ids: Vec<u64> = (0..32).map(|i| 10_000 + i as u64).collect();
    let mut idx = IdMapIndex::new(DIM, 2).unwrap();
    idx.add_with_ids(&data, &ids).unwrap();
    let before = idx.search(&query, 8);

    idx.write(&path).unwrap();
    assert!(path.join("manifest.json").exists());
    assert!(path.join("index.ovrq").exists());
    assert!(path.join("sign.ovsb").exists());
    assert!(path.join("ids.bin").exists());

    let loaded = IdMapIndex::load(&path).unwrap();
    let after = loaded.search(&query, 8);
    assert_eq!(loaded.dim(), DIM);
    assert_eq!(loaded.bits(), 2);
    assert_eq!(loaded.len(), idx.len());
    assert_eq!(after, before);

    cleanup(&path);
}

#[test]
fn loaded_idmap_delete_updates_mapping_and_never_returns_deleted_id() {
    let path = temp_bundle("idmap-loaded-delete");
    cleanup(&path);

    let data = vectors(40, DIM);
    let ids: Vec<u64> = (0..40).map(|i| 70_000 + i as u64).collect();
    let mut idx = IdMapIndex::new(DIM, 2).unwrap();
    idx.add_with_ids(&data, &ids).unwrap();
    idx.write(&path).unwrap();

    let mut loaded = IdMapIndex::load(&path).unwrap();
    assert!(loaded.remove(70_012));
    assert!(!loaded.contains(70_012));
    assert_eq!(loaded.len(), 39);

    let (_, found) = loaded.search(&vectors(2, DIM), 39);
    assert!(!found.contains(&70_012));
    assert!(found.iter().all(|id| loaded.contains(*id)));

    cleanup(&path);
}

#[test]
fn positional_loader_rejects_idmap_bundle() {
    let path = temp_bundle("idmap-wrong-loader");
    cleanup(&path);

    let mut idx = IdMapIndex::new(DIM, 2).unwrap();
    idx.add_with_ids(&vectors(3, DIM), &[1, 2, 3]).unwrap();
    idx.write(&path).unwrap();

    let err = OrdinalIndex::load(&path).err().unwrap();
    assert_eq!(err.kind(), ErrorKind::InvalidData);
    assert!(err.to_string().contains("IdMapIndex::load"));

    cleanup(&path);
}

#[test]
fn idmap_loader_rejects_positional_bundle() {
    let path = temp_bundle("idmap-missing-ids");
    cleanup(&path);

    let mut idx = OrdinalIndex::new(DIM, 2).unwrap();
    idx.add(&vectors(3, DIM));
    idx.write(&path).unwrap();

    let err = IdMapIndex::load(&path).err().unwrap();
    assert_eq!(err.kind(), ErrorKind::InvalidData);
    assert!(err.to_string().contains("ID sidecar"));

    cleanup(&path);
}

#[test]
fn persisted_duplicate_ids_are_rejected_after_manifest_verifies() {
    let path = temp_bundle("idmap-duplicate-ids");
    cleanup(&path);

    let mut idx = IdMapIndex::new(DIM, 2).unwrap();
    idx.add_with_ids(&vectors(3, DIM), &[11, 12, 13]).unwrap();
    idx.write(&path).unwrap();

    rewrite_ids_and_update_manifest(&path, &[11, 12, 12]);

    let err = IdMapIndex::load(&path).err().unwrap();
    assert_eq!(err.kind(), ErrorKind::InvalidData);
    assert!(err.to_string().contains("duplicate ID 12"));

    cleanup(&path);
}

#[test]
fn persisted_ids_count_mismatch_is_rejected_after_manifest_verifies() {
    let path = temp_bundle("idmap-count-mismatch");
    cleanup(&path);

    let mut idx = IdMapIndex::new(DIM, 2).unwrap();
    idx.add_with_ids(&vectors(3, DIM), &[11, 12, 13]).unwrap();
    idx.write(&path).unwrap();

    let ids_path = path.join("ids.bin");
    let mut file = fs::File::create(&ids_path).unwrap();
    file.write_all(b"ODBIDS1\0").unwrap();
    file.write_all(&2u64.to_le_bytes()).unwrap();
    for id in [11u64, 12, 13] {
        file.write_all(&id.to_le_bytes()).unwrap();
    }
    file.flush().unwrap();
    update_ids_manifest_hash(&path);

    let err = IdMapIndex::load(&path).err().unwrap();
    assert_eq!(err.kind(), ErrorKind::InvalidData);
    assert!(err.to_string().contains("does not match index len"));

    cleanup(&path);
}

#[test]
fn persisted_ids_trailing_bytes_are_rejected_after_manifest_verifies() {
    let path = temp_bundle("idmap-trailing-ids");
    cleanup(&path);

    let mut idx = IdMapIndex::new(DIM, 2).unwrap();
    idx.add_with_ids(&vectors(3, DIM), &[11, 12, 13]).unwrap();
    idx.write(&path).unwrap();

    let ids_path = path.join("ids.bin");
    let mut file = fs::OpenOptions::new().append(true).open(&ids_path).unwrap();
    file.write_all(b"x").unwrap();
    update_ids_manifest_hash(&path);

    let err = IdMapIndex::load(&path).err().unwrap();
    assert_eq!(err.kind(), ErrorKind::InvalidData);
    assert!(err.to_string().contains("trailing bytes"));

    cleanup(&path);
}

#[test]
fn open_from_verified_plan_round_trips_id_search() {
    use ordinaldb::manifest::{verify_for_load, VerifyOptions};
    use ordinaldb::DenseLoadOptions;

    let path = temp_bundle("idmap-open-from-plan");
    cleanup(&path);

    let data = vectors(24, DIM);
    let query = vectors(2, DIM);
    let ids: Vec<u64> = (0..24).map(|i| 500 + i as u64).collect();
    let mut idx = IdMapIndex::new(DIM, 2).unwrap();
    idx.add_with_ids(&data, &ids).unwrap();
    let before = idx.search(&query, 8);
    idx.write(&path).unwrap();

    // Verify the manifest once, then reuse the plan to open the dense side.
    let plan = verify_for_load(path.join("manifest.json"), VerifyOptions::default()).unwrap();
    let opened = IdMapIndex::open_from_verified_plan(&plan, DenseLoadOptions::default()).unwrap();
    let after = opened.search(&query, 8);

    assert_eq!(opened.dim(), DIM);
    assert_eq!(opened.bits(), 2);
    assert_eq!(opened.len(), idx.len());
    assert_eq!(after, before);

    cleanup(&path);
}

fn rewrite_ids_and_update_manifest(path: &Path, ids: &[u64]) {
    let ids_path = path.join("ids.bin");
    let mut file = fs::File::create(&ids_path).unwrap();
    file.write_all(b"ODBIDS1\0").unwrap();
    file.write_all(&(ids.len() as u64).to_le_bytes()).unwrap();
    for id in ids {
        file.write_all(&id.to_le_bytes()).unwrap();
    }
    file.flush().unwrap();
    update_ids_manifest_hash(path);
}

fn update_ids_manifest_hash(path: &Path) {
    let manifest_path = path.join("manifest.json");
    let ids_hash = sha256_file(path.join("ids.bin")).unwrap();
    let mut document = load_manifest_file(&manifest_path).unwrap();
    let ids_artifact = document
        .manifest
        .auxiliary_artifacts
        .iter_mut()
        .find(|artifact| artifact.name == "ordinaldb.ids")
        .unwrap();
    ids_artifact.sha256 = ids_hash.sha256;
    ids_artifact.file_size_bytes = ids_hash.size_bytes;
    write_manifest_file(&document.manifest, manifest_path).unwrap();
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

#[test]
fn add_with_ids_2d_reports_dim_invalid_for_out_of_range_dims() {
    // dim validity must surface as DimInvalid, matching
    // OrdinalIndex::add_2d's contract — not as a buffer-shape error.
    let mut idx = IdMapIndex::new(64, 2).unwrap();
    for bad_dim in [0usize, 1, (u16::MAX as usize) + 1] {
        let err = idx
            .add_with_ids_2d(&[0.0; 8], bad_dim, &[1])
            .expect_err("out-of-range dim must be rejected");
        assert!(
            matches!(err, AddError::DimInvalid(d) if d == bad_dim),
            "dim {bad_dim}: expected DimInvalid, got {err:?}"
        );
    }
    // Buffer-shape errors remain their own variant for valid dims.
    let err = idx
        .add_with_ids_2d(&[0.0; 7], 64, &[1])
        .expect_err("misaligned buffer must be rejected");
    assert!(matches!(err, AddError::VectorBufferNotMultipleOfDim { .. }));
}
