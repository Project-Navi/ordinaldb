//! Regression tests for crash debris under `vectors/`.
//!
//! A SIGKILL during generation replacement left a double-suffixed temp
//! directory (`..g000000000005.odb.tmp-<pid>-<nanos>.tmp-<pid>-<nanos>`)
//! behind. Debris under `vectors/` must never brick verification or GC of an
//! otherwise healthy store: it is reclaimable, never fatal — while symlinks
//! and canonically-named anomalies still fail closed.

mod common;

use std::fs;

use ordinaldb_adapter_store::{
    acquire_writer_lock, open_verified, record_generation_gc, remove_generation_debris,
    scan_generation_directory, write_legacy_snapshot, GenerationGcUpdate, StoreRevision,
};

/// The exact double-suffixed debris name left behind by a SIGKILL during
/// an interrupted generation replacement.
const DOUBLE_SUFFIXED_DEBRIS_NAME: &str =
    "..g000000000005.odb.tmp-211848-1783014389442708489.tmp-211848-1783014389442721724";
const SINGLE_SUFFIX_DEBRIS: &str = ".g000000000003.odb.tmp-217216-1783014480710827089";

fn store_with_debris(name: &str) -> std::path::PathBuf {
    let root = common::temp_root(name);
    common::write_index_at(&root, common::GENERATION_INDEX_PATH);
    write_legacy_snapshot(
        &root,
        None,
        common::payloads(false, common::GENERATION_INDEX_PATH),
    )
    .unwrap();
    let vectors = root.join("vectors");
    fs::create_dir(vectors.join(DOUBLE_SUFFIXED_DEBRIS_NAME)).unwrap();
    fs::create_dir(vectors.join(SINGLE_SUFFIX_DEBRIS)).unwrap();
    fs::write(vectors.join("stray-file.txt"), b"junk").unwrap();
    root
}

#[test]
fn open_verified_tolerates_crash_debris_under_vectors() {
    let root = store_with_debris("debris-open");
    let store = open_verified(&root, Some("langchain")).unwrap();
    assert_eq!(
        store.active_generation_path(),
        Some(common::GENERATION_INDEX_PATH)
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn scan_classifies_debris_as_reclaimable_and_never_the_active_generation() {
    let root = store_with_debris("debris-scan");
    let scan = scan_generation_directory(&root).unwrap();

    assert_eq!(
        scan.generation_paths,
        vec![common::GENERATION_INDEX_PATH.to_string()]
    );
    let debris_paths: Vec<&str> = scan
        .debris
        .iter()
        .map(|entry| entry.path.as_str())
        .collect();
    let expected_debris = [
        format!("vectors/{DOUBLE_SUFFIXED_DEBRIS_NAME}"),
        format!("vectors/{SINGLE_SUFFIX_DEBRIS}"),
        "vectors/stray-file.txt".to_string(),
    ];
    assert_eq!(
        debris_paths,
        expected_debris
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
    );

    // The interrupted-replacement temp dirs are attributed to their
    // generation ids; the stray file is unrecognized debris.
    let by_path = |path: &str| {
        scan.debris
            .iter()
            .find(|entry| entry.path.ends_with(path))
            .unwrap()
    };
    assert_eq!(by_path(DOUBLE_SUFFIXED_DEBRIS_NAME).generation_id, Some(5));
    assert_eq!(by_path(SINGLE_SUFFIX_DEBRIS).generation_id, Some(3));
    assert_eq!(by_path("stray-file.txt").generation_id, None);
    for entry in &scan.debris {
        assert!(
            entry.warning.contains("reclaimable"),
            "warning must be structured and actionable: {}",
            entry.warning
        );
        assert_ne!(entry.path, common::GENERATION_INDEX_PATH);
    }

    let _ = fs::remove_dir_all(root);
}

#[test]
fn scan_is_empty_for_store_without_vectors_directory() {
    let root = common::temp_root("debris-scan-empty");
    let scan = scan_generation_directory(&root).unwrap();
    assert!(scan.generation_paths.is_empty());
    assert!(scan.debris.is_empty());
    let _ = fs::remove_dir_all(root);
}

#[cfg(unix)]
#[test]
fn scan_fails_closed_on_symlink_entry() {
    let root = store_with_debris("debris-scan-symlink");
    std::os::unix::fs::symlink(
        root.join(common::GENERATION_INDEX_PATH),
        root.join("vectors/g000000000009.odb"),
    )
    .unwrap();
    let err = scan_generation_directory(&root).unwrap_err();
    assert!(err.to_string().contains("symlink"), "{err}");
    let _ = fs::remove_dir_all(root);
}

#[test]
fn scan_fails_closed_on_file_with_canonical_generation_name() {
    let root = store_with_debris("debris-scan-canonical-file");
    fs::write(root.join("vectors/g000000000008.odb"), b"not a bundle").unwrap();
    let err = scan_generation_directory(&root).unwrap_err();
    assert!(err.to_string().contains("directory"), "{err}");
    let _ = fs::remove_dir_all(root);
}

#[test]
fn record_generation_gc_accepts_debris_paths_through_the_lifecycle() {
    let root = store_with_debris("debris-gc-record");
    let store = open_verified(&root, None).unwrap();
    let mut expected = StoreRevision::from_manifest(&store.manifest).unwrap();
    let debris_path = format!("vectors/{DOUBLE_SUFFIXED_DEBRIS_NAME}");

    for state in ["reclaimable", "deleting", "deleted"] {
        let updated = record_generation_gc(
            &root,
            Some(expected.clone()),
            &[GenerationGcUpdate {
                generation_id: Some(5),
                path: debris_path.clone(),
                state: state.to_string(),
                reason: "crash debris reclaimed by gc".to_string(),
            }],
        )
        .unwrap();
        expected = StoreRevision::from_manifest(&updated.manifest).unwrap();
    }

    let _ = fs::remove_dir_all(root);
}

#[test]
fn remove_generation_debris_deletes_debris_and_refuses_generations() {
    let root = store_with_debris("debris-remove");
    let lock = acquire_writer_lock(&root).unwrap();

    remove_generation_debris(&root, &format!("vectors/{DOUBLE_SUFFIXED_DEBRIS_NAME}")).unwrap();
    remove_generation_debris(&root, &format!("vectors/{SINGLE_SUFFIX_DEBRIS}")).unwrap();
    remove_generation_debris(&root, "vectors/stray-file.txt").unwrap();
    assert!(!root
        .join("vectors")
        .join(DOUBLE_SUFFIXED_DEBRIS_NAME)
        .exists());
    assert!(!root.join("vectors").join(SINGLE_SUFFIX_DEBRIS).exists());
    assert!(!root.join("vectors/stray-file.txt").exists());
    // Idempotent: deleting already-deleted debris is fine.
    remove_generation_debris(&root, &format!("vectors/{DOUBLE_SUFFIXED_DEBRIS_NAME}")).unwrap();

    // Canonically-named generation bundles (the active generation and
    // anything adapter.redb can reference) are never deletable through
    // this API.
    let err = remove_generation_debris(&root, common::GENERATION_INDEX_PATH).unwrap_err();
    assert!(err.to_string().contains("generation"), "{err}");
    assert!(root.join(common::GENERATION_INDEX_PATH).is_dir());

    // Paths outside vectors/ are refused.
    let err = remove_generation_debris(&root, "adapter.redb").unwrap_err();
    assert!(err.to_string().contains("vectors"), "{err}");
    assert!(root.join("adapter.redb").is_file());

    drop(lock);
    // The store is still fully healthy after debris removal.
    open_verified(&root, Some("langchain")).unwrap();
    let scan = scan_generation_directory(&root).unwrap();
    assert!(scan.debris.is_empty());

    let _ = fs::remove_dir_all(root);
}

#[test]
fn remove_generation_debris_requires_the_writer_lock() {
    let root = store_with_debris("debris-remove-lock");
    let err = remove_generation_debris(&root, &format!("vectors/{DOUBLE_SUFFIXED_DEBRIS_NAME}"))
        .unwrap_err();
    assert!(err.to_string().contains("lock"), "{err}");
    assert!(root
        .join("vectors")
        .join(DOUBLE_SUFFIXED_DEBRIS_NAME)
        .is_dir());
    let _ = fs::remove_dir_all(root);
}

#[cfg(unix)]
#[test]
fn remove_generation_debris_refuses_symlinked_entries() {
    let root = store_with_debris("debris-remove-symlink");
    let outside = common::temp_root("debris-remove-symlink-target");
    std::os::unix::fs::symlink(&outside, root.join("vectors/.g000000000007.odb.tmp-1-2")).unwrap();
    let lock = acquire_writer_lock(&root).unwrap();
    let err = remove_generation_debris(&root, "vectors/.g000000000007.odb.tmp-1-2").unwrap_err();
    assert!(err.to_string().contains("symlink"), "{err}");
    assert!(outside.exists());
    drop(lock);
    let _ = fs::remove_dir_all(root);
    let _ = fs::remove_dir_all(outside);
}

#[test]
fn empty_lazy_store_loads_despite_crash_debris() {
    let root = common::temp_root("debris-empty-lazy");
    write_legacy_snapshot(&root, None, common::payloads(true, "index.odb")).unwrap();
    // A crash during the first vector-bearing save leaves temp debris under
    // vectors/ while adapter.redb still says empty_lazy. The store must stay
    // loadable.
    let vectors = root.join("vectors");
    fs::create_dir_all(&vectors).unwrap();
    fs::create_dir(vectors.join(SINGLE_SUFFIX_DEBRIS)).unwrap();
    open_verified(&root, Some("langchain")).unwrap();

    // A committed (canonical) generation is still a hard failure for an
    // empty_lazy store.
    fs::create_dir(vectors.join("g000000000002.odb")).unwrap();
    let err = open_verified(&root, Some("langchain")).unwrap_err();
    assert!(err.to_string().contains("empty_lazy"), "{err}");

    let _ = fs::remove_dir_all(root);
}
