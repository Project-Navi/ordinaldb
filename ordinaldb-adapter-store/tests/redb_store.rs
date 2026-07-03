use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use ordinaldb_adapter_store::{
    acquire_writer_lock, commit, open_verified, record_generation_gc, write_legacy_snapshot,
    AdapterMutation, GenerationGcUpdate, LegacyPayloads, MetadataPatch, StoreRevision,
    ADAPTER_STORE_FILE, ADAPTER_STORE_SCHEMA_VERSION,
};
use ordinaldb_core::OrdinalIndex;
use redb::{Database, ReadableDatabase, ReadableTable, ReadableTableMetadata, TableDefinition};
use serde_json::{json, Map};

const VECTORS: &[f32] = &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
const LEGACY_INDEX_PATH: &str = "index.odb";
const GENERATION_INDEX_PATH: &str = "vectors/g000000000001.odb";
const SECOND_GENERATION_INDEX_PATH: &str = "vectors/g000000000002.odb";
const WRITE_LOCK_FILE: &str = ".ordinaldb.write.lock";
const META: TableDefinition<&str, &str> = TableDefinition::new("meta");
const GENERATIONS: TableDefinition<u64, &str> = TableDefinition::new("generations");
const GC_QUEUE: TableDefinition<u64, &str> = TableDefinition::new("gc_queue");
const AUDIT_LOG: TableDefinition<u64, &str> = TableDefinition::new("audit_log");
const U64_TO_STRING: TableDefinition<u64, &str> = TableDefinition::new("u64_to_string");

#[test]
fn redb_store_round_trips_legacy_payloads_and_verifies_generation() {
    let root = temp_root("round-trip");
    write_index_at(&root, LEGACY_INDEX_PATH);
    let payloads = payloads(false, LEGACY_INDEX_PATH);

    let written = write_legacy_snapshot(&root, None, payloads.clone()).unwrap();
    assert_eq!(
        written.manifest["schema_version"],
        json!(ADAPTER_STORE_SCHEMA_VERSION)
    );
    let store_uuid = written.manifest["store_uuid"].as_str().unwrap();
    assert_eq!(store_uuid.len(), 36);
    assert_eq!(&store_uuid[14..15], "4");
    assert!(matches!(&store_uuid[19..20], "8" | "9" | "a" | "b"));
    assert_eq!(written.manifest["origin"], json!("created"));
    assert_eq!(
        written.manifest["migrated_from_json_sidecars"],
        json!(false)
    );
    assert_eq!(written.manifest["vector_count"], json!(2));
    assert!(root.join(ADAPTER_STORE_FILE).is_file());
    assert!(root.join(WRITE_LOCK_FILE).is_file());

    let loaded = open_verified(&root, Some("langchain")).unwrap();
    assert_eq!(loaded.payloads.adapter_json, payloads.adapter_json);
    assert_eq!(loaded.payloads.id_map_json, payloads.id_map_json);
    assert_eq!(loaded.payloads.documents_json, payloads.documents_json);
    assert_eq!(loaded.payloads.metadata_json, payloads.metadata_json);
    assert_eq!(loaded.adapter_name(), Some("langchain"));
    assert_eq!(loaded.bits(), Some(2));
    assert_eq!(loaded.dim(), Some(4));
    assert_eq!(loaded.vector_count(), Some(2));
    assert!(!loaded.empty_lazy());

    let _ = fs::remove_dir_all(root);
}

#[test]
fn redb_store_rejects_invalid_store_identity_manifest_fields() {
    let bad_uuid_root = temp_root("invalid-store-uuid");
    write_index_at(&bad_uuid_root, GENERATION_INDEX_PATH);
    write_legacy_snapshot(&bad_uuid_root, None, payloads(false, GENERATION_INDEX_PATH)).unwrap();

    mutate_manifest(&bad_uuid_root, |manifest| {
        manifest["store_uuid"] = json!("ordinaldb-not-a-uuid");
    });
    let err = open_verified(&bad_uuid_root, Some("langchain")).unwrap_err();
    assert!(
        err.to_string()
            .contains("store_uuid must be a random UUIDv4 string"),
        "{err}"
    );
    let _ = fs::remove_dir_all(bad_uuid_root);

    let bad_origin_root = temp_root("invalid-store-origin");
    write_index_at(&bad_origin_root, GENERATION_INDEX_PATH);
    write_legacy_snapshot(
        &bad_origin_root,
        None,
        payloads(false, GENERATION_INDEX_PATH),
    )
    .unwrap();
    mutate_manifest(&bad_origin_root, |manifest| {
        manifest["origin"] = json!("unknown");
    });
    let err = open_verified(&bad_origin_root, Some("langchain")).unwrap_err();
    assert!(
        err.to_string()
            .contains("origin must be one of created, imported_legacy_json, upgraded_schema"),
        "{err}"
    );
    let _ = fs::remove_dir_all(bad_origin_root);
}

#[test]
fn redb_store_enforces_expected_revision_on_existing_store() {
    let root = temp_root("cas-existing-store");
    write_index_at(&root, LEGACY_INDEX_PATH);
    let first = write_legacy_snapshot(&root, None, payloads(false, LEGACY_INDEX_PATH)).unwrap();

    let err = write_legacy_snapshot(&root, None, payloads(false, LEGACY_INDEX_PATH)).unwrap_err();
    assert!(
        err.to_string().contains("expected adapter store revision"),
        "{err}"
    );

    let expected = StoreRevision::from_manifest(&first.manifest).unwrap();
    let second = write_legacy_snapshot(
        &root,
        Some(expected.clone()),
        payloads(false, LEGACY_INDEX_PATH),
    )
    .unwrap();
    assert_eq!(second.manifest["commit_sequence"], json!(2));

    let err = write_legacy_snapshot(&root, Some(expected), payloads(false, LEGACY_INDEX_PATH))
        .unwrap_err();
    assert!(err.to_string().contains("stale adapter snapshot"), "{err}");

    let _ = fs::remove_dir_all(root);
}

#[test]
fn redb_store_commit_returns_revision_and_rejects_stale_expected_revision() {
    let root = temp_root("typed-commit-cas");
    write_index_at(&root, GENERATION_INDEX_PATH);
    let first = write_legacy_snapshot(&root, None, payloads(false, GENERATION_INDEX_PATH)).unwrap();
    let expected = StoreRevision::from_manifest(&first.manifest).unwrap();

    write_single_index_at(&root, SECOND_GENERATION_INDEX_PATH);
    let committed = commit(
        &root,
        expected.clone(),
        AdapterMutation::ReplaceLegacySnapshot(one_record_payloads(SECOND_GENERATION_INDEX_PATH)),
    )
    .unwrap();
    assert_eq!(committed.revision.commit_sequence, 2);
    assert_eq!(committed.revision.active_generation_id, 2);
    assert_eq!(
        committed.revision.active_generation_path,
        SECOND_GENERATION_INDEX_PATH
    );
    assert_eq!(committed.manifest["commit_sequence"], json!(2));

    let err = commit(
        &root,
        expected,
        AdapterMutation::ReplaceLegacySnapshot(one_record_payloads(SECOND_GENERATION_INDEX_PATH)),
    )
    .unwrap_err();
    assert!(err.to_string().contains("stale adapter snapshot"), "{err}");

    let _ = fs::remove_dir_all(root);
}

#[test]
fn redb_store_commit_patches_metadata_without_new_generation() {
    let root = temp_root("typed-metadata-patch");
    write_index_at(&root, GENERATION_INDEX_PATH);
    let first = write_legacy_snapshot(&root, None, payloads(false, GENERATION_INDEX_PATH)).unwrap();
    let expected = StoreRevision::from_manifest(&first.manifest).unwrap();

    let mut metadata = Map::new();
    metadata.insert("group".to_string(), json!("patched"));
    metadata.insert("rank".to_string(), json!(7));
    let committed = commit(
        &root,
        expected,
        AdapterMutation::PatchMetadata(vec![MetadataPatch {
            id: "a".to_string(),
            metadata,
        }]),
    )
    .unwrap();

    assert_eq!(committed.revision.commit_sequence, 2);
    assert_eq!(committed.revision.active_generation_id, 1);
    assert_eq!(
        committed.revision.active_generation_path,
        GENERATION_INDEX_PATH
    );
    assert_eq!(committed.manifest["table_counts"]["generations"], json!(1));
    assert_eq!(committed.manifest["table_counts"]["metadata"], json!(2));
    assert_eq!(committed.manifest["table_counts"]["audit_log"], json!(2));

    let loaded = open_verified(&root, Some("langchain")).unwrap();
    let metadata_payload: serde_json::Value =
        serde_json::from_str(&loaded.payloads.metadata_json).unwrap();
    assert_eq!(
        metadata_payload["metadata"]["a"],
        json!({"group": "patched", "rank": 7})
    );
    assert_eq!(metadata_payload["metadata"]["b"], json!({"group": "y"}));

    let err = commit(
        &root,
        StoreRevision::from_manifest(&loaded.manifest).unwrap(),
        AdapterMutation::PatchMetadata(vec![MetadataPatch {
            id: "missing".to_string(),
            metadata: Map::new(),
        }]),
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("metadata patch ID is not active"),
        "{err}"
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn redb_store_rejects_second_writer_without_store_mutation() {
    let root = temp_root("writer-lock-held");
    write_index_at(&root, LEGACY_INDEX_PATH);
    let _lock = acquire_writer_lock(&root).unwrap();

    let err = write_legacy_snapshot(&root, None, payloads(false, LEGACY_INDEX_PATH)).unwrap_err();
    assert!(err.to_string().contains("writer lock"), "{err}");
    assert!(!root.join(ADAPTER_STORE_FILE).exists());
    drop(_lock);
    assert_eq!(
        fs::read_to_string(root.join(WRITE_LOCK_FILE)).unwrap(),
        format!("pid={}\nlock=advisory-v1\n", std::process::id())
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn writer_lock_same_process_reacquire_fails_while_held() {
    let root = temp_root("writer-lock-reacquire");

    let guard = acquire_writer_lock(&root).unwrap();
    let err = match acquire_writer_lock(&root) {
        Ok(_) => panic!("second same-process acquisition must fail"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("already held"), "{err}");

    // A failed re-acquisition must not have unregistered or unlocked the
    // holder: releasing the original guard re-enables acquisition.
    drop(guard);
    let _guard = acquire_writer_lock(&root).unwrap();

    let _ = fs::remove_dir_all(root);
}

#[test]
fn writer_lock_concurrent_same_process_acquires_yield_exactly_one_guard() {
    // The in-process registry is the authoritative same-process guard (OS
    // advisory locks are per-process on some platforms/filesystems, e.g.
    // flock emulated via fcntl on NFS). Its reserve step must be atomic:
    // racing acquirers may never both succeed.
    const THREADS: usize = 8;
    const ROUNDS: usize = 50;

    let root = temp_root("writer-lock-race");
    let start = Arc::new(Barrier::new(THREADS));
    let attempted = Arc::new(Barrier::new(THREADS));

    for _ in 0..ROUNDS {
        let successes: usize = thread::scope(|scope| {
            let handles: Vec<_> = (0..THREADS)
                .map(|_| {
                    let root = &root;
                    let start = Arc::clone(&start);
                    let attempted = Arc::clone(&attempted);
                    scope.spawn(move || {
                        start.wait();
                        let result = acquire_writer_lock(root);
                        // Hold every outcome until all threads have
                        // attempted, so a winner releasing early cannot
                        // legitimize a second success within the round.
                        attempted.wait();
                        match result {
                            Ok(_guard) => 1usize,
                            Err(err) => {
                                assert!(
                                    err.to_string().contains("already held"),
                                    "loser must fail as already held: {err}"
                                );
                                0
                            }
                        }
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).sum()
        });
        assert_eq!(
            successes, 1,
            "exactly one concurrent acquirer may obtain the writer lock"
        );
    }

    let _ = fs::remove_dir_all(root);
}

#[test]
fn redb_store_ignores_stale_malformed_lock_file() {
    let root = temp_root("stale-malformed-lock");
    write_index_at(&root, LEGACY_INDEX_PATH);
    fs::write(root.join(WRITE_LOCK_FILE), vec![b'x'; 1024 * 1024]).unwrap();

    let written = write_legacy_snapshot(&root, None, payloads(false, LEGACY_INDEX_PATH)).unwrap();
    assert_eq!(written.manifest["commit_sequence"], json!(1));
    let lock_contents = fs::read_to_string(root.join(WRITE_LOCK_FILE)).unwrap();
    assert!(lock_contents.contains("lock=advisory-v1"));
    assert!(lock_contents.len() < 128);

    let _ = fs::remove_dir_all(root);
}

#[cfg(unix)]
#[test]
fn redb_store_rejects_symlinked_root_before_mutation() {
    let real = temp_root("symlink-real");
    let link = temp_root("symlink-link");
    let _ = fs::remove_dir_all(&link);
    std::os::unix::fs::symlink(&real, &link).unwrap();

    let err =
        write_legacy_snapshot(&link, None, payloads(true, GENERATION_INDEX_PATH)).unwrap_err();
    assert!(err.to_string().contains("must not be a symlink"), "{err}");
    assert!(!real.join(ADAPTER_STORE_FILE).exists());

    let _ = fs::remove_file(link);
    let _ = fs::remove_dir_all(real);
}

#[cfg(unix)]
#[test]
fn redb_store_rejects_symlinked_root_ancestor_before_mutation() {
    let real_parent = temp_root("symlink-ancestor-real");
    let link_parent = temp_root("symlink-ancestor-link");
    let _ = fs::remove_dir_all(&link_parent);
    std::os::unix::fs::symlink(&real_parent, &link_parent).unwrap();
    let root = link_parent.join("adapter-root");

    let err =
        write_legacy_snapshot(&root, None, payloads(true, GENERATION_INDEX_PATH)).unwrap_err();
    assert!(err.to_string().contains("must not be a symlink"), "{err}");
    assert!(!real_parent.join("adapter-root").exists());
    assert!(!real_parent.join(ADAPTER_STORE_FILE).exists());

    let _ = fs::remove_file(link_parent);
    let _ = fs::remove_dir_all(real_parent);
}

#[test]
fn redb_store_accepts_immutable_generation_index_path() {
    let root = temp_root("generation-path");
    write_index_at(&root, GENERATION_INDEX_PATH);
    let payloads = payloads(false, GENERATION_INDEX_PATH);

    let written = write_legacy_snapshot(&root, None, payloads.clone()).unwrap();
    assert_eq!(written.manifest["active_generation_id"], json!(1));
    assert_eq!(
        written.manifest["active_generation_path"],
        json!(GENERATION_INDEX_PATH)
    );
    assert!(root.join(ADAPTER_STORE_FILE).is_file());

    let loaded = open_verified(&root, Some("langchain")).unwrap();
    assert_eq!(loaded.payloads.adapter_json, payloads.adapter_json);
    assert_eq!(loaded.vector_count(), Some(2));
    assert!(!loaded.empty_lazy());

    let _ = fs::remove_dir_all(root);
}

#[test]
fn redb_store_derives_generation_id_from_generation_path() {
    let root = temp_root("generation-id");
    write_index_at(&root, SECOND_GENERATION_INDEX_PATH);
    let payloads = payloads(false, SECOND_GENERATION_INDEX_PATH);

    let written = write_legacy_snapshot(&root, None, payloads).unwrap();
    assert_eq!(written.manifest["active_generation_id"], json!(2));
    assert_eq!(
        written.manifest["active_generation_path"],
        json!(SECOND_GENERATION_INDEX_PATH)
    );

    let loaded = open_verified(&root, Some("langchain")).unwrap();
    assert_eq!(loaded.active_generation_id(), Some(2));
    assert_eq!(
        loaded.active_generation_path(),
        Some(SECOND_GENERATION_INDEX_PATH)
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn redb_store_updates_stable_database_and_retains_generation_audit_history() {
    let root = temp_root("stable-redb-history");
    write_index_at(&root, GENERATION_INDEX_PATH);
    let first = write_legacy_snapshot(&root, None, payloads(false, GENERATION_INDEX_PATH)).unwrap();
    let redb_path = root.join(ADAPTER_STORE_FILE);
    #[cfg(unix)]
    let first_metadata = fs::metadata(&redb_path).unwrap();

    write_index_at(&root, SECOND_GENERATION_INDEX_PATH);
    let second = write_legacy_snapshot(
        &root,
        Some(StoreRevision::from_manifest(&first.manifest).unwrap()),
        payloads(false, SECOND_GENERATION_INDEX_PATH),
    )
    .unwrap();
    #[cfg(unix)]
    let second_metadata = fs::metadata(&redb_path).unwrap();

    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        assert_eq!(first_metadata.ino(), second_metadata.ino());
    }
    assert_eq!(second.manifest["commit_sequence"], json!(2));
    assert_eq!(second.manifest["table_counts"]["generations"], json!(2));
    assert_eq!(second.manifest["table_counts"]["audit_log"], json!(2));

    let db = Database::open(&redb_path).unwrap();
    let read_txn = db.begin_read().unwrap();
    let generations = read_txn.open_table(GENERATIONS).unwrap();
    assert!(generations.get(1).unwrap().is_some());
    assert!(generations.get(2).unwrap().is_some());
    assert_eq!(generations.len().unwrap(), 2);
    let audit_log = read_txn.open_table(AUDIT_LOG).unwrap();
    assert!(audit_log.get(1).unwrap().is_some());
    assert!(audit_log.get(2).unwrap().is_some());
    assert_eq!(audit_log.len().unwrap(), 2);
    drop(read_txn);
    drop(db);

    let loaded = open_verified(&root, Some("langchain")).unwrap();
    assert_eq!(loaded.active_generation_id(), Some(2));
    assert_eq!(
        loaded.active_generation_path(),
        Some(SECOND_GENERATION_INDEX_PATH)
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn redb_store_records_generation_gc_with_revision_guard() {
    let root = temp_root("gc-record");
    write_index_at(&root, GENERATION_INDEX_PATH);
    let first = write_legacy_snapshot(&root, None, payloads(false, GENERATION_INDEX_PATH)).unwrap();
    let expected = StoreRevision::from_manifest(&first.manifest).unwrap();

    let recorded = record_generation_gc(
        &root,
        Some(expected.clone()),
        &[GenerationGcUpdate {
            generation_id: Some(1),
            path: GENERATION_INDEX_PATH.to_string(),
            state: "reclaimable".to_string(),
            reason: "test".to_string(),
        }],
    )
    .unwrap();

    assert_eq!(recorded.manifest["commit_sequence"], json!(2));
    assert_eq!(recorded.manifest["table_counts"]["gc_queue"], json!(1));
    assert_eq!(recorded.manifest["table_counts"]["audit_log"], json!(2));

    write_index_at(&root, SECOND_GENERATION_INDEX_PATH);
    let updated = write_legacy_snapshot(
        &root,
        Some(StoreRevision::from_manifest(&recorded.manifest).unwrap()),
        payloads(false, SECOND_GENERATION_INDEX_PATH),
    )
    .unwrap();
    assert_eq!(updated.manifest["commit_sequence"], json!(3));
    assert_eq!(updated.manifest["table_counts"]["gc_queue"], json!(1));
    assert_eq!(updated.manifest["table_counts"]["audit_log"], json!(3));
    let db = Database::open(root.join(ADAPTER_STORE_FILE)).unwrap();
    let read_txn = db.begin_read().unwrap();
    let gc_queue = read_txn.open_table(GC_QUEUE).unwrap();
    assert!(gc_queue.get(1).unwrap().is_some());
    assert_eq!(gc_queue.len().unwrap(), 1);
    drop(read_txn);
    drop(db);

    let err = record_generation_gc(
        &root,
        Some(expected),
        &[GenerationGcUpdate {
            generation_id: Some(1),
            path: GENERATION_INDEX_PATH.to_string(),
            state: "deleted".to_string(),
            reason: "test".to_string(),
        }],
    )
    .unwrap_err();
    assert!(err.to_string().contains("stale adapter snapshot"), "{err}");

    let _ = fs::remove_dir_all(root);
}

#[test]
fn redb_store_supports_empty_lazy_without_vector_generation() {
    let root = temp_root("empty-lazy");
    let payloads = payloads(true, GENERATION_INDEX_PATH);

    let written = write_legacy_snapshot(&root, None, payloads).unwrap();
    assert_eq!(written.manifest["empty_lazy"], json!(true));
    assert_eq!(written.manifest["vector_count"], json!(0));

    let loaded = open_verified(&root, Some("langchain")).unwrap();
    assert!(loaded.empty_lazy());
    assert_eq!(loaded.dim(), None);

    let _ = fs::remove_dir_all(root);
}

#[test]
fn redb_store_rejects_corrupt_store_file() {
    let root = temp_root("corrupt-store");
    write_index_at(&root, LEGACY_INDEX_PATH);
    write_legacy_snapshot(&root, None, payloads(false, LEGACY_INDEX_PATH)).unwrap();

    fs::write(root.join(ADAPTER_STORE_FILE), b"not a redb database").unwrap();
    let err = open_verified(&root, Some("langchain")).unwrap_err();
    assert!(
        err.to_string().contains("redb")
            || err.to_string().contains("magic")
            || err.to_string().contains("database")
            || err.to_string().contains("invalid data"),
        "{err}"
    );

    let _ = fs::remove_dir_all(root);
}

#[cfg(unix)]
#[test]
fn redb_store_rejects_symlinked_store_file() {
    let root = temp_root("symlink-redb");
    write_index_at(&root, LEGACY_INDEX_PATH);
    write_legacy_snapshot(&root, None, payloads(false, LEGACY_INDEX_PATH)).unwrap();
    let outside = root.with_extension("outside-redb");
    fs::rename(root.join(ADAPTER_STORE_FILE), &outside).unwrap();
    std::os::unix::fs::symlink(&outside, root.join(ADAPTER_STORE_FILE)).unwrap();

    let err = open_verified(&root, Some("langchain")).unwrap_err();

    assert!(err.to_string().contains("must not be a symlink"), "{err}");
    let _ = fs::remove_file(outside);
    let _ = fs::remove_dir_all(root);
}

#[test]
fn redb_store_rejects_active_generation_manifest_mismatch() {
    let root = temp_root("manifest-mismatch");
    write_index_at(&root, LEGACY_INDEX_PATH);
    write_legacy_snapshot(&root, None, payloads(false, LEGACY_INDEX_PATH)).unwrap();

    fs::write(root.join("index.odb").join("manifest.json"), "{}\n").unwrap();
    let err = open_verified(&root, Some("langchain")).unwrap_err();
    assert!(
        err.to_string().contains("manifest")
            || err.to_string().contains("metadata")
            || err.to_string().contains("schema_version"),
        "{err}"
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn redb_store_rejects_missing_active_generation_on_open() {
    let root = temp_root("missing-generation");
    write_index_at(&root, GENERATION_INDEX_PATH);
    write_legacy_snapshot(&root, None, payloads(false, GENERATION_INDEX_PATH)).unwrap();

    fs::remove_dir_all(root.join(GENERATION_INDEX_PATH)).unwrap();
    let err = open_verified(&root, Some("langchain")).unwrap_err();
    assert!(
        err.to_string()
            .contains("active generation path is missing")
            || err.to_string().contains("No such file")
            || err.to_string().contains("not found")
            || err.to_string().contains("cannot"),
        "{err}"
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn redb_store_rejects_oversized_active_generation_manifest() {
    let root = temp_root("oversized-manifest");
    write_index_at(&root, LEGACY_INDEX_PATH);
    fs::write(
        root.join(LEGACY_INDEX_PATH).join("manifest.json"),
        vec![b' '; 1024 * 1024 + 1],
    )
    .unwrap();

    let err = write_legacy_snapshot(&root, None, payloads(false, LEGACY_INDEX_PATH)).unwrap_err();
    assert!(err.to_string().contains("manifest too large"), "{err}");
    assert!(!root.join(ADAPTER_STORE_FILE).exists());

    let _ = fs::remove_dir_all(root);
}

#[test]
fn redb_store_rejects_non_file_active_generation_manifest() {
    let root = temp_root("non-file-manifest");
    write_index_at(&root, LEGACY_INDEX_PATH);
    fs::remove_file(root.join(LEGACY_INDEX_PATH).join("manifest.json")).unwrap();
    fs::create_dir(root.join(LEGACY_INDEX_PATH).join("manifest.json")).unwrap();

    let err = write_legacy_snapshot(&root, None, payloads(false, LEGACY_INDEX_PATH)).unwrap_err();
    assert!(err.to_string().contains("manifest must be a file"), "{err}");
    assert!(!root.join(ADAPTER_STORE_FILE).exists());

    let _ = fs::remove_dir_all(root);
}

#[cfg(unix)]
#[test]
fn redb_store_rejects_symlinked_active_generation_manifest() {
    let root = temp_root("symlink-manifest");
    write_index_at(&root, LEGACY_INDEX_PATH);
    let manifest_path = root.join(LEGACY_INDEX_PATH).join("manifest.json");
    let outside_manifest = root.join("outside-manifest.json");
    fs::rename(&manifest_path, &outside_manifest).unwrap();
    std::os::unix::fs::symlink(&outside_manifest, &manifest_path).unwrap();

    let err = write_legacy_snapshot(&root, None, payloads(false, LEGACY_INDEX_PATH)).unwrap_err();
    assert!(err.to_string().contains("symlink"), "{err}");
    assert!(!root.join(ADAPTER_STORE_FILE).exists());

    let _ = fs::remove_dir_all(root);
}

#[cfg(unix)]
#[test]
fn redb_store_rejects_symlinked_generation_parent() {
    let root = temp_root("symlink-generation-parent");
    let outside = temp_root("outside-generation-parent");
    write_index_at(&outside, "g000000000001.odb");
    std::os::unix::fs::symlink(&outside, root.join("vectors")).unwrap();

    let err =
        write_legacy_snapshot(&root, None, payloads(false, GENERATION_INDEX_PATH)).unwrap_err();
    assert!(
        err.to_string()
            .contains("active generation path must not contain a symlink"),
        "{err}"
    );
    assert!(!root.join(ADAPTER_STORE_FILE).exists());

    let _ = fs::remove_dir_all(root);
    let _ = fs::remove_dir_all(outside);
}

#[cfg(unix)]
#[test]
fn redb_store_rejects_symlinked_legacy_index_directory() {
    let root = temp_root("symlink-legacy-index");
    let outside = temp_root("outside-legacy-index");
    write_index_at(&outside, LEGACY_INDEX_PATH);
    std::os::unix::fs::symlink(
        outside.join(LEGACY_INDEX_PATH),
        root.join(LEGACY_INDEX_PATH),
    )
    .unwrap();

    let err = write_legacy_snapshot(&root, None, payloads(false, LEGACY_INDEX_PATH)).unwrap_err();
    assert!(
        err.to_string()
            .contains("active generation path must not contain a symlink"),
        "{err}"
    );
    assert!(!root.join(ADAPTER_STORE_FILE).exists());

    let _ = fs::remove_dir_all(root);
    let _ = fs::remove_dir_all(outside);
}

#[test]
fn redb_store_rejects_duplicate_u64_ids_before_mutation() {
    let root = temp_root("duplicate-u64");
    write_index_at(&root, LEGACY_INDEX_PATH);
    let mut payloads = payloads(false, LEGACY_INDEX_PATH);
    payloads.id_map_json = serde_json::to_string(&json!({
        "schema_version": "ordinaldb.adapter.id_map.v1",
        "next_u64_id": 3,
        "string_to_u64": {"a": 1, "b": 1},
        "u64_to_slot": {"1": 0}
    }))
    .unwrap();

    let err = write_legacy_snapshot(&root, None, payloads).unwrap_err();
    assert!(err.to_string().contains("duplicate u64 id"), "{err}");
    assert!(!root.join(ADAPTER_STORE_FILE).exists());

    let _ = fs::remove_dir_all(root);
}

#[test]
fn redb_store_rejects_duplicate_json_keys_before_mutation() {
    let root = temp_root("duplicate-json");
    write_index_at(&root, LEGACY_INDEX_PATH);
    let mut payloads = payloads(false, LEGACY_INDEX_PATH);
    payloads.documents_json = r#"{"schema_version":"ordinaldb.adapter.documents.v1","documents":{"a":"alpha","a":"shadow","b":"beta"}}"#.to_string();

    let err = write_legacy_snapshot(&root, None, payloads).unwrap_err();
    assert!(err.to_string().contains("duplicate JSON key"), "{err}");
    assert!(!root.join(ADAPTER_STORE_FILE).exists());

    let _ = fs::remove_dir_all(root);
}

#[test]
fn redb_store_rejects_extra_payload_keys_before_mutation() {
    let root = temp_root("extra-keys");
    write_index_at(&root, LEGACY_INDEX_PATH);
    let mut payloads = payloads(false, LEGACY_INDEX_PATH);
    let mut id_map: serde_json::Value = serde_json::from_str(&payloads.id_map_json).unwrap();
    id_map["unexpected"] = json!(true);
    payloads.id_map_json = serde_json::to_string(&id_map).unwrap();

    let err = write_legacy_snapshot(&root, None, payloads).unwrap_err();
    assert!(
        err.to_string().contains("id_map.json has invalid keys"),
        "{err}"
    );
    assert!(!root.join(ADAPTER_STORE_FILE).exists());

    let _ = fs::remove_dir_all(root);
}

#[test]
fn redb_store_rejects_invalid_index_path_before_mutation() {
    for invalid in [
        "vectors/index.odb",
        "vectors//g000000000001.odb",
        "vectors/g000000000001.odb/",
        r"vectors\g000000000001.odb",
        "vectors/g000000000000.odb",
    ] {
        let root = temp_root("invalid-index-path");
        write_index_at(&root, LEGACY_INDEX_PATH);
        let mut payloads = payloads(false, LEGACY_INDEX_PATH);
        let mut adapter: serde_json::Value = serde_json::from_str(&payloads.adapter_json).unwrap();
        adapter["index_path"] = json!(invalid);
        payloads.adapter_json = serde_json::to_string(&adapter).unwrap();

        let err = write_legacy_snapshot(&root, None, payloads).unwrap_err();
        assert!(
            err.to_string().contains("generation path")
                || err.to_string().contains("adapter index_path"),
            "{err}"
        );
        assert!(!root.join(ADAPTER_STORE_FILE).exists());

        let _ = fs::remove_dir_all(root);
    }
}

#[test]
fn redb_store_rejects_empty_lazy_with_stale_index_dir() {
    let root = temp_root("empty-lazy-stale-index");
    fs::create_dir_all(root.join("index.odb")).unwrap();
    fs::write(root.join("index.odb").join("manifest.json"), "{}\n").unwrap();

    let err =
        write_legacy_snapshot(&root, None, payloads(true, GENERATION_INDEX_PATH)).unwrap_err();
    assert!(err.to_string().contains("empty_lazy"), "{err}");
    assert!(!root.join(ADAPTER_STORE_FILE).exists());

    let _ = fs::remove_dir_all(root);
}

#[test]
fn redb_store_rejects_empty_lazy_with_stale_generation_dir() {
    let root = temp_root("empty-lazy-stale-generation");
    fs::create_dir_all(root.join(GENERATION_INDEX_PATH)).unwrap();
    fs::write(
        root.join(GENERATION_INDEX_PATH).join("manifest.json"),
        "{}\n",
    )
    .unwrap();

    let err =
        write_legacy_snapshot(&root, None, payloads(true, GENERATION_INDEX_PATH)).unwrap_err();
    assert!(err.to_string().contains("empty_lazy"), "{err}");
    assert!(!root.join(ADAPTER_STORE_FILE).exists());

    let _ = fs::remove_dir_all(root);
}

#[test]
fn redb_store_rejects_manifest_payload_mismatch() {
    let root = temp_root("manifest-payload-mismatch");
    write_index_at(&root, LEGACY_INDEX_PATH);
    write_legacy_snapshot(&root, None, payloads(false, LEGACY_INDEX_PATH)).unwrap();

    mutate_manifest(&root, |manifest| {
        manifest["active_id_count"] = json!(1);
    });

    let err = open_verified(&root, Some("langchain")).unwrap_err();
    assert!(
        err.to_string()
            .contains("active_id_count manifest mismatch"),
        "{err}"
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn redb_store_rejects_u64_to_string_mismatch() {
    let root = temp_root("reverse-map-mismatch");
    write_index_at(&root, LEGACY_INDEX_PATH);
    write_legacy_snapshot(&root, None, payloads(false, LEGACY_INDEX_PATH)).unwrap();

    let db = Database::open(root.join(ADAPTER_STORE_FILE)).unwrap();
    let write_txn = db.begin_write().unwrap();
    {
        let mut reverse = write_txn.open_table(U64_TO_STRING).unwrap();
        reverse.insert(1, "b").unwrap();
        reverse.insert(2, "a").unwrap();
    }
    write_txn.commit().unwrap();
    drop(db);

    let err = open_verified(&root, Some("langchain")).unwrap_err();
    assert!(
        err.to_string()
            .contains("u64_to_string table does not match payload"),
        "{err}"
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn redb_store_marks_previous_generation_retired_when_new_active_published() {
    let root = temp_root("generation-lifecycle");
    write_index_at(&root, GENERATION_INDEX_PATH);
    let first = write_legacy_snapshot(&root, None, payloads(false, GENERATION_INDEX_PATH)).unwrap();
    assert_eq!(generation_row(&root, 1)["state"], json!("active"));

    write_single_index_at(&root, SECOND_GENERATION_INDEX_PATH);
    let second = write_legacy_snapshot(
        &root,
        Some(StoreRevision::from_manifest(&first.manifest).unwrap()),
        one_record_payloads(SECOND_GENERATION_INDEX_PATH),
    )
    .unwrap();

    assert_eq!(second.manifest["active_generation_id"], json!(2));
    let first_generation = generation_row(&root, 1);
    assert_eq!(first_generation["state"], json!("retired"));
    assert_eq!(first_generation["retired_by_commit_sequence"], json!(2));
    let second_generation = generation_row(&root, 2);
    assert_eq!(second_generation["state"], json!("active"));
    assert_eq!(
        second_generation["path"],
        json!(SECOND_GENERATION_INDEX_PATH)
    );
    open_verified(&root, Some("langchain")).unwrap();

    let _ = fs::remove_dir_all(root);
}

#[test]
fn redb_store_rejects_invalid_generation_lifecycle_state() {
    let root = temp_root("invalid-generation-state");
    write_index_at(&root, GENERATION_INDEX_PATH);
    write_legacy_snapshot(&root, None, payloads(false, GENERATION_INDEX_PATH)).unwrap();
    mutate_generation_row(&root, 1, |row| {
        row["state"] = json!("unknown");
    });

    let err = open_verified(&root, Some("langchain")).unwrap_err();
    assert!(
        err.to_string()
            .contains("invalid generation lifecycle state"),
        "{err}"
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn redb_store_rejects_invalid_gc_lifecycle_state() {
    let root = temp_root("invalid-gc-state");
    write_index_at(&root, GENERATION_INDEX_PATH);
    let written =
        write_legacy_snapshot(&root, None, payloads(false, GENERATION_INDEX_PATH)).unwrap();

    let err = record_generation_gc(
        &root,
        Some(StoreRevision::from_manifest(&written.manifest).unwrap()),
        &[GenerationGcUpdate {
            generation_id: Some(1),
            path: GENERATION_INDEX_PATH.to_string(),
            state: "unknown".to_string(),
            reason: "test".to_string(),
        }],
    )
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("generation GC state must be a known generation lifecycle state"),
        "{err}"
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn redb_store_reconciles_live_tables_and_preserves_gc_history() {
    let root = temp_root("reconcile-live-tables");
    write_index_at(&root, GENERATION_INDEX_PATH);
    let first = write_legacy_snapshot(&root, None, payloads(false, GENERATION_INDEX_PATH)).unwrap();
    let recorded = record_generation_gc(
        &root,
        Some(StoreRevision::from_manifest(&first.manifest).unwrap()),
        &[GenerationGcUpdate {
            generation_id: Some(1),
            path: GENERATION_INDEX_PATH.to_string(),
            state: "retired".to_string(),
            reason: "test".to_string(),
        }],
    )
    .unwrap();

    write_single_index_at(&root, SECOND_GENERATION_INDEX_PATH);
    let updated = write_legacy_snapshot(
        &root,
        Some(StoreRevision::from_manifest(&recorded.manifest).unwrap()),
        one_record_payloads(SECOND_GENERATION_INDEX_PATH),
    )
    .unwrap();

    assert_eq!(updated.manifest["active_id_count"], json!(1));
    assert_eq!(updated.manifest["table_counts"]["string_to_u64"], json!(1));
    assert_eq!(updated.manifest["table_counts"]["u64_to_string"], json!(1));
    assert_eq!(updated.manifest["table_counts"]["documents"], json!(1));
    assert_eq!(updated.manifest["table_counts"]["metadata"], json!(1));
    assert_eq!(updated.manifest["table_counts"]["gc_queue"], json!(1));
    let db = Database::open(root.join(ADAPTER_STORE_FILE)).unwrap();
    let read_txn = db.begin_read().unwrap();
    let reverse = read_txn.open_table(U64_TO_STRING).unwrap();
    assert!(reverse.get(1).unwrap().is_none());
    assert_eq!(reverse.get(2).unwrap().unwrap().value(), "b");
    let gc_queue = read_txn.open_table(GC_QUEUE).unwrap();
    assert!(gc_queue.get(1).unwrap().is_some());
    drop(read_txn);
    drop(db);
    open_verified(&root, Some("langchain")).unwrap();

    let _ = fs::remove_dir_all(root);
}

fn payloads(empty_lazy: bool, index_path: &str) -> LegacyPayloads {
    if empty_lazy {
        return LegacyPayloads {
            adapter_json: canonical(json!({
                "schema_version": "ordinaldb.adapter.v1",
                "adapter": "langchain",
                "bits": 2,
                "dim": null,
                "empty_lazy": true,
                "index_path": index_path,
                "sidecars": {
                    "id_map.json": {"sha256": "", "file_size_bytes": 0},
                    "documents.json": {"sha256": "", "file_size_bytes": 0},
                    "metadata.json": {"sha256": "", "file_size_bytes": 0}
                }
            })),
            id_map_json: canonical(json!({
                "schema_version": "ordinaldb.adapter.id_map.v1",
                "next_u64_id": 1,
                "string_to_u64": {},
                "u64_to_slot": {}
            })),
            documents_json: canonical(json!({
                "schema_version": "ordinaldb.adapter.documents.v1",
                "documents": {}
            })),
            metadata_json: canonical(json!({
                "schema_version": "ordinaldb.adapter.metadata.v1",
                "metadata": {}
            })),
        };
    }

    LegacyPayloads {
        adapter_json: canonical(json!({
            "schema_version": "ordinaldb.adapter.v1",
            "adapter": "langchain",
            "bits": 2,
            "dim": 4,
            "empty_lazy": false,
            "index_path": index_path,
            "sidecars": {
                "id_map.json": {"sha256": "", "file_size_bytes": 0},
                "documents.json": {"sha256": "", "file_size_bytes": 0},
                "metadata.json": {"sha256": "", "file_size_bytes": 0}
            }
        })),
        id_map_json: canonical(json!({
            "schema_version": "ordinaldb.adapter.id_map.v1",
            "next_u64_id": 3,
            "string_to_u64": {"a": 1, "b": 2},
            "u64_to_slot": {"1": 0, "2": 1}
        })),
        documents_json: canonical(json!({
            "schema_version": "ordinaldb.adapter.documents.v1",
            "documents": {"a": "alpha", "b": "beta"}
        })),
        metadata_json: canonical(json!({
            "schema_version": "ordinaldb.adapter.metadata.v1",
            "metadata": {"a": {"group": "x"}, "b": {"group": "y"}}
        })),
    }
}

fn one_record_payloads(index_path: &str) -> LegacyPayloads {
    LegacyPayloads {
        adapter_json: canonical(json!({
            "schema_version": "ordinaldb.adapter.v1",
            "adapter": "langchain",
            "bits": 2,
            "dim": 4,
            "empty_lazy": false,
            "index_path": index_path,
            "sidecars": {
                "id_map.json": {"sha256": "", "file_size_bytes": 0},
                "documents.json": {"sha256": "", "file_size_bytes": 0},
                "metadata.json": {"sha256": "", "file_size_bytes": 0}
            }
        })),
        id_map_json: canonical(json!({
            "schema_version": "ordinaldb.adapter.id_map.v1",
            "next_u64_id": 3,
            "string_to_u64": {"b": 2},
            "u64_to_slot": {"2": 0}
        })),
        documents_json: canonical(json!({
            "schema_version": "ordinaldb.adapter.documents.v1",
            "documents": {"b": "beta"}
        })),
        metadata_json: canonical(json!({
            "schema_version": "ordinaldb.adapter.metadata.v1",
            "metadata": {"b": {"group": "y"}}
        })),
    }
}

fn write_index_at(root: &Path, index_path: &str) {
    let mut index = OrdinalIndex::new(4, 2).unwrap();
    index.add_2d(VECTORS, 4).unwrap();
    let index_root = root.join(index_path);
    fs::create_dir_all(index_root.parent().unwrap()).unwrap();
    index.write(index_root).unwrap();
}

fn write_single_index_at(root: &Path, index_path: &str) {
    let mut index = OrdinalIndex::new(4, 2).unwrap();
    index.add_2d(&VECTORS[..4], 4).unwrap();
    let index_root = root.join(index_path);
    fs::create_dir_all(index_root.parent().unwrap()).unwrap();
    index.write(index_root).unwrap();
}

fn canonical(value: serde_json::Value) -> String {
    serde_json::to_string(&value).unwrap()
}

fn mutate_manifest(root: &Path, mut f: impl FnMut(&mut serde_json::Value)) {
    let db = Database::open(root.join(ADAPTER_STORE_FILE)).unwrap();
    let write_txn = db.begin_write().unwrap();
    {
        let mut meta = write_txn.open_table(META).unwrap();
        let manifest_json = meta.get("manifest").unwrap().unwrap().value().to_string();
        let mut manifest: serde_json::Value = serde_json::from_str(&manifest_json).unwrap();
        f(&mut manifest);
        let updated = serde_json::to_string(&manifest).unwrap();
        meta.insert("manifest", updated.as_str()).unwrap();
    }
    write_txn.commit().unwrap();
}

fn generation_row(root: &Path, generation_id: u64) -> serde_json::Value {
    let db = Database::open(root.join(ADAPTER_STORE_FILE)).unwrap();
    let read_txn = db.begin_read().unwrap();
    let generations = read_txn.open_table(GENERATIONS).unwrap();
    let row = generations
        .get(generation_id)
        .unwrap()
        .unwrap()
        .value()
        .to_string();
    serde_json::from_str(&row).unwrap()
}

fn mutate_generation_row(
    root: &Path,
    generation_id: u64,
    mut f: impl FnMut(&mut serde_json::Value),
) {
    let db = Database::open(root.join(ADAPTER_STORE_FILE)).unwrap();
    let write_txn = db.begin_write().unwrap();
    {
        let mut generations = write_txn.open_table(GENERATIONS).unwrap();
        let row = generations
            .get(generation_id)
            .unwrap()
            .unwrap()
            .value()
            .to_string();
        let mut generation: serde_json::Value = serde_json::from_str(&row).unwrap();
        f(&mut generation);
        let updated = serde_json::to_string(&generation).unwrap();
        generations.insert(generation_id, updated.as_str()).unwrap();
    }
    write_txn.commit().unwrap();
}

fn temp_root(name: &str) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!("ordinaldb-adapter-store-{name}-{stamp}"));
    fs::create_dir_all(&root).unwrap();
    root
}
