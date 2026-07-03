//! Regression tests for adapter.redb corruption handling.
//!
//! Byte-flipping adapter.redb must never surface as a raw panic (byte-flip
//! corruption used to produce `internal error: entered unreachable code` from
//! the storage engine's page deserialization). Corruption must always come back
//! as a structured [`AdapterStoreError`] — or, when the flipped bytes are not
//! part of the live data reachable by verification, as a successful verified
//! open (that scoping is documented in the threat model).

mod common;

use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;

use ordinaldb_adapter_store::{
    generation_gc_events, open_verified, record_generation_gc, write_legacy_snapshot,
    GenerationGcUpdate, StoreRevision, ADAPTER_STORE_FILE,
};

fn build_store(name: &str) -> PathBuf {
    let root = common::temp_root(name);
    common::write_index_at(&root, common::GENERATION_INDEX_PATH);
    write_legacy_snapshot(
        &root,
        None,
        common::payloads(false, common::GENERATION_INDEX_PATH),
    )
    .unwrap();
    root
}

/// Runs `operation` against a copy of the store whose adapter.redb has
/// `bytes` XOR-flipped at `offset`, reporting whether it panicked.
fn flips_panic<T>(
    root: &std::path::Path,
    original: &[u8],
    offset: usize,
    len: usize,
    operation: impl FnOnce() -> T,
) -> bool {
    let store_file = root.join(ADAPTER_STORE_FILE);
    let mut corrupted = original.to_vec();
    for byte in corrupted.iter_mut().skip(offset).take(len) {
        *byte ^= 0xFF;
    }
    fs::write(&store_file, &corrupted).unwrap();
    let panicked = catch_unwind(AssertUnwindSafe(operation)).is_err();
    fs::write(&store_file, original).unwrap();
    panicked
}

#[test]
fn corrupted_adapter_store_never_panics_on_open() {
    let root = build_store("corruption-open-sweep");
    let original = fs::read(root.join(ADAPTER_STORE_FILE)).unwrap();

    // Silence the default panic hook for the duration of the sweep so a
    // pre-fix run does not flood stderr; the sweep records every panic.
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    let stride = (original.len() / 4096).max(1);
    let mut panic_offsets = Vec::new();
    let mut clean_errors = 0usize;
    let mut silent_ok = 0usize;
    for offset in (0..original.len()).step_by(stride) {
        let mut panicked = false;
        let mut result = None;
        {
            let root = &root;
            let store_file = root.join(ADAPTER_STORE_FILE);
            let mut corrupted = original.clone();
            corrupted[offset] ^= 0xFF;
            fs::write(&store_file, &corrupted).unwrap();
            match catch_unwind(AssertUnwindSafe(|| open_verified(root, None))) {
                Ok(outcome) => result = Some(outcome),
                Err(_) => panicked = true,
            }
            fs::write(&store_file, &original).unwrap();
        }
        if panicked {
            panic_offsets.push(offset);
        } else {
            match result.unwrap() {
                Ok(_) => silent_ok += 1,
                Err(_) => clean_errors += 1,
            }
        }
    }

    std::panic::set_hook(previous_hook);
    println!(
        "single-byte flip sweep over {} bytes (stride {stride}): {} clean errors, {} silent OKs, {} panics",
        original.len(),
        clean_errors,
        silent_ok,
        panic_offsets.len(),
    );
    assert!(
        panic_offsets.is_empty(),
        "open_verified panicked instead of returning AdapterStoreError at offsets {panic_offsets:?}"
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn corrupted_adapter_store_never_panics_on_midfile_chunk_flip() {
    // Mid-file chunk corruption: 64 bytes flipped near the middle of
    // adapter.redb.
    let root = build_store("corruption-open-chunk");
    let original = fs::read(root.join(ADAPTER_STORE_FILE)).unwrap();

    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let mut panics = Vec::new();
    for offset in [
        0usize,
        original.len() / 4,
        original.len() / 2,
        original.len().saturating_sub(64),
    ] {
        if flips_panic(&root, &original, offset, 64, || open_verified(&root, None)) {
            panics.push(offset);
        }
    }
    std::panic::set_hook(previous_hook);
    assert!(panics.is_empty(), "64-byte flips panicked at {panics:?}");

    let _ = fs::remove_dir_all(root);
}

#[test]
fn corrupted_adapter_store_never_panics_on_gc_paths() {
    let root = build_store("corruption-gc");
    let store = open_verified(&root, None).unwrap();
    let expected = StoreRevision::from_manifest(&store.manifest).unwrap();
    let original = fs::read(root.join(ADAPTER_STORE_FILE)).unwrap();

    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let mut panics = Vec::new();
    let stride = (original.len() / 512).max(1);
    for offset in (0..original.len()).step_by(stride) {
        if flips_panic(&root, &original, offset, 1, || {
            let _ = generation_gc_events(&root);
            let _ = record_generation_gc(
                &root,
                Some(expected.clone()),
                &[GenerationGcUpdate {
                    generation_id: Some(1),
                    path: common::GENERATION_INDEX_PATH.to_string(),
                    state: "retired".to_string(),
                    reason: "corruption sweep".to_string(),
                }],
            );
        }) {
            panics.push(offset);
        }
    }
    std::panic::set_hook(previous_hook);
    assert!(panics.is_empty(), "gc paths panicked at offsets {panics:?}");

    let _ = fs::remove_dir_all(root);
}
