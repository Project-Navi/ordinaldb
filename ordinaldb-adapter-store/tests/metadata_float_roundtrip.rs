//! Regression tests for float canonicalization in metadata payloads.
//!
//! Real embedding vectors (f32 values promoted to f64, serialized with
//! shortest-round-trip reprs by Python's json module) triggered
//! `metadata table does not match payload` at save time: serde_json's
//! default best-effort float parsing is not correctly rounded, so
//! `parse(serialize(parse(s)))` can drift by one ULP between the metadata
//! table row and the verbatim payload. Verification then reports a healthy
//! store as corrupt. These values are exact f32→f64 promotions whose
//! shortest reprs are known hard parsing cases.

mod common;

use std::fs;

use ordinaldb_adapter_store::{
    commit, open_verified, write_legacy_snapshot, AdapterMutation, MetadataPatch, StoreRevision,
};
use serde_json::json;

/// Exact f32→f64 promotions whose shortest decimal reprs round-trip only
/// under correctly rounded parsing (found by exhaustive f32-space sweep
/// against serde_json 1.0.150 defaults).
const ADVERSARIAL_EMBEDDING_VALUES: &[f64] = &[
    1.0000005268295808e-9,  // f32 bits 0x30897064
    1.0000011929633956e-9,  // f32 bits 0x3089706a
    1.0000026362533276e-9,  // f32 bits 0x30897077
    6.9875747923557e-41,    // subnormal f32 promotion
    1.39751495847114e-40,   // subnormal f32 promotion
    -1.0000047456770744e-9, // negative, f32 bits 0x3089708a
];

// The digit strings deliberately mirror what Python's json module writes
// (shortest-round-trip reprs of f64 values), including ones clippy considers
// longer than necessary.
#[allow(clippy::excessive_precision)]
fn embedding_metadata() -> serde_json::Value {
    // A realistic embedding-bearing metadata object: mostly ordinary values
    // with a few near-zero components that hit the hard parsing cases.
    let mut embedding: Vec<f64> = vec![
        0.038493849337100983,
        -0.061524353176355362,
        0.12250971794128418,
        -0.0035618704278022051,
    ];
    embedding.extend_from_slice(ADVERSARIAL_EMBEDDING_VALUES);
    json!({
        "a": {"group": "x"},
        "b": {
            "group": "y",
            "embedding": embedding,
            "score": 0.10000000149011612_f64, // 0.1f32 promoted to f64
        }
    })
}

#[test]
fn metadata_with_real_embedding_floats_survives_write_and_verify() {
    let root = common::temp_root("float-roundtrip-write");
    common::write_index_at(&root, common::GENERATION_INDEX_PATH);
    let payloads =
        common::payloads_with_metadata(common::GENERATION_INDEX_PATH, embedding_metadata());

    write_legacy_snapshot(&root, None, payloads.clone())
        .expect("embedding floats must survive the write -> verify round trip");
    let store = open_verified(&root, Some("langchain")).unwrap();
    assert_eq!(store.payloads.metadata_json, payloads.metadata_json);

    let _ = fs::remove_dir_all(root);
}

#[test]
fn metadata_patch_with_real_embedding_floats_survives_commit() {
    let root = common::temp_root("float-roundtrip-patch");
    common::write_index_at(&root, common::GENERATION_INDEX_PATH);
    let written = write_legacy_snapshot(
        &root,
        None,
        common::payloads(false, common::GENERATION_INDEX_PATH),
    )
    .unwrap();
    let expected = StoreRevision::from_manifest(&written.manifest).unwrap();

    let patch_metadata = json!({
        "group": "y",
        "embedding": ADVERSARIAL_EMBEDDING_VALUES,
    });
    let serde_json::Value::Object(patch_object) = patch_metadata else {
        unreachable!("json! object literal");
    };
    commit(
        &root,
        expected,
        AdapterMutation::PatchMetadata(vec![MetadataPatch {
            id: "b".to_string(),
            metadata: patch_object,
        }]),
    )
    .expect("embedding floats must survive metadata patch commits");
    open_verified(&root, Some("langchain")).unwrap();

    let _ = fs::remove_dir_all(root);
}

#[test]
fn every_reopen_of_an_embedding_bearing_store_stays_verified() {
    // Round-trip stability must hold across repeated open/verify cycles,
    // not just the first write.
    let root = common::temp_root("float-roundtrip-reopen");
    common::write_index_at(&root, common::GENERATION_INDEX_PATH);
    write_legacy_snapshot(
        &root,
        None,
        common::payloads_with_metadata(common::GENERATION_INDEX_PATH, embedding_metadata()),
    )
    .expect("embedding floats must survive the write -> verify round trip");
    for _ in 0..3 {
        open_verified(&root, Some("langchain")).unwrap();
    }
    let _ = fs::remove_dir_all(root);
}
