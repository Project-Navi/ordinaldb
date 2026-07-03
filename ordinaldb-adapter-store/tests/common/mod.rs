//! Shared helpers for the adapter-store integration tests.

#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use ordinaldb_adapter_store::LegacyPayloads;
use ordinaldb_core::OrdinalIndex;
use serde_json::json;

pub const VECTORS: &[f32] = &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
pub const GENERATION_INDEX_PATH: &str = "vectors/g000000000001.odb";

pub fn temp_root(name: &str) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!("ordinaldb-adapter-store-{name}-{stamp}"));
    fs::create_dir_all(&root).unwrap();
    root
}

pub fn canonical(value: serde_json::Value) -> String {
    serde_json::to_string(&value).unwrap()
}

pub fn write_index_at(root: &Path, index_path: &str) {
    let mut index = OrdinalIndex::new(4, 2).unwrap();
    index.add_2d(VECTORS, 4).unwrap();
    let index_root = root.join(index_path);
    fs::create_dir_all(index_root.parent().unwrap()).unwrap();
    index.write(index_root).unwrap();
}

pub fn payloads(empty_lazy: bool, index_path: &str) -> LegacyPayloads {
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
    payloads_with_metadata(
        index_path,
        json!({"a": {"group": "x"}, "b": {"group": "y"}}),
    )
}

/// Two-record payloads (`a`, `b`) with caller-provided metadata objects.
pub fn payloads_with_metadata(index_path: &str, metadata: serde_json::Value) -> LegacyPayloads {
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
            "metadata": metadata
        })),
    }
}
