//! Core-index tests using real all-MiniLM-L6-v2 embeddings.
//!
//! Synthetic vectors systematically miss bugs that real embedding
//! bit-patterns expose (see tests/fixtures/real_embeddings/generate.py for
//! provenance). These tests exercise add/search/persist/delete with the
//! committed fixture corpus: 32 documents in four domains and 8 queries
//! with expected-domain labels, all verified to rank correctly under exact
//! dot-product at fixture-generation time.

use ordinaldb::{IdMapIndex, OrdinalIndex};

const DIM: usize = 384;
const DOCS: &[u8] = include_bytes!("../../tests/fixtures/real_embeddings/minilm_docs_f32.bin");
const QUERIES: &[u8] =
    include_bytes!("../../tests/fixtures/real_embeddings/minilm_queries_f32.bin");

/// Row layout of texts.json: 0..8 docs, 8..16 tickets, 16..24 notes,
/// 24..32 papers. Queries: see texts.json expect_domain fields.
const DOMAIN_OF_ROW: [&str; 4] = ["docs", "tickets", "notes", "papers"];
const QUERY_EXPECT: [&str; 8] = [
    "tickets", "tickets", "notes", "notes", "papers", "papers", "docs", "docs",
];

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    // chunks_exact silently drops a trailing partial chunk; without this
    // assert, 1-3 corrupt trailing bytes on a committed fixture would pass
    // through undetected because the decoded element count is unaffected.
    assert_eq!(
        bytes.len() % 4,
        0,
        "fixture byte length {} is not 4-byte aligned",
        bytes.len()
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn docs() -> Vec<f32> {
    let v = decode_f32(DOCS);
    assert_eq!(v.len(), 32 * DIM, "docs fixture shape drifted");
    v
}

fn queries() -> Vec<f32> {
    let v = decode_f32(QUERIES);
    assert_eq!(v.len(), 8 * DIM, "queries fixture shape drifted");
    v
}

fn domain_of(row: usize) -> &'static str {
    DOMAIN_OF_ROW[row / 8]
}

/// A corrupted fixture with 1-3 trailing bytes must be rejected rather than
/// silently truncated by `chunks_exact` (the decoded element count alone
/// does not change, so the shape asserts in `docs()`/`queries()` would not
/// have caught this).
#[test]
fn decode_f32_rejects_misaligned_byte_length() {
    for len in [1usize, 2, 3, 5, 6, 7] {
        let bytes = vec![0u8; len];
        let result = std::panic::catch_unwind(|| decode_f32(&bytes));
        assert!(
            result.is_err(),
            "decode_f32 should reject {len}-byte input as misaligned"
        );
    }
    // Aligned input must still decode normally.
    let aligned = vec![0u8; 8];
    assert_eq!(decode_f32(&aligned), vec![0.0f32, 0.0]);
}

fn temp_bundle(name: &str) -> std::path::PathBuf {
    // PID alone is not unique enough: PIDs are reused over time and can
    // collide across concurrent/containerized runs, which would make
    // `remove_dir_all` on this path unsafe (see sibling test files
    // ordinal_index.rs, id_map.rs, boundary_api.rs for the established
    // pattern of pairing the PID with extra per-call entropy).
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "ordinaldb-real-emb-{}-{}-{}",
        name,
        std::process::id(),
        stamp
    ))
}

/// Quantized top-3 must still surface the semantically expected domain for
/// every fixture query — the retrieval-quality contract on real data.
#[test]
fn real_embedding_search_hits_expected_domain_in_top3() {
    let mut idx = OrdinalIndex::new(DIM, 2).unwrap();
    idx.add_2d(&docs(), DIM).unwrap();

    let q = queries();
    for (qi, expect) in QUERY_EXPECT.iter().enumerate() {
        let one = &q[qi * DIM..(qi + 1) * DIM];
        let results = idx.search_checked(one, 3).unwrap();
        let domains: Vec<&str> = results
            .indices
            .iter()
            .map(|&row| domain_of(row as usize))
            .collect();
        assert!(
            domains.contains(expect),
            "query {qi}: expected domain {expect:?} in quantized top-3, got {domains:?} \
             (indices {:?})",
            results.indices
        );
    }
}

/// Persistence round-trip with real embeddings: results after write+load
/// must be identical to in-memory results.
#[test]
fn real_embedding_write_load_roundtrip_is_lossless() {
    let path = temp_bundle("roundtrip");
    let _ = std::fs::remove_dir_all(&path);

    let mut idx = OrdinalIndex::new(DIM, 2).unwrap();
    idx.add_2d(&docs(), DIM).unwrap();
    let before = idx.search_checked(&queries(), 5).unwrap();
    idx.write(&path).unwrap();

    let loaded = OrdinalIndex::load(&path).unwrap();
    let after = loaded.search_checked(&queries(), 5).unwrap();
    assert_eq!(before.indices, after.indices);
    assert_eq!(before.scores, after.scores);

    let _ = std::fs::remove_dir_all(&path);
}

/// IdMapIndex lifecycle on real embeddings: stable IDs, allowlist
/// filtering as pre-search restriction, and delete + re-query.
#[test]
fn real_embedding_idmap_allowlist_and_delete() {
    let mut ids = IdMapIndex::new(DIM, 2).unwrap();
    let external: Vec<u64> = (0..32).map(|r| 1000 + r as u64).collect();
    ids.add_with_ids_2d(&docs(), DIM, &external).unwrap();

    let q = queries();
    // q-002 ("SSO login loop") restricted to the papers rows only: the
    // allowlist must be authoritative even when the best matches (tickets)
    // are excluded.
    let papers_only: Vec<u64> = (24..32).map(|r| 1000 + r as u64).collect();
    let sso = &q[DIM..2 * DIM];
    let (_scores, found) = ids
        .search_checked_with_allowlist(sso, 3, Some(&papers_only))
        .unwrap();
    assert!(!found.is_empty());
    assert!(
        found.iter().all(|id| papers_only.contains(id)),
        "allowlist leaked: {found:?}"
    );

    // Delete the top match for q-001 (duplicate-billing ticket, row 10 =
    // id 1010) and confirm it never resurfaces.
    let billing = &q[..DIM];
    let (_s, before) = ids.search_checked(billing, 1).unwrap();
    let top = before[0];
    assert!(ids.remove(top), "expected removal of id {top}");
    let (_s, after) = ids.search_checked(billing, 5).unwrap();
    assert!(
        !after.contains(&top),
        "deleted id {top} resurfaced in {after:?}"
    );

    // A stale allowlist entry referencing the deleted id must be a clean
    // error, not a panic.
    let stale = ids.search_checked_with_allowlist(billing, 3, Some(&[top]));
    assert!(stale.is_err(), "stale allowlist id should error");
}
