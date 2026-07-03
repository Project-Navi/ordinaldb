use ordinaldb::IdMapIndex;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

const DIM: usize = 64;
static NEXT_TEMP: AtomicUsize = AtomicUsize::new(0);

#[test]
fn pi_runtime_persistence_rewrite_and_search_are_stable() {
    let path = temp_bundle("pi-runtime-rewrite");
    cleanup(&path);

    let data = vectors(512, DIM);
    let queries = vectors(4, DIM);
    let ids: Vec<u64> = (0..512).map(|idx| 50_000 + idx as u64).collect();
    let removed_id = ids[17];

    let mut idx = IdMapIndex::new(DIM, 2).unwrap();
    idx.add_with_ids(&data, &ids).unwrap();

    let before = idx.search(&queries, 16);
    for _ in 0..4 {
        assert_eq!(idx.search(&queries, 16), before);
    }

    idx.write(&path).unwrap();
    assert!(path.join("manifest.json").exists());
    assert!(path.join("index.ovrq").exists());
    assert!(path.join("sign.ovsb").exists());
    assert!(path.join("ids.bin").exists());

    let mut loaded = IdMapIndex::load(&path).unwrap();
    assert_eq!(loaded.search(&queries, 16), before);

    assert!(loaded.remove(removed_id));
    loaded.write(&path).unwrap();
    assert!(path.join("sign.ovsb").exists());

    let reloaded = IdMapIndex::load(&path).unwrap();
    assert_eq!(reloaded.len(), ids.len() - 1);
    assert!(!reloaded.contains(removed_id));

    let after_delete = reloaded.search(&queries, 32);
    assert!(!after_delete.1.contains(&removed_id));
    for _ in 0..4 {
        assert_eq!(reloaded.search(&queries, 32), after_delete);
    }

    cleanup(&path);
}

fn vectors(n: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0; n * dim];
    for row in 0..n {
        for col in 0..dim {
            let x = (((row + 17) * (col + 5) + row * 29 + col * 11) % 97) as f32 - 48.0;
            out[row * dim + col] = x / 49.0;
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
