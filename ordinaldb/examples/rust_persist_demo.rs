use std::error::Error;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use ordinaldb::{IdMapIndex, OrdinalIndex};

const DIM: usize = 64;

fn main() -> Result<(), Box<dyn Error>> {
    let vectors = demo_vectors(32, DIM);
    let queries = demo_vectors(2, DIM);

    let demo_root = create_demo_root()?;
    let demo_path = demo_root.join("demo.odb");
    let demo_ids_path = demo_root.join("demo_ids.odb");

    let mut idx = OrdinalIndex::new(DIM, 2)?;
    idx.add_2d(&vectors, DIM)?;
    let results = idx.search_checked(&queries, 5)?;
    idx.write(&demo_path)?;
    let loaded = OrdinalIndex::load(&demo_path)?;
    assert_eq!(loaded.search_checked(&queries, 5)?.indices, results.indices);

    let mut ids = IdMapIndex::new(DIM, 2)?;
    let external_ids: Vec<u64> = (0..32).map(|row| 10_000 + row as u64).collect();
    ids.add_with_ids_2d(&vectors, DIM, &external_ids)?;
    let (_scores, found) = ids.search_checked(&queries, 5)?;
    ids.write(&demo_ids_path)?;
    let loaded_ids = IdMapIndex::load(&demo_ids_path)?;
    let (_loaded_scores, loaded_found) = loaded_ids.search_checked(&queries, 5)?;
    assert_eq!(loaded_found, found);

    println!("verified {}", demo_path.display());
    println!("verified {}", demo_ids_path.display());
    let _ = std::fs::remove_dir_all(&demo_root);
    Ok(())
}

fn create_demo_root() -> Result<PathBuf, Box<dyn Error>> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    for attempt in 0..100u32 {
        let root = std::env::temp_dir().join(format!(
            "ordinaldb_rust_persist_demo_{}_{}_{}",
            std::process::id(),
            now,
            attempt
        ));
        match std::fs::create_dir(&root) {
            Ok(()) => return Ok(root),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(Box::new(error)),
        }
    }
    Err("failed to create a unique demo directory".into())
}

fn demo_vectors(n: usize, dim: usize) -> Vec<f32> {
    let mut values = vec![0.0; n * dim];
    for row in 0..n {
        for col in 0..dim {
            let value = (((row + 3) * (col + 11) + row * 17 + col * 5) % 41) as f32 - 20.0;
            values[row * dim + col] = value / 21.0;
        }
    }
    values
}
