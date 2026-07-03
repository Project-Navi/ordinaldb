use std::error::Error;

use ordinaldb::{IdMapIndex, OrdinalIndex};

const DIM: usize = 64;

fn main() -> Result<(), Box<dyn Error>> {
    let vectors = demo_vectors(16, DIM);
    let queries = demo_vectors(2, DIM);

    let mut idx = OrdinalIndex::new(DIM, 2)?;
    idx.add_2d(&vectors, DIM)?;
    let results = idx.search_checked(&queries, 4)?;
    assert_eq!(results.nq, 2);
    assert_eq!(results.k, 4);

    let positional_path = std::env::temp_dir().join("ordinaldb-rust-basic.odb");
    let _ = std::fs::remove_dir_all(&positional_path);
    idx.write(&positional_path)?;
    let loaded = OrdinalIndex::load(&positional_path)?;
    assert_eq!(loaded.search_checked(&queries, 4)?.indices, results.indices);

    let mut ids = IdMapIndex::new(DIM, 2)?;
    let external_ids: Vec<u64> = (0..16).map(|row| 10_000 + row as u64).collect();
    ids.add_with_ids_2d(&vectors, DIM, &external_ids)?;
    let (_scores, found) =
        ids.search_checked_with_allowlist(&queries, 3, Some(&[10_001, 10_004, 10_007]))?;
    assert!(found.iter().all(|id| [10_001, 10_004, 10_007].contains(id)));
    ids.remove(10_004);

    let id_path = std::env::temp_dir().join("ordinaldb-rust-basic-ids.odb");
    let _ = std::fs::remove_dir_all(&id_path);
    ids.write(&id_path)?;
    let loaded_ids = IdMapIndex::load(&id_path)?;
    assert!(!loaded_ids.contains(10_004));

    let _ = std::fs::remove_dir_all(positional_path);
    let _ = std::fs::remove_dir_all(id_path);
    Ok(())
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
