use ordinaldb::OrdinalIndex;
use std::env;
use std::fs;
use std::path::PathBuf;

const DIM: usize = 64;

fn unique_temp_dir() -> PathBuf {
    env::temp_dir().join(format!(
        "ordinaldb-bare-relative-write-{}",
        std::process::id()
    ))
}

// Regression test: `idx.write("bundle.odb")` with a bare relative name (no
// directory component) must succeed. `Path::new("bundle.odb").parent()`
// returns `Some("")`, and unguarded `sync_directory(parent)` calls fail with
// `NotFound` when handed the empty path.
//
// This lives in its own integration-test file so it runs in its own process:
// changing the process working directory cannot race other test binaries.
#[test]
fn write_and_reload_with_bare_relative_bundle_name() {
    let dir = unique_temp_dir();
    fs::create_dir_all(&dir).unwrap();
    let original_cwd = env::current_dir().unwrap();
    env::set_current_dir(&dir).unwrap();

    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let mut idx = OrdinalIndex::new(DIM, 2)?;
        let vectors: Vec<f32> = (0..3 * DIM).map(|i| (i % 17) as f32 * 0.25 - 2.0).collect();
        idx.add(&vectors);
        idx.write("bare_bundle.odb")?;
        let loaded = OrdinalIndex::load("bare_bundle.odb")?;
        let query: Vec<f32> = vectors[..DIM].to_vec();
        let _ = loaded.search(&query, 1);
        Ok(())
    })();

    env::set_current_dir(&original_cwd).unwrap();
    fs::remove_dir_all(&dir).ok();
    result.unwrap();
}
