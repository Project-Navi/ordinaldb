//! Embedding backends. `FastEmbedEmbedder` is the real semantic embedder the
//! root README recommends pairing with OrdinalDB (`fastembed`,
//! `AllMiniLML6V2`, 384-dim, CPU-only). `HashEmbedder` is a deterministic,
//! non-semantic fallback kept only so the crate documents what happens if
//! `fastembed`'s ONNX runtime cannot be used in a given environment -- it
//! produces a vector from a seeded PRNG over the text's hash, the same
//! pattern the root README's Python adapter snippets use as a
//! network-free placeholder. It has no notion of meaning: swapping it in
//! turns query class 2 (the paraphrase query) into a demonstration of dense
//! search *failing* rather than succeeding, which is called out explicitly
//! wherever this embedder is selected.
//!
//! In this run, `fastembed` downloaded its ~90MB ONNX model from Hugging
//! Face and produced embeddings in a few seconds, well inside the 20-minute
//! budget, so `HashEmbedder` is not needed for the actual demo -- it is
//! included so the fallback path is real code, not just a claim in a
//! comment.

pub const DIM: usize = 384;

pub trait Embedder {
    /// Human-readable name for the printed demo output.
    fn name(&self) -> &'static str;
    fn embed_batch(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>>;
    fn embed_one(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        Ok(self.embed_batch(&[text])?.remove(0))
    }
}

pub struct FastEmbedEmbedder {
    model: std::sync::Mutex<fastembed::TextEmbedding>,
}

impl FastEmbedEmbedder {
    pub fn try_new() -> anyhow::Result<Self> {
        use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::AllMiniLML6V2).with_show_download_progress(true),
        )?;
        Ok(Self {
            model: std::sync::Mutex::new(model),
        })
    }
}

impl Embedder for FastEmbedEmbedder {
    fn name(&self) -> &'static str {
        "fastembed/AllMiniLML6V2 (384-dim, CPU)"
    }

    fn embed_batch(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        let model = self.model.lock().expect("fastembed model mutex poisoned");
        let owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let embeddings = model.embed(owned, None)?;
        Ok(embeddings)
    }
}

/// Deterministic, non-semantic fallback embedder (see module docs).
pub struct HashEmbedder;

impl Embedder for HashEmbedder {
    fn name(&self) -> &'static str {
        "hash-embedder (deterministic, NON-semantic fallback)"
    }

    fn embed_batch(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| hash_embed(t)).collect())
    }
}

fn hash_embed(text: &str) -> Vec<f32> {
    let seed = fnv1a64(text.as_bytes());
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut out = Vec::with_capacity(DIM);
    for _ in 0..DIM {
        state = splitmix64(state);
        // Map to [-1, 1].
        let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
        out.push((unit * 2.0 - 1.0) as f32);
    }
    l2_normalize(&mut out);
    out
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

pub fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}
