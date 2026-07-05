# Matryoshka (MRL) Embeddings

Matryoshka Representation Learning (MRL) trains embedding models so that a
prefix of the full vector is itself a usable lower-dimensional embedding.
Modern models (Qwen3-Embedding, Nomic, OpenAI text-embedding-3, and others)
ship with MRL, letting you trade retrieval quality for index size and speed
by truncating vectors at index time.

## No special handling required — including no renormalization

OrdinalDB's codes are rank- and sign-based: `RankQuant` depends only on the
ordering of coordinate values within each vector, and the `SignBitmap`
candidate stage only on their signs. Both are invariant to per-vector
positive scaling, so truncating an MRL embedding changes its norm but not
the ranks or signs of the retained prefix.

The practical consequence: **truncate and index — no renormalization step.**
Stores that score with cosine or dot products require truncated MRL vectors
to be re-normalized or their scores skew; OrdinalDB structurally cannot get
this wrong.

L2 renormalization is exactly a per-vector positive scaling, so it leaves
every rank and sign untouched. OrdinalDB therefore returns the identical
candidate set and ranking whether or not you renormalize a truncated MRL
prefix — the step is a no-op here by construction, not by tuning.

## Choosing a truncation dimension

- **Prefer multiples of 64.** `bits=2` indexes with `dim % 64 == 0` get the
  `SignBitmap` two-stage fast path. Common MRL truncation points (64, 128,
  256, 512, 768, 1024, 2048) and the native Qwen3 dims (1024, 2560, 4096)
  all qualify.
- Any dimension divisible by 4 works for `bits=2` (for example 300); the
  index simply runs without the sign-stage sidecar.
- The ceiling is `u16` (65,535); out-of-range or bits-incompatible
  dimensions are rejected with a descriptive error at construction.

## What truncation costs

Two costs stack when you truncate an MRL embedding and index it:

- **MRL truncation itself** — scoring a shorter prefix instead of the full
  vector. Every vector store pays this identically; it is a property of the
  embedding model, not of OrdinalDB.
- **Ordinal quantization** — OrdinalDB's rank/sign coding applied on top of
  the truncated prefix.

Both degrade gracefully as the dimension shrinks rather than falling off a
cliff, and on-disk size stays roughly 10x smaller than raw `f32` at every
truncation point. The right operating point is corpus- and model-specific:
measure recall at your own dimensions and corpus size before committing to a
truncation dimension.
