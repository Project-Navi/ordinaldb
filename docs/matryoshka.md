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

Measured (200 documents, 8B-parameter Qwen3 embeddings at 512 dims, raw
truncation vs. truncation + L2 renormalization): retrieved indices and
ranking were identical across all 220 probe queries; score differences were
at the level of one float32 ULP (arithmetic noise from the normalization
division itself).

## Choosing a truncation dimension

- **Prefer multiples of 64.** `bits=2` indexes with `dim % 64 == 0` get the
  `SignBitmap` two-stage fast path. Common MRL truncation points (64, 128,
  256, 512, 768, 1024, 2048) and the native Qwen3 dims (1024, 2560, 4096)
  all qualify.
- Any dimension divisible by 4 works for `bits=2` (for example 300); the
  index simply runs without the sign-stage sidecar.
- The ceiling is `u16` (65,535); out-of-range or bits-incompatible
  dimensions are rejected with a descriptive error at construction.

## What truncation costs (measured)

Recall@10 of OrdinalDB search against exact float dot-product over the same
truncated prefix isolates the ordinal-quantization cost; against the
full-dimension exact ranking it shows the total cost including MRL
truncation itself (which any store pays):

| dim | vs. prefix-exact | vs. full-exact |
| --- | --- | --- |
| 4096 | 0.895 | 0.895 |
| 2048 | 0.885 | 0.830 |
| 1024 | 0.845 | 0.800 |
| 512 | 0.805 | 0.720 |
| 256 | 0.785 | 0.655 |
| 128 | 0.740 | 0.585 |
| 64 | 0.640 | 0.495 |

Quantization loss degrades gracefully down the ladder; the steeper
full-exact column is MRL truncation's own cost and is identical for any
vector store. On-disk compression held at roughly 10x vs. raw `f32` at
every dimension, and single-query latency stayed in the tens of
microseconds at this corpus size (p50 31.5 µs at 4096 dims).

Method notes: 200 documents / 20 labeled queries, real Qwen3-Embedding-8B
(GGUF Q4_K_M) vectors generated locally via Ollama, single workstation,
`bits=2` throughout. Small-corpus numbers are for shape, not absolute
benchmark claims; rerun at your corpus size before relying on a specific
operating point.
