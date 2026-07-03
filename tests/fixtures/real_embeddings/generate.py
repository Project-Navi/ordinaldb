#!/usr/bin/env python3
"""Regenerate the real-embedding test fixtures.

These fixtures exist because synthetic vectors systematically miss real
bugs: uniformly random floats never exercised the float-canonicalization
path that real all-MiniLM-L6-v2 embeddings corrupt, so the test suite
stayed green while real users hit data-integrity failures. Tests must use
real embedding bit-patterns wherever a vector crosses a serialization or
storage boundary.

Regeneration (only needed if texts.json changes; outputs are committed):

    python tests/fixtures/real_embeddings/generate.py

Requires sentence-transformers with the pinned model + revision below. The
committed artifacts are the source of truth; this script documents their
provenance.

Outputs (all little-endian float32, row-major):
    minilm_docs_f32.bin     -- N_docs x 384
    minilm_queries_f32.bin  -- N_queries x 384
    adversarial_floats.json -- real embedding values with pathological
                               f64 shortest-round-trip representations,
                               plus curated IEEE-754 edge cases
    manifest.json           -- model id, versions, shapes, sha256 digests
"""

import hashlib
import json
import sys
from pathlib import Path

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# Immutable commit hash on the model repo, not a mutable branch/tag: pins
# regeneration to the exact weights that produced the committed fixtures so
# an unrelated upstream model update can't silently change embeddings.
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DIM = 384

HERE = Path(__file__).resolve().parent


def main() -> None:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    corpus = json.loads((HERE / "texts.json").read_text())
    doc_texts = [d["text"] for d in corpus["documents"]]
    query_texts = [q["text"] for q in corpus["queries"]]

    model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device="cpu")
    docs = model.encode(doc_texts, convert_to_numpy=True).astype(np.float32)
    queries = model.encode(query_texts, convert_to_numpy=True).astype(np.float32)
    assert docs.shape == (len(doc_texts), DIM), docs.shape
    assert queries.shape == (len(query_texts), DIM), queries.shape

    (HERE / "minilm_docs_f32.bin").write_bytes(docs.tobytes(order="C"))
    (HERE / "minilm_queries_f32.bin").write_bytes(queries.tobytes(order="C"))

    # Real embedding values whose float32 -> float64 promotion needs a long
    # shortest-round-trip decimal representation (the exact class of value
    # that exposed the metadata canonicalization bug), harvested from the
    # actual embeddings rather than invented.
    promoted = docs.astype(np.float64).ravel()
    reprs = [(abs(v), len(repr(float(v))), float(v)) for v in promoted[:4096]]
    longest = sorted(reprs, key=lambda t: -t[1])[:64]
    adversarial = {
        "from_real_embeddings_f32_promoted": [v for _, _, v in longest],
        "curated_ieee754": [
            0.10000000149011612,  # 0.1f32 promoted to f64
            -0.0,
            5e-324,               # smallest positive subnormal f64
            2.2250738585072014e-308,  # smallest positive normal f64
            1.7976931348623157e308,   # f64::MAX
            9.999999999999999e20,     # below serde exponent threshold
            1e21,                     # at/above exponent-notation threshold
            0.1 + 0.2,                # 0.30000000000000004
            1.0000000000000002,       # 1.0 + f64::EPSILON
            -1.5e-45,                 # f32 subnormal magnitude, promoted
        ],
    }
    adversarial_json = json.dumps(adversarial, indent=1) + "\n"
    (HERE / "adversarial_floats.json").write_text(adversarial_json)

    import sentence_transformers
    import torch

    manifest = {
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dim": DIM,
        "dtype": "float32 little-endian, row-major",
        "documents": len(doc_texts),
        "queries": len(query_texts),
        "generator_versions": {
            "python": sys.version.split()[0],
            "sentence_transformers": sentence_transformers.__version__,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "sha256": {
            "minilm_docs_f32.bin": hashlib.sha256(docs.tobytes()).hexdigest(),
            "minilm_queries_f32.bin": hashlib.sha256(queries.tobytes()).hexdigest(),
            "texts.json": hashlib.sha256(
                (HERE / "texts.json").read_bytes()
            ).hexdigest(),
            "adversarial_floats.json": hashlib.sha256(
                adversarial_json.encode("utf-8")
            ).hexdigest(),
        },
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {len(doc_texts)}x{DIM} docs, {len(query_texts)}x{DIM} queries")
    print(json.dumps(manifest["sha256"], indent=1))


if __name__ == "__main__":
    main()
