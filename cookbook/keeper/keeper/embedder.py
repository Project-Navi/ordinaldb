"""Local CPU embedder for Keeper.

Wraps sentence-transformers/all-MiniLM-L6-v2 behind the duck-typed interface
the OrdinalDB Agno adapter looks for (``get_embedding``). Loaded once per
process and reused for every embed call.
"""

from __future__ import annotations

from functools import lru_cache

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384


class LocalEmbedder:
    """CPU sentence-transformers embedder, agno-adapter compatible."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device="cpu")

    def get_embedding(self, text: str) -> list[float]:
        vector = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vector.astype("float32").tolist()

    def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return [v.astype("float32").tolist() for v in vectors]


@lru_cache(maxsize=1)
def shared_embedder() -> LocalEmbedder:
    """Process-local singleton so repeated calls in one process reuse the model."""
    return LocalEmbedder()
