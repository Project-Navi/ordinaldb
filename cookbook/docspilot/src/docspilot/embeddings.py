"""LangChain Embeddings wrapper around sentence-transformers, CPU-only.

We hand-roll this instead of pulling in langchain-huggingface because the
task only asked for sentence-transformers + langchain-core, and the
Embeddings interface is two methods -- adding a whole extra package for that
would be the kind of unnecessary dependency the project's rules warn against.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class MiniLMEmbeddings(Embeddings):
    """all-MiniLM-L6-v2 on CPU, exposed as a LangChain Embeddings implementation."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model = SentenceTransformer(model_name, device="cpu")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(
            [text],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vector[0].tolist()
