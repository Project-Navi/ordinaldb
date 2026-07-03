"""Thin helpers around ordinaldb.langchain.OrdinalDBVectorStore.

Kept deliberately small: the adapter's public surface (from_documents,
add_documents, similarity_search, delete, save_local/load_local) is already
the right shape for a retrieval app, so there is little to wrap. What lives
here is just the two operations DocsPilot needs beyond single calls: building
a fresh store from a documents+ids batch, and re-opening a persisted one.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from ordinaldb.langchain import OrdinalDBVectorStore

from docspilot.chunking import Chunk
from docspilot.embeddings import EMBEDDING_DIM, MiniLMEmbeddings

DEFAULT_BITS = 2


def build_store(
    store_path: Path,
    embedding: MiniLMEmbeddings,
    chunks: list[Chunk],
) -> OrdinalDBVectorStore:
    """Create a brand-new adapter store at `store_path` and ingest `chunks`."""
    documents: list[Document] = [chunk.document for chunk in chunks]
    ids = [chunk.id for chunk in chunks]
    return OrdinalDBVectorStore.from_documents(
        documents,
        embedding=embedding,
        ids=ids,
        path=store_path,
        dim=EMBEDDING_DIM,
        bits=DEFAULT_BITS,
    )


def open_store(store_path: Path, embedding: MiniLMEmbeddings) -> OrdinalDBVectorStore:
    """Re-open a persisted adapter store. Raises if `store_path` has no store."""
    return OrdinalDBVectorStore.load_local(store_path, embeddings=embedding)
