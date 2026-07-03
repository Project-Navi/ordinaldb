"""Core PaperScout library: embeddings, indexing, and LlamaIndex wiring.

Keeps demo.py focused on the narrative (build -> query -> filter -> persist
-> reload) by putting the reusable plumbing here.
"""

from __future__ import annotations

from pathlib import Path

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.llms import MockLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from ordinaldb.llama_index import OrdinalDBVectorStore

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384
BITS = 2


def configure_local_settings() -> HuggingFaceEmbedding:
    """Wire local, API-key-free defaults into LlamaIndex's global Settings.

    sentence-transformers/all-MiniLM-L6-v2 is used as the embed_model. A
    MockLLM stands in for Settings.llm: as_query_engine() resolves a
    default OpenAI LLM even for response_mode="no_text" retrieval-only
    engines, and raises ImportError if llama-index-llms-openai isn't
    installed. MockLLM (ships in llama-index-core, no extra package, no
    API key) avoids that -- a LlamaIndex quirk, not an OrdinalDB one, but
    the first wall a no-API-key user hits on the idiomatic query-engine
    path.
    """
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    Settings.embed_model = embed_model
    Settings.llm = MockLLM()
    return embed_model


def paper_to_document(paper: dict) -> Document:
    """Convert a paper metadata dict into a LlamaIndex Document.

    No manual `ref_doc_id` stamping needed: `OrdinalDBVectorStore.add()`
    stamps `ref_doc_id`/`doc_id`/`document_id` into stored metadata itself,
    mirroring LlamaIndex's own `node_to_metadata_dict()` convention -- the
    same thing `delete_ref_doc()` and metadata filters look for.
    """
    text = f"{paper['title']}. {paper['abstract']}"
    metadata = {
        "category": paper["category"],
        "year": paper["year"],
        "authors_count": paper["authors_count"],
        "source": paper["source"],
        "title": paper["title"],
    }
    return Document(text=text, metadata=metadata, doc_id=paper["id"])


def build_documents(papers: list[dict]) -> list[Document]:
    return [paper_to_document(paper) for paper in papers]


def open_vector_store(path: str | Path) -> OrdinalDBVectorStore:
    """Open (or lazily create) an OrdinalDB-backed LlamaIndex vector store.

    Bad paths fail closed instead of silently starting an empty store: an
    existing plain file raises `AdapterStoreError` at construction, and a
    directory with unrelated stray content emits `AdapterPathWarning`
    before starting fresh.
    """
    return OrdinalDBVectorStore(path=str(path), dim=EMBED_DIM, bits=BITS)


def build_index(
    documents: list[Document],
    vector_store: OrdinalDBVectorStore,
    embed_model: HuggingFaceEmbedding,
) -> VectorStoreIndex:
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
    )


def load_index_from_store(
    vector_store: OrdinalDBVectorStore,
    embed_model: HuggingFaceEmbedding,
) -> VectorStoreIndex:
    """Rebuild a VectorStoreIndex handle around an already-populated store.

    This is the reload path: no documents are re-added, we just point a
    fresh VectorStoreIndex at the existing OrdinalDB-backed vector store.
    """
    return VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
