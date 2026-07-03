"""Local-only LangChain adapter smoke for OrdinalDB.

Run with: python examples/python_adapters/langchain_edge_smoke.py
Requires: pip install 'ordinaldb[langchain]'
"""

from __future__ import annotations

import json
import math
import tempfile

from langchain_core.embeddings import Embeddings

from ordinaldb.langchain import OrdinalDBVectorStore


class LocalEmbeddings(Embeddings):
    axes = (
        ("edge", "local", "offline", "factory"),
        ("agent", "memory", "search", "rag"),
        ("cloud", "remote", "network", "hosted"),
        ("privacy", "security", "private"),
        ("sensor", "telemetry", "cache"),
        ("recipe", "kitchen", "bread"),
        ("persist", "load", "delete"),
        ("adapter", "metadata", "filter"),
    )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        lowered = text.lower()
        values = [float(sum(lowered.count(term) for term in axis)) for axis in self.axes]
        if not any(values):
            values[0] = 1.0
        norm = math.sqrt(sum(value * value for value in values))
        return [value / norm for value in values]


with tempfile.TemporaryDirectory() as tmp:
    embeddings = LocalEmbeddings()
    store = OrdinalDBVectorStore(embedding=embeddings, dim=8, bits=2)
    store.add_texts(
        [
            "Edge agent stores private local memory",
            "Cloud RAG service uses remote network embeddings",
            "Factory laptop keeps telemetry cache offline",
        ],
        metadatas=[
            {"tier": "edge", "kind": "agent"},
            {"tier": "cloud", "kind": "rag"},
            {"tier": "edge", "kind": "telemetry"},
        ],
        ids=["edge-agent", "cloud-rag", "factory-cache"],
    )

    hits = store.similarity_search("local edge agent memory", k=2)
    filtered = store.similarity_search("remote network", k=3, filter={"tier": "edge"})
    assert [doc.id for doc in hits][0] == "edge-agent"
    assert all(doc.metadata["tier"] == "edge" for doc in filtered)
    assert store.get_by_ids(["missing"]) == []
    store.persist(tmp)
    loaded = OrdinalDBVectorStore.load_local(tmp, embeddings=embeddings)
    loaded.delete(["cloud-rag", "missing"])
    assert loaded.get_by_ids(["cloud-rag"]) == []

    print(json.dumps({"hits": [doc.id for doc in hits], "filtered": [doc.id for doc in filtered]}))
