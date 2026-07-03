"""Local-only Haystack adapter smoke for OrdinalDB."""

from __future__ import annotations

import json
import math
import os
import tempfile

from haystack import Document
from haystack.document_stores.types import DuplicatePolicy

from ordinaldb.haystack import OrdinalDocumentStore, OrdinalEmbeddingRetriever

os.environ.setdefault("HAYSTACK_TELEMETRY_ENABLED", "False")
os.environ.setdefault("HAYSTACK_DISABLE_TELEMETRY", "1")
os.environ.setdefault("POSTHOG_DISABLED", "1")


def embed(text: str) -> list[float]:
    axes = (
        ("edge", "local", "agent"),
        ("haystack", "retriever", "document"),
        ("latency", "cache", "runtime"),
        ("privacy", "telemetry", "egress"),
    )
    lowered = text.lower()
    values = [float(sum(lowered.count(term) for term in axis)) for axis in axes]
    if not any(values):
        values[0] = 1.0
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


with tempfile.TemporaryDirectory() as tmp:
    store = OrdinalDocumentStore(dim=4, bits=2)
    docs = [
        Document(id="edge-design", content="Local edge agent memory", meta={"status": "approved"}, embedding=embed("local edge agent")),
        Document(id="telemetry-risk", content="Disable telemetry egress", meta={"status": "approved"}, embedding=embed("privacy telemetry egress")),
        Document(id="cloud-base", content="Remote cloud baseline", meta={"status": "rejected"}, embedding=embed("remote cloud")),
    ]
    assert store.write_documents(docs, policy=DuplicatePolicy.FAIL) == 3
    assert store.write_documents(docs[:1], policy=DuplicatePolicy.SKIP) == 0
    retriever = OrdinalEmbeddingRetriever(document_store=store, top_k=2)
    result = retriever.run(query_embedding=embed("privacy telemetry"), filters={"field": "meta.status", "operator": "==", "value": "approved"})
    assert all(doc.meta["status"] == "approved" for doc in result["documents"])
    store.save(tmp)
    loaded = OrdinalDocumentStore.load(tmp)
    loaded.delete_documents(["cloud-base", "missing"])
    assert [doc.id for doc in loaded.filter_documents({"field": "meta.status", "operator": "==", "value": "rejected"})] == []
    print(json.dumps({"hits": [doc.id for doc in result["documents"]]}))
