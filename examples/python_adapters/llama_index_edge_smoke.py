"""Local-only LlamaIndex adapter smoke for OrdinalDB."""

from __future__ import annotations

import json
import tempfile

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

from ordinaldb.llama_index import OrdinalDBVectorStore


def embed(text: str) -> list[float]:
    axes = (
        ("edge", "local", "offline"),
        ("agent", "workflow", "tool"),
        ("privacy", "egress", "network"),
        ("device", "pi", "thermal"),
        ("document", "metadata", "filter"),
        ("cloud", "remote", "hosted"),
        ("persist", "reload", "delete"),
        ("smoke", "prototype", "runbook"),
    )
    lowered = text.lower()
    values = [float(sum(lowered.count(term) for term in axis)) for axis in axes]
    if not any(values):
        values[0] = 1.0
    return values


with tempfile.TemporaryDirectory() as tmp:
    store = OrdinalDBVectorStore(dim=8, bits=2)
    nodes = [
        TextNode(id_="edge-plan", text="Edge agent workflow stays local", metadata={"topic": "plan", "priority": 5}, embedding=embed("edge local agent workflow")),
        TextNode(id_="privacy", text="Privacy policy blocks network egress", metadata={"topic": "policy", "priority": 4}, embedding=embed("privacy network egress")),
        TextNode(id_="device", text="Pi thermal runbook for local device", metadata={"topic": "device", "priority": 2}, embedding=embed("pi device thermal")),
    ]
    store.add(nodes)
    result = store.query(VectorStoreQuery(query_embedding=embed("local edge workflow"), similarity_top_k=2))
    filtered = store.query(
        VectorStoreQuery(
            query_embedding=embed("network policy"),
            similarity_top_k=3,
            filters=MetadataFilters(filters=[MetadataFilter(key="topic", value="policy", operator=FilterOperator.EQ)]),
        )
    )
    assert "edge-plan" in (result.ids or [])
    assert filtered.ids == ["privacy"]
    assert store.get_nodes(["missing"]) == []
    store.persist(tmp)
    loaded = OrdinalDBVectorStore(path=tmp)
    loaded.delete_nodes(node_ids=["device", "missing"])
    assert loaded.get_nodes(["device"]) == []
    print(json.dumps({"hits": result.ids, "filtered": filtered.ids}))
