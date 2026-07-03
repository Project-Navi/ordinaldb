"""Local-only Agno adapter smoke for OrdinalDB."""

from __future__ import annotations

import json
import tempfile

from agno.knowledge.document import Document

from ordinaldb.agno import OrdinalDb


class LocalEmbedder:
    def get_embedding(self, text: str) -> list[float]:
        lowered = text.lower()
        if "beta" in lowered:
            return [0.0, 1.0, 0.0, 0.0]
        if "gamma" in lowered:
            return [0.0, 0.0, 1.0, 0.0]
        return [1.0, 0.0, 0.0, 0.0]


with tempfile.TemporaryDirectory() as tmp:
    db = OrdinalDb(path=tmp, embedder=LocalEmbedder(), dim=4, bits=2, auto_save=True)
    db.create()
    assert db.exists()
    db.insert(
        "content-alpha",
        [
            Document(id="alpha", content="alpha local note", name="alpha-note", meta_data={"stage": "draft"}),
            Document(id="beta", content="beta ready checklist", name="beta-note", meta_data={"stage": "ready"}),
        ],
        embeddings=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    )
    assert [doc.id for doc in db.search("find beta", filters={"stage": "ready"})] == ["beta"]
    db.upsert(
        "content-alpha",
        [Document(id="alpha-v2", content="alpha published note", meta_data={"stage": "published"})],
        embeddings=[[1.0, 0.0, 0.0, 0.0]],
    )
    assert not db.id_exists("alpha")
    loaded = OrdinalDb.load(tmp, embedder=LocalEmbedder())
    assert [doc.id for doc in loaded.search_by_vector([1.0, 0.0, 0.0, 0.0], filters={"stage": "published"})] == ["alpha-v2"]
    print(json.dumps({"remaining": [doc.id for doc in loaded.search_by_vector([1.0, 0.0, 0.0, 0.0], limit=2)]}))
