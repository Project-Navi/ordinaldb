"""Keeper's durable memory store.

Thin wrapper around ``ordinaldb.agno.OrdinalDb`` that speaks in terms of
``Memory`` records instead of raw Agno ``Document``/embedding lists. This is
the only place Keeper touches the OrdinalDB adapter surface.
"""

from __future__ import annotations

from pathlib import Path

from ordinaldb.agno import OrdinalDb

from .embedder import EMBED_DIM, LocalEmbedder, shared_embedder
from .memory import Memory, Recalled


class KeeperStore:
    """Durable long-term memory for one agent, backed by an OrdinalDB adapter store."""

    def __init__(
        self,
        path: str | Path,
        embedder: LocalEmbedder | None = None,
        dim: int = EMBED_DIM,
        bits: int = 2,
        auto_save: bool = False,
    ) -> None:
        self.path = str(path)
        self.embedder = embedder or shared_embedder()
        self._db = OrdinalDb(
            path=self.path,
            embedder=self.embedder,
            dim=dim,
            bits=bits,
            auto_save=auto_save,
        )

    def remember(self, memory: Memory) -> str:
        """Embed and store one memory. Returns the stored memory id."""
        document = {
            "id": memory.memory_id,
            "content": memory.text,
            "meta_data": memory.to_metadata(),
        }
        vector = self.embedder.get_embedding(memory.text)
        written_ids = self._db.insert(memory.memory_id, [document], embeddings=[vector])
        return written_ids[0]

    def remember_many(self, memories: list[Memory]) -> list[str]:
        if not memories:
            return []
        documents = [
            {"id": m.memory_id, "content": m.text, "meta_data": m.to_metadata()}
            for m in memories
        ]
        vectors = self.embedder.get_embeddings_batch([m.text for m in memories])
        return self._db.insert(memories[0].memory_id, documents, embeddings=vectors)

    def recall(
        self,
        query: str,
        k: int = 5,
        session_id: str | None = None,
        kind: str | None = None,
    ) -> list[Recalled]:
        filters: dict[str, str] = {}
        if session_id is not None:
            filters["session_id"] = session_id
        if kind is not None:
            filters["kind"] = kind
        docs = self._db.search(query, limit=k, filters=filters or None)
        return [
            Recalled(
                memory_id=doc.id,
                text=doc.content,
                session_id=doc.meta_data.get("session_id", ""),
                kind=doc.meta_data.get("kind", ""),
                timestamp=doc.meta_data.get("timestamp", 0.0),
                score=doc.reranking_score,
            )
            for doc in docs
        ]

    def forget(self, memory_id: str) -> bool:
        return self._db.delete_by_id(memory_id)

    def forget_by_metadata(self, metadata: dict[str, str]) -> bool:
        return self._db.delete_by_metadata(metadata)

    def id_exists(self, memory_id: str) -> bool:
        return self._db.id_exists(memory_id)

    def exists(self) -> bool:
        return self._db.exists()

    def __len__(self) -> int:
        return self._db.get_count()

    def save(self) -> None:
        self._db.save(self.path)
