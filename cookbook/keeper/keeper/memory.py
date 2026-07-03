"""Memory record shape for Keeper.

A Memory is the unit an agent session stores and recalls: free text plus the
metadata an SRE-grade memory store needs to reason about staleness and
provenance (session_id, kind, timestamp).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Literal

MemoryKind = Literal["fact", "preference", "event"]


@dataclass(frozen=True)
class Memory:
    text: str
    session_id: str
    kind: MemoryKind
    timestamp: float = field(default_factory=time.time)
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_metadata(self) -> dict[str, str | float]:
        return {
            "session_id": self.session_id,
            "kind": self.kind,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class Recalled:
    memory_id: str
    text: str
    session_id: str
    kind: str
    timestamp: float
    score: float | None
