#!/usr/bin/env python
"""Standalone re-open check, run as a genuinely separate process.

demo.py shells out to this script (a fresh `python` invocation, fresh
interpreter, fresh imports -- no shared state) to prove the persisted
OrdinalDB adapter store survives a process restart. Usage:

    python reopen_check.py <store_path> <query>

Prints a single JSON line: {"hits": [{"source": ..., "section": ..., "score": ...}, ...]}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from docspilot.embeddings import MiniLMEmbeddings  # noqa: E402
from docspilot.store import open_store  # noqa: E402


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: reopen_check.py <store_path> <query>", file=sys.stderr)
        raise SystemExit(2)
    store_path = Path(sys.argv[1])
    query = sys.argv[2]

    embedding = MiniLMEmbeddings()
    store = open_store(store_path, embedding)
    results = store.similarity_search_with_score(query, k=3)
    payload = {
        "hits": [
            {"source": doc.metadata.get("source"), "section": doc.metadata.get("section"), "score": score}
            for doc, score in results
        ]
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
