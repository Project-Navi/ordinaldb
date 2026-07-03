#!/usr/bin/env python
"""Run one simulated agent session as its own OS process against a Keeper store.

Real-world shape: an agent process starts, opens its durable memory store,
recalls relevant context from prior sessions, does some work (storing new
facts/preferences/events and retiring outdated ones), then persists and exits.
Each invocation of this script IS one session/process -- the multi-session
lifecycle trial drives this via subprocess.run(), not in-process function
calls, so session N+1 has no Python state left over from session N.

Usage:
    python session_runner.py <store_path> <script.json>

script.json shape:
{
  "session_id": "session-3",
  "recall_first": {"query": "...", "k": 5},
  "remember": [{"text": "...", "kind": "fact"}, ...],
  "forget_ids": ["<memory_id>", ...]
}

Prints one JSON object to stdout: recalled-before, ids written, ids forgotten,
recalled-after, and store size after the session.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from keeper import KeeperStore, Memory  # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: session_runner.py <store_path> <script.json>", file=sys.stderr)
        return 2

    store_path, script_path = sys.argv[1], sys.argv[2]
    script = json.loads(Path(script_path).read_text())
    session_id = script["session_id"]

    t_open = time.time()
    store = KeeperStore(store_path)
    open_s = time.time() - t_open

    result: dict = {"session_id": session_id, "pid": None, "open_seconds": open_s}
    import os

    result["pid"] = os.getpid()

    recall_before = []
    if script.get("recall_first"):
        spec = script["recall_first"]
        t = time.time()
        hits = store.recall(spec["query"], k=spec.get("k", 5))
        recall_before = [
            {"id": h.memory_id, "text": h.text, "kind": h.kind, "session_id": h.session_id}
            for h in hits
        ]
        result["recall_before_seconds"] = time.time() - t
    result["recall_before"] = recall_before

    written_ids: list[str] = []
    for item in script.get("remember", []):
        mem = Memory(text=item["text"], session_id=session_id, kind=item["kind"])
        written_ids.append(store.remember(mem))
    result["written_ids"] = written_ids

    forgotten = []
    for mem_id in script.get("forget_ids", []):
        changed = store.forget(mem_id)
        forgotten.append({"id": mem_id, "changed": changed})
    result["forgotten"] = forgotten

    recall_after = []
    if script.get("recall_last"):
        spec = script["recall_last"]
        hits = store.recall(spec["query"], k=spec.get("k", 5))
        recall_after = [
            {"id": h.memory_id, "text": h.text, "kind": h.kind, "session_id": h.session_id}
            for h in hits
        ]
    result["recall_after"] = recall_after

    t_save = time.time()
    store.save()
    result["save_seconds"] = time.time() - t_save
    result["store_size_after"] = len(store)

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
