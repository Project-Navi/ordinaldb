#!/usr/bin/env python
"""Keeper demo: a 5-session agent lifecycle against one durable OrdinalDB store.

Each "session" below is launched as its own OS subprocess (session_runner.py)
against the SAME on-disk store directory, so this is a real test of
cross-process durability, not an in-memory simulation. The narrative: an AI
coding assistant remembers facts/preferences/events about a user and project
across 5 separate runs, updates a stale preference, and recalls the current
state at the end.

Run: python demo.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent
STORE = ROOT / "store" / "demo_lifecycle"
RUNNER = ROOT / "session_runner.py"


def run_session(script: dict) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(script, f)
        script_path = f.name
    proc = subprocess.run(
        [sys.executable, str(RUNNER), str(STORE), script_path],
        capture_output=True,
        text=True,
    )
    Path(script_path).unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(f"session {script['session_id']} failed:\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    shutil.rmtree(STORE, ignore_errors=True)

    # ---- Session 1: onboarding ------------------------------------------
    banner("SESSION 1 (pid separate from this script) -- onboarding")
    s1 = run_session(
        {
            "session_id": "session-1",
            "remember": [
                {"text": "The user's name is Alex Rivera.", "kind": "fact"},
                {"text": "The project is called Aurora, a Rust CLI tool.", "kind": "fact"},
                {
                    "text": "The user prefers terse commit messages under 50 characters.",
                    "kind": "preference",
                },
            ],
        }
    )
    print(f"pid={s1['pid']}  open={s1['open_seconds']:.2f}s  save={s1['save_seconds']:.2f}s")
    print(f"wrote {len(s1['written_ids'])} memories: {s1['written_ids']}")
    stale_pref_id = s1["written_ids"][2]  # the terse-commit-message preference

    # ---- Session 2: next day, preference changes -------------------------
    banner("SESSION 2 -- recalls session 1, supersedes a preference")
    s2 = run_session(
        {
            "session_id": "session-2",
            "recall_first": {
                "query": "what commit message style does the user want?",
                "k": 2,
            },
            "remember": [
                {
                    "text": "On 2026-06-28 the user asked to switch from cargo test to cargo nextest.",
                    "kind": "event",
                },
                {
                    "text": "The user now prefers verbose commit messages with a body explaining why.",
                    "kind": "preference",
                },
            ],
            "forget_ids": [stale_pref_id],
        }
    )
    print(f"pid={s2['pid']}  open={s2['open_seconds']:.2f}s")
    print("recalled before writing (should surface the OLD terse-commit preference):")
    for r in s2["recall_before"]:
        print(f"  [{r['kind']:10s}] {r['text']}")
    print(f"forgot stale preference {stale_pref_id}: {s2['forgotten']}")
    fresh_pref_id = s2["written_ids"][1]

    # ---- Session 3: verify the stale preference is gone -------------------
    banner("SESSION 3 -- verifies deleted memory no longer surfaces")
    s3 = run_session(
        {
            "session_id": "session-3",
            "recall_first": {
                "query": "what commit message style does the user want?",
                "k": 3,
            },
            "remember": [
                {"text": "The user's timezone is America/Chicago.", "kind": "fact"},
                {"text": "On 2026-06-29 the user shipped v1.2.0 of Aurora.", "kind": "event"},
            ],
        }
    )
    print(f"pid={s3['pid']}  open={s3['open_seconds']:.2f}s")
    print("recalled (should show the NEW verbose-commit preference, NOT the deleted one):")
    for r in s3["recall_before"]:
        print(f"  [{r['kind']:10s}] {r['text']}")
    recalled_ids = {r["id"] for r in s3["recall_before"]}
    assert stale_pref_id not in recalled_ids, "DELETED memory resurfaced -- durability bug"
    assert fresh_pref_id in recalled_ids, "fresh preference failed to surface"
    print("ASSERTION PASSED: deleted memory did not resurface; fresh preference did.")

    # ---- Session 4: CI incident ------------------------------------------
    banner("SESSION 4 -- new event, broad status recall")
    s4 = run_session(
        {
            "session_id": "session-4",
            "recall_first": {"query": "give me a project status update", "k": 5},
            "remember": [
                {
                    "text": "On 2026-06-30 CI started failing on ARM64 runners due to a flaky test.",
                    "kind": "event",
                },
            ],
        }
    )
    print(f"pid={s4['pid']}  open={s4['open_seconds']:.2f}s")
    print("broad status recall:")
    for r in s4["recall_before"]:
        print(f"  [{r['kind']:10s}] {r['text']}")

    # ---- Session 5: final recall-quality check -----------------------------
    banner("SESSION 5 -- final recall quality + store size check")
    s5 = run_session(
        {
            "session_id": "session-5",
            "recall_first": {
                "query": "what do you know about the user and the project?",
                "k": 10,
            },
        }
    )
    print(f"pid={s5['pid']}  open={s5['open_seconds']:.2f}s")
    print(f"final store size: {s5['store_size_after']} memories")
    print("full recall (all live memories, ranked by relevance to a broad query):")
    for r in s5["recall_before"]:
        print(f"  [{r['kind']:10s}][{r['session_id']}] {r['text']}")

    expected_size = 3 + 2 - 1 + 2 + 1  # session1 - forgotten in session2 + session3 + session4
    assert s5["store_size_after"] == expected_size, (
        f"expected {expected_size} live memories, got {s5['store_size_after']}"
    )
    print(f"\nASSERTION PASSED: store size == {expected_size} across 5 separate-process sessions.")
    print("\nDEMO COMPLETE -- durability across 5 subprocess sessions verified.")


if __name__ == "__main__":
    main()
