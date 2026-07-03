#!/usr/bin/env python
"""Writer process for the durability demo.

Continuously writes batches of memories to a Keeper store and calls save()
in a loop, logging a fsync'd line before/after each round so the caller
(durability_demo.py) can tell, after killing this process, which rounds had
actually finished committing.

Usage: durability_writer.py <store_path> <log_path> <rounds> <batch_size>
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from keeper import KeeperStore, Memory  # noqa: E402


def main() -> int:
    if len(sys.argv) != 5:
        print(
            "usage: durability_writer.py <store_path> <log_path> <rounds> <batch_size>",
            file=sys.stderr,
        )
        return 2

    store_path, log_path, rounds, batch_size = (
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]),
        int(sys.argv[4]),
    )

    logf = open(log_path, "a", buffering=1)

    def log(msg: str) -> None:
        logf.write(f"{time.time():.6f} {msg}\n")
        logf.flush()
        os.fsync(logf.fileno())

    store = KeeperStore(store_path)
    log(f"READY pid={os.getpid()}")

    for round_i in range(rounds):
        batch = [
            Memory(
                text=f"durability-demo round={round_i} idx={j} nonce={os.urandom(4).hex()}",
                session_id="durability-writer",
                kind="event",
            )
            for j in range(batch_size)
        ]
        store.remember_many(batch)
        log(f"BEFORE_SAVE round={round_i}")
        store.save()
        log(f"AFTER_SAVE round={round_i}")

    log("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
