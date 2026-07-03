#!/usr/bin/env python
"""Keeper durability demo: crash-recovery, verify, and gc, end to end.

Keeper's memory store is a real OrdinalDB adapter directory, so its
durability guarantees are the ordinaldb-cli's, not something Keeper
reimplements. This walkthrough:

  1. kills a writer process mid-write (SIGKILL, no graceful shutdown) and
     reopens the store from a fresh process to show recovery never returns
     partial/corrupt data;
  2. runs `ordinaldb verify` against the recovered store;
  3. runs `ordinaldb adapter gc` to reclaim the old generations a crash (or
     ordinary repeated saves) leaves behind;
  4. flips bytes in an on-disk artifact and shows `verify` catches it and
     fails closed, instead of silently loading corrupted vectors.

Run: python durability_demo.py

This kills a subprocess with SIGKILL as part of the demo -- that's
intentional, not a bug in this script.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
STORE_PATH = PROJECT_DIR / "store" / "durability_demo"
WRITER = PROJECT_DIR / "durability_writer.py"
LOG_PATH = PROJECT_DIR / "store" / "durability_demo_writer.log"

ROUNDS = 15
BATCH_SIZE = 80


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def find_repo_root(start: Path) -> Path:
    """Walk upward from `start` looking for the OrdinalDB repo root, marked
    by a Cargo.toml alongside an ordinaldb-cli/ crate directory."""
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "Cargo.toml").is_file() and (candidate / "ordinaldb-cli").is_dir():
            return candidate
    raise FileNotFoundError(f"could not locate the OrdinalDB repo root walking up from {start}")


def cli_command(repo_root: Path) -> list[str]:
    """Prefer an already-built release binary; fall back to `cargo run`."""
    built = repo_root / "target" / "release" / "ordinaldb"
    if built.is_file():
        return [str(built)]
    return ["cargo", "run", "-q", "-p", "ordinaldb-cli", "--"]


def run_cli(repo_root: Path, *args: str) -> dict:
    # NOTE: don't treat a non-zero returncode alone as failure. `ordinaldb
    # verify --json` legitimately exits 1 when the store it inspects is
    # invalid (see `Ok(valid) => ... Ok(false) => process::exit(1)` in
    # ordinaldb-cli/src/main.rs) while still printing a well-formed JSON
    # report -- section_tamper_evidence() below depends on exactly that
    # ("valid": false is the expected, successful outcome of that step).
    # The reliable signal that the CLI actually failed to run (crashed,
    # bad args, panic) rather than just reporting a negative result is
    # that it never gets to print its JSON report at all, so stdout won't
    # parse as JSON.
    cmd = [*cli_command(repo_root), *args]
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=120,
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"CLI command failed to produce JSON output (exit {proc.returncode}): {' '.join(cmd)}\n"
            f"error: {exc}\n"
            f"stdout:\n{proc.stdout[:2000].strip()}\n"
            f"stderr:\n{proc.stderr.strip()}"
        ) from exc
    return {"returncode": proc.returncode, "json": payload, "stderr": proc.stderr.strip()}


def wait_for_ready(log_path: Path, proc: subprocess.Popen, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        if log_path.exists() and "READY" in log_path.read_text():
            return True
        time.sleep(0.005)
    return False


def parse_log(log_path: Path) -> dict:
    completed_rounds = set()
    in_flight_round = None
    before_rounds = set()
    for line in log_path.read_text().splitlines():
        _, _, rest = line.partition(" ")
        if rest.startswith("BEFORE_SAVE"):
            before_rounds.add(int(rest.split("=")[1]))
        elif rest.startswith("AFTER_SAVE"):
            completed_rounds.add(int(rest.split("=")[1]))
    for round_i in sorted(before_rounds):
        if round_i not in completed_rounds:
            in_flight_round = round_i
            break
    return {"num_completed": len(completed_rounds), "in_flight_round": in_flight_round}


def reopen_and_check(store_path: Path) -> dict:
    """Reopen the store in a brand-new process (not this one) and recall."""
    code = (
        "import sys, json; sys.path.insert(0, %r)\n"
        "from keeper import KeeperStore\n"
        "try:\n"
        "    s = KeeperStore(%r)\n"
        "    hits = s.recall('durability-demo', k=3)\n"
        "    print(json.dumps({'ok': True, 'size': len(s), 'sample_hit': hits[0].text if hits else None}))\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'ok': False, 'error': f'{type(exc).__name__}: {exc}'}))\n"
    ) % (str(PROJECT_DIR), str(store_path))
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"ok": False, "error": f"no parseable output; stderr={proc.stderr[-2000:]}"}


def section_crash_recovery() -> Path:
    banner("1. CRASH RECOVERY: kill -9 a writer mid-write, reopen fresh")
    shutil.rmtree(STORE_PATH, ignore_errors=True)
    LOG_PATH.unlink(missing_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen([sys.executable, str(WRITER), str(STORE_PATH), str(LOG_PATH), str(ROUNDS), str(BATCH_SIZE)])
    if not wait_for_ready(LOG_PATH, proc):
        raise RuntimeError("writer process never became ready")

    delay = random.uniform(0.1, 0.9)
    print(f"writer is running ({ROUNDS} rounds x {BATCH_SIZE} memories); killing it after {delay:.2f}s")
    time.sleep(delay)

    still_running = proc.poll() is None
    if still_running:
        os.kill(proc.pid, signal.SIGKILL)
    proc.wait(timeout=10)
    print(f"writer died from SIGKILL: {still_running}")

    log_info = parse_log(LOG_PATH)
    if log_info["in_flight_round"] is not None:
        print(
            f"writer's log: round {log_info['in_flight_round']} was still in flight when killed "
            "(the store's actual on-disk state is checked next, independently of this log)"
        )
    else:
        print(f"writer's log: {log_info['num_completed']} round(s) had already committed before the kill")

    reopened = reopen_and_check(STORE_PATH)
    print(f"reopen in a fresh process: {reopened}")
    if not reopened.get("ok"):
        raise RuntimeError(f"store failed to reopen after the crash: {reopened}")
    print(f"recovered store has {reopened['size']} memories -- no partial writes, no corruption")
    return STORE_PATH


def section_resume_writing(store_path: Path) -> None:
    """A recovered store is a store you can keep using. Run a few more
    ordinary (non-killed) save rounds so the gc walkthrough below has more
    than one generation to reclaim -- every save leaves its predecessor's
    generation on disk until something garbage-collects it."""
    banner("2. RESUME: keep writing normally after recovery")
    LOG_PATH.unlink(missing_ok=True)
    proc = subprocess.run(
        [sys.executable, str(WRITER), str(store_path), str(LOG_PATH), "4", "50"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"resumed writer failed: {proc.stderr}")
    reopened = reopen_and_check(store_path)
    print(f"4 more save rounds completed normally; store now has {reopened.get('size')} memories")


def section_verify(store_path: Path, repo_root: Path) -> None:
    banner("3. VERIFY: ordinaldb verify --json")
    result = run_cli(repo_root, "verify", "--json", str(store_path))
    print(json.dumps(result["json"], indent=2))
    if not result["json"].get("valid"):
        raise RuntimeError(f"verify failed on a store that should be healthy: {result}")
    print("verify: valid=true")


def section_gc(store_path: Path, repo_root: Path) -> None:
    banner("4. GC: reclaim old generations left behind by repeated saves")
    stats_before = run_cli(repo_root, "stats", "--json", str(store_path))["json"]
    print(f"generations before gc: {stats_before['generation_count']} "
          f"(orphaned: {stats_before['orphan_generation_count']})")

    dry_run = run_cli(repo_root, "adapter", "gc", str(store_path), "--retain", "1", "--dry-run", "--json")["json"]
    print(f"gc --dry-run would reclaim: {dry_run['reclaimable_generation_paths']}")

    real = run_cli(repo_root, "adapter", "gc", str(store_path), "--retain", "1", "--json")["json"]
    print(f"gc reclaimed: {real['deleted_generation_paths']}")

    stats_after = run_cli(repo_root, "stats", "--json", str(store_path))["json"]
    print(f"generations after gc: {stats_after['generation_count']} "
          f"(orphaned: {stats_after['orphan_generation_count']})")

    verify_after = run_cli(repo_root, "verify", "--json", str(store_path))["json"]
    print(f"verify after gc: valid={verify_after.get('valid')}")


def section_tamper_evidence(store_path: Path, repo_root: Path) -> None:
    banner("5. TAMPER EVIDENCE: flip bytes on disk, confirm verify fails closed")
    inspect = run_cli(repo_root, "inspect", "--json", str(store_path))["json"]
    active_generation = store_path / inspect["active_generation_path"]
    artifact = active_generation / "index.ovrq"

    data = bytearray(artifact.read_bytes())
    n = len(data)
    flip_start, flip_end = max(0, n - 40), max(0, n - 8)
    for i in range(flip_start, flip_end):
        data[i] ^= 0xFF
    artifact.write_bytes(data)
    print(f"flipped {flip_end - flip_start} bytes near the end of {artifact.relative_to(store_path)}")

    tampered = run_cli(repo_root, "verify", "--json", str(store_path))["json"]
    print(f"verify on tampered store: {tampered}")
    if tampered.get("valid"):
        raise RuntimeError("verify accepted a tampered artifact -- tamper detection is broken")
    print("verify correctly fails closed instead of silently loading corrupted vectors")

    for i in range(flip_start, flip_end):
        data[i] ^= 0xFF
    artifact.write_bytes(data)
    restored = run_cli(repo_root, "verify", "--json", str(store_path))["json"]
    print(f"restored original bytes; verify again: valid={restored.get('valid')}")


def main() -> None:
    repo_root = find_repo_root(PROJECT_DIR)
    print(f"repo root: {repo_root}")
    store_path = section_crash_recovery()
    section_resume_writing(store_path)
    section_verify(store_path, repo_root)
    section_gc(store_path, repo_root)
    section_tamper_evidence(store_path, repo_root)
    banner("DONE")


if __name__ == "__main__":
    main()
