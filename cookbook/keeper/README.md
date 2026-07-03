# Keeper

Durable long-term memory for a local AI agent, built on OrdinalDB's Agno
integration (`ordinaldb.agno.OrdinalDb`) with local CPU embeddings
(`sentence-transformers/all-MiniLM-L6-v2`, no LLM, no API key).

## The problem

An agent that remembers things across sessions needs that memory to survive
more than a happy path: process restarts, crashes mid-write, and routine
maintenance like reclaiming old data. Keeper is a small memory layer
(`remember` / `recall` / `forget`) over an OrdinalDB adapter store, plus a
walkthrough of what happens when a writer gets killed mid-save, how to
verify a store's integrity, and how to garbage-collect it.

## Quickstart

From this directory, with OrdinalDB already built from source at the repo
root (`cargo build --release` and
`maturin build --release -m ordinaldb-python/Cargo.toml`, per the top-level
README's "Build from source" section):

```bash
cd cookbook/keeper
python3 -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install "$(ls ../../target/wheels/ordinaldb-*.whl)[agno]"
pip install -r requirements.txt
python demo.py
python durability_demo.py
```

The wildcard has to resolve via command substitution before the `[agno]`
extras marker is appended -- see the top-level README's "Build from source"
section for why.

`durability_demo.py` also needs the `ordinaldb-cli` binary; it uses
`target/release/ordinaldb` if it's already built (`cargo build --release -p
ordinaldb-cli`), or falls back to `cargo run -p ordinaldb-cli` otherwise.
This script deliberately sends `SIGKILL` to a subprocess -- that's the point
of the demo, not a bug.

## What to look at in the code

```
keeper/embedder.py    -- local CPU sentence-transformers embedder (agno-compatible)
keeper/memory.py      -- Memory / Recalled record types
keeper/store.py       -- KeeperStore: remember / recall / forget over OrdinalDb
session_runner.py     -- runs ONE session as its own OS process (so "separate
                          process" in demo.py is literal, not simulated)
demo.py               -- 5-session agent lifecycle demo
durability_writer.py  -- writer subprocess used by durability_demo.py
durability_demo.py    -- crash-recovery + verify + gc + tamper-evidence walkthrough
```

`demo.py` spawns 5 separate Python processes against one on-disk store, in
order: onboarding, a preference update that supersedes and deletes an old
one, verification that the deleted memory never resurfaces, a new event, and
a final broad recall across everything still live.

`durability_demo.py` kills a writer process mid-write with `SIGKILL`,
reopens the store from a fresh process to show recovery never returns
partial or corrupt data, runs `ordinaldb verify`, runs `ordinaldb adapter
gc` to reclaim the generations left behind by repeated saves, and finally
flips bytes in an on-disk artifact to show `verify` catches it and fails
closed instead of silently loading corrupted vectors.

## Differentiators demonstrated

- **Crash-safe persistence** -- killing a writer process with `SIGKILL` at a
  random point never leaves a store that reopens with partial or torn data:
  either the last completed save is there, or the one before it is, never
  something in between.
- **Verify/gc ops story** -- `ordinaldb verify` and `ordinaldb adapter gc`
  are real, scriptable operational commands, not internal test helpers:
  `durability_demo.py` runs both against a store that Keeper itself wrote.
- **Tamper evidence** -- flipping bytes in an on-disk vector artifact makes
  `verify` fail closed with a specific "manifest verification failed"
  error, instead of silently returning corrupted search results.
