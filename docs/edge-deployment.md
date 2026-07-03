# Edge And Local Adapter Deployment

This guide covers OrdinalDB as an embedded vector store inside a local agent,
desktop workflow, edge service, or offline integration test. It focuses on the
optional Python framework adapters. The Rust core has no framework dependency
and remains vector-only.

For launch positioning, treat edge deployment as a supported local retrieval
shape, not as a claim that OrdinalDB ships models or manages inference. Your
application owns embeddings. OrdinalDB owns compact vector persistence,
structurally checked reloads, the adapter control plane, and the Rust query
path.

## Supported Boundary

Use the adapters when the application owns embedding generation and passes
vectors into OrdinalDB or into the framework object that wraps it. The adapters
do not download models, call hosted embedding APIs, or run a server.

Install the narrowest extra that matches the framework:

```bash
pip install 'ordinaldb[langchain]'
pip install 'ordinaldb[llama-index]'
pip install 'ordinaldb[haystack]'
pip install 'ordinaldb[agno]'
```

`ordinaldb[adapters]` is useful for CI and compatibility smoke tests, but it
pulls all framework dependency graphs. Those graphs include network-capable
packages such as HTTP clients, LangSmith, OpenAI client packages, or telemetry
packages depending on the selected framework. Installing them does not mean the
OrdinalDB adapter will make network calls, but local deployments should still
pin, audit, and vendor the environment.

## Embedding Dimensions

Adapter dimensions must satisfy the same OrdVec RankQuant constraints as the
core index:

| bits | required dimension multiple |
| --- | --- |
| 1 | 8 |
| 2 | 4 |
| 4 | 16 |

Dimensions must also be in the range `2..=65535`. For example, `bits=2,
dim=8` is valid, while `bits=2, dim=6` is rejected before the Rust core is
constructed. This is most visible with hand-written smoke-test embedders.

## Persistence And Durability

Adapters keep framework text, metadata, string IDs, and checkpoints in
`adapter.redb`, beside immutable vector-only `.odb` generations:

```text
adapter-store/
    adapter.redb
    vectors/
        g000000000001.odb/
```

Writes are in memory until the framework persistence method is called:

- LangChain: `persist(...)`, `save_local(...)`, or `dump(...)`
- LlamaIndex: `persist(...)`
- Haystack: `save(...)`
- Agno: `save(...)`

Agno also supports `auto_save=True` for callers that want each insert, upsert,
delete, or drop to persist immediately to the configured path. The default
stays explicit-save to avoid surprising existing callers.

Every save writes a complete new vector generation — the whole current vector
set, not an incremental diff — so its cost scales with total store size (see
[`persistence.md`](persistence.md#adapter-directories) for measured numbers).
`auto_save=True` pays that full-rewrite cost on every single mutation, so it
is a poor fit for loops that insert one record at a time on a store that
already holds many rows. Prefer batching: accumulate a batch of adds/upserts
with `auto_save=False` (the default), then call `save()` once per batch.

Use `ordinaldb verify PATH` in release, backup, or startup checks. For
redb-backed adapter stores, verification treats `adapter.redb` as authoritative
and checks the active vector generation it publishes. JSON sidecars are derived
artifacts, not live verification inputs:

```bash
ordinaldb adapter export-json adapter-store
ordinaldb adapter import-legacy legacy-adapter-store --output imported-store
ordinaldb adapter gc adapter-store --retain 2
```

## Offline Wheelhouse

Build or download wheels on a connected machine, then install from a local
wheelhouse on the edge host.

```bash
python -m pip wheel -w wheelhouse -c constraints/edge-adapters.txt \
  './ordinaldb-python[langchain]'

PIP_NO_INDEX=1 python -m pip install --find-links wheelhouse \
  -c constraints/edge-adapters.txt 'ordinaldb[langchain]'
```

For all adapters:

```bash
python -m pip wheel -w wheelhouse -c constraints/edge-adapters.txt \
  './ordinaldb-python[adapters]'

PIP_NO_INDEX=1 python -m pip install --find-links wheelhouse \
  -c constraints/edge-adapters.txt 'ordinaldb[adapters]'
```

Source builds need the Rust toolchain and the `maturin>=1.12,<2.0` build
backend. Wheel installs should not discover build dependencies at first boot.
This is an offline install pattern, not a complete production lock; build a
target-platform wheelhouse with all transitive wheels before moving to an
offline host.

## Telemetry And Egress

The OrdinalDB adapters do not make model or telemetry calls. Some framework
packages include optional telemetry or hosted-client integrations. Set the
framework controls before importing and constructing pipelines when no-egress
behavior matters, and enforce egress at the OS, container, or firewall layer for
strict no-network deployments:

```bash
export HAYSTACK_TELEMETRY_ENABLED=False
export HAYSTACK_DISABLE_TELEMETRY=1
export POSTHOG_DISABLED=1
export AGNO_TELEMETRY=false
```

Avoid importing hosted model components in local-only processes. Use
deterministic local embedders in tests and device smoke checks.

The adapter CI also runs `examples/python_adapters/blocked_egress_smoke.py`,
which executes the local adapter examples with Python socket connection and DNS
entry points blocked. This catches accidental adapter-example network calls, but
strict deployments should still enforce egress outside the Python process.

## Missing IDs And Unsupported Modes

Framework adapter methods follow their framework conventions:

- missing gets return no result;
- missing deletes are no-ops unless a framework method documents otherwise;
- filters that match no records return an empty result without calling core
  vector search;
- MMR, sparse, hybrid, text-search, semantic-hybrid, and normalized relevance
  score APIs are rejected instead of returning misleading results.

Scores returned through adapter search methods are OrdinalDB similarity scores,
not cosine scores or normalized relevance scores.

## Raspberry Pi And ARM64

Use 64-bit Linux ARM64 for Raspberry Pi class deployments. Build Rust and Python
wheels ahead of time when possible, and prefer SSD or NVMe storage over microSD
for write-heavy adapter persistence.

See [`raspberry-pi.md`](raspberry-pi.md) for the Rust ARM64 path, QEMU and AWS
Graviton testing notes, thread-count tuning, and profiling checklist.
