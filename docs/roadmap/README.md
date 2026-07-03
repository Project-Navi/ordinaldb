# Roadmap

Public engineering roadmaps for OrdinalDB. These are working design
documents, published so users and contributors can see where the project is
headed and why — they describe intent and dependency order, not committed
dates, and they change as the work teaches us things.

| Document | Scope | Status |
|----------|-------|--------|
| [`0.2.0-feature-parity-spec.md`](0.2.0-feature-parity-spec.md) | Storage and adapter feature parity for the 0.2.0 release | Draft for the 0.2.0 release train |
| [`ltr-hybrid-production-spec.md`](ltr-hybrid-production-spec.md) | Hybrid (BM25 + dense + RRF) and learning-to-rank production readiness | Active |
| [`0.3.0-api-async-streaming-spec.md`](0.3.0-api-async-streaming-spec.md) | API concurrency contract, async integration, result streaming | Design target after 0.2.0 |
| [`ordinaldb-trust-spec.md`](ordinaldb-trust-spec.md) | Cryptographically sealed index bundles (Ed25519 seals) | Draft — proposed, not scheduled |

Anything promised in a roadmap but not yet in `CHANGELOG.md` is not shipped.
The README and [`docs/api.md`](../api.md) describe only what exists today.
