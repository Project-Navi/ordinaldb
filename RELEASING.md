# Releasing OrdinalDB

OrdinalDB builds Rust crate archives and Python wheel/sdist artifacts from
tag-triggered GitHub Actions workflows. Publication authority is intentionally
not wired into these workflows — `release-crates.yml` and `release-pypi.yml`
stop at `upload-artifact`. The actual `cargo publish` / `twine upload` step is
a manual, human-run action taken once, at 0.2.0 launch (see
[Manual publish at launch](#manual-publish-at-launch-final-step) below), not on
every tag push. `tests/release_publish_invariants.sh` enforces this by failing
if a release workflow ever grows a publish step. Future automation, if any,
should use Trusted Publishing where available and human-reviewed GitHub
Environments.

## Tags

- Rust crate: `ordinaldb-vMAJOR.MINOR.PATCH` (e.g. `ordinaldb-v0.2.0`)
- Python package: `ordinaldb-py-vMAJOR.MINOR.PATCH` (e.g. `ordinaldb-py-v0.2.0`)

Both workflows enforce strict SemVer tag names before any packaging job runs.
The `ordinaldb-` prefix is deliberate: bare `v*` / `py-v*` tags (e.g. inherited
from upstream history) must NOT trigger an OrdinalDB release, so never `git
push --tags` — push the specific `ordinaldb-v*` tag explicitly.

## Required GitHub Settings

- Default branch: `main`
- Branch protection on `main` requires pull requests, code owner review, stale
  review dismissal, last-push approval, conversation resolution, and the
  OrdinalDB CI matrix.
- Environments `crates-io` and `pypi` should require reviewer team
  `Project-Navi/stewards` when the repository plan supports environment
  reviewers, and restrict deployments to the matching tag pattern.
- Repository Actions workflow permissions should remain read-only by default.

## Pre-Tag Checklist

1. Confirm `main` is green.
2. Confirm `CHANGELOG.md` has the release notes.
3. Run `make release-invariants` (`bash tests/release_publish_invariants.sh`) —
   checks that the release workflows trigger only on the namespaced tag glob,
   contain no publish step yet, and actually upload artifacts.
4. Run `make release-crate-smoke` (`scripts/release_crate_package_smoke.sh`).
   This packages `ordinaldb-hybrid`, `ordinaldb`, `ordinaldb-adapter-store`,
   `ordinaldb-ltr`, and `ordinaldb-cli`, stages each produced `.crate` in a
   temporary local registry, verifies a non-workspace downstream crate against
   those staged artifacts instead of unpublished crates.io state, and installs
   the staged `ordinaldb-cli` binary crate.
5. Run `make hostile-input-smoke` (`scripts/hostile_input_smoke.py`) — a
   deterministic hostile-input smoke test for adapter storage (corrupt
   `redb`/`json` bundles, duplicate JSON keys, non-finite values, symlinked
   store files). Requires the `ordinaldb` Python package installed in the
   active environment (see the README's "Build from source" section).
6. Run `make limits-report` (`scripts/limits_report.py`) — generates a
   persistence and filter limits report (10K/100K-row core and adapter
   timings, footprint, and filter-selectivity measurements) at
   `benchmark-results/limits-report.json`. Same Python package requirement as
   step 5.
   - `make release-checklist` runs steps 3–6 in one command.
7. Build, install, import, and test the Python sdist in an isolated
   environment.
8. Audit environment settings with `bash tests/release_environment_settings.sh`
   if you have a token that can read repository environments.
9. Create and push the appropriate tag.

## Manual publish at launch (final step)

Publishing to crates.io and PyPI is **not** automated, on purpose. Do this by
hand, once, when actually launching a version publicly — never wire it into a
tag-triggered workflow without updating `tests/release_publish_invariants.sh`
first (it will fail the build if a publish step appears in a release
workflow).

### One-time account prerequisites

Set these up once, ahead of the first publish, and keep the credentials out of
the repo and out of any GitHub Actions log:

- **crates.io** — an account with an API token. Either run `cargo login` once
  locally (stores the token in `~/.cargo/credentials.toml`) or export
  `CARGO_REGISTRY_TOKEN` for the publish session only.
- **PyPI** — either:
  - [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) configured
    for the `ordinaldb` PyPI project against this repository (no long-lived
    token to manage), or
  - a PyPI API token scoped to the `ordinaldb` project, exported as
    `TWINE_USERNAME=__token__` and `TWINE_PASSWORD=<token>` for the publish
    session only.

### Publish order (crates.io)

Publish in dependency order. crates.io indexing takes a little time to
propagate; wait for each crate's page (`https://crates.io/crates/<name>`) to
show the new version before publishing the next crate that depends on it.

```bash
cargo publish -p ordinaldb-hybrid --locked
# wait for ordinaldb-hybrid to be indexed on crates.io, then:
cargo publish -p ordinaldb --locked
# wait for ordinaldb to be indexed on crates.io, then:
cargo publish -p ordinaldb-adapter-store --locked
cargo publish -p ordinaldb-ltr --locked
# wait for ordinaldb-ltr to be indexed on crates.io, then:
cargo publish -p ordinaldb-cli --locked
```

### Publish Python (PyPI)

Prefer downloading the wheel/sdist artifacts that `release-pypi.yml` already
built, installed, and import-tested for the tag being released, rather than
rebuilding locally, so what gets published is exactly what CI verified. Then:

```bash
python -m pip install --upgrade twine
twine check dist/*
twine upload dist/*
```

If building locally instead (e.g. `ordinaldb-python`'s wheel/sdist are not
available as workflow artifacts for some reason):

```bash
cd ordinaldb-python
pip install maturin
maturin build --release --out dist
maturin sdist --out dist
cd ..
python -m pip install --upgrade twine
twine check ordinaldb-python/dist/*
twine upload ordinaldb-python/dist/*
```
