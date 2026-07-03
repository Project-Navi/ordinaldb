<!--
Before opening this PR, there should be an issue behind it where the
approach was discussed — see CONTRIBUTING.md. Typo fixes, docs-only
changes, and one-line obvious bugfixes may go straight to a PR.
-->

## Issue

Closes #

## What changed

<!-- A few bullets are enough. -->

## Why

<!-- The reasoning, or a pointer to the issue if it already says it all. -->

## Verification

<!--
Tick only what you actually ran. Recall- or speed-affecting changes
should include before/after numbers.
-->

- [ ] `cargo fmt --all --check`
- [ ] `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] `cargo test --workspace`
- [ ] `python -m unittest discover -s ordinaldb-python/tests`
- [ ] Adapter extras tests (or adapter behavior is untouched)
- [ ] `maturin sdist --out dist` (or packaging is untouched)
- [ ] `cargo deny check advisories bans licenses sources` (or dependencies are untouched)
- [ ] `bash tests/release_publish_invariants.sh` (only when release/CI files changed)
- [ ] `CHANGELOG.md` entry under `Unreleased` for user-facing changes
