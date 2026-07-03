#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

fail() {
  echo "::error::release-publish invariant violated: $*"
  exit 1
}

# Structural checks below parse the workflow YAML with PyYAML instead of
# grepping full lines of implementation detail. Full-line substring greps
# (e.g. matching an exact python-version list, or an exact pinned-tool
# version string in a comment) false-fail on harmless refactors without
# catching anything a real CI run wouldn't already catch. PyYAML ships on
# GitHub-hosted runner images; fail loudly and early if it is ever missing
# so this doesn't silently pass on a broken environment.
python3 -c "import yaml" 2>/dev/null \
  || fail "python3 'yaml' module (PyYAML) is required for structural workflow checks"

for f in ordinaldb.cdx.json ordinaldb-cli/ordinaldb-cli.cdx.json ordinaldb-python/ordinaldb-python.cdx.json; do
  git check-ignore -q -- "$f" || fail "$f is not gitignored"
done

# A release workflow must (a) trigger on exactly its namespaced tag glob,
# (b) contain no step with publish authority (that is a manual, launch-day
# step per RELEASING.md), and (c) actually upload build artifacts.
check_release_workflow() {
  local workflow="$1"
  local expected_tag="$2"

  python3 - "$workflow" "$expected_tag" <<'PY'
import re
import sys

import yaml

workflow, expected_tag = sys.argv[1:]

with open(workflow, encoding="utf-8") as handle:
    doc = yaml.safe_load(handle)

# YAML 1.1 parses the bare `on:` key as the boolean True, not the string
# "on" - PyYAML's safe_load follows that resolver, so check both.
on = doc.get(True)
if on is None:
    on = doc.get("on")
if not isinstance(on, dict):
    sys.exit(f"{workflow}: missing top-level 'on:' mapping")

push = on.get("push")
if not isinstance(push, dict):
    sys.exit(f"{workflow}: on.push must be a mapping with tags")

tags = push.get("tags")
if isinstance(tags, str):
    tags = [tags]
if tags != [expected_tag]:
    sys.exit(f"{workflow}: on.push.tags must be exactly [{expected_tag!r}]; found {tags!r}")

jobs = doc.get("jobs")
if not isinstance(jobs, dict) or not jobs:
    sys.exit(f"{workflow}: no jobs found")

publish_patterns = [
    re.compile(r"cargo\s+publish"),
    re.compile(r"\btwine\b"),
    re.compile(r"maturin\s+(publish|upload)"),
    re.compile(r"gh-action-pypi-publish"),
]

has_artifact_upload = False
for job_name, job in jobs.items():
    if not isinstance(job, dict):
        continue
    for step in job.get("steps", []) or []:
        if not isinstance(step, dict):
            continue
        uses = str(step.get("uses") or "")
        run = str(step.get("run") or "")
        haystack = f"{uses}\n{run}"
        for pattern in publish_patterns:
            if pattern.search(haystack):
                sys.exit(
                    f"{workflow}: job {job_name!r} step contains publish authority "
                    f"matching {pattern.pattern!r}; publishing must stay manual until launch"
                )
        if uses.startswith("actions/upload-artifact"):
            has_artifact_upload = True

if not has_artifact_upload:
    sys.exit(f"{workflow}: no actions/upload-artifact step found; release workflow must produce artifacts")

print(f"OK: {workflow}")
PY
}

check_release_workflow .github/workflows/release-crates.yml 'ordinaldb-v*' \
  || fail "release-crates.yml failed structural release-workflow checks"
check_release_workflow .github/workflows/release-pypi.yml 'ordinaldb-py-v*' \
  || fail "release-pypi.yml failed structural release-workflow checks"

# All workflow files (not only release ones) must pin `uses:` references to
# a full commit SHA rather than a mutable tag/branch - a supply-chain
# invariant, checked structurally against each step's `uses:` field.
python3 - <<'PY' || fail "workflow actions must be pinned to commit SHAs"
import pathlib
import re
import sys

import yaml

sha_re = re.compile(r"^[0-9a-fA-F]{40}$")
violations = []

for workflow in sorted(pathlib.Path(".github/workflows").glob("*.yml")):
    doc = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    jobs = doc.get("jobs", {}) if isinstance(doc, dict) else {}
    for job_name, job in jobs.items():
        if not isinstance(job, dict):
            continue
        for step in job.get("steps", []) or []:
            if not isinstance(step, dict):
                continue
            uses = step.get("uses")
            if not uses or "@" not in uses:
                continue
            ref = uses.rsplit("@", 1)[1]
            if not sha_re.match(ref):
                violations.append(f"{workflow}: job {job_name!r} uses {uses!r} (not pinned to a commit SHA)")

if violations:
    print("\n".join(violations))
    sys.exit(1)
PY

echo "OK: release-publish invariants hold."
