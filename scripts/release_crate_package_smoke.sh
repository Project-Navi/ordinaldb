#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Preflight: a release tag must never carry git-sourced dependencies — the
# offline-registry staging below reads Cargo.lock and cannot supply git+
# sources as registry crates. The pre-publish [patch.crates-io] block in
# the workspace root must be removed before tagging (see RELEASING.md).
if grep -q '^source = "git+' Cargo.lock; then
    echo "ERROR: Cargo.lock contains git-sourced dependencies; remove the" >&2
    echo "workspace [patch.crates-io] block before tagging a release." >&2
    exit 1
fi


if ! command -v cargo-local-registry >/dev/null 2>&1; then
  echo "error: cargo-local-registry is required; install with: cargo install cargo-local-registry --version 0.2.12 --locked" >&2
  exit 1
fi

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

registry_dir="$tmp_dir/local-registry"
config_path="$tmp_dir/cargo-config.toml"
metadata_path="$tmp_dir/metadata.json"

cargo metadata --format-version 1 --no-deps --locked >"$metadata_path"
cargo local-registry sync Cargo.lock "$registry_dir"

cat >"$config_path" <<EOF
[source.crates-io]
replace-with = "ordinaldb-staged"

[source.ordinaldb-staged]
local-registry = "$registry_dir"
EOF

package_args=(--locked --config "$config_path")
if [[ "${ALLOW_DIRTY:-}" == "1" ]]; then
  package_args+=(--allow-dirty)
fi

crate_version() {
  local crate="$1"

  python3 - "$metadata_path" "$crate" <<'PY'
import json
import sys

metadata_path, crate = sys.argv[1:]
metadata = json.load(open(metadata_path, encoding="utf-8"))
for package in metadata["packages"]:
    if package["name"] == crate:
        print(package["version"])
        break
else:
    raise SystemExit(f"unknown workspace crate {crate!r}")
PY
}

stage_crate() {
  local crate="$1"
  local crate_file
  local version

  version="$(crate_version "$crate")"

  crate_file="target/package/${crate}-${version}.crate"
  if [[ ! -f "$crate_file" ]]; then
    echo "error: expected packaged crate missing: $crate_file" >&2
    exit 1
  fi
  cp "$crate_file" "$registry_dir/"

  python3 - "$metadata_path" "$registry_dir" "$crate" "$crate_file" <<'PY'
import hashlib
import json
import pathlib
import sys

metadata_path = pathlib.Path(sys.argv[1])
registry_dir = pathlib.Path(sys.argv[2])
crate_name = sys.argv[3]
crate_file = pathlib.Path(sys.argv[4])

metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
package = next(
    package for package in metadata["packages"] if package["name"] == crate_name
)


def index_path(name: str) -> pathlib.Path:
    if len(name) == 1:
        return pathlib.Path("1") / name
    if len(name) == 2:
        return pathlib.Path("2") / name
    if len(name) == 3:
        return pathlib.Path("3") / name[0] / name
    return pathlib.Path(name[:2]) / name[2:4] / name


def dependency_entry(dependency: dict) -> dict:
    rename = dependency.get("rename")
    return {
        "name": rename or dependency["name"],
        "req": dependency["req"],
        "features": dependency.get("features", []),
        "optional": dependency.get("optional", False),
        "default_features": dependency.get("uses_default_features", True),
        "target": dependency.get("target"),
        "kind": dependency.get("kind"),
        "package": dependency["name"] if rename else None,
    }


entry = {
    "name": package["name"],
    "vers": package["version"],
    "deps": [dependency_entry(dependency) for dependency in package["dependencies"]],
    "cksum": hashlib.sha256(crate_file.read_bytes()).hexdigest(),
    "features": package.get("features", {}),
    "yanked": False,
}

destination = registry_dir / "index" / index_path(package["name"])
destination.parent.mkdir(parents=True, exist_ok=True)
lines = []
if destination.exists():
    lines = [
        line
        for line in destination.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line)["vers"] != package["version"]
    ]
lines.append(json.dumps(entry, separators=(",", ":"), sort_keys=True))
destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

verify_downstream_consumer() {
  local downstream_dir="$tmp_dir/downstream-consumer"
  local install_root="$tmp_dir/install-root"
  local ordinaldb_version
  local adapter_store_version
  local hybrid_version
  local ltr_version
  local cli_version

  ordinaldb_version="$(crate_version ordinaldb)"
  adapter_store_version="$(crate_version ordinaldb-adapter-store)"
  hybrid_version="$(crate_version ordinaldb-hybrid)"
  ltr_version="$(crate_version ordinaldb-ltr)"
  cli_version="$(crate_version ordinaldb-cli)"

  mkdir -p "$downstream_dir/src"
  cat >"$downstream_dir/Cargo.toml" <<EOF
[package]
name = "ordinaldb-downstream-staged-smoke"
version = "0.0.0"
edition = "2021"
publish = false

[dependencies]
ordinaldb = "$ordinaldb_version"
ordinaldb-adapter-store = "$adapter_store_version"
ordinaldb-hybrid = "$hybrid_version"
ordinaldb-ltr = "$ltr_version"
EOF

  cat >"$downstream_dir/src/main.rs" <<'EOF'
fn main() {
    let _ = ordinaldb::OrdinalIndex::new(4, 2).expect("construct ordinal index");
    let _ = ordinaldb_adapter_store::ADAPTER_STORE_SCHEMA_VERSION;
    let _ = ordinaldb_hybrid::RankedBatch::empty(1);
    let _ = ordinaldb_ltr::FEATURE_CACHE_SCHEMA_VERSION;
}
EOF

  cargo generate-lockfile --manifest-path "$downstream_dir/Cargo.toml" --config "$config_path"
  cargo check --manifest-path "$downstream_dir/Cargo.toml" --locked --config "$config_path"
  cargo install ordinaldb-cli \
    --version "$cli_version" \
    --config "$config_path" \
    --root "$install_root"
  "$install_root/bin/ordinaldb" --help >/dev/null
}

for crate in \
  ordinaldb-hybrid \
  ordinaldb \
  ordinaldb-adapter-store \
  ordinaldb-ltr \
  ordinaldb-cli
do
  cargo package -p "$crate" "${package_args[@]}"
  stage_crate "$crate"
done

verify_downstream_consumer

echo "OK: packaged Rust crates verified by downstream staged local registry checks."
