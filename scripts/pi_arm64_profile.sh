#!/usr/bin/env bash
set -euo pipefail

run_timed() {
  local label="$1"
  shift

  echo "::group::${label}"
  if [ -x /usr/bin/time ]; then
    /usr/bin/time -v "$@"
  else
    time "$@"
  fi
  echo "::endgroup::"
}

echo "::group::host"
date -u +"utc=%Y-%m-%dT%H:%M:%SZ"
uname -a
rustc -vV
echo "RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-unset}"

if [ -r /proc/cpuinfo ]; then
  echo "--- /proc/cpuinfo ---"
  sed -n '1,80p' /proc/cpuinfo
fi

if command -v free >/dev/null 2>&1; then
  echo "--- memory ---"
  free -h
fi

if command -v df >/dev/null 2>&1; then
  echo "--- filesystem ---"
  df -h .
fi

if command -v lsblk >/dev/null 2>&1; then
  echo "--- block devices ---"
  lsblk -o NAME,TYPE,SIZE,MODEL,ROTA,MOUNTPOINTS || true
fi

if command -v vcgencmd >/dev/null 2>&1; then
  echo "--- raspberry pi firmware state ---"
  vcgencmd measure_temp || true
  vcgencmd measure_clock arm || true
  vcgencmd get_throttled || true
fi
echo "::endgroup::"

run_timed "ordinaldb tests with one Rayon worker" \
  env RAYON_NUM_THREADS=1 cargo test -p ordinaldb --release --locked

run_timed "ordinaldb tests with two Rayon workers" \
  env RAYON_NUM_THREADS=2 cargo test -p ordinaldb --release --locked

run_timed "downstream smoke with four Rayon workers" \
  env RAYON_NUM_THREADS=4 cargo run --release --manifest-path examples/downstream-smoke/Cargo.toml
