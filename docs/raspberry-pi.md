# Raspberry Pi

OrdinalDB supports Rust on Raspberry Pi class hardware through the Linux ARM64
target (`aarch64-unknown-linux-gnu`). Use a Raspberry Pi 4 or Raspberry Pi 5
with a 64-bit OS for the supported path.

For Python framework adapters on ARM64, start with the edge/local deployment
guide: [`edge-deployment.md`](edge-deployment.md). It covers wheelhouses,
optional extras, valid adapter dimensions, local embeddings, and telemetry
controls.

The 32-bit Raspberry Pi OS target (`armv7-unknown-linux-gnueabihf`) is not part
of the current CI matrix. It may work from source, but treat it as best-effort
until OrdinalDB carries an explicit 32-bit ARM lane.

## Build

Install a current Rust toolchain on the Pi, then build normally:

```bash
cargo build --release -p ordinaldb
cargo run --release --manifest-path examples/downstream-smoke/Cargo.toml
```

CI runs the Rust test suite and downstream consumer smoke on a native
`ubuntu-24.04-arm` runner. The stable ARM check context is
`Rust (linux-arm64)`. A separate `Rust Pi runtime (linux-arm64)` job runs the
Rust crate under multiple Rayon worker counts and exercises persistence rewrite
behavior.

For a repeatable ARM64 smoke/profile pass, run:

```bash
bash scripts/pi_arm64_profile.sh
```

The script records host details, Rust compiler details, memory and block-device
state, optional Raspberry Pi firmware throttling state, and wraps the
Pi-specific Rust commands with `/usr/bin/time -v` when available.

## Local Emulation

You can use virt-manager/libvirt with an ARM64 QEMU VM to test the Rust path
without owning a Raspberry Pi. This is useful for build compatibility,
persistence smoke tests, and low-resource runtime checks:

- create an `aarch64` VM with the QEMU `virt` machine type and UEFI firmware;
- install a standard Debian or Ubuntu ARM64 image;
- give the VM 1 to 4 vCPUs and 2 to 4 GiB RAM to approximate small-device
  scheduling pressure;
- run the same commands as the supported build path inside the VM.

On an x86_64 host, QEMU uses CPU emulation for this setup. Treat it as a
compatibility and correctness lane, not a performance profile. Timing data from
that VM mostly measures emulation overhead. If the host is ARM64 and libvirt can
use KVM acceleration, the VM is a better performance signal, but it still does
not model Pi thermals, memory bandwidth, microSD behavior, power throttling, or
the exact Raspberry Pi SoC.

AWS Graviton instances are a practical native ARM64 profiling target when no Pi
is available. Use a short-lived `t4g.medium` for Pi-class smoke or a less bursty
`c7g.large` for steadier CPU profiling. Graviton is still not Pi hardware; use
it to catch ARM64 runtime issues and broad performance regressions, not to claim
Raspberry Pi thermal, power, or microSD behavior.

## Runtime Tuning

OrdinalDB uses OrdVec, which uses Rayon internally. Set `RAYON_NUM_THREADS`
before process start to control CPU use:

```bash
RAYON_NUM_THREADS=2 cargo run --release --example rust_basic
```

Suggested starting points:

- `1` for thermal, memory-sensitive, or shared devices;
- `2` for general Pi 4/5 use while leaving CPU for the rest of the system;
- `4` for a dedicated Pi doing sustained indexing or search.

## Storage

OrdinalDB `.odb` bundles are directory-backed indexes. For write-heavy
persistence workloads, prefer USB SSD or NVMe storage over microSD. Use a clean
shutdown path and reliable power for production deployments.

The write path uses a temporary bundle, verifies the written artifacts, and then
renames the bundle into place. This protects normal rewrite failures, but the
current MVP does not claim power-loss durability across all filesystems because
it does not fsync every file and parent directory.

## Benchmark Checklist

Before a Pi benchmark, capture enough host state to interpret the result:

```bash
uname -a
rustc -vV
echo "RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-unset}"
vcgencmd measure_temp || true
vcgencmd measure_clock arm || true
vcgencmd get_throttled || true
```

For sustained runs, use active cooling and a power supply appropriate for the
board. If `vcgencmd get_throttled` reports current undervoltage, frequency
capping, throttling, or soft-temperature limiting, treat the benchmark as a
hardware-limited run rather than an OrdinalDB result.
