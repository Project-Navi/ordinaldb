BENCH_DATASET ?= scifact
BENCH_BITS ?= 2
BENCH_K ?= 10
BENCH_HARRIER_QUANT ?= q8_0
BENCH_LLAMA_GPU_LAYERS ?= 999
BENCH_CUDA_ARCHITECTURES ?= 86
BENCH_CMAKE_TOP_LEVEL_INCLUDES ?= $(abspath benchmarks/beir-rust/cmake/single-gpu-cuda.cmake)
BENCH_OUTPUT ?= benchmark-results/beir-scifact-test.json
BENCH_ARGS ?=
BENCH_CARGO_ARGS ?=

LIMITS_REPORT_BIN ?= target/release/ordinaldb
LIMITS_REPORT_OUTPUT ?= benchmark-results/limits-report.json
LIMITS_REPORT_WORK_DIR ?= .ordinaldb-limits-report-cache
LIMITS_REPORT_SIZES ?= 10000 100000

.DEFAULT_GOAL := help

.PHONY: help run-benchmark clean-benchmark limits-report hostile-input-smoke \
	release-crate-smoke release-invariants release-checklist

help:
	@printf '%s\n' \
		"Targets:" \
		"  run-benchmark        Run the BEIR benchmark explicitly." \
		"  clean-benchmark      Remove benchmark cache, results, and target artifacts." \
		"  limits-report        Generate the persistence/filter limits report (requires the" \
		"                       ordinaldb Python package installed; see README 'Build from source')." \
		"  hostile-input-smoke  Run the deterministic hostile-input adapter-storage smoke test" \
		"                       (same Python package requirement as limits-report)." \
		"  release-crate-smoke  Package every release crate and verify it via a staged local registry." \
		"  release-invariants   Run the release-workflow structural invariant checks." \
		"  release-checklist    Run the pre-tag automatable checks: release-invariants," \
		"                       hostile-input-smoke, limits-report, release-crate-smoke." \
		"                       See RELEASING.md for the full pre-tag checklist, including the" \
		"                       remaining manual steps."

run-benchmark:
	mkdir -p benchmark-results
	CMAKE_CUDA_ARCHITECTURES="$(BENCH_CUDA_ARCHITECTURES)" \
		CMAKE_PROJECT_TOP_LEVEL_INCLUDES="$(BENCH_CMAKE_TOP_LEVEL_INCLUDES)" \
		cargo run --locked --release --manifest-path benchmarks/beir-rust/Cargo.toml $(BENCH_CARGO_ARGS) -- \
		--dataset "$(BENCH_DATASET)" \
		--bits "$(BENCH_BITS)" \
		--k "$(BENCH_K)" \
		--harrier-quant "$(BENCH_HARRIER_QUANT)" \
		--llama-gpu-layers "$(BENCH_LLAMA_GPU_LAYERS)" \
		--output "$(BENCH_OUTPUT)" \
		$(BENCH_ARGS)

clean-benchmark:
	rm -rf .ordinaldb-benchmark-cache benchmark-results benchmarks/beir-rust/target

limits-report:
	cargo build --release --locked -p ordinaldb-cli
	mkdir -p "$(dir $(LIMITS_REPORT_OUTPUT))"
	python3 scripts/limits_report.py \
		--output "$(LIMITS_REPORT_OUTPUT)" \
		--work-dir "$(LIMITS_REPORT_WORK_DIR)" \
		--ordinaldb-bin "$(LIMITS_REPORT_BIN)" \
		--sizes $(LIMITS_REPORT_SIZES) \
		--clean

hostile-input-smoke:
	python3 scripts/hostile_input_smoke.py

release-crate-smoke:
	bash scripts/release_crate_package_smoke.sh

release-invariants:
	bash tests/release_publish_invariants.sh

release-checklist: release-invariants hostile-input-smoke limits-report release-crate-smoke
	@echo "Automated pre-tag checks passed. Remaining manual RELEASING.md steps:" \
		"confirm main is green, confirm CHANGELOG.md has the release notes, build/install/" \
		"import/test the Python sdist, optionally audit environment settings, then tag."
