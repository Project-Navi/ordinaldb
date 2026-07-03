import importlib
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

import ordinaldb
import ordinaldb.adapters._common as adapters_common
from ordinaldb.adapters import AdapterStore, AdapterStoreError, adapter_store_markers_exist


VECTORS = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)
QUERY = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
GENERATION_INDEX_PATH = Path("vectors") / "g000000000001.odb"
GENERATION_INDEX_PATH_STR = "vectors/g000000000001.odb"
SECOND_GENERATION_INDEX_PATH = Path("vectors") / "g000000000002.odb"
SECOND_GENERATION_INDEX_PATH_STR = "vectors/g000000000002.odb"
LEGACY_ROOT_INDEX_FIXTURE = (
    Path(__file__).parent / "fixtures" / "legacy-root-index-adapter"
)


def _assert_writer_lock_released(testcase: unittest.TestCase, path: Path) -> None:
    try:
        lock = adapters_common._AdapterStateStore.acquire_writer_lock(path)
    except ValueError as exc:
        testcase.fail(f"writer lock should be releasable: {exc}")
    with lock:
        pass


def _convert_saved_store_to_legacy_root_layout(path: Path) -> None:
    redb_path = path / "adapter.redb"
    if redb_path.exists():
        redb_path.unlink()
    generation_path = path / GENERATION_INDEX_PATH
    if generation_path.exists():
        generation_path.rename(path / "index.odb")
    vectors_path = path / "vectors"
    if vectors_path.exists():
        vectors_path.rmdir()
    adapter_path = path / "adapter.json"
    adapter = _read_json(adapter_path)
    adapter["index_path"] = "index.odb"
    _write_json(adapter_path, adapter)


class AdapterImportTests(unittest.TestCase):
    def test_top_level_import_does_not_import_frameworks(self):
        self.assertTrue(hasattr(ordinaldb, "OrdinalIndex"))
        for module in ("langchain_core", "llama_index", "haystack", "agno"):
            self.assertNotIn(module, sys.modules)

    def test_top_level_import_is_framework_free_in_clean_interpreter(self):
        script = (
            "import sys, ordinaldb; "
            "blocked = ['langchain_core', 'llama_index', 'haystack', 'agno']; "
            "present = [name for name in blocked if name in sys.modules]; "
            "assert not present, present"
        )
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_missing_framework_extras_raise_install_hint(self):
        modules = [
            ("ordinaldb.langchain", "langchain_core", "ordinaldb[langchain]"),
            ("ordinaldb.llama_index", "llama_index", "ordinaldb[llama-index]"),
            ("ordinaldb.haystack", "haystack", "ordinaldb[haystack]"),
            ("ordinaldb.agno", "agno", "ordinaldb[agno]"),
        ]
        for module_name, dependency_name, hint in modules:
            if importlib.util.find_spec(dependency_name) is not None:
                continue
            with self.subTest(module=module_name):
                with self.assertRaisesRegex(ImportError, hint.replace("[", r"\[").replace("]", r"\]")):
                    importlib.import_module(module_name)


class AdapterStoreTests(unittest.TestCase):
    def test_adapter_store_markers_require_authoritative_sentinel(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "partial"
            path.mkdir()
            self.assertFalse(adapter_store_markers_exist(path))

            (path / "metadata.json").write_text("{}", encoding="utf-8")
            self.assertFalse(adapter_store_markers_exist(path))
            (path / "metadata.json").unlink()

            (path / "documents.json").write_text("[]", encoding="utf-8")
            self.assertFalse(adapter_store_markers_exist(path))
            (path / "documents.json").unlink()

            (path / "vectors").mkdir()
            self.assertFalse(adapter_store_markers_exist(path))
            (path / "vectors").rmdir()

            (path / "adapter.json").write_text("{}", encoding="utf-8")
            self.assertFalse(adapter_store_markers_exist(path))
            (path / "adapter.json").write_text(
                json.dumps(
                    {
                        "schema_version": "ordinaldb.adapter.v1",
                        "adapter": "common",
                        "bits": 2,
                        "dim": None,
                        "empty_lazy": True,
                        "index_path": GENERATION_INDEX_PATH_STR,
                        "sidecars": {
                            "id_map.json": {
                                "sha256": "0" * 64,
                                "file_size_bytes": 0,
                            },
                            "documents.json": {
                                "sha256": "1" * 64,
                                "file_size_bytes": 0,
                            },
                            "metadata.json": {
                                "sha256": "2" * 64,
                                "file_size_bytes": 0,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(adapter_store_markers_exist(path))
            (path / "adapter.json").unlink()

            (path / "adapter.redb").write_bytes(b"not a redb database")
            self.assertTrue(adapter_store_markers_exist(path))

    def test_save_load_stable_id_map_and_vector_only_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lc"
            store = AdapterStore(bits=2, dim=4, adapter_name="langchain")
            store.add(
                ids=["a", "b"],
                embeddings=VECTORS[:2],
                documents=["alpha", "beta"],
                metadatas=[{"group": "x"}, {"group": "y"}],
            )
            store.save(path, adapter_name="langchain")

            self.assertTrue((path / "adapter.redb").exists())
            _assert_writer_lock_released(self, path)
            self.assertTrue((path / GENERATION_INDEX_PATH / "manifest.json").exists())
            self.assertFalse((path / GENERATION_INDEX_PATH / "ids.bin").exists())
            self.assertFalse((path / "index.odb").exists())
            adapter = _read_json(path / "adapter.json")
            self.assertEqual(adapter["index_path"], GENERATION_INDEX_PATH_STR)
            id_map = _read_json(path / "id_map.json")
            self.assertEqual(id_map["next_u64_id"], 3)
            self.assertEqual(id_map["string_to_u64"], {"a": 1, "b": 2})

            loaded = AdapterStore.load(path, expected_adapter="langchain")
            self.assertEqual(loaded.ids(), ["a", "b"])
            self.assertEqual(loaded.get(["a"])[0].document, "alpha")
            results = loaded.search_by_vector(QUERY, k=10, filter={"group": "y"})
            self.assertEqual([record.id for record in results], ["b"])

    def test_save_publishes_next_generation_in_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "store"
            store = AdapterStore(bits=2, dim=4)
            store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            store.save(path)
            first_index = ordinaldb.OrdinalIndex.load(path / GENERATION_INDEX_PATH)
            self.assertEqual(len(first_index), 1)

            store.add(ids=["b"], embeddings=VECTORS[1:2], documents=["beta"])
            store.save(path)

            adapter = _read_json(path / "adapter.json")
            self.assertEqual(adapter["index_path"], SECOND_GENERATION_INDEX_PATH_STR)
            self.assertTrue((path / GENERATION_INDEX_PATH / "manifest.json").exists())
            self.assertTrue((path / SECOND_GENERATION_INDEX_PATH / "manifest.json").exists())
            self.assertFalse((path / "index.odb").exists())
            self.assertEqual(len(ordinaldb.OrdinalIndex.load(path / GENERATION_INDEX_PATH)), 1)

            loaded = AdapterStore.load(path)
            self.assertEqual(loaded.ids(), ["a", "b"])

    def test_unreferenced_generation_does_not_change_visible_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "orphan"
            store = AdapterStore(bits=2, dim=4)
            store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            store.save(path)

            shutil.copytree(path / GENERATION_INDEX_PATH, path / SECOND_GENERATION_INDEX_PATH)

            loaded = AdapterStore.load(path)
            self.assertEqual(loaded.ids(), ["a"])
            self.assertEqual(_read_json(path / "adapter.json")["index_path"], GENERATION_INDEX_PATH_STR)

    def test_failed_redb_publish_leaves_previous_generation_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "redb-publish-failure"
            store = AdapterStore(bits=2, dim=4)
            store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            store.save(path)
            store.add(ids=["b"], embeddings=VECTORS[1:2], documents=["beta"])
            original = adapters_common._write_adapter_state

            def fail_publish(*args, **kwargs):
                raise AdapterStoreError("injected redb publish failure")

            adapters_common._write_adapter_state = fail_publish
            try:
                with self.assertRaisesRegex(AdapterStoreError, "injected redb publish failure"):
                    store.save(path)
            finally:
                adapters_common._write_adapter_state = original

            self.assertTrue((path / SECOND_GENERATION_INDEX_PATH / "manifest.json").exists())
            loaded = AdapterStore.load(path)
            self.assertEqual(loaded.ids(), ["a"])
            self.assertEqual(_read_json(path / "adapter.json")["index_path"], GENERATION_INDEX_PATH_STR)

    def test_json_export_refresh_failure_is_committed_after_redb_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "json-export-failure"
            store = AdapterStore(bits=2, dim=4)
            store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            store.save(path)
            store.add(ids=["b"], embeddings=VECTORS[1:2], documents=["beta"])
            original = adapters_common._write_compatibility_exports

            def fail_exports(*args, **kwargs):
                raise AdapterStoreError("injected export refresh failure")

            adapters_common._write_compatibility_exports = fail_exports
            try:
                store.save(path)
            finally:
                adapters_common._write_compatibility_exports = original

            loaded = AdapterStore.load(path)
            self.assertEqual(loaded.ids(), ["a", "b"])
            self.assertEqual(_read_json(path / "adapter.json")["index_path"], GENERATION_INDEX_PATH_STR)

    def test_load_uses_redb_when_compatibility_export_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing-export"
            store = AdapterStore(bits=2, dim=4)
            store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            store.save(path)
            (path / "documents.json").unlink()

            loaded = AdapterStore.load(path)
            self.assertEqual(loaded.ids(), ["a"])
            self.assertEqual(loaded.get(["a"])[0].document, "alpha")

    def test_empty_lazy_persistence_is_adapter_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty"
            store = AdapterStore(bits=2)
            store.save(path)

            self.assertTrue((path / "adapter.redb").exists())
            self.assertFalse((path / "index.odb").exists())
            self.assertFalse((path / "vectors").exists())
            adapter = _read_json(path / "adapter.json")
            self.assertEqual(adapter["index_path"], GENERATION_INDEX_PATH_STR)
            self.assertTrue(adapter["empty_lazy"])
            self.assertIsNone(adapter["dim"])

            loaded = AdapterStore.load(path)
            self.assertEqual(len(loaded), 0)
            self.assertIsNone(loaded.dim_opt)

    def test_load_ignores_newer_json_exports_when_redb_is_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stale-redb"
            store = AdapterStore(bits=2, dim=4, adapter_name="langchain")
            store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            store.save(path, adapter_name="langchain")

            documents_path = path / "documents.json"
            documents = _read_json(documents_path)
            documents["documents"]["a"] = "alpha-new"
            _write_json(documents_path, documents)

            adapter_path = path / "adapter.json"
            adapter = _read_json(adapter_path)
            adapter["sidecars"]["documents.json"] = _file_digest(documents_path)
            _write_json(adapter_path, adapter)

            newer_than_redb = (path / "adapter.redb").stat().st_mtime + 2
            for sidecar in (adapter_path, documents_path):
                os.utime(sidecar, (newer_than_redb, newer_than_redb))

            loaded = AdapterStore.load(path, expected_adapter="langchain")
            self.assertEqual([record.document for record in loaded.get(["a"])], ["alpha"])

    def test_load_tolerates_older_json_exports_after_redb_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old-json-exports"
            store = AdapterStore(bits=2, dim=4, adapter_name="langchain")
            store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            store.save(path, adapter_name="langchain")

            documents_path = path / "documents.json"
            documents = _read_json(documents_path)
            documents["documents"]["a"] = "old-export"
            _write_json(documents_path, documents)
            before_redb = (path / "adapter.redb").stat().st_mtime - 2
            os.utime(documents_path, (before_redb, before_redb))

            loaded = AdapterStore.load(path, expected_adapter="langchain")
            self.assertEqual(loaded.get(["a"])[0].document, "alpha")

    def test_load_rejects_missing_redb_for_generation_layout_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing-redb-new-layout"
            store = AdapterStore(bits=2, dim=4, adapter_name="langchain")
            store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            store.save(path, adapter_name="langchain")

            (path / "adapter.redb").unlink()

            with self.assertRaisesRegex(AdapterStoreError, "adapter.redb is required"):
                AdapterStore.load(path, expected_adapter="langchain")

    def test_load_rejects_symlinked_adapter_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "real"
            link = Path(tmp) / "link"
            store = AdapterStore(bits=2, dim=4, adapter_name="langchain")
            store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            store.save(real, adapter_name="langchain")
            try:
                link.symlink_to(real, target_is_directory=True)
            except (NotImplementedError, OSError):
                self.skipTest("directory symlink unavailable")

            with self.assertRaisesRegex(AdapterStoreError, "must not be a symlink"):
                AdapterStore.load(link, expected_adapter="langchain")

    def test_legacy_load_accepts_root_index_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy-root-index"
            store = AdapterStore(bits=2, dim=4, adapter_name="langchain")
            store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            store.save(path, adapter_name="langchain")
            _convert_saved_store_to_legacy_root_layout(path)

            loaded = AdapterStore.load(path, expected_adapter="langchain")
            self.assertEqual(loaded.ids(), ["a"])
            self.assertEqual(loaded.get(["a"])[0].document, "alpha")

    def test_committed_previous_layout_fixture_loads(self):
        self.assertFalse((LEGACY_ROOT_INDEX_FIXTURE / "adapter.redb").exists())
        self.assertTrue((LEGACY_ROOT_INDEX_FIXTURE / "index.odb" / "manifest.json").exists())

        loaded = AdapterStore.load(
            LEGACY_ROOT_INDEX_FIXTURE,
            expected_adapter="langchain",
        )

        self.assertEqual(loaded.ids(), ["legacy-a"])
        self.assertEqual(loaded.get(["legacy-a"])[0].document, "legacy alpha")

    def test_committed_previous_layout_fixture_migrates_on_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "legacy-source"
            target = Path(tmp) / "migrated-target"
            shutil.copytree(LEGACY_ROOT_INDEX_FIXTURE, source)
            source_before = _tree_digest(source)
            loaded = AdapterStore.load(source, expected_adapter="langchain")
            loaded.add(
                ids=["legacy-b"],
                embeddings=VECTORS[1:2],
                documents=["legacy beta"],
                metadatas=[{"group": "legacy"}],
            )

            with self.assertRaisesRegex(AdapterStoreError, "different target"):
                loaded.save(source, adapter_name="langchain")
            self.assertEqual(_tree_digest(source), source_before)

            loaded.save(target, adapter_name="langchain")

            self.assertEqual(_tree_digest(source), source_before)
            self.assertFalse((source / "adapter.redb").exists())
            self.assertFalse((source / "vectors").exists())
            adapter = _read_json(target / "adapter.json")
            self.assertEqual(adapter["index_path"], GENERATION_INDEX_PATH_STR)
            self.assertTrue((source / "index.odb" / "manifest.json").exists())
            self.assertTrue((target / GENERATION_INDEX_PATH / "manifest.json").exists())
            migrated = AdapterStore.load(target, expected_adapter="langchain")
            self.assertEqual(migrated.ids(), ["legacy-a", "legacy-b"])

    def test_save_rejects_active_rust_writer_lock_without_mutating_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "locked"
            store = AdapterStore(bits=2, dim=4)
            store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            store.save(path)
            lock_path = _writer_lock_path(path)
            before = _tree_digest(path, exclude={lock_path})

            with adapters_common._AdapterStateStore.acquire_writer_lock(path):
                with self.assertRaisesRegex(AdapterStoreError, "writer lock"):
                    store.save(path)
                self.assertEqual(_tree_digest(path, exclude={lock_path}), before)
                self.assertEqual(list(path.parent.glob(f".{path.name}.tmp-*")), [])
            self.assertIn("lock=advisory-v1", lock_path.read_text(encoding="ascii"))

    def test_malformed_writer_lock_file_is_only_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diagnostic-lock"
            store = AdapterStore(bits=2, dim=4)
            store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            store.save(path)
            lock_path = _writer_lock_path(path)
            lock_path.write_bytes(b"malformed lock data" * 8192)

            store.add(ids=["b"], embeddings=VECTORS[1:2], documents=["beta"])
            store.save(path)

            lock_text = lock_path.read_text(encoding="ascii")
            self.assertIn(f"pid={os.getpid()}", lock_text)
            self.assertIn("lock=advisory-v1", lock_text)
            self.assertLess(lock_path.stat().st_size, 128)

    def test_json_atomic_writer_matches_payload_digest_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.json"
            payload = {"schema_version": "test.v1", "value": "line\nbreak"}

            adapters_common._write_json_atomic(path, payload)

            self.assertEqual(
                adapters_common._file_digest(path),
                adapters_common._payload_digest(payload),
            )
            self.assertNotIn(b"\r\n", path.read_bytes())

    def test_save_ignores_stale_pid_text_before_generation_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rust-locked"
            store = AdapterStore(bits=2, dim=4)
            store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            store.save(path)
            store.add(ids=["b"], embeddings=VECTORS[1:2], documents=["beta"])
            lock_path = _writer_lock_path(path)
            lock_path.write_text("pid=held-by-rust\n", encoding="utf-8")
            store.save(path)
            self.assertTrue((path / SECOND_GENERATION_INDEX_PATH).exists())
            self.assertIn("lock=advisory-v1", lock_path.read_text(encoding="ascii"))

    def test_save_ignores_non_ascii_writer_lock_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "binary-lock"
            store = AdapterStore(bits=2, dim=4)
            store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            store.save(path)
            lock_path = _writer_lock_path(path)
            lock_path.write_bytes(b"\xff")
            store.add(ids=["b"], embeddings=VECTORS[1:2], documents=["beta"])
            store.save(path)
            self.assertIn("lock=advisory-v1", lock_path.read_text(encoding="ascii"))
            self.assertEqual(AdapterStore.load(path).ids(), ["a", "b"])

    def test_save_ignores_oversized_numeric_writer_lock_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "oversized-pid-lock"
            store = AdapterStore(bits=2, dim=4)
            store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            store.save(path)
            lock_path = _writer_lock_path(path)
            lock_path.write_text("pid=999999999999999999999999999999\n", encoding="ascii")
            store.add(ids=["b"], embeddings=VECTORS[1:2], documents=["beta"])
            store.save(path)
            lock_text = lock_path.read_text(encoding="ascii")
            self.assertIn(f"pid={os.getpid()}", lock_text)
            self.assertIn("lock=advisory-v1", lock_text)
            self.assertLess(lock_path.stat().st_size, 128)

    def test_publish_killpoints_are_crash_consistent(self):
        pre_redb_stages = {
            "after_generation_temp_write",
            "after_generation_temp_fsync",
            "after_generation_temp_verified",
            "after_generation_rename",
            "before_adapter_state_publish",
        }
        stages = [
            *sorted(pre_redb_stages),
            "after_adapter_state_publish",
            f"after_export_{adapters_common.ID_MAP_FILE}",
            f"after_export_{adapters_common.DOCUMENTS_FILE}",
            f"after_export_{adapters_common.METADATA_FILE}",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for stage in stages:
                with self.subTest(stage=stage):
                    path = Path(tmp) / stage.replace("/", "-")
                    store = AdapterStore(bits=2, dim=4)
                    store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
                    store.save(path)
                    store.add(ids=["b"], embeddings=VECTORS[1:2], documents=["beta"])

                    def fail_at(current_stage, _root):
                        if current_stage == stage:
                            raise AdapterStoreError(f"injected {stage}")

                    adapters_common._PUBLISH_TEST_HOOK = fail_at
                    try:
                        if stage in pre_redb_stages:
                            with self.assertRaisesRegex(AdapterStoreError, f"injected {stage}"):
                                store.save(path)
                        else:
                            store.save(path)
                    finally:
                        adapters_common._PUBLISH_TEST_HOOK = None

                    loaded = AdapterStore.load(path)
                    expected_ids = ["a"] if stage in pre_redb_stages else ["a", "b"]
                    self.assertEqual(loaded.ids(), expected_ids)
                    _assert_writer_lock_released(self, path)
                    if stage.startswith("after_generation_temp"):
                        temp_generations = list((path / "vectors").glob(".g*.tmp-*"))
                        self.assertEqual(temp_generations, [])

    def test_subprocess_sigkill_publish_killpoints_are_crash_consistent(self):
        if not hasattr(signal, "SIGKILL"):
            self.skipTest("SIGKILL is not available on this platform")
        pre_redb_stages = {
            "after_generation_temp_write",
            "after_generation_temp_fsync",
            "after_generation_temp_verified",
            "after_generation_rename",
            "before_adapter_state_publish",
        }
        stages = [
            *sorted(pre_redb_stages),
            "after_adapter_state_publish",
            f"after_export_{adapters_common.ID_MAP_FILE}",
            f"after_export_{adapters_common.DOCUMENTS_FILE}",
            f"after_export_{adapters_common.METADATA_FILE}",
            f"after_export_{adapters_common.ADAPTER_FILE}",
        ]
        child_code = """
import os
import signal
import sys
from pathlib import Path

import numpy as np

import ordinaldb.adapters._common as adapters_common
from ordinaldb.adapters import AdapterStore

path = Path(sys.argv[1])
stage = sys.argv[2]

def kill_at(current_stage, _root):
    if current_stage == stage:
        os.kill(os.getpid(), signal.SIGKILL)

adapters_common._PUBLISH_TEST_HOOK = kill_at
store = AdapterStore.load(path)
store.add(
    ids=["b"],
    embeddings=np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
    documents=["beta"],
)
store.save(path)
raise SystemExit(42)
"""
        with tempfile.TemporaryDirectory() as tmp:
            for stage in stages:
                with self.subTest(stage=stage):
                    path = Path(tmp) / stage.replace("/", "-")
                    store = AdapterStore(bits=2, dim=4)
                    store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
                    store.save(path)

                    result = subprocess.run(
                        [sys.executable, "-c", child_code, str(path), stage],
                        check=False,
                    )

                    self.assertEqual(result.returncode, -signal.SIGKILL)
                    loaded = AdapterStore.load(path)
                    expected_ids = ["a"] if stage in pre_redb_stages else ["a", "b"]
                    self.assertEqual(loaded.ids(), expected_ids)

                    loaded.add(
                        ids=["c"],
                        embeddings=VECTORS[2:3],
                        documents=["gamma"],
                    )
                    loaded.save(path)
                    _assert_writer_lock_released(self, path)
                    recovered = AdapterStore.load(path)
                    self.assertIn("c", recovered.ids())

    def test_subprocess_sigkill_releases_engine_writer_lock(self):
        if not hasattr(signal, "SIGKILL"):
            self.skipTest("SIGKILL is not available on this platform")
        child_code = """
import os
import sys
import time
from pathlib import Path

import ordinaldb.adapters._common as adapters_common

path = Path(sys.argv[1])
ready_path = Path(sys.argv[2])

with adapters_common._AdapterStateStore.acquire_writer_lock(path):
    ready_path.write_text(str(os.getpid()), encoding="ascii")
    while True:
        time.sleep(1)
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "killed-lock-holder"
            ready_path = Path(tmp) / "lock-ready"
            path.mkdir()

            child = subprocess.Popen(
                [sys.executable, "-c", child_code, str(path), str(ready_path)],
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                deadline = time.monotonic() + 10
                while (
                    not ready_path.exists()
                    and child.poll() is None
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.05)
                if not ready_path.exists():
                    stderr = child.stderr.read() if child.stderr is not None else ""
                    self.fail(
                        "child did not acquire writer lock: "
                        f"returncode={child.poll()} stderr={stderr}"
                    )

                with self.assertRaisesRegex(ValueError, "already held"):
                    adapters_common._AdapterStateStore.acquire_writer_lock(path)

                os.kill(child.pid, signal.SIGKILL)
                self.assertEqual(child.wait(timeout=10), -signal.SIGKILL)
                _assert_writer_lock_released(self, path)

                store = AdapterStore(bits=2, dim=4)
                store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
                store.save(path)
                self.assertEqual(AdapterStore.load(path).ids(), ["a"])
            finally:
                if child.poll() is None:
                    os.kill(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
                if child.stderr is not None:
                    child.stderr.close()

    def test_keyboard_interrupt_during_generation_write_cleans_lock_and_temp(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "interrupt-generation"
            store = AdapterStore(bits=2, dim=4)
            store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            store.save(path)
            store.add(ids=["b"], embeddings=VECTORS[1:2], documents=["beta"])

            def interrupt_at_temp_write(stage, _root):
                if stage == "after_generation_temp_write":
                    raise KeyboardInterrupt("injected generation interrupt")

            adapters_common._PUBLISH_TEST_HOOK = interrupt_at_temp_write
            try:
                with self.assertRaisesRegex(KeyboardInterrupt, "injected generation"):
                    store.save(path)
            finally:
                adapters_common._PUBLISH_TEST_HOOK = None

            _assert_writer_lock_released(self, path)
            self.assertEqual(AdapterStore.load(path).ids(), ["a"])
            self.assertEqual(list((path / "vectors").glob(".g*.tmp-*")), [])

    def test_keyboard_interrupt_before_publish_keeps_previous_visible_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "interrupt-before-publish"
            store = AdapterStore(bits=2, dim=4)
            store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            store.save(path)
            store.add(ids=["b"], embeddings=VECTORS[1:2], documents=["beta"])

            def interrupt_before_publish(stage, _root):
                if stage == "before_adapter_state_publish":
                    raise KeyboardInterrupt("injected publish interrupt")

            adapters_common._PUBLISH_TEST_HOOK = interrupt_before_publish
            try:
                with self.assertRaisesRegex(KeyboardInterrupt, "injected publish"):
                    store.save(path)
            finally:
                adapters_common._PUBLISH_TEST_HOOK = None

            _assert_writer_lock_released(self, path)
            self.assertEqual(AdapterStore.load(path).ids(), ["a"])
            self.assertTrue((path / SECOND_GENERATION_INDEX_PATH / "manifest.json").exists())

    def test_stale_loaded_writer_save_rejects_without_losing_newer_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stale-writer"
            base = AdapterStore(bits=2, dim=4)
            base.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            base.save(path)

            stale_writer = AdapterStore.load(path)
            fresh_writer = AdapterStore.load(path)
            fresh_writer.add(ids=["b"], embeddings=VECTORS[1:2], documents=["beta"])
            fresh_writer.save(path)

            stale_writer.add(ids=["c"], embeddings=VECTORS[2:3], documents=["gamma"])
            with self.assertRaisesRegex(AdapterStoreError, "stale adapter snapshot"):
                stale_writer.save(path)

            loaded = AdapterStore.load(path)
            self.assertEqual(loaded.ids(), ["a", "b"])
            self.assertEqual([record.document for record in loaded.get(["a", "b"])], ["alpha", "beta"])
            self.assertFalse((path / "vectors" / "g000000000003.odb").exists())

    def test_stale_empty_loaded_writer_save_rejects_without_losing_first_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stale-empty-writer"
            AdapterStore(bits=2).save(path)

            writer_a = AdapterStore.load(path)
            writer_b = AdapterStore.load(path)
            writer_a.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            writer_a.save(path)

            writer_b.add(ids=["b"], embeddings=VECTORS[1:2], documents=["beta"])
            with self.assertRaisesRegex(AdapterStoreError, "stale adapter snapshot"):
                writer_b.save(path)

            loaded = AdapterStore.load(path)
            self.assertEqual(loaded.ids(), ["a"])
            self.assertEqual([record.document for record in loaded.get(["a"])], ["alpha"])
            self.assertFalse((path / "vectors" / "g000000000002.odb").exists())

    def test_unbased_writer_cannot_attach_to_persisted_empty_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attach-empty"
            AdapterStore(bits=2).save(path)

            writer = AdapterStore(bits=2, dim=4)
            writer.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            with self.assertRaisesRegex(AdapterStoreError, "without a loaded base commit token"):
                writer.save(path)

            loaded = AdapterStore.load(path)
            self.assertEqual(loaded.ids(), [])

    def test_new_writer_cannot_overwrite_non_empty_store_without_loaded_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unbased-writer"
            base = AdapterStore(bits=2, dim=4)
            base.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            base.save(path)

            unbased = AdapterStore(bits=2, dim=4)
            unbased.add(ids=["b"], embeddings=VECTORS[1:2], documents=["beta"])

            with self.assertRaisesRegex(AdapterStoreError, "without a loaded base commit token|different path"):
                unbased.save(path)

            loaded = AdapterStore.load(path)
            self.assertEqual(loaded.ids(), ["a"])

    def test_keyboard_interrupt_during_json_atomic_write_removes_temp_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "payload.json"
            original = adapters_common._json_dumps

            def interrupting_json_dumps(_payload):
                raise KeyboardInterrupt("injected json interrupt")

            adapters_common._json_dumps = interrupting_json_dumps
            try:
                with self.assertRaisesRegex(KeyboardInterrupt, "injected json"):
                    adapters_common._write_json_atomic(path, {"a": 1})
            finally:
                adapters_common._json_dumps = original

            self.assertFalse(path.exists())
            self.assertEqual(list(Path(tmp).glob(".payload.json.tmp-*")), [])

    def test_load_rejects_missing_active_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing-generation"
            store = AdapterStore(bits=2, dim=4)
            store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            store.save(path)

            shutil.rmtree(path / GENERATION_INDEX_PATH)

            with self.assertRaisesRegex(
                AdapterStoreError,
                "active generation path is missing|No such file|not found",
            ):
                AdapterStore.load(path)

    def test_load_rejects_symlinked_generation_parent(self):
        if not hasattr(os, "symlink"):
            self.skipTest("symlink not available")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "symlink-parent"
            store = AdapterStore(bits=2, dim=4)
            store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            store.save(path)
            outside = Path(tmp) / "outside-vectors"
            (path / "vectors").rename(outside)
            os.symlink(outside, path / "vectors")

            with self.assertRaisesRegex(AdapterStoreError, "active generation path.*symlink"):
                AdapterStore.load(path)

    def test_load_rejects_symlinked_redb_store(self):
        if not hasattr(os, "symlink"):
            self.skipTest("symlink not available")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "redb-symlink"
            store = AdapterStore(bits=2, dim=4)
            store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            store.save(path)
            outside = Path(tmp) / "outside.redb"
            (path / "adapter.redb").rename(outside)
            os.symlink(outside, path / "adapter.redb")

            with self.assertRaisesRegex(AdapterStoreError, "adapter.redb must not be a symlink"):
                AdapterStore.load(path)

    def test_duplicate_batch_ids_do_not_mutate(self):
        store = AdapterStore(bits=2, dim=4)

        with self.assertRaisesRegex(AdapterStoreError, "duplicate string IDs.*dup"):
            store.add(
                ids=["dup", "dup"],
                embeddings=VECTORS[:2],
                documents=["one", "two"],
            )

        self.assertEqual(len(store), 0)

    def test_invalid_rankquant_dim_fails_before_core_construction(self):
        with self.assertRaisesRegex(
            AdapterStoreError,
            "common adapter requires dim divisible by 4 when bits=2; got dim=6",
        ):
            AdapterStore(bits=2, dim=6)
        with self.assertRaisesRegex(
            AdapterStoreError,
            "common adapter requires dim divisible by 16 when bits=4; got dim=8",
        ):
            AdapterStore(bits=4, dim=8)

    def test_lazy_add_rejects_invalid_rankquant_dim_without_mutation(self):
        store = AdapterStore(bits=2, adapter_name="langchain")

        with self.assertRaisesRegex(
            AdapterStoreError,
            "langchain adapter embedding batch requires dim divisible by 4 when bits=2; got dim=6",
        ):
            store.add(
                ids=["a"],
                embeddings=np.ones((1, 6), dtype=np.float32),
                documents=["alpha"],
            )

        self.assertEqual(len(store), 0)
        self.assertIsNone(store.dim_opt)

    def test_empty_add_is_noop_without_embedding_preflight(self):
        store = AdapterStore(bits=2)

        self.assertEqual(
            store.add(ids=[], embeddings=None, documents=[], metadatas=[]),
            [],
        )
        self.assertEqual(len(store), 0)
        self.assertIsNone(store.dim_opt)

    def test_add_accepts_iterables(self):
        store = AdapterStore(bits=2, dim=4)

        added = store.add(
            ids=(value for value in ["a", "b"]),
            embeddings=VECTORS[:2],
            documents=(value for value in ["alpha", "beta"]),
            metadatas=(value for value in [{"group": "x"}, {"group": "y"}]),
        )

        self.assertEqual(added, ["a", "b"])
        self.assertEqual([record.document for record in store.get()], ["alpha", "beta"])

    def test_get_and_delete_accept_single_string_id(self):
        store = AdapterStore(bits=2, dim=4)
        store.add(
            ids=["doc-12"],
            embeddings=VECTORS[:1],
            documents=["alpha"],
        )

        self.assertEqual([record.id for record in store.get("doc-12")], ["doc-12"])
        self.assertTrue(store.delete("doc-12"))
        self.assertEqual(store.get("doc-12"), [])

    def test_upsert_replacement_gets_fresh_u64(self):
        store = AdapterStore(bits=2, dim=4)
        store.add(
            ids=["doc"],
            embeddings=VECTORS[:1],
            documents=["old"],
            metadatas=[{"version": 1}],
        )
        old_u64 = store.get(["doc"])[0].u64_id

        store.add(
            ids=["doc"],
            embeddings=VECTORS[1:2],
            documents=["new"],
            metadatas=[{"version": 2}],
            upsert=True,
        )

        record = store.get(["doc"])[0]
        self.assertEqual(record.document, "new")
        self.assertNotEqual(record.u64_id, old_u64)
        self.assertEqual(len(store), 1)

    def test_save_reload_after_upsert_and_delete_to_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "store"
            store = AdapterStore(bits=2, dim=4)
            store.add(
                ids=["a", "b"],
                embeddings=VECTORS[:2],
                documents=["alpha", "beta"],
                metadatas=[{"version": 1}, {"version": 1}],
            )
            store.add(
                ids=["a"],
                embeddings=VECTORS[2:3],
                documents=["alpha-new"],
                metadatas=[{"version": 2}],
                upsert=True,
            )
            store.save(path)

            loaded = AdapterStore.load(path)
            self.assertEqual(loaded.get(["a"])[0].document, "alpha-new")
            self.assertEqual(loaded.get(["a"])[0].metadata["version"], 2)
            loaded.delete(["a", "b"])
            loaded.save(path)

            empty = AdapterStore.load(path)
            self.assertEqual(len(empty), 0)
            self.assertEqual(empty.dim_opt, 4)

    def test_save_rejects_non_finite_metadata_without_publishing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nan-metadata-save"
            store = AdapterStore(bits=2, dim=4)
            store.add(
                ids=["a"],
                embeddings=VECTORS[:1],
                documents=["alpha"],
                metadatas=[{"score": float("nan")}],
            )

            with self.assertRaisesRegex(AdapterStoreError, "non-finite number"):
                store.save(path)

            _assert_writer_lock_released(self, path)
            self.assertFalse((path / "adapter.redb").exists())
            self.assertFalse((path / GENERATION_INDEX_PATH).exists())

    def test_loaded_reader_snapshot_stays_on_original_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snapshot"
            writer = AdapterStore(bits=2, dim=4)
            writer.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            writer.save(path)
            old_reader = AdapterStore.load(path)

            writer.add(ids=["b"], embeddings=VECTORS[1:2], documents=["beta"])
            writer.save(path)

            self.assertEqual(old_reader.ids(), ["a"])
            self.assertEqual(
                [record.id for record in old_reader.search_by_vector(QUERY, k=10)],
                ["a"],
            )
            new_reader = AdapterStore.load(path)
            self.assertEqual(new_reader.ids(), ["a", "b"])

    def test_failed_add_upsert_delete_commits_leave_visible_store_unchanged(self):
        def fail_before_publish(stage, _root):
            if stage == "before_adapter_state_publish":
                raise AdapterStoreError("injected p2 commit failure")

        scenarios = {
            "add": lambda store: store.add(
                ids=["c"],
                embeddings=VECTORS[2:3],
                documents=["gamma"],
                metadatas=[{"version": 1}],
            ),
            "upsert": lambda store: store.add(
                ids=["a"],
                embeddings=VECTORS[2:3],
                documents=["alpha-new"],
                metadatas=[{"version": 2}],
                upsert=True,
            ),
            "delete": lambda store: store.delete(["b"]),
        }
        with tempfile.TemporaryDirectory() as tmp:
            for name, mutate in scenarios.items():
                with self.subTest(name=name):
                    path = Path(tmp) / name
                    base = AdapterStore(bits=2, dim=4)
                    base.add(
                        ids=["a", "b"],
                        embeddings=VECTORS[:2],
                        documents=["alpha", "beta"],
                        metadatas=[{"version": 1}, {"version": 1}],
                    )
                    base.save(path)
                    writer = AdapterStore.load(path)
                    mutate(writer)

                    adapters_common._PUBLISH_TEST_HOOK = fail_before_publish
                    try:
                        with self.assertRaisesRegex(AdapterStoreError, "injected p2"):
                            writer.save(path)
                    finally:
                        adapters_common._PUBLISH_TEST_HOOK = None

                    visible = AdapterStore.load(path)
                    self.assertEqual(visible.ids(), ["a", "b"])
                    self.assertEqual(visible.get(["a"])[0].document, "alpha")
                    self.assertEqual(visible.get(["b"])[0].document, "beta")

    def test_portable_filter_selectivities_use_pre_search_allowlist(self):
        store = AdapterStore(bits=2, dim=4)
        ids = [f"doc-{index:03d}" for index in range(100)]
        embeddings = np.vstack([VECTORS[index % len(VECTORS)] for index in range(100)])
        documents = [f"document {index}" for index in range(100)]
        metadatas = [
            {
                "one": "yes" if index == 7 else "no",
                "p01": "yes" if index == 0 else "no",
                "half": "yes" if index < 50 else "no",
                "all": "yes",
            }
            for index in range(100)
        ]
        store.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        recorder = _SearchRecorder(store._index)
        store._index = recorder

        cases = [
            ({"one": "missing"}, set()),
            ({"one": "yes"}, {"doc-007"}),
            ({"p01": "yes"}, {"doc-000"}),
            ({"half": "yes"}, set(ids[:50])),
            ({"all": "yes"}, set(ids)),
        ]
        for filter_value, expected_ids in cases:
            with self.subTest(filter=filter_value):
                records = store.search_by_vector(QUERY, k=100, filter=filter_value)
                self.assertEqual({record.id for record in records}, expected_ids)
                if expected_ids:
                    mask = recorder.calls[-1]["mask"]
                    self.assertIsNotNone(mask)
                    self.assertEqual(int(mask.sum()), len(expected_ids))

    def test_portable_filter_rejects_unsupported_values_and_mixed_types(self):
        store = AdapterStore(bits=2, dim=4)
        store.add(
            ids=["a", "b", "c"],
            embeddings=VECTORS,
            documents=["alpha", "beta", "gamma"],
            metadatas=[
                {"rank": 1, "group": "x", "flag": True},
                {"rank": "1", "group": "y", "flag": False},
                {"tags": ["x"]},
            ],
        )

        self.assertEqual(store.search_by_vector(QUERY, k=3, filter={"missing": "x"}), [])
        with self.assertRaisesRegex(AdapterStoreError, "JSON scalar"):
            store.search_by_vector(QUERY, k=3, filter={"group": ["x"]})
        with self.assertRaisesRegex(AdapterStoreError, "mixed metadata type"):
            store.search_by_vector(QUERY, k=3, filter={"rank": "1"})
        with self.assertRaisesRegex(AdapterStoreError, "non-scalar metadata"):
            store.search_by_vector(QUERY, k=3, filter={"tags": "x"})

    def test_empty_filter_result_returns_without_core_search(self):
        store = AdapterStore(bits=2, dim=4)
        store.add(
            ids=["a"],
            embeddings=VECTORS[:1],
            documents=["alpha"],
            metadatas=[{"group": "x"}],
        )

        self.assertEqual(store.search_by_vector(QUERY, k=5, filter={"group": "none"}), [])

    def test_callable_filter_internal_type_error_is_not_retried(self):
        store = AdapterStore(bits=2, dim=4)
        store.add(
            ids=["a"],
            embeddings=VECTORS[:1],
            documents=["alpha"],
            metadatas=[{"group": "x"}],
        )

        def broken_filter(metadata):
            raise TypeError("internal filter failure")

        with self.assertRaisesRegex(TypeError, "internal filter failure"):
            store.filter_to_u64_allowlist(broken_filter)

    def test_callable_filter_can_accept_document_and_metadata(self):
        store = AdapterStore(bits=2, dim=4)
        store.add(
            ids=["a"],
            embeddings=VECTORS[:1],
            documents=["alpha"],
            metadatas=[{"group": "x"}],
        )

        def two_arg_filter(document, metadata):
            return document == "alpha" and metadata["group"] == "x"

        self.assertEqual(store.filter_to_u64_allowlist(two_arg_filter), [1])

    def test_callable_filter_prefers_document_metadata_when_optional(self):
        store = AdapterStore(bits=2, dim=4)
        store.add(
            ids=["a"],
            embeddings=VECTORS[:1],
            documents=["alpha"],
            metadatas=[{"group": "x"}],
        )

        def optional_two_arg_filter(document=None, metadata=None):
            return (
                document == "alpha"
                and metadata is not None
                and metadata["group"] == "x"
            )

        self.assertEqual(store.filter_to_u64_allowlist(optional_two_arg_filter), [1])

    def test_load_rejects_corrupt_redb_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad-redb"
            store = AdapterStore(bits=2, dim=4)
            store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            store.save(path)

            (path / "adapter.redb").write_bytes(b"not a redb database")
            with self.assertRaisesRegex(AdapterStoreError, "adapter state store verification"):
                AdapterStore.load(path)

    def test_load_ignores_redb_json_export_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "redb-json-drift"
            store = AdapterStore(bits=2, dim=4)
            store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            store.save(path)

            documents_path = path / "documents.json"
            documents = _read_json(documents_path)
            documents["documents"]["a"] = "alpha-drift"
            _write_json(documents_path, documents)
            redb_mtime = (path / "adapter.redb").stat().st_mtime
            os.utime(documents_path, (redb_mtime + 2, redb_mtime + 2))

            loaded = AdapterStore.load(path)
            self.assertEqual([record.document for record in loaded.get(["a"])], ["alpha"])

    def test_load_rejects_empty_lazy_with_stale_index_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty-stale-index"
            store = AdapterStore(bits=2)
            store.save(path)
            (path / "index.odb").mkdir()
            (path / "index.odb" / "manifest.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(AdapterStoreError, "empty_lazy.*index.odb"):
                AdapterStore.load(path)

    def test_load_rejects_empty_lazy_with_stale_generation_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty-stale-generation"
            store = AdapterStore(bits=2)
            store.save(path)
            (path / GENERATION_INDEX_PATH).mkdir(parents=True)
            (path / GENERATION_INDEX_PATH / "manifest.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(AdapterStoreError, "empty_lazy.*vectors"):
                AdapterStore.load(path)

    def test_legacy_load_rejects_corrupt_and_unknown_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad"

            def write_legacy_fixture():
                if path.exists():
                    shutil.rmtree(path)
                store = AdapterStore(bits=2, dim=4)
                store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
                store.save(path)
                _convert_saved_store_to_legacy_root_layout(path)

            write_legacy_fixture()

            adapter = _read_json(path / "adapter.json")
            adapter["schema_version"] = "ordinaldb.adapter.v0"
            (path / "adapter.json").write_text(json.dumps(adapter), encoding="utf-8")
            with self.assertRaisesRegex(AdapterStoreError, "unsupported adapter schema"):
                AdapterStore.load(path)

            write_legacy_fixture()
            (path / "documents.json").write_text("{not json", encoding="utf-8")
            with self.assertRaisesRegex(AdapterStoreError, "sidecar integrity"):
                AdapterStore.load(path)

    def test_load_wraps_missing_sidecar_io_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing"
            store = AdapterStore(bits=2, dim=4)
            store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            store.save(path)
            _convert_saved_store_to_legacy_root_layout(path)

            (path / "documents.json").unlink()
            with self.assertRaisesRegex(AdapterStoreError, "cannot read sidecar documents.json"):
                AdapterStore.load(path)

    def test_legacy_load_rejects_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "duplicate-json"
            store = AdapterStore(bits=2, dim=4)
            store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            store.save(path)
            _convert_saved_store_to_legacy_root_layout(path)

            id_map_path = path / "id_map.json"
            id_map_path.write_text(
                '{"schema_version":"ordinaldb.adapter.id_map.v1",'
                '"next_u64_id":2,'
                '"string_to_u64":{"a":1,"a":2},'
                '"u64_to_slot":{"1":0}}\n',
                encoding="utf-8",
            )
            adapter = _read_json(path / "adapter.json")
            adapter["sidecars"]["id_map.json"] = _file_digest(id_map_path)
            _write_json(path / "adapter.json", adapter)

            with self.assertRaisesRegex(AdapterStoreError, "duplicate JSON key"):
                AdapterStore.load(path)

    def test_legacy_load_wraps_invalid_utf8_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid-utf8-json"
            store = AdapterStore(bits=2, dim=4)
            store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            store.save(path)
            _convert_saved_store_to_legacy_root_layout(path)
            (path / "adapter.json").write_bytes(b'{"schema_version":"\xca"}')

            with self.assertRaisesRegex(AdapterStoreError, "corrupt JSON in adapter.json"):
                AdapterStore.load(path)

    def test_legacy_load_rejects_oversized_adapter_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "oversized-adapter-json"
            store = AdapterStore(bits=2, dim=4)
            store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            store.save(path)
            _convert_saved_store_to_legacy_root_layout(path)
            (path / "adapter.json").write_text(
                " " * (adapters_common.MAX_ADAPTER_JSON_BYTES + 1),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(AdapterStoreError, "adapter.json exceeds"):
                AdapterStore.load(path)

    def test_legacy_load_rejects_count_and_sidecar_mismatches(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "count-mismatch"
            store = AdapterStore(bits=2, dim=4)
            store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            store.save(path)
            _convert_saved_store_to_legacy_root_layout(path)

            documents_path = path / "documents.json"
            documents = _read_json(documents_path)
            documents["documents"]["stale"] = "stale"
            _write_json(documents_path, documents)
            adapter = _read_json(path / "adapter.json")
            adapter["sidecars"]["documents.json"] = _file_digest(documents_path)
            _write_json(path / "adapter.json", adapter)

            with self.assertRaisesRegex(AdapterStoreError, "documents sidecar keys"):
                AdapterStore.load(path)

    def test_legacy_load_rejects_hostile_metadata_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hostile-metadata"
            store = AdapterStore(bits=2, dim=4)
            store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            store.save(path)
            _convert_saved_store_to_legacy_root_layout(path)

            metadata_path = path / "metadata.json"
            metadata = _read_json(metadata_path)
            metadata["metadata"]["a"] = ["not", "an", "object"]
            _write_json(metadata_path, metadata)
            adapter = _read_json(path / "adapter.json")
            adapter["sidecars"]["metadata.json"] = _file_digest(metadata_path)
            _write_json(path / "adapter.json", adapter)

            with self.assertRaisesRegex(AdapterStoreError, "metadata must be a JSON object"):
                AdapterStore.load(path)

    def test_legacy_load_rejects_non_finite_metadata_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nonfinite-metadata"
            store = AdapterStore(bits=2, dim=4)
            store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
            store.save(path)
            _convert_saved_store_to_legacy_root_layout(path)

            metadata_path = path / "metadata.json"
            metadata_path.write_text(
                '{"metadata":{"a":{"score":NaN}},'
                '"schema_version":"ordinaldb.adapter.metadata.v1"}\n',
                encoding="utf-8",
            )
            adapter = _read_json(path / "adapter.json")
            adapter["sidecars"]["metadata.json"] = _file_digest(metadata_path)
            _write_json(path / "adapter.json", adapter)

            with self.assertRaisesRegex(AdapterStoreError, "non-finite JSON number"):
                AdapterStore.load(path)

    def test_legacy_load_rejects_non_empty_null_dim_and_string_u64_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad-dim"

            def write_legacy_fixture():
                if path.exists():
                    shutil.rmtree(path)
                store = AdapterStore(bits=2, dim=4)
                store.add(ids=["a"], embeddings=VECTORS[:1], documents=["alpha"])
                store.save(path)
                _convert_saved_store_to_legacy_root_layout(path)

            write_legacy_fixture()

            adapter_path = path / "adapter.json"
            adapter = _read_json(adapter_path)
            adapter["dim"] = None
            _write_json(adapter_path, adapter)
            with self.assertRaisesRegex(AdapterStoreError, "non-empty adapter sidecar must have dim"):
                AdapterStore.load(path)

            write_legacy_fixture()
            id_map_path = path / "id_map.json"
            id_map = _read_json(id_map_path)
            id_map["next_u64_id"] = "2"
            _write_json(id_map_path, id_map)
            adapter = _read_json(path / "adapter.json")
            adapter["sidecars"]["id_map.json"] = _file_digest(id_map_path)
            _write_json(path / "adapter.json", adapter)
            with self.assertRaisesRegex(AdapterStoreError, "next_u64_id.*unsigned 64-bit"):
                AdapterStore.load(path)

            write_legacy_fixture()
            id_map_path = path / "id_map.json"
            id_map = _read_json(id_map_path)
            id_map["string_to_u64"]["a"] = "1"
            _write_json(id_map_path, id_map)
            adapter = _read_json(path / "adapter.json")
            adapter["sidecars"]["id_map.json"] = _file_digest(id_map_path)
            _write_json(path / "adapter.json", adapter)
            with self.assertRaisesRegex(AdapterStoreError, "u64 id.*unsigned 64-bit"):
                AdapterStore.load(path)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload):
    path.write_bytes(
        (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
    )


def _file_digest(path: Path):
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "file_size_bytes": len(data)}


def _tree_digest(path: Path, *, exclude=None):
    excluded = {item.resolve() for item in exclude or set()}
    out = []
    for file_path in sorted(child for child in path.rglob("*") if child.is_file()):
        if file_path.resolve() in excluded:
            continue
        out.append((file_path.relative_to(path).as_posix(), _file_digest(file_path)))
    return out


def _writer_lock_path(path: Path):
    return path / ".ordinaldb.write.lock"


class _SearchRecorder:
    def __init__(self, inner):
        self._inner = inner
        self.calls = []

    def search(self, *args, **kwargs):
        self.calls.append(dict(kwargs))
        return self._inner.search(*args, **kwargs)

    def __len__(self):
        return len(self._inner)

    def __getattr__(self, name):
        return getattr(self._inner, name)


if __name__ == "__main__":
    unittest.main()
