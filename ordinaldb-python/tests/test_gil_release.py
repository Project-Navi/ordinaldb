"""Regression tests: long-running native calls must release the GIL.

Before the fix, no binding released the GIL, so a single ``search`` call
serialized every other Python thread for its full duration -- defeating the
core index's internal Rayon parallelism for concurrent callers.

Strategy: run a slow native call in a worker thread while the main thread
spins in a short-sleep loop. If the GIL is released, the main thread keeps
making progress (many loop iterations). If the GIL is held for the whole
native call, the main thread blocks after at most a handful of iterations.

Runs under both pytest and the CI ``unittest`` discovery.
"""

import threading
import time
import unittest

import numpy as np

from ordinaldb import IdMapIndex, OrdinalIndex

RNG = np.random.default_rng(42)
DIM = 256
N_VECTORS = 100_000
N_QUERIES = 2_048
N_ADD_VECTORS = 300_000

# With the GIL held for the whole call, the main thread gets at most a few
# iterations (only while the worker is still in Python-level code around the
# native call). With the GIL released, the yielding spin makes thousands.
MIN_ITERATIONS = 30
MIN_CALL_SECONDS = 0.15


def _vectors(n: int) -> np.ndarray:
    vectors = RNG.standard_normal((n, DIM), dtype=np.float32)
    vectors *= 0.1
    return vectors


def _main_thread_progress_during(native_call):
    """Run native_call in a worker thread; count main-thread iterations."""
    started = threading.Event()
    done = threading.Event()
    error = []

    def worker() -> None:
        started.set()
        try:
            native_call()
        except BaseException as exc:  # pragma: no cover - defensive
            error.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=worker)
    begin = time.perf_counter()
    thread.start()
    started.wait()
    iterations = 0
    while not done.is_set():
        # sleep(0) is a pure scheduler yield with no minimum: sleep(0.001)
        # rounds up to the OS timer tick (~15ms on Windows), which caps
        # iterations far below MIN_ITERATIONS for a sub-second call
        # regardless of GIL state. A bare yield counts actual GIL
        # availability: thousands of iterations when released, ~0 when held.
        time.sleep(0)
        iterations += 1
    thread.join()
    elapsed = time.perf_counter() - begin
    if error:
        raise error[0]
    return iterations, elapsed


class SearchReleasesGilTests(unittest.TestCase):
    def _require_slow_enough(self, elapsed: float) -> None:
        if elapsed < MIN_CALL_SECONDS:
            self.skipTest(
                f"native call finished in {elapsed:.3f}s; too fast on this "
                "machine to measure GIL contention reliably"
            )

    def _assert_progress(self, iterations: int, elapsed: float, call: str) -> None:
        self._require_slow_enough(elapsed)
        self.assertGreaterEqual(
            iterations,
            MIN_ITERATIONS,
            f"main thread made only {iterations} iterations during a "
            f"{elapsed:.3f}s {call}: the GIL was not released",
        )

    def test_ordinal_index_search_releases_gil(self):
        idx = OrdinalIndex(dim=DIM, bits=2)
        idx.add(_vectors(N_VECTORS))
        queries = _vectors(N_QUERIES)

        iterations, elapsed = _main_thread_progress_during(
            lambda: idx.search(queries, k=10)
        )
        self._assert_progress(iterations, elapsed, "search")

    def test_id_map_index_search_releases_gil(self):
        idx = IdMapIndex(dim=DIM, bits=2)
        ids = np.arange(N_VECTORS, dtype=np.uint64)
        idx.add_with_ids(_vectors(N_VECTORS), ids)
        queries = _vectors(N_QUERIES)

        iterations, elapsed = _main_thread_progress_during(
            lambda: idx.search(queries, k=10)
        )
        self._assert_progress(iterations, elapsed, "search")

    def test_add_releases_gil(self):
        idx = OrdinalIndex(dim=DIM, bits=2)
        vectors = _vectors(N_ADD_VECTORS)

        iterations, elapsed = _main_thread_progress_during(
            lambda: idx.add(vectors)
        )
        self._assert_progress(iterations, elapsed, "add")


class ConcurrentSearchCorrectnessTests(unittest.TestCase):
    def test_two_threads_search_same_index_get_identical_results(self):
        idx = IdMapIndex(dim=DIM, bits=2)
        n = 2_000
        ids = np.arange(n, dtype=np.uint64)
        idx.add_with_ids(_vectors(n), ids)
        queries = _vectors(16)

        expected_scores, expected_ids = idx.search(queries, k=5)
        results = [None, None]

        def worker(slot: int) -> None:
            for _ in range(20):
                results[slot] = idx.search(queries, k=5)

        threads = [
            threading.Thread(target=worker, args=(slot,)) for slot in (0, 1)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        for scores, found in results:
            np.testing.assert_array_equal(scores, expected_scores)
            np.testing.assert_array_equal(found, expected_ids)


if __name__ == "__main__":
    unittest.main()
