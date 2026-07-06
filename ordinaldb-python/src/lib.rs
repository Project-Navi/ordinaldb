use std::path::PathBuf;

use ordinaldb_adapter_store::{
    acquire_writer_lock, open_verified, write_legacy_snapshot,
    write_legacy_snapshot_with_existing_lock, LegacyPayloads, StoreRevision, WriterLockGuard,
};
use ordinaldb_core::{
    AddError, BuildOptions, ConstructError, DenseError, IdMapIndex, OrdinalIndex, SearchResults,
    SignPolicy,
};
use pyo3::buffer::{Element, PyBuffer};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList, PyModule, PyType};

/// Slot-addressed ordinal-quantized vector index.
///
/// Rows are addressed by their insertion slot (`0..len()`). Use
/// `IdMapIndex` instead when rows need stable 64-bit identifiers that
/// survive removals.
///
/// Args:
///     dim: Vector dimensionality. Omit to create a lazy index whose
///         dimensionality is locked by the first `add` call.
///     bits: Quantization bit width (1, 2, or 4). Defaults to 2.
///     bit_width: Alias for `bits`; provide at most one of the two.
///     sign: Sign-sidecar build policy: "disabled", "optional"
///         (default), or "required". A sidecar needs `bits == 2` and
///         `dim` a multiple of 64; "required" raises when `(dim, bits)`
///         cannot carry one (for a lazy index, on the first `add`).
///
/// Raises:
///     ValueError: If `bits` is unsupported, both aliases are given,
///         `sign` is not one of the three policies, or `sign="required"`
///         cannot be honored for `(dim, bits)`.
#[pyclass(module = "ordinaldb._ordinaldb", name = "OrdinalIndex")]
struct PyOrdinalIndex {
    inner: OrdinalIndex,
}

#[pymethods]
impl PyOrdinalIndex {
    #[new]
    #[pyo3(signature = (dim=None, bits=None, bit_width=None, sign="optional"))]
    #[pyo3(text_signature = "(dim=None, bits=None, bit_width=None, sign=\"optional\")")]
    fn new(
        dim: Option<usize>,
        bits: Option<u8>,
        bit_width: Option<u8>,
        sign: &str,
    ) -> PyResult<Self> {
        let bits = resolve_bits(bits, bit_width)?;
        let options = BuildOptions {
            sign: resolve_sign_policy(sign)?,
        };
        let inner =
            match dim {
                Some(dim) => OrdinalIndex::new_with_build_options(dim, bits, options)
                    .map_err(construct_err)?,
                None => OrdinalIndex::new_lazy_with_build_options(bits, options)
                    .map_err(construct_err)?,
            };
        Ok(Self { inner })
    }

    /// Whether the index currently maintains a sign sidecar for two-stage
    /// search.
    #[getter]
    fn has_sign_sidecar(&self) -> bool {
        self.inner.has_sign_sidecar()
    }

    /// Append vectors to the index.
    ///
    /// Releases the GIL while quantizing, so other Python threads keep
    /// running during large batch inserts.
    ///
    /// Args:
    ///     vectors: 2D C-contiguous NumPy array with dtype float32 of
    ///         shape `(n, dim)`. Values must be finite with
    ///         `|value| < 1e16`.
    ///
    /// Raises:
    ///     ValueError: On dtype/layout/dimension mismatch or non-finite
    ///         values; the index is left unmodified.
    #[pyo3(text_signature = "($self, vectors)")]
    fn add(&mut self, py: Python<'_>, vectors: &Bound<'_, PyAny>) -> PyResult<()> {
        let (slice, dim) = vectors_2d(py, vectors, "vectors")?;
        let inner = &mut self.inner;
        py.detach(|| inner.add_2d(&slice, dim)).map_err(add_err)
    }

    /// Search for the top-`k` nearest rows per query.
    ///
    /// Releases the GIL for the duration of the native search, so
    /// concurrent Python callers are not serialized.
    ///
    /// Args:
    ///     queries: 2D C-contiguous NumPy array with dtype float32 of
    ///         shape `(nq, dim)`.
    ///     k: Number of results per query. Capped at the number of
    ///         searchable rows.
    ///     mask: Optional 1D NumPy bool array of length `len(self)`;
    ///         only slots where the mask is True are searched.
    ///
    /// Returns:
    ///     Tuple `(scores, indices)` of flat NumPy arrays (float32,
    ///     int64), each of length `nq * effective_k`, row-major per
    ///     query.
    ///
    /// Raises:
    ///     ValueError: On malformed queries (wrong dtype/layout/dim,
    ///         NaN/Inf/out-of-range values) or a mask length mismatch.
    #[pyo3(signature = (queries, k, mask=None))]
    #[pyo3(text_signature = "($self, queries, k, mask=None)")]
    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: &Bound<'py, PyAny>,
        k: usize,
        mask: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let (query_slice, _dim) = queries_2d(py, &self.inner, queries)?;
        let mask_values = match mask {
            Some(mask) => Some(mask_1d(mask, self.inner.len())?),
            None => None,
        };
        let inner = &self.inner;
        let results = py
            .detach(|| -> Result<SearchResults, DenseError> {
                match &mask_values {
                    Some(mask_values) => {
                        let candidates = mask_candidates(mask_values)?;
                        inner.search_checked_with_candidates(&query_slice, k, &candidates)
                    }
                    None => inner.search_checked(&query_slice, k),
                }
            })
            .map_err(dense_err)?;
        results_to_py(py, results)
    }

    /// Remove the row at `idx`, moving the last row into its slot.
    ///
    /// Args:
    ///     idx: Slot index of the row to remove.
    ///
    /// Returns:
    ///     The slot index the previously-last row moved from.
    ///
    /// Raises:
    ///     ValueError: If `idx` is out of range.
    #[pyo3(text_signature = "($self, idx)")]
    fn swap_remove(&mut self, idx: usize) -> PyResult<usize> {
        if idx >= self.inner.len() {
            return Err(value_err(format!(
                "index {idx} out of range for index of length {}",
                self.inner.len()
            )));
        }
        Ok(self.inner.swap_remove(idx))
    }

    /// Persist the index as a verified `.odb` bundle directory.
    ///
    /// Releases the GIL while writing to disk.
    ///
    /// Args:
    ///     path: Destination bundle directory path.
    ///
    /// Raises:
    ///     ValueError: If the index cannot be persisted (e.g. lazy with
    ///         no dimensionality yet).
    ///     OSError: On filesystem errors.
    #[pyo3(text_signature = "($self, path)")]
    fn write(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let inner = &self.inner;
        py.detach(|| inner.write(path)).map_err(io_err)
    }

    /// Load an index from a verified `.odb` bundle directory.
    ///
    /// Releases the GIL while reading from disk.
    ///
    /// Args:
    ///     path: Bundle directory previously produced by `write`.
    ///
    /// Returns:
    ///     A new `OrdinalIndex`.
    ///
    /// Raises:
    ///     ValueError: If the bundle is malformed or was written by a
    ///         different index type.
    ///     OSError: On filesystem errors.
    #[classmethod]
    #[pyo3(text_signature = "($cls, path)")]
    fn load(_cls: &Bound<'_, PyType>, py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            inner: py.detach(|| OrdinalIndex::load(path)).map_err(io_err)?,
        })
    }

    /// Return the number of rows stored in the index.
    #[pyo3(text_signature = "($self)")]
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Return True if the index holds no rows.
    #[pyo3(text_signature = "($self)")]
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Return the vector dimensionality.
    ///
    /// Returns 0 if the index is lazy and no dimensionality has been
    /// locked by an `add` yet; use `dim_opt` to distinguish that case.
    #[pyo3(text_signature = "($self)")]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Return the vector dimensionality, or None for an unlocked lazy index.
    #[pyo3(text_signature = "($self)")]
    fn dim_opt(&self) -> Option<usize> {
        self.inner.dim_opt()
    }

    /// Return the quantization bit width (1, 2, or 4).
    #[pyo3(text_signature = "($self)")]
    fn bits(&self) -> u8 {
        self.inner.bits()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

/// Vector index addressing rows by stable unsigned 64-bit identifiers.
///
/// Unlike `OrdinalIndex`, rows keep their external id across removals,
/// so callers never observe slot reshuffling.
///
/// Args:
///     dim: Vector dimensionality. Omit to create a lazy index whose
///         dimensionality is locked by the first `add_with_ids` call.
///     bits: Quantization bit width (1, 2, or 4). Defaults to 2.
///     bit_width: Alias for `bits`; provide at most one of the two.
///     sign: Sign-sidecar build policy: "disabled", "optional"
///         (default), or "required". A sidecar needs `bits == 2` and
///         `dim` a multiple of 64; "required" raises when `(dim, bits)`
///         cannot carry one (for a lazy index, on the first
///         `add_with_ids`).
///
/// Raises:
///     ValueError: If `bits` is unsupported, both aliases are given,
///         `sign` is not one of the three policies, or `sign="required"`
///         cannot be honored for `(dim, bits)`.
#[pyclass(module = "ordinaldb._ordinaldb", name = "IdMapIndex")]
struct PyIdMapIndex {
    inner: IdMapIndex,
}

/// Low-level verified snapshot store used by the Python adapter layer.
///
/// Internal API: prefer `ordinaldb.adapters` for application code. All
/// methods are static; snapshots are written atomically with
/// compare-and-swap revision checks.
#[pyclass(module = "ordinaldb._ordinaldb", name = "_AdapterStateStore")]
struct PyAdapterStateStore;

/// Exclusive writer lock over an adapter store directory.
///
/// Internal API: returned by `_AdapterStateStore.acquire_writer_lock`.
/// Use as a context manager; the lock is released on `__exit__`.
#[pyclass(module = "ordinaldb._ordinaldb", name = "_AdapterWriteLock")]
struct PyAdapterWriteLock {
    inner: Option<WriterLockGuard>,
}

#[pymethods]
impl PyAdapterStateStore {
    /// Acquire the exclusive writer lock for an adapter store directory.
    ///
    /// Releases the GIL while acquiring, so a contended lock does not
    /// stall other Python threads.
    ///
    /// Args:
    ///     path: Adapter store directory.
    ///
    /// Returns:
    ///     An `_AdapterWriteLock` context manager holding the lock.
    ///
    /// Raises:
    ///     ValueError: If the lock cannot be acquired.
    #[staticmethod]
    #[pyo3(text_signature = "(path)")]
    fn acquire_writer_lock(py: Python<'_>, path: PathBuf) -> PyResult<PyAdapterWriteLock> {
        let inner = py
            .detach(|| acquire_writer_lock(path))
            .map_err(adapter_store_err)?;
        Ok(PyAdapterWriteLock { inner: Some(inner) })
    }

    /// Atomically publish a legacy-format snapshot of the adapter state.
    ///
    /// Acquires the writer lock internally. If `expected_revision_json`
    /// is given, the write is a compare-and-swap: it fails unless the
    /// store's current revision matches, protecting against concurrent
    /// writers. Releases the GIL for the duration of the write.
    ///
    /// Args:
    ///     path: Adapter store directory.
    ///     adapter_json: Serialized `adapter.json` payload.
    ///     id_map_json: Serialized `id_map.json` payload.
    ///     documents_json: Serialized `documents.json` payload.
    ///     metadata_json: Serialized `metadata.json` payload.
    ///     expected_revision_json: Optional manifest JSON of the revision
    ///         this write is based on.
    ///
    /// Returns:
    ///     The committed manifest as a JSON string.
    ///
    /// Raises:
    ///     ValueError: On lock failure, revision mismatch, or malformed
    ///         payloads.
    #[staticmethod]
    #[pyo3(signature = (
        path,
        adapter_json,
        id_map_json,
        documents_json,
        metadata_json,
        expected_revision_json=None
    ))]
    #[pyo3(text_signature = "(path, adapter_json, id_map_json, documents_json, \
metadata_json, expected_revision_json=None)")]
    fn write_legacy_snapshot(
        py: Python<'_>,
        path: PathBuf,
        adapter_json: String,
        id_map_json: String,
        documents_json: String,
        metadata_json: String,
        expected_revision_json: Option<String>,
    ) -> PyResult<String> {
        let expected = parse_store_revision(expected_revision_json.as_deref())?;
        let verified = py
            .detach(|| {
                write_legacy_snapshot(
                    path,
                    expected,
                    LegacyPayloads {
                        adapter_json,
                        id_map_json,
                        documents_json,
                        metadata_json,
                    },
                )
            })
            .map_err(adapter_store_err)?;
        serde_json_string(&verified.manifest)
    }

    /// Publish a legacy-format snapshot under an already-held writer lock.
    ///
    /// Same compare-and-swap semantics as `write_legacy_snapshot`, but
    /// assumes the caller already holds the `_AdapterWriteLock` for
    /// `path`. Releases the GIL for the duration of the write.
    ///
    /// Args:
    ///     path: Adapter store directory.
    ///     adapter_json: Serialized `adapter.json` payload.
    ///     id_map_json: Serialized `id_map.json` payload.
    ///     documents_json: Serialized `documents.json` payload.
    ///     metadata_json: Serialized `metadata.json` payload.
    ///     expected_revision_json: Optional manifest JSON of the revision
    ///         this write is based on.
    ///
    /// Returns:
    ///     The committed manifest as a JSON string.
    ///
    /// Raises:
    ///     ValueError: On revision mismatch or malformed payloads.
    #[staticmethod]
    #[pyo3(signature = (
        path,
        adapter_json,
        id_map_json,
        documents_json,
        metadata_json,
        expected_revision_json=None
    ))]
    #[pyo3(text_signature = "(path, adapter_json, id_map_json, documents_json, \
metadata_json, expected_revision_json=None)")]
    fn write_legacy_snapshot_with_existing_lock(
        py: Python<'_>,
        path: PathBuf,
        adapter_json: String,
        id_map_json: String,
        documents_json: String,
        metadata_json: String,
        expected_revision_json: Option<String>,
    ) -> PyResult<String> {
        let expected = parse_store_revision(expected_revision_json.as_deref())?;
        let verified = py
            .detach(|| {
                write_legacy_snapshot_with_existing_lock(
                    path,
                    expected,
                    LegacyPayloads {
                        adapter_json,
                        id_map_json,
                        documents_json,
                        metadata_json,
                    },
                )
            })
            .map_err(adapter_store_err)?;
        serde_json_string(&verified.manifest)
    }

    /// Load and verify the current snapshot payloads from a store.
    ///
    /// Releases the GIL while reading and verifying.
    ///
    /// Args:
    ///     path: Adapter store directory.
    ///     expected_adapter: Optional adapter name that must match the
    ///         stored snapshot.
    ///
    /// Returns:
    ///     Tuple `(adapter_json, id_map_json, documents_json,
    ///     metadata_json)` of the verified payload strings.
    ///
    /// Raises:
    ///     ValueError: If the store is missing, corrupt, fails
    ///         verification, or was written by a different adapter.
    #[staticmethod]
    #[pyo3(signature = (path, expected_adapter=None))]
    #[pyo3(text_signature = "(path, expected_adapter=None)")]
    fn load_legacy_snapshot(
        py: Python<'_>,
        path: PathBuf,
        expected_adapter: Option<String>,
    ) -> PyResult<(String, String, String, String)> {
        let verified = py
            .detach(|| open_verified(path, expected_adapter.as_deref()))
            .map_err(adapter_store_err)?;
        Ok((
            verified.payloads.adapter_json,
            verified.payloads.id_map_json,
            verified.payloads.documents_json,
            verified.payloads.metadata_json,
        ))
    }

    /// Verify a store's integrity and return its manifest.
    ///
    /// Releases the GIL while reading and verifying.
    ///
    /// Args:
    ///     path: Adapter store directory.
    ///     expected_adapter: Optional adapter name that must match the
    ///         stored snapshot.
    ///
    /// Returns:
    ///     The verified manifest as a JSON string.
    ///
    /// Raises:
    ///     ValueError: If verification fails.
    #[staticmethod]
    #[pyo3(signature = (path, expected_adapter=None))]
    #[pyo3(text_signature = "(path, expected_adapter=None)")]
    fn verify(py: Python<'_>, path: PathBuf, expected_adapter: Option<String>) -> PyResult<String> {
        let verified = py
            .detach(|| open_verified(path, expected_adapter.as_deref()))
            .map_err(adapter_store_err)?;
        serde_json_string(&verified.manifest)
    }
}

#[pymethods]
impl PyAdapterWriteLock {
    /// Enter the context manager; the lock is already held.
    fn __enter__(slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf
    }

    /// Release the writer lock. Exceptions are never suppressed.
    fn __exit__(
        &mut self,
        _exc_type: &Bound<'_, PyAny>,
        _exc: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> bool {
        self.inner.take();
        false
    }
}

#[pymethods]
impl PyIdMapIndex {
    #[new]
    #[pyo3(signature = (dim=None, bits=None, bit_width=None, sign="optional"))]
    #[pyo3(text_signature = "(dim=None, bits=None, bit_width=None, sign=\"optional\")")]
    fn new(
        dim: Option<usize>,
        bits: Option<u8>,
        bit_width: Option<u8>,
        sign: &str,
    ) -> PyResult<Self> {
        let bits = resolve_bits(bits, bit_width)?;
        let options = BuildOptions {
            sign: resolve_sign_policy(sign)?,
        };
        let inner = match dim {
            Some(dim) => {
                IdMapIndex::new_with_build_options(dim, bits, options).map_err(construct_err)?
            }
            None => {
                IdMapIndex::new_lazy_with_build_options(bits, options).map_err(construct_err)?
            }
        };
        Ok(Self { inner })
    }

    /// Whether the index currently maintains a sign sidecar for two-stage
    /// search.
    #[getter]
    fn has_sign_sidecar(&self) -> bool {
        self.inner.has_sign_sidecar()
    }

    /// Append vectors under caller-provided stable ids.
    ///
    /// The batch is validated up front and applied atomically: on any
    /// error (including duplicate ids) the index is left unmodified.
    /// Releases the GIL while quantizing.
    ///
    /// Args:
    ///     vectors: 2D C-contiguous NumPy array with dtype float32 of
    ///         shape `(n, dim)`. Values must be finite with
    ///         `|value| < 1e16`.
    ///     ids: 1D C-contiguous NumPy array with dtype uint64 of length
    ///         `n`; ids must be unique and not already present.
    ///
    /// Raises:
    ///     ValueError: On dtype/layout/dimension mismatch, non-finite
    ///         values, or duplicate ids.
    #[pyo3(text_signature = "($self, vectors, ids)")]
    fn add_with_ids(
        &mut self,
        py: Python<'_>,
        vectors: &Bound<'_, PyAny>,
        ids: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let (vector_slice, dim) = vectors_2d(py, vectors, "vectors")?;
        let ids = ids_1d(py, ids, "ids")?;
        let inner = &mut self.inner;
        py.detach(|| inner.add_with_ids_2d(&vector_slice, dim, &ids))
            .map_err(add_err)
    }

    /// Search for the top-`k` nearest rows per query.
    ///
    /// Releases the GIL for the duration of the native search, so
    /// concurrent Python callers are not serialized.
    ///
    /// Args:
    ///     queries: 2D C-contiguous NumPy array with dtype float32 of
    ///         shape `(nq, dim)`.
    ///     k: Number of results per query. Capped at the number of
    ///         searchable rows.
    ///     allowlist: Optional 1D NumPy uint64 array; only these ids are
    ///         searched. Every id must be present in the index.
    ///
    /// Returns:
    ///     Tuple `(scores, ids)` of flat NumPy arrays (float32, uint64),
    ///     each of length `nq * effective_k`, row-major per query.
    ///
    /// Raises:
    ///     ValueError: On malformed queries (wrong dtype/layout/dim,
    ///         NaN/Inf/out-of-range values) or an allowlist id that is
    ///         not present in the index.
    #[pyo3(signature = (queries, k, allowlist=None))]
    #[pyo3(text_signature = "($self, queries, k, allowlist=None)")]
    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: &Bound<'py, PyAny>,
        k: usize,
        allowlist: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let (query_slice, _dim) = id_queries_2d(py, &self.inner, queries)?;
        let allowlist_ids = match allowlist {
            Some(allowlist) => Some(ids_1d(py, allowlist, "allowlist")?),
            None => None,
        };
        let inner = &self.inner;
        let (scores, ids) = py
            .detach(|| {
                inner.search_checked_with_allowlist(&query_slice, k, allowlist_ids.as_deref())
            })
            .map_err(dense_err)?;
        Ok((array_f32(py, scores)?, array_u64(py, ids)?))
    }

    /// Remove the row with the given id.
    ///
    /// Args:
    ///     id: Stable id of the row to remove.
    ///
    /// Returns:
    ///     True if the id was present and removed, False otherwise.
    #[pyo3(text_signature = "($self, id)")]
    fn remove(&mut self, id: u64) -> bool {
        self.inner.remove(id)
    }

    /// Persist the index as a verified `.odb` bundle directory.
    ///
    /// Releases the GIL while writing to disk.
    ///
    /// Args:
    ///     path: Destination bundle directory path.
    ///
    /// Raises:
    ///     ValueError: If the index cannot be persisted (e.g. lazy with
    ///         no dimensionality yet).
    ///     OSError: On filesystem errors.
    #[pyo3(text_signature = "($self, path)")]
    fn write(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        let inner = &self.inner;
        py.detach(|| inner.write(path)).map_err(io_err)
    }

    /// Load an index from a verified `.odb` bundle directory.
    ///
    /// Releases the GIL while reading from disk.
    ///
    /// Args:
    ///     path: Bundle directory previously produced by `write`.
    ///
    /// Returns:
    ///     A new `IdMapIndex`.
    ///
    /// Raises:
    ///     ValueError: If the bundle is malformed or was written by a
    ///         different index type.
    ///     OSError: On filesystem errors.
    #[classmethod]
    #[pyo3(text_signature = "($cls, path)")]
    fn load(_cls: &Bound<'_, PyType>, py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        Ok(Self {
            inner: py.detach(|| IdMapIndex::load(path)).map_err(io_err)?,
        })
    }

    /// Return True if the given id is present in the index.
    #[pyo3(text_signature = "($self, id)")]
    fn contains(&self, id: u64) -> bool {
        self.inner.contains(id)
    }

    /// Return the number of rows stored in the index.
    #[pyo3(text_signature = "($self)")]
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Return True if the index holds no rows.
    #[pyo3(text_signature = "($self)")]
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Return the vector dimensionality.
    ///
    /// Returns 0 if the index is lazy and no dimensionality has been
    /// locked by an `add_with_ids` yet; use `dim_opt` to distinguish
    /// that case.
    #[pyo3(text_signature = "($self)")]
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Return the vector dimensionality, or None for an unlocked lazy index.
    #[pyo3(text_signature = "($self)")]
    fn dim_opt(&self) -> Option<usize> {
        self.inner.dim_opt()
    }

    /// Return the quantization bit width (1, 2, or 4).
    #[pyo3(text_signature = "($self)")]
    fn bits(&self) -> u8 {
        self.inner.bits()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

/// Native extension module backing the `ordinaldb` Python package.
///
/// Exposes `OrdinalIndex` and `IdMapIndex` plus the internal adapter
/// state-store primitives used by `ordinaldb.adapters`.
#[pymodule]
fn _ordinaldb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOrdinalIndex>()?;
    m.add_class::<PyIdMapIndex>()?;
    m.add_class::<PyAdapterStateStore>()?;
    m.add_class::<PyAdapterWriteLock>()?;
    Ok(())
}

fn resolve_bits(bits: Option<u8>, bit_width: Option<u8>) -> PyResult<u8> {
    match (bits, bit_width) {
        (Some(_), Some(_)) => Err(value_err("provide only one of bits or bit_width")),
        (Some(bits), None) | (None, Some(bits)) => Ok(bits),
        (None, None) => Ok(2),
    }
}

fn resolve_sign_policy(sign: &str) -> PyResult<SignPolicy> {
    match sign {
        "disabled" => Ok(SignPolicy::Disabled),
        "optional" => Ok(SignPolicy::Optional),
        "required" => Ok(SignPolicy::Required),
        other => Err(value_err(format!(
            "sign must be one of \"disabled\", \"optional\", or \"required\"; got {other:?}"
        ))),
    }
}

fn vectors_2d(
    py: Python<'_>,
    vectors: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<(Vec<f32>, usize)> {
    let buffer = f32_buffer(vectors, name)?;
    require_2d(&buffer, name)?;
    let dim = buffer.shape()[1];
    Ok((buffer_vec(py, &buffer, name)?, dim))
}

fn queries_2d(
    py: Python<'_>,
    index: &OrdinalIndex,
    queries: &Bound<'_, PyAny>,
) -> PyResult<(Vec<f32>, usize)> {
    let (query_slice, dim) = vectors_2d(py, queries, "queries")?;
    match index.dim_opt() {
        Some(existing) if existing != dim => Err(value_err(format!(
            "query dim mismatch: index dim={existing}, query dim={dim}"
        ))),
        Some(_) => Ok((query_slice, dim)),
        None => Err(value_err(
            "cannot search a lazy empty index before its dim is set",
        )),
    }
}

fn id_queries_2d(
    py: Python<'_>,
    index: &IdMapIndex,
    queries: &Bound<'_, PyAny>,
) -> PyResult<(Vec<f32>, usize)> {
    let (query_slice, dim) = vectors_2d(py, queries, "queries")?;
    match index.dim_opt() {
        Some(existing) if existing != dim => Err(value_err(format!(
            "query dim mismatch: index dim={existing}, query dim={dim}"
        ))),
        Some(_) => Ok((query_slice, dim)),
        None => Err(value_err(
            "cannot search a lazy empty index before its dim is set",
        )),
    }
}

fn ids_1d(py: Python<'_>, ids: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<u64>> {
    let buffer = u64_buffer(ids, name)?;
    require_1d(&buffer, name)?;
    buffer_vec(py, &buffer, name)
}

fn mask_1d(mask: &Bound<'_, PyAny>, expected_len: usize) -> PyResult<Vec<bool>> {
    let message = "mask must be a 1D C-contiguous NumPy array with dtype bool";
    let ndim: usize = mask
        .getattr("ndim")
        .and_then(|ndim| ndim.extract())
        .map_err(|_| value_err(message))?;
    if ndim != 1 {
        return Err(value_err(message));
    }

    let c_contiguous: bool = mask
        .getattr("flags")
        .and_then(|flags| flags.getattr("c_contiguous"))
        .and_then(|flag| flag.extract())
        .map_err(|_| value_err(message))?;
    if !c_contiguous {
        return Err(value_err(message));
    }

    let dtype_kind: String = mask
        .getattr("dtype")
        .and_then(|dtype| dtype.getattr("kind"))
        .and_then(|kind| kind.extract())
        .map_err(|_| value_err(message))?;
    if dtype_kind != "b" {
        return Err(value_err(message));
    }

    let values: Vec<bool> = mask
        .call_method0("tolist")
        .and_then(|values| values.extract())
        .map_err(|_| value_err(message))?;
    if values.len() != expected_len {
        return Err(value_err(format!(
            "mask length {} does not match index length {expected_len}",
            values.len()
        )));
    }
    Ok(values)
}

fn mask_candidates(mask: &[bool]) -> Result<Vec<u32>, DenseError> {
    let mut candidates = Vec::new();
    for (slot, allowed) in mask.iter().enumerate() {
        if *allowed {
            let slot = u32::try_from(slot).map_err(|_| DenseError::SlotIndexOverflow(slot))?;
            candidates.push(slot);
        }
    }
    Ok(candidates)
}

fn f32_buffer(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<PyBuffer<f32>> {
    PyBuffer::get(obj).map_err(|_| {
        value_err(format!(
            "{name} must be a C-contiguous NumPy array with dtype float32"
        ))
    })
}

fn u64_buffer(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<PyBuffer<u64>> {
    PyBuffer::get(obj).map_err(|_| {
        value_err(format!(
            "{name} must be a C-contiguous NumPy array with dtype uint64"
        ))
    })
}

fn require_2d<T: Element>(buffer: &PyBuffer<T>, name: &str) -> PyResult<()> {
    if buffer.dimensions() != 2 {
        return Err(value_err(format!("{name} must be a 2D NumPy array")));
    }
    if !buffer.is_c_contiguous() {
        return Err(contiguous_err(name));
    }
    Ok(())
}

fn require_1d<T: Element>(buffer: &PyBuffer<T>, name: &str) -> PyResult<()> {
    if buffer.dimensions() != 1 {
        return Err(value_err(format!("{name} must be a 1D NumPy array")));
    }
    if !buffer.is_c_contiguous() {
        return Err(contiguous_err(name));
    }
    Ok(())
}

fn buffer_vec<T: Element>(py: Python<'_>, buffer: &PyBuffer<T>, name: &str) -> PyResult<Vec<T>> {
    buffer
        .as_slice(py)
        .ok_or_else(|| contiguous_err(name))
        .map(|slice| slice.iter().map(|cell| cell.get()).collect())
}

fn results_to_py<'py>(
    py: Python<'py>,
    results: SearchResults,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    Ok((
        array_f32(py, results.scores)?,
        array_i64(py, results.indices)?,
    ))
}

fn array_f32(py: Python<'_>, values: Vec<f32>) -> PyResult<Bound<'_, PyAny>> {
    array_from_list(py, values, "float32")
}

fn array_i64(py: Python<'_>, values: Vec<i64>) -> PyResult<Bound<'_, PyAny>> {
    array_from_list(py, values, "int64")
}

fn array_u64(py: Python<'_>, values: Vec<u64>) -> PyResult<Bound<'_, PyAny>> {
    array_from_list(py, values, "uint64")
}

fn array_from_list<'py, T>(
    py: Python<'py>,
    values: Vec<T>,
    dtype: &str,
) -> PyResult<Bound<'py, PyAny>>
where
    T: IntoPyObject<'py>,
{
    let numpy = PyModule::import(py, "numpy")?;
    let list = PyList::new(py, values)?;
    let dtype = numpy.getattr(dtype)?;
    numpy.getattr("asarray")?.call1((list, dtype))
}

fn construct_err(err: ConstructError) -> PyErr {
    value_err(err.to_string())
}

fn add_err(err: AddError) -> PyErr {
    value_err(err.to_string())
}

fn dense_err(err: DenseError) -> PyErr {
    match err {
        DenseError::Io(err) => io_err(err),
        other => value_err(other.to_string()),
    }
}

fn io_err(err: std::io::Error) -> PyErr {
    match err.kind() {
        std::io::ErrorKind::InvalidData | std::io::ErrorKind::InvalidInput => {
            value_err(err.to_string())
        }
        _ => PyOSError::new_err(err.to_string()),
    }
}

fn parse_store_revision(input: Option<&str>) -> PyResult<Option<StoreRevision>> {
    let Some(input) = input else {
        return Ok(None);
    };
    let manifest: serde_json::Value =
        serde_json::from_str(input).map_err(|err| value_err(err.to_string()))?;
    StoreRevision::from_manifest(&manifest)
        .map(Some)
        .map_err(adapter_store_err)
}

fn adapter_store_err(err: ordinaldb_adapter_store::AdapterStoreError) -> PyErr {
    value_err(err.to_string())
}

fn serde_json_string(value: &serde_json::Value) -> PyResult<String> {
    serde_json::to_string(value).map_err(|err| value_err(err.to_string()))
}

fn contiguous_err(name: &str) -> PyErr {
    value_err(format!("{name} must be a C-contiguous NumPy array"))
}

fn value_err(message: impl Into<String>) -> PyErr {
    PyValueError::new_err(message.into())
}
