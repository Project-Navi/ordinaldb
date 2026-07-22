//! [`IdMapIndex`]: an [`OrdinalIndex`] with a caller-facing `u64` ID space.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::{Duration, Instant};

use ordvec_manifest::VerifyOptions;

use crate::manifest::AuxiliaryArtifactDeclaration;
#[cfg(feature = "hybrid")]
use crate::ordinal::{validate_query_buffer, validate_single_query_buffer};
use crate::ordinal::{
    BuildOptions, DenseBundleInspectReport, DenseLoadOptions, DenseSearchOptions, DenseSearchPlan,
    DenseSearchTimings, VerifiedBundleReport,
};
use crate::{AddError, ConstructError, DenseError, OrdinalIndex, SearchResults};

/// Detailed result of [`IdMapIndex::search_with_report`]: the mapped
/// scores/IDs plus the dense search's execution plan and a timing
/// breakdown.
pub struct IdMapSearchReport {
    /// Scores for every returned row, descending (highest similarity
    /// first), parallel to `ids`.
    pub scores: Vec<f32>,
    /// External row IDs corresponding to `scores`, in the same order.
    pub ids: Vec<u64>,
    /// The execution plan [`OrdinalIndex::search_with_report`] resolved
    /// for this query (which search mode ran, effective `k`, candidate
    /// pool size, ...).
    pub dense_plan: DenseSearchPlan,
    /// Timing breakdown for the underlying dense search (validation,
    /// candidate generation, rerank, ...), excluding ID mapping.
    pub dense_timings: DenseSearchTimings,
    /// Time spent translating internal slots back to external row IDs
    /// after the dense search completed.
    pub id_mapping: Duration,
    /// Total wall-clock time for the whole call, including ID mapping.
    pub total: Duration,
}

/// A dense [`OrdinalIndex`] plus a bidirectional mapping between
/// caller-supplied `u64` row IDs and internal slots.
///
/// Use this instead of a bare [`OrdinalIndex`] whenever rows need stable
/// external identity across deletions: [`OrdinalIndex`] identifies rows by
/// their slot position, which [`OrdinalIndex::swap_remove`] can reassign,
/// whereas `IdMapIndex` keeps the ID a caller sees stable until that
/// specific ID is [`Self::remove`]d.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use ordinaldb::IdMapIndex;
///
/// // 4-dimensional vectors, 2-bit RankQuant codes. For `bits == 2`, `dim`
/// // must be a multiple of 4 (preflight with `rankquant_compatible`; see
/// // `ConstructError::DimNotCompatibleWithBits`).
/// let mut index = IdMapIndex::new(4, 2)?;
///
/// // Three row-major vectors with caller-chosen external IDs.
/// index.add_with_ids_2d(
///     &[
///         1.0, 2.0, 3.0, 4.0, // id 10: ascending
///         4.0, 3.0, 2.0, 1.0, // id 20: descending
///         1.0, 3.0, 2.0, 4.0, // id 30: mixed
///     ],
///     4,
///     &[10, 20, 30],
/// )?;
///
/// // Search returns external IDs, not slots; scores are sorted descending.
/// let (_scores, ids) = index.search_checked(&[1.0, 2.0, 3.0, 4.0], 1)?;
/// assert_eq!(ids, vec![10]);
///
/// // Restrict the same query to an allowlist of external IDs.
/// let (_scores, ids) =
///     index.search_checked_with_allowlist(&[1.0, 2.0, 3.0, 4.0], 1, Some(&[20, 30]))?;
/// assert_eq!(ids, vec![30]);
///
/// // Removing an ID keeps the other IDs stable, even though the underlying
/// // slot storage swap-removes rows.
/// assert!(index.remove(10));
/// assert!(!index.contains(10));
/// assert_eq!(index.len(), 2);
/// let (_scores, ids) = index.search_checked(&[1.0, 2.0, 3.0, 4.0], 1)?;
/// assert_eq!(ids, vec![30]);
/// # Ok(())
/// # }
/// ```
pub struct IdMapIndex {
    inner: OrdinalIndex,
    slot_to_id: Vec<u64>,
    id_to_slot: HashMap<u64, usize>,
}

impl IdMapIndex {
    /// Construct an empty index with a fixed `dim` and RankQuant `bits`
    /// (`1`, `2`, or `4`).
    ///
    /// # Errors
    /// Returns [`ConstructError`] if `bits` is unsupported or `dim` is
    /// invalid or incompatible with `bits`.
    pub fn new(dim: usize, bits: u8) -> Result<Self, ConstructError> {
        Self::new_with_build_options(dim, bits, BuildOptions::default())
    }

    /// Construct an empty index with a fixed `dim` and RankQuant `bits`,
    /// with explicit [`BuildOptions`].
    ///
    /// # Errors
    /// Returns [`ConstructError`] if `bits` is unsupported, `dim` is
    /// invalid or incompatible with `bits`, or
    /// [`ConstructError::SignSidecarUnsupported`] when
    /// [`crate::SignPolicy::Required`] cannot be honored for `(dim,
    /// bits)`.
    pub fn new_with_build_options(
        dim: usize,
        bits: u8,
        options: BuildOptions,
    ) -> Result<Self, ConstructError> {
        Ok(Self {
            inner: OrdinalIndex::new_with_build_options(dim, bits, options)?,
            slot_to_id: Vec::new(),
            id_to_slot: HashMap::new(),
        })
    }

    /// Construct an empty index with `bits` fixed but `dim` left
    /// undetermined; `dim` is inferred from the first non-empty batch
    /// passed to [`Self::add_with_ids_2d`].
    ///
    /// # Errors
    /// Returns [`ConstructError::UnsupportedBits`] if `bits` is not `1`,
    /// `2`, or `4`.
    pub fn new_lazy(bits: u8) -> Result<Self, ConstructError> {
        Self::new_lazy_with_build_options(bits, BuildOptions::default())
    }

    /// Like [`Self::new_lazy`], with explicit [`BuildOptions`].
    ///
    /// The sign-sidecar decision is deferred to the first non-empty
    /// [`Self::add_with_ids_2d`], which commits `dim`: under
    /// [`crate::SignPolicy::Required`], a first batch whose `dim` cannot
    /// carry a sidecar is rejected with
    /// [`AddError::SignSidecarUnsupported`] and the index stays lazy.
    ///
    /// # Errors
    /// Returns [`ConstructError::UnsupportedBits`] if `bits` is not `1`,
    /// `2`, or `4`, or [`ConstructError::SignSidecarUnsupportedBits`] if
    /// [`crate::SignPolicy::Required`] is requested with `bits` `1` or `4`.
    pub fn new_lazy_with_build_options(
        bits: u8,
        options: BuildOptions,
    ) -> Result<Self, ConstructError> {
        Ok(Self {
            inner: OrdinalIndex::new_lazy_with_build_options(bits, options)?,
            slot_to_id: Vec::new(),
            id_to_slot: HashMap::new(),
        })
    }

    /// Add row-major vectors with their IDs, using the `dim` the index has
    /// already committed to.
    ///
    /// # Panics
    /// Panics if the index is still lazy (constructed via
    /// [`Self::new_lazy`]) and has not received a first add yet; use
    /// [`Self::add_with_ids_2d`] to establish `dim` on the first call.
    ///
    /// # Errors
    /// See [`Self::add_with_ids_2d`].
    pub fn add_with_ids(&mut self, vectors: &[f32], ids: &[u64]) -> Result<(), AddError> {
        let dim = self.inner.dim_opt().expect(
            "IdMapIndex dim is not set; use add_with_ids_2d(vectors, dim, ids) on the \
             first add or construct with IdMapIndex::new(dim, bits)",
        );
        self.add_with_ids_2d(vectors, dim, ids)
    }

    /// Add row-major vectors with their IDs, checking (and, for a lazy
    /// index, establishing) `dim` explicitly.
    ///
    /// `vectors` is a flat, row-major `f32` buffer of `ids.len()` rows of
    /// `dim` coordinates each. The add is all-or-nothing: IDs are checked
    /// for duplicates (against both the index and the batch itself) before
    /// any vector is inserted, so a rejected batch leaves the index
    /// unchanged. An empty `ids`/`vectors` batch is a no-op that returns
    /// `Ok(())`.
    ///
    /// # Errors
    /// - [`AddError::DimInvalid`] if `dim` is less than 2 or does not fit
    ///   in a `u16` (matching [`OrdinalIndex::add_2d`]'s contract).
    /// - [`AddError::VectorBufferNotMultipleOfDim`] if `vectors.len()` is
    ///   not a multiple of `dim`.
    /// - [`AddError::IdsCountMismatch`] if `ids.len()` does not match the
    ///   implied row count.
    /// - [`AddError::IdAlreadyPresent`] if an ID is already in the index or
    ///   duplicated within `ids`.
    /// - [`AddError::DimMismatch`], [`AddError::DimNotCompatibleWithBits`],
    ///   or [`AddError::InvalidInputValue`] as propagated from the
    ///   underlying [`OrdinalIndex::add_2d`].
    pub fn add_with_ids_2d(
        &mut self,
        vectors: &[f32],
        dim: usize,
        ids: &[u64],
    ) -> Result<(), AddError> {
        if dim < 2 || dim > u16::MAX as usize {
            return Err(AddError::DimInvalid(dim));
        }
        if !vectors.len().is_multiple_of(dim) {
            return Err(AddError::VectorBufferNotMultipleOfDim {
                vectors_len: vectors.len(),
                dim,
            });
        }

        let row_count = vectors.len() / dim;
        if ids.len() != row_count {
            return Err(AddError::IdsCountMismatch {
                expected: row_count,
                got: ids.len(),
            });
        }

        // Validate the entire ID batch up front — against the index and
        // against itself — so a rejected add never leaves the index
        // partially updated.
        let mut batch_ids = HashSet::with_capacity(ids.len());
        for &id in ids {
            let clashes_with_index = self.id_to_slot.contains_key(&id);
            if clashes_with_index || !batch_ids.insert(id) {
                return Err(AddError::IdAlreadyPresent(id));
            }
        }

        if row_count == 0 {
            return Ok(());
        }

        let first_new_slot = self.inner.len();
        self.inner.add_2d(vectors, dim)?;
        for (offset, &id) in ids.iter().enumerate() {
            self.id_to_slot.insert(id, first_new_slot + offset);
        }
        self.slot_to_id.extend_from_slice(ids);

        Ok(())
    }

    /// Remove the row identified by `id`. Returns `false` — with the index
    /// untouched — when the ID is unknown.
    ///
    /// Removal is O(1): the underlying [`OrdinalIndex`] backfills the freed
    /// slot from its tail row, and the ID bookkeeping mirrors that move, so
    /// every surviving ID keeps resolving to its original vector.
    pub fn remove(&mut self, id: u64) -> bool {
        let slot = match self.id_to_slot.remove(&id) {
            Some(slot) => slot,
            None => return false,
        };

        let backfilled_from = self.inner.swap_remove(slot);
        debug_assert_eq!(
            backfilled_from,
            self.slot_to_id.len() - 1,
            "inner index must backfill the freed slot from its tail"
        );

        // Mirror the inner move in the ID tables. After the swap-remove,
        // the former tail ID (if any) occupies `slot` and needs its
        // reverse mapping refreshed; removing the tail itself leaves
        // nothing at `slot`.
        let evicted = self.slot_to_id.swap_remove(slot);
        debug_assert_eq!(evicted, id, "slot table out of sync with ID map");
        if let Some(&relocated_id) = self.slot_to_id.get(slot) {
            self.id_to_slot.insert(relocated_id, slot);
        }
        true
    }

    /// Search for the `k` nearest rows to each query, returning flat
    /// `(scores, ids)` parallel vectors sorted by descending score.
    ///
    /// `queries` is a flat, row-major `f32` buffer of `dim`-sized query
    /// vectors. Equivalent to
    /// `self.search_with_allowlist(queries, k, None)`; see that method for
    /// the shared panics.
    pub fn search(&self, queries: &[f32], k: usize) -> (Vec<f32>, Vec<u64>) {
        self.search_with_allowlist(queries, k, None)
    }

    /// Like [`Self::search`], but restricted to `allowlist` (a set of
    /// external row IDs) when `Some`.
    ///
    /// # Panics
    /// Panics if `allowlist` contains an ID that is not present in the
    /// index, or (only reachable with far more than `u32::MAX` rows) if an
    /// internal slot does not fit in a `u32`. Prefer
    /// [`Self::search_checked_with_allowlist`] when the allowlist may
    /// contain stale or untrusted IDs.
    pub fn search_with_allowlist(
        &self,
        queries: &[f32],
        k: usize,
        allowlist: Option<&[u64]>,
    ) -> (Vec<f32>, Vec<u64>) {
        let results = if let Some(ids) = allowlist {
            let mut candidates = Vec::with_capacity(ids.len());
            for &id in ids {
                let Some(&slot) = self.id_to_slot.get(&id) else {
                    panic!("id {id} in allowlist is not present in index");
                };
                let slot = u32::try_from(slot).expect("slot index exceeds u32 range");
                candidates.push(slot);
            }
            candidates.sort_unstable();
            candidates.dedup();
            self.inner.search_with_candidates(queries, k, &candidates)
        } else {
            self.inner.search(queries, k)
        };
        let ids = results
            .indices
            .iter()
            .map(|slot| self.slot_to_id[*slot as usize])
            .collect();
        (results.scores, ids)
    }

    /// `Result`-returning equivalent of [`Self::search`].
    ///
    /// # Errors
    /// See [`Self::search_checked_with_allowlist`].
    pub fn search_checked(
        &self,
        queries: &[f32],
        k: usize,
    ) -> Result<(Vec<f32>, Vec<u64>), DenseError> {
        self.search_checked_with_allowlist(queries, k, None)
    }

    /// `Result`-returning equivalent of [`Self::search_with_allowlist`].
    ///
    /// # Errors
    /// Returns [`DenseError::AllowlistRowIdMissing`] if `allowlist`
    /// contains an ID not present in the index, or
    /// [`DenseError::SlotIndexOverflow`] if an internal slot does not fit
    /// in a `u32`. Also propagates [`DenseError::InvalidQueryDim`] /
    /// [`DenseError::InvalidQueryValue`] for a malformed `queries` buffer.
    pub fn search_checked_with_allowlist(
        &self,
        queries: &[f32],
        k: usize,
        allowlist: Option<&[u64]>,
    ) -> Result<(Vec<f32>, Vec<u64>), DenseError> {
        let results = self.search_results_checked(queries, k, allowlist)?;
        self.scores_ids_from_results(results)
    }

    /// Search like [`Self::search_checked`], additionally returning the
    /// resolved execution plan and a timing breakdown. Does not support an
    /// allowlist.
    ///
    /// # Errors
    /// See [`Self::search_checked`].
    pub fn search_with_report(
        &self,
        queries: &[f32],
        k: usize,
        options: DenseSearchOptions,
    ) -> Result<IdMapSearchReport, DenseError> {
        let total_started = Instant::now();
        let dense = self.inner.search_with_report(queries, k, options)?;
        let crate::ordinal::DenseSearchReport {
            results,
            plan,
            timings,
        } = dense;
        let mapping_started = Instant::now();
        let (scores, ids) = self.scores_ids_from_results(results)?;
        let id_mapping = mapping_started.elapsed();
        Ok(IdMapSearchReport {
            scores,
            ids,
            dense_plan: plan,
            dense_timings: timings,
            id_mapping,
            total: total_started.elapsed(),
        })
    }

    fn search_results_checked(
        &self,
        queries: &[f32],
        k: usize,
        allowlist: Option<&[u64]>,
    ) -> Result<SearchResults, DenseError> {
        let Some(ids) = allowlist else {
            return self.inner.search_checked(queries, k);
        };
        let candidates = self.candidate_slots_from_allowlist(ids)?;
        self.inner
            .search_checked_with_candidates(queries, k, &candidates)
    }

    fn candidate_slots_from_allowlist(&self, ids: &[u64]) -> Result<Vec<u32>, DenseError> {
        let mut candidates = Vec::with_capacity(ids.len());
        for &id in ids {
            let Some(&slot) = self.id_to_slot.get(&id) else {
                return Err(DenseError::AllowlistRowIdMissing(id));
            };
            let slot = u32::try_from(slot).map_err(|_| DenseError::SlotIndexOverflow(slot))?;
            candidates.push(slot);
        }
        candidates.sort_unstable();
        candidates.dedup();
        Ok(candidates)
    }

    fn scores_ids_from_results(
        &self,
        results: SearchResults,
    ) -> Result<(Vec<f32>, Vec<u64>), DenseError> {
        Self::validate_search_results_shape(&results)?;
        let mut scores = Vec::with_capacity(results.scores.len());
        let mut ids = Vec::with_capacity(results.indices.len());
        for (score, slot) in results.scores.into_iter().zip(results.indices) {
            let Some(row_id) = self.row_id_for_slot(slot)? else {
                continue;
            };
            scores.push(score);
            ids.push(row_id);
        }
        Ok((scores, ids))
    }

    fn row_id_for_slot(&self, slot: i64) -> Result<Option<u64>, DenseError> {
        if slot < 0 {
            return Ok(None);
        }
        let slot = slot as usize;
        let id = *self
            .slot_to_id
            .get(slot)
            .ok_or_else(|| DenseError::metadata_mismatch(format!("slot {slot} out of range")))?;
        Ok(Some(id))
    }

    #[cfg(feature = "hybrid")]
    fn ranked_lists_from_results(
        &self,
        results: SearchResults,
    ) -> Result<Vec<Vec<crate::hybrid::ScoredRow>>, DenseError> {
        Self::validate_search_results_shape(&results)?;
        let mut rows = Vec::with_capacity(results.nq);
        for query_idx in 0..results.nq {
            let start = query_idx * results.k;
            let end = start + results.k;
            let mut query_rows = Vec::with_capacity(results.k);
            for i in start..end {
                let Some(row_id) = self.row_id_for_slot(results.indices[i])? else {
                    continue;
                };
                query_rows.push(crate::hybrid::ScoredRow {
                    row_id,
                    score: results.scores[i],
                });
            }
            rows.push(query_rows);
        }
        Ok(rows)
    }

    #[cfg(feature = "hybrid")]
    fn ranked_batch_from_results(
        &self,
        results: SearchResults,
    ) -> Result<crate::hybrid::RankedBatch, DenseError> {
        let rows = self.ranked_lists_from_results(results)?;
        crate::hybrid::RankedBatch::from_ranked_lists(rows)
            .map_err(|err| DenseError::metadata_mismatch(err.to_string()))
    }

    fn validate_search_results_shape(results: &SearchResults) -> Result<(), DenseError> {
        let expected = results
            .nq
            .checked_mul(results.k)
            .ok_or_else(|| DenseError::Limit("nq * k overflow in dense search results".into()))?;
        if results.scores.len() != expected || results.indices.len() != expected {
            return Err(DenseError::metadata_mismatch(format!(
                "dense search result shape mismatch: nq={}, k={}, scores={}, indices={}",
                results.nq,
                results.k,
                results.scores.len(),
                results.indices.len()
            )));
        }
        Ok(())
    }

    /// Search a single query, returning up to `k`
    /// [`crate::hybrid::ScoredRow`]s (`row_id`/`score` pairs) sorted by
    /// descending score. Only available with the `hybrid` Cargo feature.
    ///
    /// # Errors
    /// Returns [`DenseError::InvalidQueryDim`] if `query.len()` does not
    /// equal the index's `dim`, or [`DenseError::InvalidQueryValue`] for a
    /// non-finite or out-of-range coordinate.
    #[cfg(feature = "hybrid")]
    pub fn search_rows(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<crate::hybrid::ScoredRow>, DenseError> {
        validate_single_query_buffer(query, self.inner.dim_opt())?;
        let batch = self.search_batch_rows(query, k)?;
        Ok(batch.hits().to_vec())
    }

    /// Batched form of [`Self::search_rows`]: `queries` is a flat,
    /// row-major buffer of `dim`-sized queries, returned as a single
    /// [`crate::hybrid::RankedBatch`]. Only available with the `hybrid`
    /// Cargo feature.
    ///
    /// # Errors
    /// See [`Self::search_checked`].
    #[cfg(feature = "hybrid")]
    pub fn search_batch_rows(
        &self,
        queries: &[f32],
        k: usize,
    ) -> Result<crate::hybrid::RankedBatch, DenseError> {
        let results = self.search_results_checked(queries, k, None)?;
        self.ranked_batch_from_results(results)
    }

    /// Like [`Self::search_rows`], restricted to `allowlist` (a set of
    /// external row IDs) when `Some`. Only available with the `hybrid`
    /// Cargo feature.
    ///
    /// # Errors
    /// See [`Self::search_checked_with_allowlist`].
    #[cfg(feature = "hybrid")]
    pub fn search_rows_with_allowlist(
        &self,
        query: &[f32],
        k: usize,
        allowlist: Option<&[u64]>,
    ) -> Result<Vec<crate::hybrid::ScoredRow>, DenseError> {
        validate_single_query_buffer(query, self.inner.dim_opt())?;
        let batch = self.search_batch_rows_with_allowlists(query, k, allowlist.map(|ids| [ids]))?;
        Ok(batch.hits().to_vec())
    }

    /// Batched form of [`Self::search_rows_with_allowlist`]: `allowlists`
    /// yields one allowlist per query in `queries` (or `None` to search
    /// unrestricted). Only available with the `hybrid` Cargo feature.
    ///
    /// # Errors
    /// Returns [`DenseError::MetadataMismatch`] if the number of allowlists
    /// does not match the number of queries; otherwise as
    /// [`Self::search_checked_with_allowlist`].
    #[cfg(feature = "hybrid")]
    pub fn search_batch_rows_with_allowlists<'a, I>(
        &self,
        queries: &[f32],
        k: usize,
        allowlists: Option<I>,
    ) -> Result<crate::hybrid::RankedBatch, DenseError>
    where
        I: IntoIterator<Item = &'a [u64]>,
    {
        let Some(allowlists) = allowlists else {
            return self.search_batch_rows(queries, k);
        };
        let Some(dim) = self.inner.dim_opt() else {
            if queries.is_empty() {
                return Ok(crate::hybrid::RankedBatch::empty(0));
            }
            return Err(DenseError::InvalidQueryDim {
                len: queries.len(),
                dim: 0,
            });
        };
        let nq = validate_query_buffer(queries, dim)?;
        let lists = allowlists.into_iter().collect::<Vec<_>>();
        if lists.len() != nq {
            return Err(DenseError::metadata_mismatch(format!(
                "allowlist count {} does not match query count {nq}",
                lists.len()
            )));
        }
        let mut rows = Vec::with_capacity(nq);
        for (query_idx, query) in queries.chunks_exact(dim).enumerate() {
            let results = self.search_results_checked(query, k, Some(lists[query_idx]))?;
            let mut query_rows = self.ranked_lists_from_results(results)?;
            let query_rows = query_rows.pop().unwrap_or_default();
            rows.push(query_rows);
        }
        crate::hybrid::RankedBatch::from_ranked_lists(rows)
            .map_err(|err| DenseError::metadata_mismatch(err.to_string()))
    }

    /// Whether the index currently holds a row for `id`.
    pub fn contains(&self, id: u64) -> bool {
        self.id_to_slot.contains_key(&id)
    }

    /// How many rows the index currently holds.
    pub fn len(&self) -> usize {
        self.slot_to_id.len()
    }

    /// Whether the index holds no rows at all.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The vector dimension every row must match. A still-lazy index (one
    /// built with [`Self::new_lazy`] that has yet to see its first add)
    /// reports `0` here; reach for [`Self::dim_opt`] when that case must
    /// be told apart, since a committed `dim` of `0` is impossible.
    pub fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// [`Self::dim`] as an `Option`: `None` until a lazy index has its
    /// dimension established by the first add.
    pub fn dim_opt(&self) -> Option<usize> {
        self.inner.dim_opt()
    }

    /// The RankQuant bit width (`1`, `2`, or `4`) chosen at construction;
    /// it never changes afterward.
    pub fn bits(&self) -> u8 {
        self.inner.bits()
    }

    /// Returns `true` if the index currently maintains a sign sidecar for
    /// two-stage search.
    pub fn has_sign_sidecar(&self) -> bool {
        self.inner.has_sign_sidecar()
    }

    /// Write the index as a `.odb` bundle (including the ID sidecar) to
    /// `path`, atomically.
    ///
    /// # Errors
    /// Returns an `InvalidInput` [`std::io::Error`] if the index is still
    /// lazy and `dim` has not been established, or any I/O/verification
    /// error encountered while writing.
    pub fn write(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let inner = self.inner.rankquant().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "cannot persist a lazy IdMapIndex before its dim is set",
            )
        })?;
        crate::io::write_id_map_bundle(path, inner, self.inner.sign_bitmap(), &self.slot_to_id)
    }

    /// Like [`Self::write`], but with explicit manifest creation options
    /// and extra caller-supplied auxiliary artifacts, returning a
    /// [`VerifiedBundleReport`] describing what was written.
    ///
    /// # Errors
    /// Returns [`DenseError::MetadataMismatch`] if the index is still
    /// lazy, or any I/O/manifest error encountered while writing.
    pub fn write_verified_bundle(
        &self,
        path: impl AsRef<Path>,
        manifest_options: crate::manifest::CreateManifestOptions,
        auxiliary_artifacts: Vec<AuxiliaryArtifactDeclaration>,
    ) -> Result<VerifiedBundleReport, DenseError> {
        let path = path.as_ref();
        let inner = self.inner.rankquant().ok_or_else(|| {
            DenseError::metadata_mismatch("cannot persist a lazy IdMapIndex before its dim is set")
        })?;
        crate::io::write_id_map_bundle_with_options(
            path,
            inner,
            self.inner.sign_bitmap(),
            &self.slot_to_id,
            crate::io::BundleWriteOptions {
                manifest_options,
                auxiliary_artifacts,
            },
        )?;
        Ok(VerifiedBundleReport {
            path: path.to_path_buf(),
            manifest_path: path.join(crate::io::MANIFEST_FILE),
            dim: self.dim(),
            bits: self.bits(),
            row_count: self.len(),
            has_sign: self.inner.has_sign_sidecar(),
            has_ids: true,
        })
    }

    /// Load a bundle directory previously written by [`Self::write`] or
    /// [`Self::write_verified_bundle`], with default manifest
    /// verification.
    ///
    /// # Errors
    /// Returns an `InvalidData` [`std::io::Error`] if the bundle is
    /// missing its ID sidecar (it was written by a bare [`OrdinalIndex`] —
    /// load it with [`OrdinalIndex::load`] instead), fails manifest
    /// verification, or is otherwise malformed.
    pub fn load(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let artifacts = crate::io::load_id_map_bundle(path)?;
        let inner = OrdinalIndex::from_loaded_parts(artifacts.rankquant, artifacts.sign)?;
        Self::from_loaded_parts(inner, artifacts.ids)
    }

    /// Load a bundle from an explicit `manifest.json` path with
    /// caller-controlled [`VerifyOptions`] and [`DenseLoadOptions`].
    ///
    /// # Errors
    /// Returns [`DenseError::RowIdentity`] if the verified bundle has no ID
    /// sidecar (load it with [`OrdinalIndex::open_verified`] instead), or
    /// any manifest/verification/I/O error from the underlying load.
    pub fn open_verified(
        manifest_path: impl AsRef<Path>,
        verify_options: VerifyOptions,
        load_options: DenseLoadOptions,
    ) -> Result<Self, DenseError> {
        let (rankquant, sign, ids_path) = crate::ordinal::load_verified_ordinal_parts(
            manifest_path.as_ref(),
            verify_options,
            load_options,
        )?;
        let ids_path = ids_path.ok_or_else(|| {
            DenseError::row_identity("verified bundle is missing required OrdinalDB ID sidecar")
        })?;
        let ids = crate::io::read_ids_file(&ids_path, rankquant.len())?;
        let inner = OrdinalIndex::from_loaded_parts(rankquant, sign)?;
        Ok(Self::from_loaded_parts(inner, ids)?)
    }

    /// Summarize the in-memory index's shape (dim, bits, row count, whether
    /// a sign sidecar and ID sidecar are present). `manifest_path`/
    /// `index_path` are always `None`: this describes the index in memory,
    /// not a bundle on disk.
    pub fn inspect(&self) -> DenseBundleInspectReport {
        DenseBundleInspectReport {
            manifest_path: None,
            index_path: None,
            dim: self.dim(),
            bits: self.bits(),
            row_count: self.len(),
            has_sign: self.inner.has_sign_sidecar(),
            has_ids: true,
            row_identity_kind: "row_id_identity".to_string(),
        }
    }

    fn from_loaded_parts(inner: OrdinalIndex, slot_to_id: Vec<u64>) -> std::io::Result<Self> {
        if inner.len() != slot_to_id.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "ID sidecar count {} does not match index len {}",
                    slot_to_id.len(),
                    inner.len()
                ),
            ));
        }
        let mut id_to_slot = HashMap::with_capacity(slot_to_id.len());
        for (slot, &id) in slot_to_id.iter().enumerate() {
            if id_to_slot.insert(id, slot).is_some() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("duplicate ID {id} in persisted ID sidecar"),
                ));
            }
        }
        Ok(Self {
            inner,
            slot_to_id,
            id_to_slot,
        })
    }
}

#[cfg(all(test, feature = "hybrid"))]
mod tests {
    use super::*;

    #[test]
    fn dense_ranked_batch_filters_negative_slots_without_score_misalignment() {
        let index = fixture_index();
        let results = SearchResults {
            scores: vec![1.0, 999.0, 0.9, 0.8, 777.0, 666.0],
            indices: vec![0, -1, 1, 2, -1, -1],
            nq: 2,
            k: 3,
        };

        let batch = index.ranked_batch_from_results(results).unwrap();
        assert_eq!(batch.query_count(), 2);
        assert_eq!(
            batch.hits_for_query(0).unwrap(),
            &[
                crate::hybrid::ScoredRow {
                    row_id: 10,
                    score: 1.0
                },
                crate::hybrid::ScoredRow {
                    row_id: 20,
                    score: 0.9
                }
            ]
        );
        assert_eq!(
            batch.hits_for_query(1).unwrap(),
            &[crate::hybrid::ScoredRow {
                row_id: 30,
                score: 0.8
            }]
        );
    }

    #[test]
    fn dense_flat_scores_and_ids_filter_negative_slots_together() {
        let index = fixture_index();
        let results = SearchResults {
            scores: vec![1.0, 999.0, 0.9],
            indices: vec![0, -1, 1],
            nq: 1,
            k: 3,
        };

        let (scores, ids) = index.scores_ids_from_results(results).unwrap();
        assert_eq!(scores, vec![1.0, 0.9]);
        assert_eq!(ids, vec![10, 20]);
    }

    fn fixture_index() -> IdMapIndex {
        let mut index = IdMapIndex::new(64, 2).unwrap();
        let vectors = vec![0.125; 3 * 64];
        index.add_with_ids(&vectors, &[10, 20, 30]).unwrap();
        index
    }
}
