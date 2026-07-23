//! [`OrdinalIndex`]: the dense, slot-addressed RankQuant index, plus the
//! search/build/load option types and reports it shares with
//! [`crate::IdMapIndex`].

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use ordvec::{RankQuant, SignBitmap, SubsetScratch};
use ordvec_manifest::{ManifestIndexKind, ManifestIndexParams, VerifiedLoadPlan, VerifyOptions};
use rayon::prelude::*;

use crate::id_map::IdMapIndex;
use crate::manifest::{AuxiliaryArtifactDeclaration, VerifiedAuxiliaryArtifactExt};
use crate::{AddError, ConstructError, DenseError};

/// Top-k search results, laid out as `nq` contiguous blocks of `k`, sorted
/// by descending score (highest similarity first) within each block. See
/// `ordvec::SearchResults` for the field-level contract; `indices` holds
/// `-1` for any unfilled slot (fewer than `k` candidates were available).
pub type SearchResults = ordvec::SearchResults;

const MAX_INPUT_MAGNITUDE: f32 = 1e16;
const TWO_STAGE_MIN_CANDIDATES: usize = 256;
const TWO_STAGE_K_MULTIPLIER: usize = 32;
const TWO_STAGE_MAX_SCORE_CELLS: usize = 1_048_576;

/// Dense, slot-addressed RankQuant index.
///
/// Vectors are appended row-major and quantized into `bits`-per-coordinate
/// RankQuant codes; a vector's position in insertion order (its "slot",
/// `0..len()`) doubles as its row identity (`row_id_identity`) — there is
/// no separate ID space. [`Self::swap_remove`] can reassign a slot's
/// occupant, so use [`crate::IdMapIndex`] instead when rows need external
/// IDs that stay stable across deletions.
///
/// When `bits == 2` and `dim` is a multiple of `64` (and
/// [`BuildOptions::sign`] permits one), the index also maintains a
/// `SignBitmap` sidecar used by [`DenseSearchMode::SignTwoStage`] to
/// generate a candidate shortlist before an exact RankQuant rerank.
pub struct OrdinalIndex {
    dim: Option<usize>,
    bits: u8,
    inner: Option<RankQuant>,
    sign: Option<SignBitmap>,
    sign_policy: SignPolicy,
}

/// Policy controlling whether an index builds and maintains a `SignBitmap`
/// sidecar for two-stage search. A sidecar is only *possible* when
/// `bits == 2` and `dim` is a multiple of `64` (see [`sign_compatible`]);
/// the policy decides what happens when `(dim, bits)` cannot carry one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignPolicy {
    /// Never build a sign sidecar, even when `(dim, bits)` supports one.
    Disabled,
    /// Build a sign sidecar when `(dim, bits)` supports one; construct
    /// without one otherwise. This is the default.
    Optional,
    /// Fail construction when `(dim, bits)` cannot carry a sign sidecar. A
    /// lazy index rejects a bit width that can never support the sidecar up
    /// front; for `bits == 2`, the remaining `dim` check runs when the first
    /// non-empty add commits `dim`, surfacing as
    /// [`AddError::SignSidecarUnsupported`].
    Required,
}

/// Options controlling how a new index is built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BuildOptions {
    /// Whether to build and maintain a `SignBitmap` sidecar for two-stage
    /// search, and what to do when `(dim, bits)` cannot carry one — see
    /// [`SignPolicy`]. Use [`sign_compatible`] to check a `(dim, bits)`
    /// pair up front.
    pub sign: SignPolicy,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            sign: SignPolicy::Optional,
        }
    }
}

/// Policy controlling whether a verified load requires, tolerates, or
/// rejects a sign sidecar on the bundle (see [`DenseLoadOptions::sign`]).
/// This is the load-side counterpart of the build-side [`SignPolicy`]: it
/// never changes what was written, only whether the load accepts it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignLoadPolicy {
    /// Require the sidecar whenever the bundle's `(dim, bits)` can carry
    /// one (see [`sign_compatible`]), failing with
    /// [`DenseError::MissingSignSidecar`] when it is absent; bundles that
    /// cannot carry a sidecar load without one. This is the default.
    RequireIfSupported,
    /// Fail the load with [`DenseError::MissingSignSidecar`] unless the
    /// bundle declares a sign sidecar.
    Require,
    /// Load the sidecar when the bundle declares one and proceed without
    /// it otherwise; never fail over sidecar presence.
    Any,
    /// Fail the load with [`DenseError::SignSidecarForbidden`] when the
    /// bundle declares a sign sidecar.
    Forbid,
}

/// Options controlling how a verified bundle is loaded (see
/// [`OrdinalIndex::open_verified`] / [`crate::IdMapIndex::open_verified`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseLoadOptions {
    /// Sign-sidecar load policy — see [`SignLoadPolicy`]. Defaults to
    /// [`SignLoadPolicy::RequireIfSupported`]: a bundle whose `(dim, bits)`
    /// can carry a sidecar must have one to load.
    pub sign: SignLoadPolicy,
    /// If `Some`, fail the load with [`DenseError::MetadataMismatch`] when
    /// the bundle's manifest dim does not match.
    pub expected_dim: Option<usize>,
    /// If `Some`, fail the load with [`DenseError::MetadataMismatch`] when
    /// the bundle's manifest RankQuant bit width does not match.
    pub expected_bits: Option<u8>,
}

impl Default for DenseLoadOptions {
    fn default() -> Self {
        Self {
            sign: SignLoadPolicy::RequireIfSupported,
            expected_dim: None,
            expected_bits: None,
        }
    }
}

/// Which search strategy to use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DenseSearchMode {
    /// Use the sign sidecar's two-stage candidate generation if the index
    /// has one; otherwise silently fall back to an exact RankQuant scan.
    /// This fallback never surfaces as an error or panic, even through the
    /// panicking search entry points — use
    /// [`OrdinalIndex::search_with_report`] to see which strategy actually
    /// ran ([`DenseSearchPlan::execution`]).
    #[default]
    Auto,
    /// Always brute-force score every vector's RankQuant code, ignoring
    /// any sign sidecar.
    ExactRankQuant,
    /// Always use the sign sidecar's two-stage candidate generation
    /// followed by an exact rerank of the shortlist. Requires a sign
    /// sidecar: the checked/report search APIs return
    /// [`DenseError::MissingSignSidecar`] if the index has none, while the
    /// panicking `search`/`search_with_options` fall back to
    /// [`Self::ExactRankQuant`] instead of panicking in that specific
    /// case (with no report of the downgrade — use
    /// [`OrdinalIndex::search_with_report`] for execution visibility).
    SignTwoStage,
}

/// Sizing policy for [`DenseSearchMode::SignTwoStage`]'s candidate
/// shortlist.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TwoStageOptions {
    /// Minimum number of candidates to generate, regardless of `k`.
    pub min_candidates: usize,
    /// Candidates scale linearly with `k`: at least `k * k_multiplier`
    /// candidates are requested (subject to `min_candidates` and
    /// `max_candidates`).
    pub k_multiplier: usize,
    /// Upper bound on the candidate pool size, if any.
    pub max_candidates: Option<usize>,
}

impl TwoStageOptions {
    /// A fixed-size candidate pool: always exactly `candidates` (clamped to
    /// the search space), regardless of `k`.
    pub fn fixed_candidates(candidates: usize) -> Self {
        Self {
            min_candidates: candidates,
            k_multiplier: 0,
            max_candidates: Some(candidates),
        }
    }

    /// Resolve the candidate pool size for a search requesting `k` results
    /// over `search_space` total vectors: `max(min_candidates, k *
    /// k_multiplier)`, clamped to `max_candidates` and to `search_space`.
    /// Returns `0` if `k == 0` or `search_space == 0`.
    pub fn candidate_count(&self, k: usize, search_space: usize) -> usize {
        if k == 0 || search_space == 0 {
            return 0;
        }
        let mut count = self.min_candidates.max(k.saturating_mul(self.k_multiplier));
        if let Some(max_candidates) = self.max_candidates {
            count = count.min(max_candidates);
        }
        count.min(search_space)
    }
}

impl Default for TwoStageOptions {
    fn default() -> Self {
        Self {
            min_candidates: TWO_STAGE_MIN_CANDIDATES,
            k_multiplier: TWO_STAGE_K_MULTIPLIER,
            max_candidates: None,
        }
    }
}

/// Options for the checked/report search entry points
/// (`search_checked_with_options`, `search_with_report`, ...).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseSearchOptions {
    /// Which search strategy to use.
    pub mode: DenseSearchMode,
    /// Candidate shortlist sizing policy, used only when the resolved
    /// strategy is [`DenseSearchMode::SignTwoStage`].
    pub two_stage: TwoStageOptions,
}

impl Default for DenseSearchOptions {
    fn default() -> Self {
        Self {
            mode: DenseSearchMode::Auto,
            two_stage: TwoStageOptions::default(),
        }
    }
}

impl DenseSearchOptions {
    /// Options that force an exact RankQuant scan, ignoring any sign
    /// sidecar.
    pub fn exact_rankquant() -> Self {
        Self {
            mode: DenseSearchMode::ExactRankQuant,
            ..Self::default()
        }
    }

    /// Options that force the sign-sidecar two-stage search with the given
    /// candidate-pool policy.
    pub fn sign_two_stage(two_stage: TwoStageOptions) -> Self {
        Self {
            mode: DenseSearchMode::SignTwoStage,
            two_stage,
        }
    }
}

/// Which search strategy a resolved [`DenseSearchOptions::mode`] of
/// [`DenseSearchMode::Auto`] actually ran as. See [`DenseSearchPlan`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseSearchExecution {
    /// An exact brute-force RankQuant scan of every candidate ran.
    ExactRankQuant,
    /// The sign-sidecar two-stage candidate generation plus rerank ran.
    SignTwoStage,
}

/// The resolved execution plan for a dense search, as returned by
/// [`OrdinalIndex::search_with_report`] / [`crate::IdMapIndex::search_with_report`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DenseSearchPlan {
    /// Which strategy actually ran (relevant when `mode` was
    /// [`DenseSearchMode::Auto`]).
    pub execution: DenseSearchExecution,
    /// The index's vector dimension.
    pub dim: usize,
    /// Number of queries in the batch (`nq`).
    pub query_count: usize,
    /// The `k` the caller requested.
    pub requested_k: usize,
    /// The `k` actually used, after clamping to the search/candidate space.
    pub effective_k: usize,
    /// Total number of vectors the search ran against (`len()` at search
    /// time).
    pub search_space: usize,
    /// Size of the candidate shortlist, for [`DenseSearchExecution::SignTwoStage`]
    /// only; `None` for an exact scan.
    pub candidate_count: Option<usize>,
}

/// Timing breakdown for a dense search, as returned by
/// [`OrdinalIndex::search_with_report`] / [`crate::IdMapIndex::search_with_report`].
/// Only the fields relevant to the [`DenseSearchExecution`] that ran are
/// populated; the rest are left at their zero `Duration::default()`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DenseSearchTimings {
    /// Time spent validating the query buffer.
    pub validation: Duration,
    /// Time spent generating the sign-sidecar candidate shortlist
    /// ([`DenseSearchExecution::SignTwoStage`] only).
    pub candidate_generation: Duration,
    /// Time spent reranking the candidate shortlist against exact
    /// RankQuant codes ([`DenseSearchExecution::SignTwoStage`] only).
    pub rerank: Duration,
    /// Time spent on the brute-force exact scan
    /// ([`DenseSearchExecution::ExactRankQuant`] only).
    pub exact_search: Duration,
    /// Total time for the search, including validation.
    pub total: Duration,
}

/// Bundled result of [`OrdinalIndex::search_with_report`]: the raw results
/// plus the plan and timings used to produce them.
pub struct DenseSearchReport {
    /// The search results.
    pub results: SearchResults,
    /// The resolved execution plan.
    pub plan: DenseSearchPlan,
    /// The timing breakdown.
    pub timings: DenseSearchTimings,
}

/// Report returned by `write_verified_bundle` describing what was written
/// to disk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedBundleReport {
    /// Bundle directory that was written.
    pub path: PathBuf,
    /// Path to the bundle's `manifest.json`.
    pub manifest_path: PathBuf,
    /// The index's vector dimension.
    pub dim: usize,
    /// The index's RankQuant bit width.
    pub bits: u8,
    /// Number of rows written.
    pub row_count: usize,
    /// Whether a sign sidecar was written.
    pub has_sign: bool,
    /// Whether an ID sidecar was written (`true` for
    /// [`crate::IdMapIndex`], `false` for a bare [`OrdinalIndex`]).
    pub has_ids: bool,
}

/// Report returned by `inspect()` describing an index's in-memory shape.
/// `manifest_path`/`index_path` are always `None`: this describes the
/// index as it currently exists in memory, not a bundle on disk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DenseBundleInspectReport {
    /// Always `None`; reserved for symmetry with bundle-inspection
    /// tooling.
    pub manifest_path: Option<PathBuf>,
    /// Always `None`; reserved for symmetry with bundle-inspection
    /// tooling.
    pub index_path: Option<PathBuf>,
    /// The index's vector dimension.
    pub dim: usize,
    /// The index's RankQuant bit width.
    pub bits: u8,
    /// Number of rows currently in the index.
    pub row_count: usize,
    /// Whether the index maintains a sign sidecar.
    pub has_sign: bool,
    /// Whether the index has an ID sidecar (`true` for
    /// [`crate::IdMapIndex`], `false` for a bare [`OrdinalIndex`]).
    pub has_ids: bool,
    /// The row-identity kind the index uses; always `"row_id_identity"`
    /// today.
    pub row_identity_kind: String,
}

/// Incremental, one-row-at-a-time builder for a `.odb` bundle.
///
/// Despite its name, `OrdinalIndexBuilder` is backed by a
/// [`crate::IdMapIndex`], not a bare [`OrdinalIndex`]: every row is added
/// with an explicit `row_id`, and the bundle it writes has an ID sidecar
/// (its [`VerifiedBundleReport::has_ids`] is `true`). Use this when rows
/// are produced one at a time with their IDs already known (e.g. streaming
/// ingest); use [`OrdinalIndex::add_2d`] / [`crate::IdMapIndex::add_with_ids_2d`]
/// directly for batch inserts.
pub struct OrdinalIndexBuilder {
    index: IdMapIndex,
}

impl OrdinalIndex {
    /// Construct an empty index with a fixed `dim` and RankQuant `bits`
    /// (`1`, `2`, or `4`), using [`BuildOptions::default`].
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
    /// [`SignPolicy::Required`] cannot be honored for `(dim, bits)`.
    pub fn new_with_build_options(
        dim: usize,
        bits: u8,
        options: BuildOptions,
    ) -> Result<Self, ConstructError> {
        validate_bits(bits)?;
        validate_dim_bits(dim, bits)?;
        let sign = maybe_new_sign(dim, bits, options.sign)?;
        Ok(Self {
            dim: Some(dim),
            bits,
            inner: Some(RankQuant::new(dim, bits)),
            sign,
            sign_policy: options.sign,
        })
    }

    /// Construct an empty index with `bits` fixed but `dim` left
    /// undetermined; `dim` is inferred from the first non-empty batch
    /// passed to [`Self::add_2d`]. The default [`SignPolicy::Optional`]
    /// applies: a sign sidecar is built if the eventual `dim` supports one.
    ///
    /// # Errors
    /// Returns [`ConstructError::UnsupportedBits`] if `bits` is not `1`,
    /// `2`, or `4`.
    pub fn new_lazy(bits: u8) -> Result<Self, ConstructError> {
        Self::new_lazy_with_build_options(bits, BuildOptions::default())
    }

    /// Like [`Self::new_lazy`], with explicit [`BuildOptions`].
    ///
    /// For [`SignPolicy::Required`], bit widths that can never carry a sign
    /// sidecar are rejected immediately. With `bits == 2`, the remaining
    /// sign-sidecar decision is deferred to the first non-empty
    /// [`Self::add_2d`], which commits `dim`: a batch whose `dim` cannot
    /// carry a sidecar is rejected with
    /// [`AddError::SignSidecarUnsupported`] and the index stays lazy.
    ///
    /// # Errors
    /// Returns [`ConstructError::UnsupportedBits`] if `bits` is not `1`,
    /// `2`, or `4`, or [`ConstructError::SignSidecarUnsupportedBits`] if
    /// [`SignPolicy::Required`] is requested with `bits` `1` or `4`.
    pub fn new_lazy_with_build_options(
        bits: u8,
        options: BuildOptions,
    ) -> Result<Self, ConstructError> {
        validate_bits(bits)?;
        if options.sign == SignPolicy::Required && sign_required_multiple(bits).is_none() {
            return Err(ConstructError::SignSidecarUnsupportedBits { bits });
        }
        Ok(Self {
            dim: None,
            bits,
            inner: None,
            sign: None,
            sign_policy: options.sign,
        })
    }

    /// Number of rows currently in the index.
    pub fn len(&self) -> usize {
        self.inner.as_ref().map_or(0, RankQuant::len)
    }

    /// Returns `true` if the index has no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The index's vector dimension. Returns `0` if the index is still
    /// lazy (constructed via [`Self::new_lazy`]) and has not received a
    /// first add yet; use [`Self::dim_opt`] to distinguish that case.
    pub fn dim(&self) -> usize {
        self.dim.unwrap_or(0)
    }

    /// Like [`Self::dim`], but `None` while the index is still lazy and
    /// `dim` has not yet been established.
    pub fn dim_opt(&self) -> Option<usize> {
        self.dim
    }

    /// The index's RankQuant bit width (`1`, `2`, or `4`), fixed at
    /// construction.
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Returns `true` if the index currently maintains a sign sidecar for
    /// two-stage search.
    pub fn has_sign_sidecar(&self) -> bool {
        self.sign.is_some()
    }

    /// Append row-major vectors using the `dim` the index has already
    /// committed to.
    ///
    /// # Panics
    /// Panics if the index is still lazy (constructed via
    /// [`Self::new_lazy`]) with no prior add — use [`Self::add_2d`] to
    /// establish `dim` on the first call. Also panics if `vectors.len()`
    /// is not a multiple of `dim`, or if any coordinate is not finite or
    /// has magnitude `>= 1e16`. Prefer [`Self::add_2d`] for
    /// `Result`-returning validation.
    pub fn add(&mut self, vectors: &[f32]) {
        let dim = self.dim.expect(
            "OrdinalIndex dim is not set; use add_2d(vectors, dim) on the first add \
             or construct with OrdinalIndex::new(dim, bits)",
        );
        let n = vectors.len() / dim;
        assert_eq!(
            vectors.len(),
            n * dim,
            "vectors length must be a multiple of dim",
        );
        if n == 0 {
            return;
        }
        assert_valid_values(vectors, dim, "input");
        self.inner
            .as_mut()
            .expect("committed OrdinalIndex has no RankQuant inner")
            .add(vectors);
        if let Some(sign) = &mut self.sign {
            sign.add(vectors);
        }
    }

    /// Append row-major vectors, checking (and, for a lazy index,
    /// establishing) `dim` explicitly. An empty `vectors` batch is a no-op
    /// that returns `Ok(())` (and, on a lazy index, leaves `dim`
    /// unestablished).
    ///
    /// # Errors
    /// - [`AddError::DimInvalid`] / [`AddError::DimNotCompatibleWithBits`]
    ///   if `dim` is invalid or incompatible with the index's `bits`.
    /// - [`AddError::VectorBufferNotMultipleOfDim`] if `vectors.len()` is
    ///   not a multiple of `dim`.
    /// - [`AddError::DimMismatch`] if the index already committed to a
    ///   different `dim`.
    /// - [`AddError::InvalidInputValue`] if a coordinate is not finite or
    ///   has magnitude `>= 1e16`.
    /// - [`AddError::SignSidecarUnsupported`] if this add commits `dim` on
    ///   a lazy index built with [`SignPolicy::Required`] and `(dim,
    ///   bits)` cannot carry a sign sidecar; the batch is rejected in full
    ///   and the index stays lazy.
    pub fn add_2d(&mut self, vectors: &[f32], dim: usize) -> Result<(), AddError> {
        validate_add_dim_bits(dim, self.bits)?;
        if !vectors.len().is_multiple_of(dim) {
            return Err(AddError::VectorBufferNotMultipleOfDim {
                vectors_len: vectors.len(),
                dim,
            });
        }
        let n = vectors.len() / dim;
        match self.dim {
            Some(existing) if existing != dim => {
                return Err(AddError::DimMismatch { existing, got: dim });
            }
            Some(_) => {}
            None if n == 0 => return Ok(()),
            None => {}
        }

        if let Some((vector_index, coord_index, value)) = first_invalid_coord(vectors, dim) {
            return Err(AddError::InvalidInputValue {
                vector_index,
                coord_index,
                value,
            });
        }
        if n == 0 {
            return Ok(());
        }

        if self.inner.is_none() {
            // Committing dim: enforce the sign policy before mutating
            // anything, so a rejected batch leaves the index lazy.
            let sign = maybe_new_sign(dim, self.bits, self.sign_policy).map_err(|_| {
                AddError::SignSidecarUnsupported {
                    dim,
                    bits: self.bits,
                    required_multiple: sign_required_multiple(self.bits),
                }
            })?;
            self.inner = Some(RankQuant::new(dim, self.bits));
            self.sign = sign;
            self.dim = Some(dim);
        }
        self.inner
            .as_mut()
            .expect("OrdinalIndex inner is initialized")
            .add(vectors);
        if let Some(sign) = &mut self.sign {
            sign.add(vectors);
        }
        Ok(())
    }

    /// Search for the `k` nearest rows to each query using
    /// [`DenseSearchOptions::default`] (`Auto` mode).
    ///
    /// `queries` is a flat, row-major `f32` buffer of `dim`-sized query
    /// vectors. If the index is still lazy (no `dim` established), returns
    /// an empty [`SearchResults`] regardless of `queries`' contents.
    ///
    /// # Panics
    /// Panics if `queries.len()` is not a multiple of `dim`, or if any
    /// coordinate is not finite or has magnitude `>= 1e16`. Prefer
    /// [`Self::search_checked`] for `Result`-returning validation.
    pub fn search(&self, queries: &[f32], k: usize) -> SearchResults {
        self.search_with_options(queries, k, DenseSearchOptions::default())
    }

    /// `Result`-returning equivalent of [`Self::search`].
    ///
    /// # Errors
    /// Returns [`DenseError::InvalidQueryDim`] / [`DenseError::InvalidQueryValue`]
    /// for a malformed `queries` buffer, or [`DenseError::Limit`] if `nq *
    /// k` would overflow.
    pub fn search_checked(&self, queries: &[f32], k: usize) -> Result<SearchResults, DenseError> {
        self.search_checked_with_options(queries, k, DenseSearchOptions::default())
    }

    /// Search using an explicit [`DenseSearchMode`] and [`TwoStageOptions`]
    /// (via `options`).
    ///
    /// # Panics
    /// See [`Self::search`]. Additionally, if `options.two_stage` resolves
    /// to zero candidates for a non-empty `k` (e.g. a misconfigured
    /// [`TwoStageOptions::fixed_candidates`]), this panics — except when
    /// `options.mode` is [`DenseSearchMode::SignTwoStage`] and the index
    /// has no sign sidecar, in which case it transparently retries as
    /// [`DenseSearchMode::ExactRankQuant`] instead of panicking.
    pub fn search_with_options(
        &self,
        queries: &[f32],
        k: usize,
        options: DenseSearchOptions,
    ) -> SearchResults {
        let Some(dim) = self.dim else {
            return empty_results(0);
        };
        let nq = queries.len() / dim;
        assert_eq!(
            queries.len(),
            nq * dim,
            "queries length must be a multiple of dim",
        );
        assert_valid_values(queries, dim, "query");
        self.search_validated(queries, k, nq, dim, options)
    }

    /// `Result`-returning equivalent of [`Self::search_with_options`].
    ///
    /// # Errors
    /// Returns [`DenseError::InvalidQueryDim`] / [`DenseError::InvalidQueryValue`]
    /// for a malformed `queries` buffer, [`DenseError::Limit`] if `nq * k`
    /// would overflow or a two-stage search resolves to zero candidates
    /// for a non-empty query, or [`DenseError::MissingSignSidecar`] if
    /// `options.mode` is [`DenseSearchMode::SignTwoStage`] and the index
    /// has no sign sidecar.
    pub fn search_checked_with_options(
        &self,
        queries: &[f32],
        k: usize,
        options: DenseSearchOptions,
    ) -> Result<SearchResults, DenseError> {
        let Some(dim) = self.dim else {
            return Ok(empty_results(0));
        };
        let nq = validate_query_buffer(queries, dim)?;
        checked_result_buffer_len(nq, k.min(self.len()))?;
        let mut timings = DenseSearchTimings::default();
        let (results, _) =
            self.search_validated_with_timings(queries, k, nq, dim, options, &mut timings)?;
        Ok(results)
    }

    /// Search like [`Self::search_checked_with_options`], additionally
    /// returning the resolved execution plan and a timing breakdown. If
    /// the index is still lazy, returns a zeroed plan/timings alongside an
    /// empty result set.
    ///
    /// # Errors
    /// See [`Self::search_checked_with_options`].
    pub fn search_with_report(
        &self,
        queries: &[f32],
        k: usize,
        options: DenseSearchOptions,
    ) -> Result<DenseSearchReport, DenseError> {
        let total_started = Instant::now();
        let validation_started = Instant::now();
        let Some(dim) = self.dim else {
            return Ok(DenseSearchReport {
                results: empty_results(0),
                plan: DenseSearchPlan {
                    execution: DenseSearchExecution::ExactRankQuant,
                    dim: 0,
                    query_count: 0,
                    requested_k: k,
                    effective_k: 0,
                    search_space: 0,
                    candidate_count: None,
                },
                timings: DenseSearchTimings {
                    validation: validation_started.elapsed(),
                    total: total_started.elapsed(),
                    ..DenseSearchTimings::default()
                },
            });
        };
        let nq = validate_query_buffer(queries, dim)?;
        checked_result_buffer_len(nq, k.min(self.len()))?;
        let validation = validation_started.elapsed();

        let mut timings = DenseSearchTimings {
            validation,
            ..DenseSearchTimings::default()
        };
        let (results, plan) =
            self.search_validated_with_timings(queries, k, nq, dim, options, &mut timings)?;
        timings.total = total_started.elapsed();
        Ok(DenseSearchReport {
            results,
            plan,
            timings,
        })
    }

    /// Search restricted to the slots where `mask[slot]` is `true` (or
    /// unrestricted, like [`Self::search`], when `mask` is `None`).
    ///
    /// Masked search always reranks the selected candidates with an exact
    /// RankQuant scan — [`DenseSearchOptions`] does not apply, since a
    /// candidate subset is already given.
    ///
    /// # Panics
    /// Panics if `mask` is `Some` and its length does not equal `len()`.
    /// When `mask` is `None`, panics as [`Self::search`] does.
    pub fn search_with_mask(
        &self,
        queries: &[f32],
        k: usize,
        mask: Option<&[bool]>,
    ) -> SearchResults {
        let Some(dim) = self.dim else {
            return empty_results(0);
        };
        let nq = queries.len() / dim;
        assert_eq!(
            queries.len(),
            nq * dim,
            "queries length must be a multiple of dim",
        );
        assert_valid_values(queries, dim, "query");

        let inner = self
            .inner
            .as_ref()
            .expect("committed OrdinalIndex has no RankQuant inner");

        let Some(mask) = mask else {
            return self.search_validated(queries, k, nq, dim, DenseSearchOptions::default());
        };

        assert_eq!(
            mask.len(),
            inner.len(),
            "mask length {} does not match index size {}",
            mask.len(),
            inner.len(),
        );

        let candidates: Vec<u32> = mask
            .iter()
            .enumerate()
            .filter(|&(_, allowed)| *allowed)
            .map(|(idx, _)| u32::try_from(idx).expect("slot index exceeds u32 range"))
            .collect();
        self.search_with_candidates(queries, k, &candidates)
    }

    fn search_validated(
        &self,
        queries: &[f32],
        k: usize,
        nq: usize,
        dim: usize,
        options: DenseSearchOptions,
    ) -> SearchResults {
        let mut timings = DenseSearchTimings::default();
        match self.search_validated_with_timings(queries, k, nq, dim, options, &mut timings) {
            Ok((results, _)) => results,
            Err(DenseError::MissingSignSidecar) => {
                self.search_validated_with_timings(
                    queries,
                    k,
                    nq,
                    dim,
                    DenseSearchOptions::exact_rankquant(),
                    &mut timings,
                )
                .expect("exact RankQuant fallback should not require a sign sidecar")
                .0
            }
            Err(error) => panic!("search validation failed: {error}"),
        }
    }

    fn search_validated_with_timings(
        &self,
        queries: &[f32],
        k: usize,
        nq: usize,
        dim: usize,
        options: DenseSearchOptions,
        timings: &mut DenseSearchTimings,
    ) -> Result<(SearchResults, DenseSearchPlan), DenseError> {
        let inner = self
            .inner
            .as_ref()
            .expect("committed OrdinalIndex has no RankQuant inner");
        let n = inner.len();
        let exact_effective_k = k.min(n);
        let use_sign = match options.mode {
            DenseSearchMode::Auto => self.sign.as_ref(),
            DenseSearchMode::ExactRankQuant => None,
            DenseSearchMode::SignTwoStage => {
                Some(self.sign.as_ref().ok_or(DenseError::MissingSignSidecar)?)
            }
        };

        if let Some(sign) = use_sign {
            let candidate_count = options.two_stage.candidate_count(exact_effective_k, n);
            if exact_effective_k > 0 && candidate_count == 0 {
                return Err(DenseError::Limit(
                    "two-stage candidate_count must be > 0 for non-empty searches".into(),
                ));
            }
            let effective_k = exact_effective_k.min(candidate_count);
            let results = search_two_stage(
                inner,
                sign,
                queries,
                TwoStageRun {
                    effective_k,
                    candidate_count,
                    nq,
                    dim,
                },
                timings,
            );
            Ok((
                results,
                DenseSearchPlan {
                    execution: DenseSearchExecution::SignTwoStage,
                    dim,
                    query_count: nq,
                    requested_k: k,
                    effective_k,
                    search_space: n,
                    candidate_count: Some(candidate_count),
                },
            ))
        } else {
            let started = Instant::now();
            let results = inner.search_asymmetric(queries, exact_effective_k);
            timings.exact_search = started.elapsed();
            Ok((
                results,
                DenseSearchPlan {
                    execution: DenseSearchExecution::ExactRankQuant,
                    dim,
                    query_count: nq,
                    requested_k: k,
                    effective_k: exact_effective_k,
                    search_space: n,
                    candidate_count: None,
                },
            ))
        }
    }

    pub(crate) fn search_with_candidates(
        &self,
        queries: &[f32],
        k: usize,
        candidates: &[u32],
    ) -> SearchResults {
        let Some(dim) = self.dim else {
            return empty_results(0);
        };
        let nq = queries.len() / dim;
        assert_eq!(
            queries.len(),
            nq * dim,
            "queries length must be a multiple of dim",
        );
        assert_valid_values(queries, dim, "query");
        assert!(
            candidates
                .iter()
                .all(|&candidate| (candidate as usize) < self.len()),
            "candidate id out of range"
        );

        let effective_k = k.min(candidates.len());
        if effective_k == 0 {
            return SearchResults {
                scores: Vec::new(),
                indices: Vec::new(),
                nq,
                k: 0,
            };
        }

        let inner = self
            .inner
            .as_ref()
            .expect("committed OrdinalIndex has no RankQuant inner");
        search_repeated_candidates_parallel(inner, queries, candidates, effective_k, nq, dim)
    }

    /// Search restricted to an explicit, caller-provided list of candidate
    /// slots, reranked with an exact RankQuant scan (no sign-sidecar
    /// candidate generation).
    ///
    /// # Errors
    /// Returns [`DenseError::InvalidQueryDim`] / [`DenseError::InvalidQueryValue`]
    /// for a malformed `queries` buffer, [`DenseError::Limit`] if `nq * k`
    /// would overflow, or [`DenseError::CandidateSlotOutOfRange`] if any
    /// entry in `candidates` is `>= len()`.
    pub fn search_checked_with_candidates(
        &self,
        queries: &[f32],
        k: usize,
        candidates: &[u32],
    ) -> Result<SearchResults, DenseError> {
        let Some(dim) = self.dim else {
            return Ok(empty_results(0));
        };
        let nq = validate_query_buffer(queries, dim)?;
        checked_result_buffer_len(nq, k.min(candidates.len()))?;
        for &candidate in candidates {
            if candidate as usize >= self.len() {
                return Err(DenseError::CandidateSlotOutOfRange(candidate));
            }
        }
        Ok(self.search_with_candidates(queries, k, candidates))
    }

    /// Search restricted to `allowlist` (a set of row IDs) when `Some`,
    /// otherwise unrestricted like [`Self::search_checked`].
    ///
    /// A bare `OrdinalIndex` has no separate ID space, so each entry in
    /// `allowlist` is interpreted directly as a slot index
    /// (`row_id_identity`) — this differs from
    /// [`crate::IdMapIndex::search_checked_with_allowlist`], where
    /// allowlist entries are external IDs translated through that index's
    /// ID map.
    ///
    /// # Errors
    /// Returns [`DenseError::AllowlistRowIdMissing`] if an ID does not fit
    /// a `usize` or is `>= len()`, [`DenseError::SlotIndexOverflow`] if a
    /// slot does not fit a `u32`, or otherwise as
    /// [`Self::search_checked_with_candidates`].
    pub fn search_checked_with_allowlist(
        &self,
        queries: &[f32],
        k: usize,
        allowlist: Option<&[u64]>,
    ) -> Result<SearchResults, DenseError> {
        let Some(allowlist) = allowlist else {
            return self.search_checked(queries, k);
        };
        let mut candidates = Vec::with_capacity(allowlist.len());
        for &row_id in allowlist {
            let slot =
                usize::try_from(row_id).map_err(|_| DenseError::AllowlistRowIdMissing(row_id))?;
            if slot >= self.len() {
                return Err(DenseError::AllowlistRowIdMissing(row_id));
            }
            let slot = u32::try_from(slot).map_err(|_| DenseError::SlotIndexOverflow(slot))?;
            candidates.push(slot);
        }
        candidates.sort_unstable();
        candidates.dedup();
        self.search_checked_with_candidates(queries, k, &candidates)
    }

    /// Search a single query, returning up to `k`
    /// [`crate::hybrid::ScoredRow`]s (`row_id`/`score` pairs, where
    /// `row_id` is the slot index) sorted by descending score. Only
    /// available with the `hybrid` Cargo feature.
    ///
    /// # Errors
    /// See [`Self::search_checked`].
    #[cfg(feature = "hybrid")]
    pub fn search_rows(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<crate::hybrid::ScoredRow>, DenseError> {
        validate_single_query_buffer(query, self.dim)?;
        let batch = self.search_batch_rows(query, k)?;
        Ok(batch.hits().to_vec())
    }

    /// Batched form of [`Self::search_rows`]. Only available with the
    /// `hybrid` Cargo feature.
    ///
    /// # Errors
    /// See [`Self::search_checked`].
    #[cfg(feature = "hybrid")]
    pub fn search_batch_rows(
        &self,
        queries: &[f32],
        k: usize,
    ) -> Result<crate::hybrid::RankedBatch, DenseError> {
        let results = self.search_checked(queries, k)?;
        crate::hybrid::RankedBatch::from_flat_scores_i64_indices(
            results.nq,
            results.k,
            results.scores,
            results.indices,
        )
        .map_err(|err| DenseError::metadata_mismatch(err.to_string()))
    }

    /// Like [`Self::search_rows`], restricted to `allowlist` (slot indices)
    /// when `Some`. Only available with the `hybrid` Cargo feature.
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
        validate_single_query_buffer(query, self.dim)?;
        let batch = self.search_batch_rows_with_allowlists(query, k, allowlist.map(|ids| [ids]))?;
        Ok(batch.hits().to_vec())
    }

    /// Batched form of [`Self::search_rows_with_allowlist`]: `allowlists`
    /// yields one allowlist per query in `queries` (or `None` to search
    /// unrestricted). Only available with the `hybrid` Cargo feature.
    ///
    /// # Errors
    /// Returns [`DenseError::MetadataMismatch`] if the number of
    /// allowlists does not match the number of queries; otherwise as
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
        let Some(dim) = self.dim else {
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
        let mut offsets = Vec::with_capacity(nq + 1);
        let mut candidates = Vec::new();
        offsets.push(0);
        let mut max_candidates = 0usize;
        for row_ids in lists {
            let mut row_candidates = Vec::with_capacity(row_ids.len());
            for &row_id in row_ids {
                let slot = usize::try_from(row_id)
                    .map_err(|_| DenseError::AllowlistRowIdMissing(row_id))?;
                if slot >= self.len() {
                    return Err(DenseError::AllowlistRowIdMissing(row_id));
                }
                let slot = u32::try_from(slot).map_err(|_| DenseError::SlotIndexOverflow(slot))?;
                row_candidates.push(slot);
            }
            row_candidates.sort_unstable();
            row_candidates.dedup();
            max_candidates = max_candidates.max(row_candidates.len());
            candidates.extend(row_candidates);
            offsets.push(candidates.len());
        }
        let effective_k = k.min(max_candidates);
        checked_result_buffer_len(nq, effective_k)?;
        let inner = self
            .inner
            .as_ref()
            .expect("committed OrdinalIndex has no RankQuant inner");
        let results =
            search_candidate_csr_serial(inner, queries, &offsets, &candidates, effective_k, nq);
        crate::hybrid::RankedBatch::from_flat_scores_i64_indices(
            results.nq,
            results.k,
            results.scores,
            results.indices,
        )
        .map_err(|err| DenseError::metadata_mismatch(err.to_string()))
    }

    /// Remove the row at slot `idx`, moving the last row into its place
    /// (like `Vec::swap_remove`). Returns the slot the moved row came from
    /// (i.e. `len() - 1` before removal), so callers maintaining an
    /// external slot-keyed mapping (as [`crate::IdMapIndex`] does) can keep
    /// it in sync. Any sign sidecar is kept consistent automatically.
    ///
    /// # Panics
    /// Panics if the index is still lazy (no `dim` established) — there is
    /// nothing to remove.
    pub fn swap_remove(&mut self, idx: usize) -> usize {
        assert!(
            self.dim.is_some(),
            "cannot remove from a lazy uncommitted OrdinalIndex"
        );
        let moved_from = self
            .inner
            .as_mut()
            .expect("cannot remove from a lazy uncommitted OrdinalIndex")
            .swap_remove(idx);
        if let Some(sign) = &mut self.sign {
            sign.swap_remove(idx);
        }
        moved_from
    }

    /// Write the index as a `.odb` bundle (no ID sidecar) to `path`,
    /// atomically.
    ///
    /// # Errors
    /// Returns an `InvalidInput` [`std::io::Error`] if the index is still
    /// lazy and `dim` has not been established, or any I/O/verification
    /// error encountered while writing.
    pub fn write(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let inner = self.inner.as_ref().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "cannot persist a lazy OrdinalIndex before its dim is set",
            )
        })?;
        crate::io::write_ordinal_bundle(path, inner, self.sign.as_ref())
    }

    /// Like [`Self::write`], but with explicit manifest creation options
    /// and extra caller-supplied auxiliary artifacts, returning a
    /// [`VerifiedBundleReport`] describing what was written.
    ///
    /// # Errors
    /// Returns [`DenseError::MetadataMismatch`] if the index is still lazy,
    /// or any I/O/manifest error encountered while writing.
    pub fn write_verified_bundle(
        &self,
        path: impl AsRef<Path>,
        manifest_options: crate::manifest::CreateManifestOptions,
        auxiliary_artifacts: Vec<AuxiliaryArtifactDeclaration>,
    ) -> Result<VerifiedBundleReport, DenseError> {
        let path = path.as_ref();
        let inner = self.inner.as_ref().ok_or_else(|| {
            DenseError::metadata_mismatch(
                "cannot persist a lazy OrdinalIndex before its dim is set",
            )
        })?;
        crate::io::write_ordinal_bundle_with_options(
            path,
            inner,
            self.sign.as_ref(),
            crate::io::BundleWriteOptions {
                manifest_options,
                auxiliary_artifacts,
            },
        )?;
        Ok(self.verified_bundle_report(path, false))
    }

    /// Load a bundle directory previously written by [`Self::write`] or
    /// [`Self::write_verified_bundle`], with default manifest
    /// verification and the default sign-sidecar load policy
    /// ([`SignLoadPolicy::RequireIfSupported`]).
    ///
    /// # Errors
    /// Returns an `InvalidData` [`std::io::Error`] if the bundle carries an
    /// ID sidecar (it was written by [`crate::IdMapIndex`] — load it with
    /// [`crate::IdMapIndex::load`] instead), fails manifest verification,
    /// is a sign-capable bundle missing its sign sidecar (use
    /// [`Self::load_with_options`] with [`SignLoadPolicy::Any`] to load it
    /// without one), or is otherwise malformed.
    pub fn load(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let artifacts = crate::io::load_ordinal_bundle(path)?;
        Self::from_loaded_parts(artifacts.rankquant, artifacts.sign)
    }

    /// Load a bundle directory with default manifest verification and
    /// caller-controlled [`DenseLoadOptions`].
    ///
    /// This directory-based entry point performs the same interrupted-publication
    /// recovery as [`Self::load`] before opening the verified manifest. Use
    /// [`Self::open_verified`] only when recovery is not desired or the manifest
    /// is not stored at the standard bundle path.
    ///
    /// # Errors
    /// Returns [`DenseError::RowIdentity`] if the verified bundle carries an ID
    /// sidecar, or any recovery, manifest, verification, or I/O error from the
    /// underlying load.
    pub fn load_with_options(
        path: impl AsRef<Path>,
        load_options: DenseLoadOptions,
    ) -> Result<Self, DenseError> {
        let path = path.as_ref();
        crate::io::recover_bundle_if_missing(path)?;
        Self::open_verified(
            path.join(crate::artifacts::MANIFEST_FILE),
            VerifyOptions::default(),
            load_options,
        )
    }

    /// Load a bundle from an explicit `manifest.json` path with
    /// caller-controlled [`VerifyOptions`] and [`DenseLoadOptions`].
    ///
    /// # Errors
    /// Returns [`DenseError::RowIdentity`] if the verified bundle carries
    /// an ID sidecar (load it with [`crate::IdMapIndex::open_verified`]
    /// instead), or any manifest/verification/I/O error from the
    /// underlying load.
    pub fn open_verified(
        manifest_path: impl AsRef<Path>,
        verify_options: VerifyOptions,
        load_options: DenseLoadOptions,
    ) -> Result<Self, DenseError> {
        let loaded = load_verified_bundle(manifest_path.as_ref(), verify_options, load_options)?;
        if loaded.ids_path.is_some() {
            return Err(DenseError::row_identity(
                "verified bundle contains OrdinalDB IDs; load it with IdMapIndex::open_verified",
            ));
        }
        Self::from_loaded_parts(loaded.rankquant, loaded.sign).map_err(DenseError::from)
    }

    /// Summarize the in-memory index's shape (dim, bits, row count,
    /// whether a sign sidecar is present). `manifest_path`/`index_path`
    /// are always `None`, and `has_ids` is always `false`: this describes
    /// a bare `OrdinalIndex` in memory, not a bundle on disk.
    pub fn inspect(&self) -> DenseBundleInspectReport {
        DenseBundleInspectReport {
            manifest_path: None,
            index_path: None,
            dim: self.dim(),
            bits: self.bits,
            row_count: self.len(),
            has_sign: self.sign.is_some(),
            has_ids: false,
            row_identity_kind: "row_id_identity".to_string(),
        }
    }

    pub(crate) fn rankquant(&self) -> Option<&RankQuant> {
        self.inner.as_ref()
    }

    pub(crate) fn sign_bitmap(&self) -> Option<&SignBitmap> {
        self.sign.as_ref()
    }

    pub(crate) fn from_loaded_parts(
        rankquant: RankQuant,
        sign: Option<SignBitmap>,
    ) -> std::io::Result<Self> {
        let dim = rankquant.dim();
        let bits = rankquant.bits();
        validate_dim_bits(dim, bits)
            .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err.to_string()))?;
        if let Some(sign) = &sign {
            if !sign_compatible(dim, bits) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "sign sidecar is only valid for bits=2 and dim divisible by 64",
                ));
            }
            if sign.dim() != dim || sign.len() != rankquant.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "sign sidecar shape mismatch: sign dim={}, len={} but RankQuant dim={}, len={}",
                        sign.dim(),
                        sign.len(),
                        dim,
                        rankquant.len()
                    ),
                ));
            }
        }
        Ok(Self {
            dim: Some(dim),
            bits,
            inner: Some(rankquant),
            sign,
            // A loaded index's sidecar decision is already made; the
            // policy only matters for a lazy first add, which a loaded
            // (dim-committed) index never performs.
            sign_policy: SignPolicy::Optional,
        })
    }

    fn verified_bundle_report(&self, path: &Path, has_ids: bool) -> VerifiedBundleReport {
        VerifiedBundleReport {
            path: path.to_path_buf(),
            manifest_path: path.join(crate::io::MANIFEST_FILE),
            dim: self.dim(),
            bits: self.bits,
            row_count: self.len(),
            has_sign: self.sign.is_some(),
            has_ids,
        }
    }
}

fn validate_bits(bits: u8) -> Result<(), ConstructError> {
    if matches!(bits, 1 | 2 | 4) {
        Ok(())
    } else {
        Err(ConstructError::UnsupportedBits(bits))
    }
}

/// The RankQuant packing (`8 / bits`) and bucket-balance (`2^bits`)
/// moduli for a supported `bits`. Callers must have validated `bits`.
fn rankquant_moduli(bits: u8) -> (usize, usize) {
    ((8 / bits) as usize, 1usize << bits)
}

/// Which `(dim, bits)` shape constraint a pair violates. The single source
/// of the modulus math shared by [`validate_dim_bits`],
/// [`validate_add_dim_bits`], and the preflight helpers.
enum DimBitsViolation {
    DimOutOfRange,
    NotCompatible(&'static str),
}

fn check_dim_bits(dim: usize, bits: u8) -> Result<(), DimBitsViolation> {
    if dim < 2 || dim > u16::MAX as usize {
        return Err(DimBitsViolation::DimOutOfRange);
    }
    let (codes_per_byte, buckets) = rankquant_moduli(bits);
    if !dim.is_multiple_of(codes_per_byte) {
        return Err(DimBitsViolation::NotCompatible(
            "dim must be divisible by 8 / bits for packed RankQuant storage",
        ));
    }
    if !dim.is_multiple_of(buckets) {
        return Err(DimBitsViolation::NotCompatible(
            "dim must be divisible by 2^bits for RankQuant bucket balance",
        ));
    }
    Ok(())
}

fn validate_dim_bits(dim: usize, bits: u8) -> Result<(), ConstructError> {
    check_dim_bits(dim, bits).map_err(|violation| match violation {
        DimBitsViolation::DimOutOfRange => ConstructError::DimInvalid(dim),
        DimBitsViolation::NotCompatible(reason) => {
            ConstructError::DimNotCompatibleWithBits { dim, bits, reason }
        }
    })
}

fn validate_add_dim_bits(dim: usize, bits: u8) -> Result<(), AddError> {
    check_dim_bits(dim, bits).map_err(|violation| match violation {
        DimBitsViolation::DimOutOfRange => AddError::DimInvalid(dim),
        DimBitsViolation::NotCompatible(reason) => {
            AddError::DimNotCompatibleWithBits { dim, bits, reason }
        }
    })
}

/// The multiple `dim` must satisfy for RankQuant codes at `bits` — the
/// least common multiple of the packing (`8 / bits`) and bucket-balance
/// (`2^bits`) constraints — or `None` if `bits` is not one of OrdinalDB's
/// supported widths (`1`, `2`, or `4`).
pub fn rankquant_required_multiple(bits: u8) -> Option<usize> {
    if validate_bits(bits).is_err() {
        return None;
    }
    let (codes_per_byte, buckets) = rankquant_moduli(bits);
    // Both moduli are powers of two, so their lcm is the larger one.
    Some(codes_per_byte.max(buckets))
}

/// Returns `true` if an index can be constructed with this `(dim, bits)`
/// pair: `bits` is supported, `dim` is in range (`2..=u16::MAX`), and
/// `dim` is a multiple of [`rankquant_required_multiple`].
pub fn rankquant_compatible(dim: usize, bits: u8) -> bool {
    validate_bits(bits).is_ok() && check_dim_bits(dim, bits).is_ok()
}

/// The multiple `dim` must satisfy for a `SignBitmap` sidecar at `bits`
/// (`64`, for `bits == 2`), or `None` if `bits` never supports one.
pub fn sign_required_multiple(bits: u8) -> Option<usize> {
    (bits == 2).then_some(64)
}

/// Returns `true` if an index with this `(dim, bits)` pair can carry a
/// `SignBitmap` sidecar: the pair is [`rankquant_compatible`] and `dim`
/// is a multiple of [`sign_required_multiple`].
pub fn sign_compatible(dim: usize, bits: u8) -> bool {
    rankquant_compatible(dim, bits)
        && sign_required_multiple(bits).is_some_and(|multiple| dim.is_multiple_of(multiple))
}

/// Resolve `policy` for a committed `(dim, bits)`: `Ok(Some)` when a
/// sidecar is built, `Ok(None)` when the policy or `(dim, bits)` opts
/// out, and [`ConstructError::SignSidecarUnsupported`] when
/// [`SignPolicy::Required`] cannot be honored.
fn maybe_new_sign(
    dim: usize,
    bits: u8,
    policy: SignPolicy,
) -> Result<Option<SignBitmap>, ConstructError> {
    let compatible = sign_compatible(dim, bits);
    match policy {
        SignPolicy::Disabled => Ok(None),
        SignPolicy::Optional => Ok(compatible.then(|| SignBitmap::new(dim))),
        SignPolicy::Required if compatible => Ok(Some(SignBitmap::new(dim))),
        SignPolicy::Required => Err(ConstructError::SignSidecarUnsupported {
            dim,
            bits,
            required_multiple: sign_required_multiple(bits),
        }),
    }
}

fn first_invalid_coord(values: &[f32], dim: usize) -> Option<(usize, usize, f32)> {
    // Large ingest batches paid a full serial pass here (measured ~0.1s per
    // GiB). Happy path: a parallel all-valid sweep. Only on failure does the
    // serial scan run, preserving exact first-offender reporting.
    const PARALLEL_THRESHOLD: usize = 1 << 20;
    if values.len() >= PARALLEL_THRESHOLD {
        let all_valid = values.par_chunks(1 << 18).all(|c| {
            c.iter()
                .all(|v| v.is_finite() && v.abs() < MAX_INPUT_MAGNITUDE)
        });
        if all_valid {
            return None;
        }
    }
    values.iter().enumerate().find_map(|(i, value)| {
        if !value.is_finite() || value.abs() >= MAX_INPUT_MAGNITUDE {
            Some((i / dim, i % dim, *value))
        } else {
            None
        }
    })
}

pub(crate) fn validate_query_buffer(queries: &[f32], dim: usize) -> Result<usize, DenseError> {
    if dim == 0 || !queries.len().is_multiple_of(dim) {
        return Err(DenseError::InvalidQueryDim {
            len: queries.len(),
            dim,
        });
    }
    if let Some((query_index, coord_index, value)) = first_invalid_coord(queries, dim) {
        return Err(DenseError::InvalidQueryValue {
            query_index,
            coord_index,
            value,
        });
    }
    Ok(queries.len() / dim)
}

#[cfg(feature = "hybrid")]
pub(crate) fn validate_single_query_buffer(
    query: &[f32],
    dim: Option<usize>,
) -> Result<(), DenseError> {
    let Some(dim) = dim else {
        if query.is_empty() {
            return Ok(());
        }
        return Err(DenseError::InvalidQueryDim {
            len: query.len(),
            dim: 0,
        });
    };
    if query.len() != dim {
        return Err(DenseError::InvalidQueryDim {
            len: query.len(),
            dim,
        });
    }
    validate_query_buffer(query, dim).map(|_| ())
}

fn assert_valid_values(values: &[f32], dim: usize, kind: &str) {
    if let Some((vector_index, coord_index, value)) = first_invalid_coord(values, dim) {
        panic!(
            "invalid {kind} value at vector {vector_index}, coord {coord_index}: {value} \
             (must be finite and |value| < {MAX_INPUT_MAGNITUDE})"
        );
    }
}

fn empty_results(nq: usize) -> SearchResults {
    SearchResults {
        scores: Vec::new(),
        indices: Vec::new(),
        nq,
        k: 0,
    }
}

fn result_buffer_len(nq: usize, k: usize) -> usize {
    nq.checked_mul(k)
        .expect("nq * k overflow while allocating search results")
}

fn checked_result_buffer_len(nq: usize, k: usize) -> Result<usize, DenseError> {
    nq.checked_mul(k)
        .ok_or_else(|| DenseError::Limit("nq * k overflow while allocating search results".into()))
}

#[derive(Clone, Copy)]
struct TwoStageRun {
    effective_k: usize,
    candidate_count: usize,
    nq: usize,
    dim: usize,
}

fn search_two_stage(
    rankquant: &RankQuant,
    sign: &SignBitmap,
    queries: &[f32],
    run: TwoStageRun,
    timings: &mut DenseSearchTimings,
) -> SearchResults {
    debug_assert_eq!(rankquant.dim(), sign.dim());
    debug_assert_eq!(rankquant.len(), sign.len());

    if run.effective_k == 0 {
        return empty_results(run.nq);
    }
    let out_len = result_buffer_len(run.nq, run.effective_k);
    let mut scores = vec![f32::NEG_INFINITY; out_len];
    let mut indices = vec![-1i64; out_len];

    let chunk_rows = two_stage_query_chunk_rows(run.candidate_count, run.nq);
    let result_chunk_len = chunk_rows * run.effective_k;
    let candidate_nanos = AtomicU64::new(0);
    let rerank_nanos = AtomicU64::new(0);
    queries
        .par_chunks(chunk_rows * run.dim)
        .zip(scores.par_chunks_mut(result_chunk_len))
        .zip(indices.par_chunks_mut(result_chunk_len))
        .for_each_init(
            SubsetScratch::new,
            |scratch, ((query_chunk, score_chunk), index_chunk)| {
                let started = Instant::now();
                let candidates =
                    sign.top_m_candidates_batched_serial_csr(query_chunk, run.candidate_count);
                add_elapsed_nanos(&candidate_nanos, started.elapsed());
                let started = Instant::now();
                rankquant.search_asymmetric_subset_batched_serial_into(
                    query_chunk,
                    &candidates.offsets,
                    &candidates.candidates,
                    run.effective_k,
                    scratch,
                    score_chunk,
                    index_chunk,
                );
                add_elapsed_nanos(&rerank_nanos, started.elapsed());
            },
        );
    timings.candidate_generation = Duration::from_nanos(candidate_nanos.load(Ordering::Relaxed));
    timings.rerank = Duration::from_nanos(rerank_nanos.load(Ordering::Relaxed));

    SearchResults {
        scores,
        indices,
        nq: run.nq,
        k: run.effective_k,
    }
}

fn add_elapsed_nanos(total: &AtomicU64, elapsed: Duration) {
    let nanos = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
    let _ = total.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
        Some(current.saturating_add(nanos))
    });
}

fn search_repeated_candidates_parallel(
    rankquant: &RankQuant,
    queries: &[f32],
    candidates: &[u32],
    effective_k: usize,
    nq: usize,
    dim: usize,
) -> SearchResults {
    if effective_k == 0 {
        return empty_results(nq);
    }

    let out_len = result_buffer_len(nq, effective_k);
    let mut scores = vec![f32::NEG_INFINITY; out_len];
    let mut indices = vec![-1i64; out_len];
    queries
        .par_chunks(dim)
        .zip(scores.par_chunks_mut(effective_k))
        .zip(indices.par_chunks_mut(effective_k))
        .for_each(|((query, score_row), index_row)| {
            let (row_scores, row_indices) =
                rankquant.search_asymmetric_subset(query, candidates, effective_k);
            debug_assert_eq!(row_scores.len(), effective_k);
            debug_assert_eq!(row_indices.len(), effective_k);
            score_row.copy_from_slice(&row_scores);
            index_row.copy_from_slice(&row_indices);
        });

    SearchResults {
        scores,
        indices,
        nq,
        k: effective_k,
    }
}

#[cfg(feature = "hybrid")]
fn search_candidate_csr_serial(
    rankquant: &RankQuant,
    queries: &[f32],
    offsets: &[usize],
    candidates: &[u32],
    effective_k: usize,
    nq: usize,
) -> SearchResults {
    if effective_k == 0 {
        return empty_results(nq);
    }
    let out_len = result_buffer_len(nq, effective_k);
    let mut scores = vec![f32::NEG_INFINITY; out_len];
    let mut indices = vec![-1i64; out_len];
    let mut scratch = SubsetScratch::new();
    rankquant.search_asymmetric_subset_batched_serial_into(
        queries,
        offsets,
        candidates,
        effective_k,
        &mut scratch,
        &mut scores,
        &mut indices,
    );
    SearchResults {
        scores,
        indices,
        nq,
        k: effective_k,
    }
}

fn two_stage_query_chunk_rows(n_vectors: usize, nq: usize) -> usize {
    if n_vectors == 0 || nq == 0 {
        return 1;
    }
    // Two forces size a chunk: the CSR memory cap (score cells) bounds it
    // above, and parallel engagement bounds it below-ish — a batch must
    // split across the rayon pool instead of landing in one chunk on one
    // core (any batch <= cells/M previously did exactly that). TILE_FLOOR
    // keeps each chunk's candidate scan shared across enough queries that
    // splitting never costs more corpus passes than the cores can absorb.
    // 128, not 32: each chunk's candidate scan streams the full sign
    // sidecar once, so queries-per-stream sets the DRAM demand. At 32 the
    // aggregate stream rate ceilings batch throughput (~8.6k q/s at 1.26M x
    // 1024, measured); 128 quarters the traffic and hands the limit to the
    // per-core compute ceiling. The optimal floor is a hardware-specific
    // bandwidth/core tradeoff — a wide pool on a cache-resident or
    // high-bandwidth corpus is better served by a lower floor (more chunks,
    // more cores busy), where a bandwidth-bound large corpus wants the
    // higher floor. `ORDINALDB_TWO_STAGE_TILE_FLOOR` lets operators tune it
    // for their hardware; unlike the manifest resource limits (a security
    // policy that must stay explicit, not ambient) this is a pure
    // performance knob whose worst case is suboptimal throughput, never a
    // correctness or safety change.
    let tile_floor = two_stage_tile_floor();
    let max_rows_by_cells = (TWO_STAGE_MAX_SCORE_CELLS / n_vectors).max(1);
    let target_rows = nq
        .div_ceil(rayon::current_num_threads().max(1))
        .max(tile_floor);
    max_rows_by_cells.min(target_rows).clamp(1, nq)
}

/// The two-stage query-chunk tile floor: `ORDINALDB_TWO_STAGE_TILE_FLOOR`
/// if set to a positive integer, else the measured default of 128. Read
/// once; a wide Rayon pool on a non-bandwidth-bound corpus can lower it to
/// keep more cores busy.
fn two_stage_tile_floor() -> usize {
    use std::sync::OnceLock;
    static FLOOR: OnceLock<usize> = OnceLock::new();
    *FLOOR.get_or_init(|| parse_tile_floor(std::env::var("ORDINALDB_TWO_STAGE_TILE_FLOOR").ok()))
}

/// Pure parse/clamp for the tile-floor env value: a positive integer wins,
/// anything else (absent, non-numeric, zero) falls back to the default 128.
fn parse_tile_floor(raw: Option<String>) -> usize {
    raw.and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&v| v >= 1)
        .unwrap_or(128)
}

struct LoadedVerifiedBundle {
    rankquant: RankQuant,
    sign: Option<SignBitmap>,
    ids_path: Option<PathBuf>,
}

fn load_verified_bundle(
    manifest_path: &Path,
    verify_options: VerifyOptions,
    load_options: DenseLoadOptions,
) -> Result<LoadedVerifiedBundle, DenseError> {
    let plan = crate::manifest::verify_for_load(manifest_path, verify_options)?;
    let metadata = plan.metadata();
    if metadata.kind != ManifestIndexKind::RankQuant {
        return Err(DenseError::metadata_mismatch(format!(
            "OrdinalDB dense bundles require a RankQuant primary artifact; got {:?}",
            metadata.kind
        )));
    }
    let ManifestIndexParams::RankQuant {
        bits: metadata_bits,
    } = metadata.params
    else {
        return Err(DenseError::metadata_mismatch(
            "OrdinalDB dense bundle primary artifact has non-RankQuant params",
        ));
    };
    if let Some(expected_dim) = load_options.expected_dim {
        if metadata.dim != expected_dim {
            return Err(DenseError::metadata_mismatch(format!(
                "verified dense dim {} does not match expected dim {expected_dim}",
                metadata.dim
            )));
        }
    }
    if let Some(expected_bits) = load_options.expected_bits {
        if metadata_bits != expected_bits {
            return Err(DenseError::metadata_mismatch(format!(
                "verified dense bits {metadata_bits} does not match expected bits {expected_bits}"
            )));
        }
    }
    if plan.row_identity().kind() != "row_id_identity" {
        return Err(DenseError::row_identity(format!(
            "OrdinalDB dense bundles currently require row_id_identity row identity; got {:?}",
            plan.row_identity().kind()
        )));
    }

    let rankquant = RankQuant::load(plan.artifact_path())?;
    if metadata.dim != rankquant.dim()
        || metadata.vector_count != rankquant.len()
        || metadata_bits != rankquant.bits()
        || plan.row_identity().row_count() != rankquant.len()
    {
        return Err(DenseError::metadata_mismatch(
            "verified manifest metadata does not match loaded RankQuant",
        ));
    }

    let sign_declared = plan.auxiliary_by_name(crate::io::SIGN_AUX_NAME).is_some();
    let sign_path = match load_options.sign {
        SignLoadPolicy::Forbid if sign_declared => {
            return Err(DenseError::SignSidecarForbidden);
        }
        SignLoadPolicy::Forbid => None,
        SignLoadPolicy::Require => required_sign_path(&plan)?,
        SignLoadPolicy::RequireIfSupported if sign_compatible(metadata.dim, metadata_bits) => {
            required_sign_path(&plan)?
        }
        SignLoadPolicy::RequireIfSupported | SignLoadPolicy::Any => {
            crate::io::auxiliary_path(&plan, crate::io::SIGN_AUX_NAME)?
        }
    };
    let sign = match sign_path {
        Some(path) => {
            let sign = SignBitmap::load(path)?;
            if sign.dim() != rankquant.dim() || sign.len() != rankquant.len() {
                return Err(DenseError::metadata_mismatch(format!(
                    "sign sidecar shape mismatch: sign dim={}, len={} but RankQuant dim={}, len={}",
                    sign.dim(),
                    sign.len(),
                    rankquant.dim(),
                    rankquant.len()
                )));
            }
            Some(sign)
        }
        None => None,
    };
    let ids_path = crate::io::auxiliary_path(&plan, crate::io::IDS_AUX_NAME)?;
    Ok(LoadedVerifiedBundle {
        rankquant,
        sign,
        ids_path,
    })
}

/// The verified path of the bundle's sign sidecar, required: a missing
/// declaration is [`DenseError::MissingSignSidecar`]; a declared but
/// unloadable sidecar is [`DenseError::Auxiliary`].
fn required_sign_path(
    plan: &ordvec_manifest::VerifiedLoadPlan,
) -> Result<Option<PathBuf>, DenseError> {
    match plan.require_auxiliary(crate::io::SIGN_AUX_NAME) {
        Ok(path) => Ok(Some(path.to_path_buf())),
        Err(crate::manifest::RequireAuxiliaryError::MissingDeclaration { .. }) => {
            Err(DenseError::MissingSignSidecar)
        }
        Err(error) => Err(DenseError::Auxiliary(error)),
    }
}

impl OrdinalIndexBuilder {
    /// Construct a new builder for a `dim`/`bits`/`options`-configured
    /// index.
    ///
    /// # Errors
    /// Returns [`DenseError::Construct`] if `bits`/`dim` are invalid; see
    /// [`ConstructError`].
    pub fn new(dim: usize, bits: u8, options: BuildOptions) -> Result<Self, DenseError> {
        Ok(Self {
            index: IdMapIndex::new_with_build_options(dim, bits, options)?,
        })
    }

    /// Add a single row tagged with `row_id`. `vector` must have exactly
    /// `dim` coordinates.
    ///
    /// # Errors
    /// Returns [`DenseError::Add`] if `row_id` is already present or
    /// `vector` is malformed; see [`AddError`].
    pub fn add(&mut self, row_id: u64, vector: &[f32]) -> Result<(), DenseError> {
        let dim = self.index.dim();
        self.index.add_with_ids_2d(vector, dim, &[row_id])?;
        Ok(())
    }

    /// Write the accumulated rows as a `.odb` bundle (with an ID sidecar,
    /// since this builder is backed by [`crate::IdMapIndex`]) to `path`.
    ///
    /// # Errors
    /// See [`crate::IdMapIndex::write_verified_bundle`].
    pub fn write_verified_bundle(
        &self,
        path: impl AsRef<Path>,
        manifest_options: crate::manifest::CreateManifestOptions,
        auxiliary_artifacts: Vec<AuxiliaryArtifactDeclaration>,
    ) -> Result<VerifiedBundleReport, DenseError> {
        self.index
            .write_verified_bundle(path, manifest_options, auxiliary_artifacts)
    }
}

pub(crate) fn load_verified_ordinal_parts(
    manifest_path: &Path,
    verify_options: VerifyOptions,
    load_options: DenseLoadOptions,
) -> Result<(RankQuant, Option<SignBitmap>, Option<PathBuf>), DenseError> {
    let loaded = load_verified_bundle(manifest_path, verify_options, load_options)?;
    Ok((loaded.rankquant, loaded.sign, loaded.ids_path))
}

/// Load the dense parts (primary RankQuant + optional sign sidecar) from an
/// already-verified [`VerifiedLoadPlan`], instead of re-running
/// `verify_for_load`. Each artifact is read once into memory, re-hashed
/// against the digest the plan recorded, and constructed from those very
/// bytes (`load_from_bytes`) — never reopened by path — so the bytes hashed
/// are the bytes loaded and a stale plan over mutable storage is rejected
/// rather than trusted, with no mutation window between check and use. This
/// is the freshness re-check that the sparse
/// `open_from_verified_plan_unchecked_freshness` deliberately skips.
pub(crate) fn load_verified_parts_from_plan(
    plan: &VerifiedLoadPlan,
    load_options: DenseLoadOptions,
) -> Result<(RankQuant, Option<SignBitmap>), DenseError> {
    let metadata = plan.metadata();
    if metadata.kind != ManifestIndexKind::RankQuant {
        return Err(DenseError::metadata_mismatch(format!(
            "OrdinalDB dense bundles require a RankQuant primary artifact; got {:?}",
            metadata.kind
        )));
    }
    let ManifestIndexParams::RankQuant {
        bits: metadata_bits,
    } = metadata.params
    else {
        return Err(DenseError::metadata_mismatch(
            "OrdinalDB dense bundle primary artifact has non-RankQuant params",
        ));
    };
    if let Some(expected_dim) = load_options.expected_dim {
        if metadata.dim != expected_dim {
            return Err(DenseError::metadata_mismatch(format!(
                "verified dense dim {} does not match expected dim {expected_dim}",
                metadata.dim
            )));
        }
    }
    if let Some(expected_bits) = load_options.expected_bits {
        if metadata_bits != expected_bits {
            return Err(DenseError::metadata_mismatch(format!(
                "verified dense bits {metadata_bits} does not match expected bits {expected_bits}"
            )));
        }
    }
    if plan.row_identity().kind() != "row_id_identity" {
        return Err(DenseError::row_identity(format!(
            "OrdinalDB dense bundles currently require row_id_identity row identity; got {:?}",
            plan.row_identity().kind()
        )));
    }

    let primary_bytes = read_primary_verified(plan)?;
    let rankquant = RankQuant::load_from_bytes(&primary_bytes)?;
    if metadata.dim != rankquant.dim()
        || metadata.vector_count != rankquant.len()
        || metadata_bits != rankquant.bits()
        || plan.row_identity().row_count() != rankquant.len()
    {
        return Err(DenseError::metadata_mismatch(
            "verified manifest metadata does not match loaded RankQuant",
        ));
    }

    // Resolve the sign sidecar under the same policy as `load_verified_bundle`
    // (Forbid / Require / RequireIfSupported / Any), but freshness-re-check a
    // loaded sidecar via `read_verified` before mapping it.
    let sign_aux = plan.auxiliary_by_name(crate::io::SIGN_AUX_NAME);
    if matches!(load_options.sign, SignLoadPolicy::Forbid) && sign_aux.is_some() {
        return Err(DenseError::SignSidecarForbidden);
    }
    let sign_required = match load_options.sign {
        SignLoadPolicy::Require => true,
        SignLoadPolicy::RequireIfSupported => sign_compatible(metadata.dim, metadata_bits),
        SignLoadPolicy::Forbid | SignLoadPolicy::Any => false,
    };
    let sign = match sign_aux {
        // A `Forbid` policy with a declared sidecar already returned above; any
        // other policy loads a declared sidecar.
        Some(aux) if !matches!(load_options.sign, SignLoadPolicy::Forbid) => {
            // Re-hash the sign sidecar against its recorded digest and load it
            // from those very bytes — never reopening the path — so the bytes
            // hashed are the bytes loaded (closes the re-hash-then-reopen
            // TOCTOU). `read_verified` is itself bounded to the recorded size.
            let sign_bytes = aux.read_verified()?;
            let sign = SignBitmap::load_from_bytes(&sign_bytes)?;
            if sign.dim() != rankquant.dim() || sign.len() != rankquant.len() {
                return Err(DenseError::metadata_mismatch(format!(
                    "sign sidecar shape mismatch: sign dim={}, len={} but RankQuant dim={}, len={}",
                    sign.dim(),
                    sign.len(),
                    rankquant.dim(),
                    rankquant.len()
                )));
            }
            Some(sign)
        }
        _ => {
            if sign_required {
                return Err(DenseError::MissingSignSidecar);
            }
            None
        }
    };
    Ok((rankquant, sign))
}

/// Read the primary RankQuant artifact from its verified path and re-check
/// the bytes against the SHA-256 and size the plan recorded at verification
/// time, returning the bytes only on an exact match. The hash is computed
/// over the single in-memory read, so the caller can construct the index
/// from these very bytes via `RankQuant::load_from_bytes` — the bytes hashed
/// are the bytes loaded, closing the re-hash-then-reopen TOCTOU that a
/// separate hash-then-`RankQuant::load(path)` left open. The read is bounded
/// to `recorded_size + 1` bytes so a post-verification swap to a huge file is
/// rejected by size before it can exhaust memory. Cheaper than a full
/// `verify_for_load`, which also re-parses and re-checks the manifest.
fn read_primary_verified(plan: &VerifiedLoadPlan) -> Result<Vec<u8>, DenseError> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    let path = plan.artifact_path();
    let expected_sha = plan.report().artifact.sha256.as_deref().ok_or_else(|| {
        DenseError::metadata_mismatch(
            "verified plan records no sha256 for the primary RankQuant artifact; \
             cannot re-check freshness",
        )
    })?;
    // `artifact.size_bytes` is the manifest-recorded, verification-checked
    // size; `metadata.file_size_bytes` is the always-present fallback. Either
    // came from a manifest that already passed size-bounded verification, so
    // both are trusted bounds.
    let recorded_size = plan
        .report()
        .artifact
        .size_bytes
        .unwrap_or_else(|| plan.metadata().file_size_bytes);

    let file = std::fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.take(recorded_size.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 != recorded_size {
        return Err(DenseError::metadata_mismatch(
            "primary RankQuant artifact changed on disk since the plan was verified \
             (stale plan reuse)",
        ));
    }
    let digest = hex::encode(Sha256::digest(&bytes));
    if !digest.eq_ignore_ascii_case(expected_sha) {
        return Err(DenseError::metadata_mismatch(
            "primary RankQuant artifact changed on disk since the plan was verified \
             (stale plan reuse)",
        ));
    }
    Ok(bytes)
}

#[cfg(test)]
mod chunk_scheduler_tests {
    use super::{two_stage_query_chunk_rows, TWO_STAGE_MAX_SCORE_CELLS};

    fn in_pool<T: Send>(threads: usize, f: impl FnOnce() -> T + Send) -> T {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("test pool")
            .install(f)
    }

    /// A realistic batch (nq=2048, M=320) must split across the pool
    /// instead of landing in one chunk on one core — the measured cliff
    /// this scheduler replaces.
    #[test]
    fn realistic_batch_engages_the_pool() {
        let chunk = in_pool(8, || two_stage_query_chunk_rows(320, 2048));
        assert_eq!(chunk, 256); // ceil(2048/8), floor and cells cap inactive
        assert!(2048usize.div_ceil(chunk) >= 8, "all 8 workers get a chunk");
    }

    /// The CSR memory cap still wins for huge candidate pools.
    #[test]
    fn memory_cap_wins_for_huge_candidate_pools() {
        let chunk = in_pool(8, || two_stage_query_chunk_rows(500_000, 2048));
        assert_eq!(chunk, (TWO_STAGE_MAX_SCORE_CELLS / 500_000).max(1));
    }

    /// Tiny batches stay in one chunk: scan-sharing beats parallel
    /// corpus passes below the tile floor.
    #[test]
    fn tiny_batch_stays_single_chunk() {
        let chunk = in_pool(8, || two_stage_query_chunk_rows(320, 8));
        assert_eq!(chunk, 8);
    }

    /// The tile floor bounds splitting: even a wide pool never shrinks
    /// chunks below TILE_FLOOR queries of shared scanning. (16 threads is
    /// enough to make the floor bind — ceil(512/16) = 32 < TILE_FLOOR —
    /// without oversubscribing small CI runners.)
    #[test]
    fn tile_floor_bounds_splitting() {
        let chunk = in_pool(16, || two_stage_query_chunk_rows(320, 512));
        assert_eq!(chunk, 128);
    }

    /// Single-threaded pools reproduce the legacy cells-cap behavior.
    #[test]
    fn single_thread_keeps_legacy_chunking() {
        let chunk = in_pool(1, || two_stage_query_chunk_rows(320, 4096));
        assert_eq!(chunk, TWO_STAGE_MAX_SCORE_CELLS / 320);
    }

    #[test]
    fn env_override_parse_is_clamped() {
        assert_eq!(super::parse_tile_floor(Some("32".into())), 32);
        assert_eq!(super::parse_tile_floor(Some("1".into())), 1);
        assert_eq!(super::parse_tile_floor(Some(" 64 ".into())), 64);
        assert_eq!(super::parse_tile_floor(Some("0".into())), 128);
        assert_eq!(super::parse_tile_floor(Some("bogus".into())), 128);
        assert_eq!(super::parse_tile_floor(None), 128);
    }

    #[test]
    fn degenerate_inputs_return_one() {
        assert_eq!(two_stage_query_chunk_rows(0, 100), 1);
        assert_eq!(two_stage_query_chunk_rows(100, 0), 1);
    }
}
