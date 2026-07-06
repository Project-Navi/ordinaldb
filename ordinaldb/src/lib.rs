#![warn(missing_docs)]

//! OrdinalDB core: a local-first, embedded vector index built on
//! [OrdVec](https://crates.io/crates/ordvec) ordinal rank quantization
//! (`RankQuant`).
//!
//! Vectors are stored as row-major, dense `f32` buffers. On `add`, each
//! vector's coordinates are rank-transformed and bucketed into `1 << bits`
//! equal-width bins (`bits` is one of `1`, `2`, or `4`), then bit-packed —
//! `dim * bits / 8` bytes per row instead of `dim * 4` bytes for the raw
//! `f32`s. Search scores every candidate with an asymmetric (raw query vs.
//! quantized document) or symmetric (quantized vs. quantized) comparison and
//! returns the highest-scoring rows first. When `bits == 2` and `dim` is a
//! multiple of `64`, a `SignBitmap` sidecar is also maintained so search can
//! run a cheap sign-cosine candidate generation pass before reranking the
//! shortlist with the exact `RankQuant` scan (see
//! [`ordinal::DenseSearchMode`]).
//!
//! # `OrdinalIndex` vs. `IdMapIndex`
//!
//! - [`OrdinalIndex`] is the low-level dense index. Rows live in append
//!   order at integer "slots" `0..len()`; a row's slot *is* its row
//!   identity (OrdinalDB calls this `row_id_identity`). There is no
//!   separate ID space — callers that need stable external IDs surviving
//!   `swap_remove`-style deletion should use [`IdMapIndex`] instead.
//! - [`IdMapIndex`] wraps an `OrdinalIndex` with a bidirectional mapping
//!   between caller-supplied `u64` IDs and internal slots (`add_with_ids`,
//!   `remove(id)`, ID-based search allowlists). This is the type most
//!   applications want, since it lets rows be deleted and re-inserted
//!   without external IDs shifting.
//!
//! Both types share the same search, persistence, and error surface
//! ([`AddError`], [`ConstructError`], [`DenseError`]).
//!
//! # `.odb` bundle persistence
//!
//! An index is persisted as a *bundle*: a directory (conventionally named
//! with a `.odb` suffix, e.g. `my_index.odb/`) containing:
//!
//! - `manifest.json` — an integrity-checked manifest (dim, bits, row count,
//!   row-identity kind, and a list of auxiliary artifacts with their SHA-256
//!   checksums). See the [`manifest`] module.
//! - `index.ovrq` — the packed `RankQuant` bucket codes.
//! - `sign.ovsb` — the optional `SignBitmap` sidecar, registered as a
//!   required auxiliary artifact when present.
//! - `ids.bin` — present only for [`IdMapIndex`] bundles: the `u64` row-ID
//!   sidecar, registered as a required auxiliary artifact.
//!
//! Bundle writes are atomic: contents are written to a temporary directory,
//! verified, `fsync`ed, and only then moved into place (with a
//! backup-and-recover step), so a bundle at `path` is always either fully
//! present or absent — never partially written. There are two ways to load
//! a bundle back:
//!
//! - `OrdinalIndex::load` / `IdMapIndex::load` open a bundle *directory*
//!   with default verification.
//! - `OrdinalIndex::open_verified` / `IdMapIndex::open_verified` open a
//!   bundle from an explicit `manifest.json` path with caller-controlled
//!   [`manifest::VerifyOptions`] (size limits, path-escape policy) and
//!   [`ordinal::DenseLoadOptions`] (require a sign sidecar, assert an
//!   expected `dim`/`bits`), reporting mismatches as a [`DenseError`]
//!   instead of a generic I/O error.
//!
//! Loading an `OrdinalIndex` bundle that actually carries an ID sidecar
//! (or vice versa) is a load-time error that names the correct type to use.
//!
//! # Example
//!
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use ordinaldb::OrdinalIndex;
//!
//! // 4-dimensional vectors, 2-bit RankQuant codes. For `bits == 2`, `dim`
//! // must be a multiple of 4 (preflight with `rankquant_compatible`; see
//! // `ConstructError::DimNotCompatibleWithBits`).
//! let mut index = OrdinalIndex::new(4, 2)?;
//!
//! // Two row-major vectors, back to back.
//! index.add_2d(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], 4)?;
//!
//! // Top-1 nearest neighbor of [1, 0, 0, 0]; scores are sorted descending.
//! let results = index.search_checked(&[1.0, 0.0, 0.0, 0.0], 1)?;
//! assert_eq!(results.indices[0], 0);
//! # Ok(())
//! # }
//! ```

/// `AddError`/`ConstructError`/`DenseError` — the error enums returned by
/// index construction, mutation, search, and persistence.
pub mod error;
/// [`id_map::IdMapIndex`]: an [`ordinal::OrdinalIndex`] with a caller-facing
/// `u64` ID space layered on top of internal slots.
pub mod id_map;
/// Bundle manifest creation/verification primitives (re-exported from
/// `ordvec-manifest`) used to build and check `.odb` bundles.
pub mod manifest;
/// [`ordinal::OrdinalIndex`]: the dense, slot-addressed `RankQuant` index.
pub mod ordinal;

/// Well-known names of the files and manifest auxiliary artifacts that make
/// up an OrdinalDB `.odb` bundle.
///
/// These constants are useful for tooling that inspects a bundle's
/// `manifest.json` directly (e.g. to look up the sign or ID sidecar by
/// name) rather than going through [`OrdinalIndex`]/[`IdMapIndex`].
pub mod artifacts {
    pub use crate::io::{
        EMBEDDING_MODEL, IDS_AUX_NAME, IDS_FILE, INDEX_FILE, MANIFEST_FILE, SIGN_AUX_NAME,
        SIGN_FILE,
    };

    /// Manifest auxiliary-artifact name for an attached learning-to-rank
    /// model, when the crate is built with the `experimental-ltr` feature.
    #[cfg(feature = "experimental-ltr")]
    pub use ordinaldb_hybrid::DEFAULT_LTR_MODEL_AUX_NAME as LTR_MODEL_AUX_NAME;
    /// Manifest auxiliary-artifact name for an attached sparse (BM25) index,
    /// when the crate is built with the `hybrid` feature.
    #[cfg(feature = "hybrid")]
    pub use ordinaldb_hybrid::DEFAULT_SPARSE_AUX_NAME as SPARSE_BM25_AUX_NAME;
}

/// Hybrid (dense + sparse) search support, re-exported from
/// `ordinaldb-hybrid`, plus the [`hybrid::HybridBundle`] one-call open for
/// a bundle's dense-plus-sparse pair. Only compiled with the `hybrid` Cargo
/// feature; the learning-to-rank reranking items additionally require
/// `experimental-ltr` (see that crate's docs for its stability status).
#[cfg(feature = "hybrid")]
pub mod hybrid;

mod io;

pub use error::{AddError, ConstructError, DenseError};
pub use id_map::{IdMapIndex, IdMapSearchReport};
pub use ordinal::{
    rankquant_compatible, rankquant_required_multiple, sign_compatible, sign_required_multiple,
    BuildOptions, DenseBundleInspectReport, DenseLoadOptions, DenseSearchExecution,
    DenseSearchMode, DenseSearchOptions, DenseSearchPlan, DenseSearchReport, DenseSearchTimings,
    OrdinalIndex, OrdinalIndexBuilder, SearchResults, SignPolicy, TwoStageOptions,
    VerifiedBundleReport,
};
