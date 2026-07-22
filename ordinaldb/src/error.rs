use std::error::Error;
use std::fmt;
use std::io;

/// Errors returned when adding vectors to an [`crate::OrdinalIndex`] or
/// [`crate::IdMapIndex`] (`add`, `add_2d`, `add_with_ids`,
/// `add_with_ids_2d`, and the [`crate::OrdinalIndexBuilder`] wrapper).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum AddError {
    /// `dim` is less than 2, or does not fit in a `u16`. Returned by the
    /// checked add methods before any bits-compatibility check; unlike
    /// [`crate::ConstructError::DimInvalid`], this occurs on a per-batch add
    /// rather than at construction.
    DimInvalid(usize),
    /// The batch's `dim` does not match the `dim` the index already
    /// committed to on a previous add (or at construction). `existing` is
    /// the index's fixed dim; `got` is the batch's dim.
    DimMismatch {
        /// The dim the index is committed to.
        existing: usize,
        /// The dim of the rejected batch.
        got: usize,
    },
    /// `dim` is not compatible with the index's RankQuant bit width: it
    /// must be a multiple of `8 / bits` (so codes pack into whole bytes)
    /// and a multiple of `2^bits` (so every rank bucket holds an equal
    /// share of coordinates). `reason` names which of the two invariants
    /// failed.
    DimNotCompatibleWithBits {
        /// The rejected dim.
        dim: usize,
        /// The index's RankQuant bit width.
        bits: u8,
        /// Which packing/bucket invariant `dim` violated.
        reason: &'static str,
    },
    /// The flat `vectors` buffer length is not a multiple of `dim`, so it
    /// cannot be split into whole row-major vectors.
    VectorBufferNotMultipleOfDim {
        /// Length of the flat `vectors` buffer.
        vectors_len: usize,
        /// The index dim it must divide into.
        dim: usize,
    },
    /// A coordinate in the input batch is not finite (`NaN`/`±inf`) or its
    /// magnitude is `>= 1e16`. `vector_index`/`coord_index` locate the bad
    /// value within the flat, row-major `vectors` buffer.
    InvalidInputValue {
        /// Row of the offending vector within the batch.
        vector_index: usize,
        /// Coordinate within that vector.
        coord_index: usize,
        /// The rejected value.
        value: f32,
    },
    /// `add_with_ids`/`add_with_ids_2d` only: the `ids` slice length does
    /// not match the number of rows implied by `vectors.len() / dim`.
    IdsCountMismatch {
        /// Row count implied by `vectors.len() / dim`.
        expected: usize,
        /// Length of the `ids` slice.
        got: usize,
    },
    /// `add_with_ids`/`add_with_ids_2d` only: an ID in the batch is either
    /// already present in the index, or duplicated within the same batch.
    /// The add is rejected in full (no partial insertion).
    IdAlreadyPresent(u64),
    /// The add would commit a lazy index to a `dim` that cannot carry the
    /// sign sidecar its [`crate::SignPolicy::Required`] build policy
    /// demands. The batch is rejected in full and the index stays lazy;
    /// the construction-time equivalent is
    /// [`crate::ConstructError::SignSidecarUnsupported`].
    SignSidecarUnsupported {
        /// The rejected dim.
        dim: usize,
        /// The index's RankQuant bit width.
        bits: u8,
        /// The multiple `dim` must satisfy for a sign sidecar at this
        /// `bits` (see [`crate::sign_required_multiple`]), or `None` if
        /// this `bits` never supports one.
        required_multiple: Option<usize>,
    },
}

impl fmt::Display for AddError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimInvalid(dim) => write!(
                f,
                "dimension {dim} is out of range: dim must be >= 2 and representable as u16"
            ),
            Self::DimMismatch { existing, got } => {
                write!(
                    f,
                    "batch dimension {got} does not match the dimension {existing} this index committed to"
                )
            }
            Self::DimNotCompatibleWithBits { dim, bits, reason } => {
                write!(
                    f,
                    "dim {dim} is not compatible with OrdVec RankQuant bits {bits}: {reason}"
                )
            }
            Self::VectorBufferNotMultipleOfDim { vectors_len, dim } => write!(
                f,
                "a {vectors_len}-element vector buffer does not split into whole rows of dim {dim}"
            ),
            Self::InvalidInputValue {
                vector_index,
                coord_index,
                value,
            } => write!(
                f,
                "invalid input value at vector {vector_index}, coord {coord_index}: {value}"
            ),
            Self::IdsCountMismatch { expected, got } => {
                write!(
                    f,
                    "ids slice holds {got} entries for a batch of {expected} rows"
                )
            }
            Self::IdAlreadyPresent(id) => write!(
                f,
                "duplicate id {id}: already stored in the index or repeated within the batch"
            ),
            Self::SignSidecarUnsupported {
                dim,
                bits,
                required_multiple,
            } => write_sign_sidecar_unsupported(f, *dim, *bits, *required_multiple),
        }
    }
}

impl Error for AddError {}

fn write_sign_sidecar_unsupported(
    f: &mut fmt::Formatter<'_>,
    dim: usize,
    bits: u8,
    required_multiple: Option<usize>,
) -> fmt::Result {
    match required_multiple {
        Some(multiple) => write!(
            f,
            "sign policy Required cannot be honored: dim {dim} is not a multiple of {multiple} (bits {bits})"
        ),
        None => write!(
            f,
            "sign policy Required cannot be honored: bits {bits} never supports a sign sidecar (requires bits 2); got dim {dim}"
        ),
    }
}

/// Errors returned when constructing an [`crate::OrdinalIndex`] or
/// [`crate::IdMapIndex`] (`new`, `new_lazy`).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConstructError {
    /// `bits` is not `1`, `2`, or `4`. OrdinalDB restricts RankQuant to
    /// these three stable retrieval widths even though the underlying
    /// `ordvec` crate also supports `8`.
    UnsupportedBits(u8),
    /// `dim` is less than 2, or does not fit in a `u16`.
    DimInvalid(usize),
    /// `dim` is not compatible with `bits`: it must be a multiple of
    /// `8 / bits` (whole-byte packing) and a multiple of `2^bits` (equal
    /// rank-bucket occupancy). `reason` names which invariant failed.
    DimNotCompatibleWithBits {
        /// The rejected dim.
        dim: usize,
        /// The requested RankQuant bit width.
        bits: u8,
        /// Which packing/bucket invariant `dim` violated.
        reason: &'static str,
    },
    /// Construction requested [`crate::SignPolicy::Required`] but `(dim,
    /// bits)` cannot carry a sign sidecar (sidecars need `bits == 2` and
    /// `dim` a multiple of `64`). On a lazy `bits == 2` index, an
    /// incompatible `dim` surfaces on the first non-empty add as
    /// [`crate::AddError::SignSidecarUnsupported`]; a lazy bit width that
    /// can never carry a sidecar is rejected earlier as
    /// [`Self::SignSidecarUnsupportedBits`].
    SignSidecarUnsupported {
        /// The rejected dim.
        dim: usize,
        /// The requested RankQuant bit width.
        bits: u8,
        /// The multiple `dim` must satisfy for a sign sidecar at this
        /// `bits` (see [`crate::sign_required_multiple`]), or `None` if
        /// this `bits` never supports one.
        required_multiple: Option<usize>,
    },
    /// Lazy construction requested [`crate::SignPolicy::Required`] with a
    /// RankQuant bit width that can never carry a sign sidecar. Because no
    /// eventual `dim` can make the configuration valid, this is rejected at
    /// construction instead of being deferred to the first add.
    SignSidecarUnsupportedBits {
        /// The requested RankQuant bit width (`1` or `4`).
        bits: u8,
    },
}

impl fmt::Display for ConstructError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedBits(bits) => write!(
                f,
                "OrdinalDB/OrdVec RankQuant supports bits 1, 2, or 4; got {bits}"
            ),
            Self::DimInvalid(dim) => write!(
                f,
                "dimension {dim} is out of range: dim must be >= 2 and representable as u16"
            ),
            Self::DimNotCompatibleWithBits { dim, bits, reason } => {
                write!(
                    f,
                    "dim {dim} is not compatible with OrdVec RankQuant bits {bits}: {reason}"
                )
            }
            Self::SignSidecarUnsupported {
                dim,
                bits,
                required_multiple,
            } => write_sign_sidecar_unsupported(f, *dim, *bits, *required_multiple),
            Self::SignSidecarUnsupportedBits { bits } => write!(
                f,
                "sign policy Required cannot be honored: bits {bits} never supports a sign sidecar (requires bits 2)"
            ),
        }
    }
}

impl Error for ConstructError {}

/// Errors returned by the checked (`Result`-returning) search and
/// persistence APIs across [`crate::OrdinalIndex`] and [`crate::IdMapIndex`]
/// (`search_checked*`, `search_with_report`, `write_verified_bundle`,
/// `open_verified`).
///
/// This is the error type to expect once a batch has already been
/// successfully added; the panicking counterparts of these APIs
/// (`search`, `search_with_options`, ...) surface most of the same
/// conditions as panics instead.
#[derive(Debug)]
#[non_exhaustive]
pub enum DenseError {
    /// A filesystem operation failed while writing or reading a bundle
    /// artifact.
    Io(io::Error),
    /// Index construction failed; see [`ConstructError`]. Only reachable
    /// through [`crate::OrdinalIndexBuilder::new`], which constructs its
    /// backing index lazily.
    Construct(ConstructError),
    /// Adding a row failed; see [`AddError`]. Only reachable through
    /// [`crate::OrdinalIndexBuilder::add`].
    Add(AddError),
    /// The bundle's `manifest.json` failed `ordvec-manifest`'s structural
    /// verification (checksum mismatch, size-limit violation, unsafe path,
    /// ...) during `open_verified` or `write_verified_bundle`'s
    /// post-write verification pass.
    ManifestVerification(ordvec_manifest::VerifiedLoadPlanError),
    /// The manifest could not be parsed, serialized, or otherwise
    /// constructed; a lower-level error than
    /// [`Self::ManifestVerification`].
    Manifest(ordvec_manifest::ManifestError),
    /// A required auxiliary artifact declared on the manifest (for example
    /// a sign sidecar requested via
    /// [`crate::ordinal::DenseLoadOptions::require_sign`]) was missing or
    /// invalid.
    Auxiliary(ordvec_manifest::RequireAuxiliaryError),
    /// The flat `queries` buffer length is not a multiple of `dim` (or
    /// `dim` is `0`, which only occurs when searching a lazily-constructed,
    /// still-empty index with a non-empty query buffer).
    InvalidQueryDim {
        /// Length of the flat `queries` buffer.
        len: usize,
        /// The index dim it must divide into.
        dim: usize,
    },
    /// A coordinate in the query batch is not finite (`NaN`/`±inf`) or its
    /// magnitude is `>= 1e16`. `query_index`/`coord_index` locate the bad
    /// value within the flat, row-major `queries` buffer.
    InvalidQueryValue {
        /// Row of the offending query within the batch.
        query_index: usize,
        /// Coordinate within that query.
        coord_index: usize,
        /// The rejected value.
        value: f32,
    },
    /// [`crate::ordinal::DenseSearchMode::SignTwoStage`] was requested
    /// explicitly but the index has no `SignBitmap` sidecar to generate
    /// candidates from (sign sidecars only exist for `bits == 2` indexes
    /// with `dim` a multiple of `64`, and only when the
    /// [`crate::SignPolicy`] build policy allows one).
    /// `DenseSearchMode::Auto` never produces this error — it silently
    /// falls back to an exact scan instead.
    MissingSignSidecar,
    /// An internal consistency check failed with a human-readable
    /// `message` — for example, a verified bundle's manifest metadata
    /// (dim/bits/row count) disagreeing with the artifact actually loaded
    /// from disk, or a dense search result buffer with an unexpected
    /// shape.
    MetadataMismatch(String),
    /// The bundle's row-identity kind is not the one the caller expected —
    /// for example, opening an ID-mapped bundle with
    /// [`crate::OrdinalIndex::open_verified`] instead of
    /// [`crate::IdMapIndex::open_verified`], or vice versa.
    RowIdentity(String),
    /// A row ID passed in a search allowlist does not map to any row in
    /// the index. For [`crate::IdMapIndex`] this means the ID was never
    /// added (or was removed); for a bare [`crate::OrdinalIndex`], IDs
    /// *are* slot indices, so this also fires when the ID is `>= len()` or
    /// does not fit in a `usize`.
    AllowlistRowIdMissing(u64),
    /// A candidate slot passed to `search_checked_with_candidates` is
    /// `>= len()`.
    CandidateSlotOutOfRange(u32),
    /// An internal slot index did not fit into a `u32`. Candidate and
    /// allowlist search APIs represent slots as `u32`; this is only
    /// reachable on indexes with more than `u32::MAX` rows.
    SlotIndexOverflow(usize),
    /// A resource or size limit was exceeded — for example, `nq * k`
    /// overflowing when sizing a result buffer, or a two-stage search
    /// resolving to zero candidates for a non-empty query.
    Limit(String),
}

impl DenseError {
    pub(crate) fn metadata_mismatch(message: impl Into<String>) -> Self {
        Self::MetadataMismatch(message.into())
    }

    pub(crate) fn row_identity(message: impl Into<String>) -> Self {
        Self::RowIdentity(message.into())
    }
}

impl fmt::Display for DenseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "{err}"),
            Self::Construct(err) => write!(f, "{err}"),
            Self::Add(err) => write!(f, "{err}"),
            Self::ManifestVerification(err) => write!(f, "{err}"),
            Self::Manifest(err) => write!(f, "{err}"),
            Self::Auxiliary(err) => write!(f, "{err}"),
            Self::InvalidQueryDim { len, dim } => {
                write!(
                    f,
                    "query buffer length {len} is not a multiple of dim {dim}"
                )
            }
            Self::InvalidQueryValue {
                query_index,
                coord_index,
                value,
            } => write!(
                f,
                "invalid query value at query {query_index}, coord {coord_index}: {value}"
            ),
            Self::MissingSignSidecar => f.write_str("verified dense load requires a sign sidecar"),
            Self::MetadataMismatch(message) => f.write_str(message),
            Self::RowIdentity(message) => f.write_str(message),
            Self::AllowlistRowIdMissing(id) => {
                write!(f, "allowlist row_id {id} is not present in index")
            }
            Self::CandidateSlotOutOfRange(slot) => {
                write!(f, "candidate slot {slot} is not present in index")
            }
            Self::SlotIndexOverflow(slot) => write!(f, "slot index {slot} exceeds u32 range"),
            Self::Limit(message) => f.write_str(message),
        }
    }
}

impl Error for DenseError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Construct(err) => Some(err),
            Self::Add(err) => Some(err),
            Self::ManifestVerification(err) => Some(err),
            Self::Manifest(err) => Some(err),
            Self::Auxiliary(err) => Some(err),
            Self::InvalidQueryDim { .. }
            | Self::InvalidQueryValue { .. }
            | Self::MissingSignSidecar
            | Self::MetadataMismatch(_)
            | Self::RowIdentity(_)
            | Self::AllowlistRowIdMissing(_)
            | Self::CandidateSlotOutOfRange(_)
            | Self::SlotIndexOverflow(_)
            | Self::Limit(_) => None,
        }
    }
}

impl From<io::Error> for DenseError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<ConstructError> for DenseError {
    fn from(value: ConstructError) -> Self {
        Self::Construct(value)
    }
}

impl From<AddError> for DenseError {
    fn from(value: AddError) -> Self {
        Self::Add(value)
    }
}

impl From<ordvec_manifest::VerifiedLoadPlanError> for DenseError {
    fn from(value: ordvec_manifest::VerifiedLoadPlanError) -> Self {
        Self::ManifestVerification(value)
    }
}

impl From<ordvec_manifest::ManifestError> for DenseError {
    fn from(value: ordvec_manifest::ManifestError) -> Self {
        Self::Manifest(value)
    }
}

impl From<ordvec_manifest::RequireAuxiliaryError> for DenseError {
    fn from(value: ordvec_manifest::RequireAuxiliaryError) -> Self {
        Self::Auxiliary(value)
    }
}
