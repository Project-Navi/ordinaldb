use std::error::Error;
use std::fmt;
use std::io;

/// Result type used by `ordinaldb-hybrid`.
pub type Result<T> = std::result::Result<T, HybridError>;

/// Typed errors for sparse mmap loading, manifest sidecar lookup, and fusion.
#[derive(Debug)]
pub enum HybridError {
    Io(io::Error),
    ManifestVerification(ordvec_manifest::VerifiedLoadPlanError),
    Auxiliary(ordvec_manifest::RequireAuxiliaryError),
    MalformedLayout(String),
    Tokenizer(String),
    RowId(String),
    Batch(String),
    #[cfg(feature = "ltr")]
    Ltr(String),
    Limit(String),
}

impl HybridError {
    pub(crate) fn malformed(message: impl Into<String>) -> Self {
        Self::MalformedLayout(message.into())
    }

    pub(crate) fn tokenizer(message: impl Into<String>) -> Self {
        Self::Tokenizer(message.into())
    }

    pub(crate) fn row_id(message: impl Into<String>) -> Self {
        Self::RowId(message.into())
    }

    pub(crate) fn batch(message: impl Into<String>) -> Self {
        Self::Batch(message.into())
    }

    #[cfg(feature = "ltr")]
    pub(crate) fn ltr(message: impl Into<String>) -> Self {
        Self::Ltr(message.into())
    }

    pub(crate) fn limit(message: impl Into<String>) -> Self {
        Self::Limit(message.into())
    }
}

impl fmt::Display for HybridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "io error: {error}"),
            Self::ManifestVerification(error) => write!(f, "manifest verification failed: {error}"),
            Self::Auxiliary(error) => write!(f, "auxiliary artifact lookup failed: {error}"),
            Self::MalformedLayout(message) => write!(f, "malformed sparse mmap: {message}"),
            Self::Tokenizer(message) => write!(f, "tokenizer error: {message}"),
            Self::RowId(message) => write!(f, "row-id error: {message}"),
            Self::Batch(message) => write!(f, "ranked batch error: {message}"),
            #[cfg(feature = "ltr")]
            Self::Ltr(message) => write!(f, "LTR model error: {message}"),
            Self::Limit(message) => write!(f, "limit exceeded: {message}"),
        }
    }
}

impl Error for HybridError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::ManifestVerification(error) => Some(error),
            Self::Auxiliary(error) => Some(error),
            Self::MalformedLayout(_)
            | Self::Tokenizer(_)
            | Self::RowId(_)
            | Self::Batch(_)
            | Self::Limit(_) => None,
            #[cfg(feature = "ltr")]
            Self::Ltr(_) => None,
        }
    }
}

impl From<io::Error> for HybridError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<ordvec_manifest::VerifiedLoadPlanError> for HybridError {
    fn from(error: ordvec_manifest::VerifiedLoadPlanError) -> Self {
        Self::ManifestVerification(error)
    }
}

impl From<ordvec_manifest::RequireAuxiliaryError> for HybridError {
    fn from(error: ordvec_manifest::RequireAuxiliaryError) -> Self {
        Self::Auxiliary(error)
    }
}
