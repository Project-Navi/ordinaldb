use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::{SystemTime, UNIX_EPOCH};

use memmap2::Mmap;
use ordvec_manifest::{verify_for_load, VerifiedLoadPlan, VerifyOptions};
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

use crate::fusion::{RankedBatch, ScoredRow};
use crate::{HybridError, Result};

pub const DEFAULT_SPARSE_AUX_NAME: &str = "ordinaldb.sparse_bm25";
const ORDINALDB_IDS_AUX_NAME: &str = "ordinaldb.ids";
const ORDINALDB_IDS_MAGIC: &[u8; 8] = b"ODBIDS1\0";

const MAGIC: &[u8; 8] = b"ODBSBM1\0";
const VERSION: u32 = 1;
const HEADER_LEN: usize = 128;
const ENDIAN_MARKER: u32 = 0x0102_0304;
const TOKENIZER_VERSION: u16 = 1;
const DEFAULT_K1: f32 = 1.2;
const DEFAULT_B: f32 = 0.75;

const TERM_ENTRY_LEN: usize = 32;
const POSTING_LEN: usize = 8;
const ROW_ID_LEN: usize = 8;
const DOC_LEN_LEN: usize = 4;
const TEMP_CREATE_ATTEMPTS: usize = 16;

const VERSION_OFFSET: usize = 8;
const HEADER_LEN_OFFSET: usize = 12;
const ENDIAN_OFFSET: usize = 16;
const FLAGS_OFFSET: usize = 20;
const TOKENIZER_KIND_OFFSET: usize = 24;
const TOKENIZER_VERSION_OFFSET: usize = 26;
const RESERVED0_OFFSET: usize = 28;
const DOC_COUNT_OFFSET: usize = 32;
const TERM_COUNT_OFFSET: usize = 40;
const POSTINGS_COUNT_OFFSET: usize = 48;
const AVG_DOC_LEN_OFFSET: usize = 56;
const K1_OFFSET: usize = 60;
const B_OFFSET: usize = 64;
const RESERVED1_OFFSET: usize = 68;
const ROW_IDS_OFFSET_OFFSET: usize = 72;
const DOC_LEN_OFFSET_OFFSET: usize = 80;
const TERM_TABLE_OFFSET_OFFSET: usize = 88;
const TERM_BYTES_OFFSET_OFFSET: usize = 96;
const POSTINGS_OFFSET_OFFSET: usize = 104;

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);
const FILE_LEN_OFFSET: usize = 112;
const RESERVED2_OFFSET: usize = 120;

/// Tokenizer recorded in sparse mmap headers.
///
/// Version 1 behavior is intentionally simple and stable:
/// ASCII alphanumeric runs are candidate tokens; other bytes, including `_`,
/// split tokens. Normalization lowercases ASCII, drops tokens shorter than
/// three bytes, and strips trailing `s` bytes while the token remains longer
/// than four bytes (so a normalized term never both exceeds four bytes and
/// ends in `s` — e.g. `users` → `user`, `access` → `acce`, `cars` stays
/// `cars`). `IdentifierSubtokens` additionally indexes camel-case subtokens
/// from each ASCII alphanumeric run, while still indexing the full
/// normalized run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum TokenizerKind {
    Simple = 1,
    IdentifierSubtokens = 2,
}

impl TokenizerKind {
    fn from_u16(value: u16) -> Result<Self> {
        match value {
            1 => Ok(Self::Simple),
            2 => Ok(Self::IdentifierSubtokens),
            _ => Err(HybridError::tokenizer(format!(
                "unknown tokenizer kind {value}"
            ))),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SparseBuildReport {
    pub row_count: usize,
    pub term_count: usize,
    pub postings_count: usize,
    pub tokenizer: TokenizerKind,
    pub tokenizer_version: u16,
    pub avg_doc_len: f32,
    pub file_size_bytes: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SparseInspectReport {
    pub row_count: usize,
    pub term_count: usize,
    pub postings_count: usize,
    pub tokenizer: TokenizerKind,
    pub tokenizer_version: u16,
    pub avg_doc_len: f32,
    pub k1: f32,
    pub b: f32,
    pub file_size_bytes: u64,
}

#[derive(Clone, Debug)]
struct PendingDoc {
    row_id: u64,
    terms: Vec<String>,
}

/// Builder for OrdinalDB BM25 sparse mmap sidecars.
#[derive(Clone, Debug)]
pub struct SparseIndexBuilder {
    tokenizer: TokenizerKind,
    docs: Vec<PendingDoc>,
    seen_row_ids: HashSet<u64>,
}

impl SparseIndexBuilder {
    pub fn new(tokenizer: TokenizerKind) -> Self {
        Self {
            tokenizer,
            docs: Vec::new(),
            seen_row_ids: HashSet::new(),
        }
    }

    pub fn add_text(&mut self, row_id: u64, text: &str) -> Result<()> {
        self.add_doc(row_id, tokenize_text(self.tokenizer, text))
    }

    pub fn add_normalized_terms<I>(&mut self, row_id: u64, terms: I) -> Result<()>
    where
        I: IntoIterator<Item = String>,
    {
        let mut normalized = Vec::new();
        for term in terms {
            if let Some(term) = normalize_term(&term) {
                normalized.push(term);
            }
        }
        self.add_doc(row_id, normalized)
    }

    fn add_doc(&mut self, row_id: u64, terms: Vec<String>) -> Result<()> {
        if !self.seen_row_ids.insert(row_id) {
            return Err(HybridError::row_id(format!("duplicate row_id {row_id}")));
        }
        self.docs.push(PendingDoc { row_id, terms });
        Ok(())
    }

    pub fn write_mmap(&self, path: impl AsRef<Path>) -> Result<SparseBuildReport> {
        let path = path.as_ref();
        if self.docs.len() > u32::MAX as usize {
            return Err(HybridError::limit(format!(
                "sparse doc count {} exceeds u32::MAX",
                self.docs.len()
            )));
        }
        ensure_writable_regular_path(path)?;

        let built = BuiltSparse::from_docs(self.tokenizer, &self.docs)?;
        let bytes = built.to_bytes()?;
        atomic_write(path, &bytes)?;

        Ok(SparseBuildReport {
            row_count: self.docs.len(),
            term_count: built.terms.len(),
            postings_count: built.postings_count,
            tokenizer: self.tokenizer,
            tokenizer_version: TOKENIZER_VERSION,
            avg_doc_len: built.avg_doc_len,
            file_size_bytes: bytes.len() as u64,
        })
    }
}

#[derive(Clone, Debug)]
struct BuiltSparse {
    tokenizer: TokenizerKind,
    row_ids: Vec<u64>,
    doc_lens: Vec<u32>,
    terms: Vec<String>,
    postings: Vec<Vec<(u32, u16)>>,
    postings_count: usize,
    avg_doc_len: f32,
}

impl BuiltSparse {
    fn from_docs(tokenizer: TokenizerKind, docs: &[PendingDoc]) -> Result<Self> {
        let mut postings = BTreeMap::<String, Vec<(u32, u16)>>::new();
        let mut row_ids = Vec::new();
        row_ids
            .try_reserve_exact(docs.len())
            .map_err(|_| HybridError::limit("row-id table allocation too large"))?;
        let mut doc_lens = Vec::new();
        doc_lens
            .try_reserve_exact(docs.len())
            .map_err(|_| HybridError::limit("doc length table allocation too large"))?;
        let mut total_len = 0usize;

        for (doc_ordinal, doc) in docs.iter().enumerate() {
            let doc_ordinal = u32::try_from(doc_ordinal)
                .map_err(|_| HybridError::limit("doc ordinal exceeds u32::MAX"))?;
            row_ids.push(doc.row_id);
            let doc_len = u32::try_from(doc.terms.len()).map_err(|_| {
                HybridError::limit(format!(
                    "document {} term count {} exceeds u32::MAX",
                    doc.row_id,
                    doc.terms.len()
                ))
            })?;
            doc_lens.push(doc_len);
            total_len = total_len
                .checked_add(doc.terms.len())
                .ok_or_else(|| HybridError::limit("total sparse document length overflow"))?;

            let mut counts = BTreeMap::<String, u32>::new();
            for term in &doc.terms {
                if term.is_empty() {
                    return Err(HybridError::tokenizer("empty normalized term"));
                }
                *counts.entry(term.clone()).or_default() += 1;
            }
            for (term, count) in counts {
                let tf = u16::try_from(count).map_err(|_| {
                    HybridError::limit(format!(
                        "term frequency {count} for row_id {} exceeds u16::MAX",
                        doc.row_id
                    ))
                })?;
                postings.entry(term).or_default().push((doc_ordinal, tf));
            }
        }

        let mut terms = Vec::new();
        terms
            .try_reserve_exact(postings.len())
            .map_err(|_| HybridError::limit("term table allocation too large"))?;
        let mut posting_lists = Vec::new();
        posting_lists
            .try_reserve_exact(postings.len())
            .map_err(|_| HybridError::limit("posting list table allocation too large"))?;
        let mut postings_count = 0usize;
        for (term, list) in postings {
            postings_count = postings_count
                .checked_add(list.len())
                .ok_or_else(|| HybridError::limit("postings count overflow"))?;
            terms.push(term);
            posting_lists.push(list);
        }
        let avg_doc_len = if docs.is_empty() {
            0.0
        } else {
            total_len as f32 / docs.len() as f32
        };
        Ok(Self {
            tokenizer,
            row_ids,
            doc_lens,
            terms,
            postings: posting_lists,
            postings_count,
            avg_doc_len,
        })
    }

    fn to_bytes(&self) -> Result<Vec<u8>> {
        let term_bytes_len = self
            .terms
            .iter()
            .map(String::len)
            .try_fold(0usize, |acc, len| acc.checked_add(len))
            .ok_or_else(|| HybridError::limit("term bytes length overflow"))?;

        let row_ids_offset = HEADER_LEN;
        let doc_len_offset = checked_offset(row_ids_offset, self.row_ids.len(), ROW_ID_LEN)?;
        let term_table_offset = checked_offset(doc_len_offset, self.doc_lens.len(), DOC_LEN_LEN)?;
        let term_bytes_offset =
            checked_offset(term_table_offset, self.terms.len(), TERM_ENTRY_LEN)?;
        let postings_offset = term_bytes_offset
            .checked_add(term_bytes_len)
            .ok_or_else(|| HybridError::limit("term bytes section overflow"))?;
        let file_len = checked_offset(postings_offset, self.postings_count, POSTING_LEN)?;

        let mut term_entries = Vec::new();
        term_entries
            .try_reserve_exact(self.terms.len())
            .map_err(|_| HybridError::limit("term entry table allocation too large"))?;
        let mut term_cursor = term_bytes_offset;
        let mut posting_cursor = postings_offset;
        for (term, postings) in self.terms.iter().zip(&self.postings) {
            term_entries.push(TermEntry {
                term_offset: term_cursor,
                term_len: term.len(),
                postings_offset: posting_cursor,
                postings_len: postings.len(),
            });
            term_cursor = term_cursor
                .checked_add(term.len())
                .ok_or_else(|| HybridError::limit("term cursor overflow"))?;
            posting_cursor = checked_offset(posting_cursor, postings.len(), POSTING_LEN)?;
        }

        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(file_len)
            .map_err(|_| HybridError::limit("sparse mmap byte buffer allocation too large"))?;
        bytes.extend_from_slice(MAGIC);
        write_u32_vec(&mut bytes, VERSION);
        write_u32_vec(&mut bytes, HEADER_LEN as u32);
        write_u32_vec(&mut bytes, ENDIAN_MARKER);
        write_u32_vec(&mut bytes, 0);
        write_u16_vec(&mut bytes, self.tokenizer as u16);
        write_u16_vec(&mut bytes, TOKENIZER_VERSION);
        write_u32_vec(&mut bytes, 0);
        write_u64_vec(&mut bytes, self.row_ids.len() as u64);
        write_u64_vec(&mut bytes, self.terms.len() as u64);
        write_u64_vec(&mut bytes, self.postings_count as u64);
        write_f32_vec(&mut bytes, self.avg_doc_len);
        write_f32_vec(&mut bytes, DEFAULT_K1);
        write_f32_vec(&mut bytes, DEFAULT_B);
        write_u32_vec(&mut bytes, 0);
        write_u64_vec(&mut bytes, row_ids_offset as u64);
        write_u64_vec(&mut bytes, doc_len_offset as u64);
        write_u64_vec(&mut bytes, term_table_offset as u64);
        write_u64_vec(&mut bytes, term_bytes_offset as u64);
        write_u64_vec(&mut bytes, postings_offset as u64);
        write_u64_vec(&mut bytes, file_len as u64);
        write_u64_vec(&mut bytes, 0);
        debug_assert_eq!(bytes.len(), HEADER_LEN);

        for row_id in &self.row_ids {
            write_u64_vec(&mut bytes, *row_id);
        }
        for len in &self.doc_lens {
            write_u32_vec(&mut bytes, *len);
        }
        for entry in &term_entries {
            write_u64_vec(&mut bytes, entry.term_offset as u64);
            write_u32_vec(&mut bytes, entry.term_len as u32);
            write_u32_vec(&mut bytes, 0);
            write_u64_vec(&mut bytes, entry.postings_offset as u64);
            write_u32_vec(&mut bytes, entry.postings_len as u32);
            write_u32_vec(&mut bytes, 0);
        }
        for term in &self.terms {
            bytes.extend_from_slice(term.as_bytes());
        }
        for postings in &self.postings {
            for &(doc_ordinal, tf) in postings {
                write_u32_vec(&mut bytes, doc_ordinal);
                write_u16_vec(&mut bytes, tf);
                write_u16_vec(&mut bytes, 0);
            }
        }
        debug_assert_eq!(bytes.len(), file_len);
        Ok(bytes)
    }
}

/// Memory-mapped BM25 sparse sidecar.
pub struct Bm25MmapIndex {
    mmap: Mmap,
    header: Header,
    doc_to_row_id: Vec<u64>,
    row_id_to_doc: HashMap<u64, u32>,
}

/// Prevalidated row-id allowlist mapped to sparse document ordinals.
#[derive(Clone, Debug)]
pub struct PreparedAllowlist {
    ordinals: HashSet<u32>,
}

struct RowIdTables {
    doc_to_row_id: Vec<u64>,
    row_id_to_doc: HashMap<u64, u32>,
}

impl Bm25MmapIndex {
    /// Open and validate a sparse mmap file. This does not perform manifest
    /// verification.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = open_regular_file(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let header = Header::parse(&mmap)?;
        let mut index = Self {
            mmap,
            header,
            doc_to_row_id: Vec::new(),
            row_id_to_doc: HashMap::new(),
        };
        let row_ids = index.validate_layout()?;
        index.doc_to_row_id = row_ids.doc_to_row_id;
        index.row_id_to_doc = row_ids.row_id_to_doc;
        Ok(index)
    }

    /// Controlled-storage entrypoint: verify the manifest and auxiliary
    /// artifact, then immediately mmap and validate the sparse sidecar.
    ///
    /// This uses `ordvec-manifest` as the sidecar verification authority and
    /// then validates the sparse mmap layout and row count against the verified
    /// primary index. Like `VerifiedLoadPlan`, it does not pin bytes against a
    /// hostile filesystem that can mutate files between verification and mmap.
    pub fn open_verified_sidecar(
        manifest_path: impl AsRef<Path>,
        aux_name: &str,
        verify_options: VerifyOptions,
    ) -> Result<Self> {
        let plan = verify_for_load(manifest_path, verify_options)?;
        Self::open_from_verified_plan_unchecked_freshness(&plan, aux_name)
    }

    /// Convenience for callers that already verified a bundle.
    ///
    /// `VerifiedLoadPlan` is a snapshot, not a byte pin. It does not hold file
    /// descriptors, locks, or copied bytes. Load immediately from controlled
    /// storage, or call [`Self::open_verified_sidecar`] to re-verify before
    /// mapping files that another actor could mutate.
    pub fn open_from_verified_plan_unchecked_freshness(
        plan: &VerifiedLoadPlan,
        aux_name: &str,
    ) -> Result<Self> {
        let path = plan.require_auxiliary(aux_name)?;
        let index = Self::open(path)?;
        index.validate_against_manifest(plan)?;
        Ok(index)
    }

    pub fn inspect(&self) -> SparseInspectReport {
        SparseInspectReport {
            row_count: self.header.doc_count,
            term_count: self.header.term_count,
            postings_count: self.header.postings_count,
            tokenizer: self.header.tokenizer,
            tokenizer_version: self.header.tokenizer_version,
            avg_doc_len: self.header.avg_doc_len,
            k1: self.header.k1,
            b: self.header.b,
            file_size_bytes: self.mmap.len() as u64,
        }
    }

    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<ScoredRow>> {
        self.search_with_allowlist(query, top_k, None)
    }

    pub fn search_with_allowlist(
        &self,
        query: &str,
        top_k: usize,
        allowlist: Option<&[u64]>,
    ) -> Result<Vec<ScoredRow>> {
        let prepared = allowlist
            .map(|row_ids| self.prepare_allowlist(row_ids))
            .transpose()?;
        self.search_with_prepared_allowlist(query, top_k, prepared.as_ref())
    }

    pub fn prepare_allowlist(&self, row_ids: &[u64]) -> Result<PreparedAllowlist> {
        let mut ordinals = HashSet::new();
        ordinals
            .try_reserve(row_ids.len())
            .map_err(|_| HybridError::limit("allowlist ordinal set allocation too large"))?;
        for &id in row_ids {
            let Some(&doc_ordinal) = self.row_id_to_doc.get(&id) else {
                return Err(HybridError::row_id(format!(
                    "allowlist row_id {id} is not present in sparse mmap"
                )));
            };
            ordinals.insert(doc_ordinal);
        }
        Ok(PreparedAllowlist { ordinals })
    }

    pub fn search_with_prepared_allowlist(
        &self,
        query: &str,
        top_k: usize,
        allowlist: Option<&PreparedAllowlist>,
    ) -> Result<Vec<ScoredRow>> {
        if self.header.doc_count == 0 || top_k == 0 {
            return Ok(Vec::new());
        }
        let query_terms = dedupe_preserving_order(tokenize_text(self.header.tokenizer, query));
        if query_terms.is_empty() {
            return Ok(Vec::new());
        }
        let mut scores = HashMap::<u32, f32>::new();
        let doc_count = self.header.doc_count as f32;
        let avg_doc_len = self.header.avg_doc_len.max(1.0);

        for term in query_terms {
            let Some(entry) = self.find_term(&term)? else {
                continue;
            };
            let df = entry.postings_len as f32;
            let idf = ((doc_count - df + 0.5) / (df + 0.5) + 1.0).ln();
            for offset in self.posting_range(entry)?.step_by(POSTING_LEN) {
                let doc_ordinal = le_u32_at(&self.mmap, offset)?;
                if allowlist
                    .as_ref()
                    .is_some_and(|allowed| !allowed.ordinals.contains(&doc_ordinal))
                {
                    continue;
                }
                let tf = le_u16_at(&self.mmap, offset + 4)? as f32;
                let doc_len = self.doc_len(doc_ordinal as usize)? as f32;
                let denom = tf
                    + self.header.k1
                        * (1.0 - self.header.b + self.header.b * doc_len / avg_doc_len);
                let score = idf * (tf * (self.header.k1 + 1.0)) / denom.max(f32::EPSILON);
                *scores.entry(doc_ordinal).or_default() += score;
            }
        }

        let mut rows = scores
            .into_iter()
            .map(|(doc_ordinal, score)| {
                Ok(ScoredRow {
                    row_id: self.pinned_row_id(doc_ordinal)?,
                    score,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if rows.len() > top_k {
            rows.select_nth_unstable_by(top_k - 1, sparse_row_order);
            rows.truncate(top_k);
        }
        rows.sort_unstable_by(sparse_row_order);
        Ok(rows)
    }

    fn validate_against_manifest(&self, plan: &VerifiedLoadPlan) -> Result<()> {
        let vector_count = plan.metadata().vector_count;
        let row_identity = plan.row_identity();
        let row_count = row_identity.row_count();
        if self.header.doc_count != vector_count || self.header.doc_count != row_count {
            return Err(HybridError::row_id(format!(
                "sparse row count {} does not match verified manifest vector_count {} and row_identity row_count {}",
                self.header.doc_count, vector_count, row_count
            )));
        }
        match row_identity.kind() {
            "row_id_identity" => {
                if let Some(ids_auxiliary) = plan.auxiliary_by_name(ORDINALDB_IDS_AUX_NAME) {
                    let ids_path = ids_auxiliary.path().ok_or_else(|| {
                        HybridError::row_id(format!(
                            "verified {ORDINALDB_IDS_AUX_NAME:?} auxiliary has no loadable path"
                        ))
                    })?;
                    let ids = read_ordinaldb_ids(ids_path, self.header.doc_count)?;
                    if ids != self.doc_to_row_id {
                        return Err(HybridError::row_id(
                            "sparse row-id table does not match verified OrdinalDB ID sidecar",
                        ));
                    }
                } else {
                    for (doc_ordinal, &row_id) in self.doc_to_row_id.iter().enumerate() {
                        let expected = doc_ordinal as u64;
                        if row_id != expected {
                            return Err(HybridError::row_id(format!(
                                "sparse row_id {row_id} at doc ordinal {doc_ordinal} does not match row_id_identity expected {expected}"
                            )));
                        }
                    }
                }
            }
            "jsonl" => {
                let path = row_identity
                    .path()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "<missing path>".to_string());
                return Err(HybridError::row_id(format!(
                    "jsonl row identity sidecars are not supported for sparse row-id binding yet: {path}"
                )));
            }
            kind => {
                return Err(HybridError::row_id(format!(
                    "unsupported verified row identity kind {kind:?}"
                )));
            }
        }
        Ok(())
    }

    pub fn search_batch<Q>(&self, queries: &[Q], top_k: usize) -> Result<RankedBatch>
    where
        Q: AsRef<str>,
    {
        let allowlists = vec![None; queries.len()];
        self.search_batch_with_allowlists(queries, top_k, &allowlists)
    }

    pub fn search_batch_with_allowlists<Q>(
        &self,
        queries: &[Q],
        top_k: usize,
        allowlists: &[Option<&[u64]>],
    ) -> Result<RankedBatch>
    where
        Q: AsRef<str>,
    {
        if allowlists.len() != queries.len() {
            return Err(HybridError::batch(format!(
                "allowlist count {} does not match query count {}",
                allowlists.len(),
                queries.len()
            )));
        }
        let mut offsets = Vec::with_capacity(queries.len() + 1);
        let mut hits = Vec::new();
        offsets.push(0);
        for (query, allowlist) in queries.iter().zip(allowlists) {
            hits.extend(self.search_with_allowlist(query.as_ref(), top_k, *allowlist)?);
            offsets.push(hits.len());
        }
        RankedBatch::from_sorted_offsets_hits(offsets, hits)
    }

    pub fn search_batch_with_prepared_allowlists<Q>(
        &self,
        queries: &[Q],
        top_k: usize,
        allowlists: &[Option<&PreparedAllowlist>],
    ) -> Result<RankedBatch>
    where
        Q: AsRef<str>,
    {
        if allowlists.len() != queries.len() {
            return Err(HybridError::batch(format!(
                "allowlist count {} does not match query count {}",
                allowlists.len(),
                queries.len()
            )));
        }
        let mut offsets = Vec::with_capacity(queries.len() + 1);
        let mut hits = Vec::new();
        offsets.push(0);
        for (query, allowlist) in queries.iter().zip(allowlists) {
            hits.extend(self.search_with_prepared_allowlist(query.as_ref(), top_k, *allowlist)?);
            offsets.push(hits.len());
        }
        RankedBatch::from_sorted_offsets_hits(offsets, hits)
    }

    fn validate_layout(&self) -> Result<RowIdTables> {
        if self.header.file_len != self.mmap.len() {
            return Err(HybridError::malformed(
                "header file_len does not match mapped length",
            ));
        }
        if self.header.row_ids_offset != HEADER_LEN {
            return Err(HybridError::malformed(
                "row-id section has noncanonical offset",
            ));
        }
        if self.header.doc_count > u32::MAX as usize {
            return Err(HybridError::limit("doc count exceeds u32::MAX"));
        }
        let row_ids_end = checked_offset(
            self.header.row_ids_offset,
            self.header.doc_count,
            ROW_ID_LEN,
        )?;
        if row_ids_end != self.header.doc_len_offset {
            return Err(HybridError::malformed(
                "row-id section has noncanonical end offset",
            ));
        }
        let doc_len_end = checked_offset(
            self.header.doc_len_offset,
            self.header.doc_count,
            DOC_LEN_LEN,
        )?;
        if doc_len_end != self.header.term_table_offset {
            return Err(HybridError::malformed(
                "doc-length section has noncanonical end offset",
            ));
        }
        let term_table_end = checked_offset(
            self.header.term_table_offset,
            self.header.term_count,
            TERM_ENTRY_LEN,
        )?;
        if term_table_end != self.header.term_bytes_offset {
            return Err(HybridError::malformed(
                "term table has noncanonical end offset",
            ));
        }
        checked_section(
            self.mmap.len(),
            self.header.row_ids_offset,
            self.header.doc_count,
            ROW_ID_LEN,
            "row ids",
        )?;
        checked_section(
            self.mmap.len(),
            self.header.doc_len_offset,
            self.header.doc_count,
            DOC_LEN_LEN,
            "doc lengths",
        )?;
        checked_section(
            self.mmap.len(),
            self.header.term_table_offset,
            self.header.term_count,
            TERM_ENTRY_LEN,
            "term table",
        )?;
        checked_section(
            self.mmap.len(),
            self.header.postings_offset,
            self.header.postings_count,
            POSTING_LEN,
            "postings",
        )?;
        if self.header.term_bytes_offset > self.header.postings_offset
            || self.header.postings_offset > self.mmap.len()
        {
            return Err(HybridError::malformed(
                "invalid term/postings section offsets",
            ));
        }
        if !self.header.avg_doc_len.is_finite() || self.header.avg_doc_len < 0.0 {
            return Err(HybridError::malformed(
                "avg_doc_len must be finite and non-negative",
            ));
        }
        if self.header.term_count > 0 && self.header.avg_doc_len <= 0.0 {
            return Err(HybridError::malformed(
                "avg_doc_len must be positive when terms are present",
            ));
        }
        if !self.header.k1.is_finite() || self.header.k1 <= 0.0 {
            return Err(HybridError::malformed("k1 must be finite and positive"));
        }
        if !self.header.b.is_finite() || !(0.0..=1.0).contains(&self.header.b) {
            return Err(HybridError::malformed("b must be finite and in [0, 1]"));
        }
        let mut doc_to_row_id = Vec::new();
        doc_to_row_id
            .try_reserve_exact(self.header.doc_count)
            .map_err(|_| HybridError::limit("row-id table allocation too large"))?;
        let mut row_id_to_doc = HashMap::new();
        row_id_to_doc
            .try_reserve(self.header.doc_count)
            .map_err(|_| HybridError::limit("row-id map allocation too large"))?;
        for idx in 0..self.header.doc_count {
            let row_id = self.row_id(idx)?;
            let doc_ordinal = u32::try_from(idx)
                .map_err(|_| HybridError::limit("doc ordinal exceeds u32::MAX"))?;
            if row_id_to_doc.insert(row_id, doc_ordinal).is_some() {
                return Err(HybridError::row_id(format!(
                    "duplicate row_id {row_id} in sparse mmap"
                )));
            }
            doc_to_row_id.push(row_id);
        }

        let mut computed_doc_lens = Vec::new();
        computed_doc_lens
            .try_reserve_exact(self.header.doc_count)
            .map_err(|_| HybridError::limit("computed doc length table allocation too large"))?;
        computed_doc_lens.resize(self.header.doc_count, 0u64);
        let mut previous: Option<Vec<u8>> = None;
        let mut term_cursor = self.header.term_bytes_offset;
        let mut posting_cursor = self.header.postings_offset;
        for idx in 0..self.header.term_count {
            let entry = self.term_entry(idx)?;
            if entry.term_offset != term_cursor {
                return Err(HybridError::malformed(
                    "term bytes are not stored contiguously",
                ));
            }
            if entry.term_len == 0 {
                return Err(HybridError::malformed("term length must be positive"));
            }
            let term = self.term_bytes(entry)?;
            let term_str = std::str::from_utf8(term)
                .map_err(|_| HybridError::malformed("term bytes are not valid UTF-8"))?;
            if !is_normalized_term_v1(term_str) {
                // term_len is attacker-controlled (bounded only by the mmap
                // term section), so embed a bounded preview instead of the
                // full term: formatting the whole term would allocate a
                // message proportional to a corrupted/malicious input.
                const TERM_PREVIEW_MAX_CHARS: usize = 128;
                let preview: String = term_str.chars().take(TERM_PREVIEW_MAX_CHARS).collect();
                let ellipsis = if term_str.len() > preview.len() {
                    "…"
                } else {
                    ""
                };
                return Err(HybridError::malformed(format!(
                    "term bytes are not normalized for tokenizer v1 (len={}): {preview:?}{ellipsis}",
                    entry.term_len
                )));
            }
            if previous
                .as_ref()
                .is_some_and(|prev| prev.as_slice() >= term)
            {
                return Err(HybridError::malformed("term table is not strictly sorted"));
            }
            previous = Some(term.to_vec());
            term_cursor = entry
                .term_offset
                .checked_add(entry.term_len)
                .ok_or_else(|| HybridError::limit("term cursor overflow"))?;

            let range = self.posting_range(entry)?;
            if range.start != posting_cursor {
                return Err(HybridError::malformed(
                    "postings are not stored contiguously",
                ));
            }
            if entry.postings_len == 0 {
                return Err(HybridError::malformed("term entry has no postings"));
            }
            let mut previous_doc_ordinal = None;
            for offset in range.clone().step_by(POSTING_LEN) {
                let doc_ordinal = le_u32_at(&self.mmap, offset)? as usize;
                if doc_ordinal >= self.header.doc_count {
                    return Err(HybridError::malformed(format!(
                        "posting doc ordinal {doc_ordinal} exceeds doc count {}",
                        self.header.doc_count
                    )));
                }
                if previous_doc_ordinal.is_some_and(|previous| previous >= doc_ordinal) {
                    return Err(HybridError::malformed(
                        "postings for a term must have strictly increasing doc ordinals",
                    ));
                }
                previous_doc_ordinal = Some(doc_ordinal);
                let tf = le_u16_at(&self.mmap, offset + 4)?;
                if tf == 0 {
                    return Err(HybridError::malformed(
                        "posting term frequency must be positive",
                    ));
                }
                let reserved = le_u16_at(&self.mmap, offset + 6)?;
                if reserved != 0 {
                    return Err(HybridError::malformed(
                        "posting reserved bytes must be zero",
                    ));
                }
                computed_doc_lens[doc_ordinal] = computed_doc_lens[doc_ordinal]
                    .checked_add(u64::from(tf))
                    .ok_or_else(|| HybridError::limit("computed document length overflow"))?;
            }
            posting_cursor = range.end;
        }
        if term_cursor != self.header.postings_offset {
            return Err(HybridError::malformed(
                "term bytes section has noncanonical end offset",
            ));
        }
        if posting_cursor != self.mmap.len() {
            return Err(HybridError::malformed(
                "postings section has noncanonical end offset",
            ));
        }
        let mut total_doc_len = 0u64;
        for (idx, &computed) in computed_doc_lens.iter().enumerate() {
            let stored = u64::from(self.doc_len(idx)?);
            if stored != computed {
                return Err(HybridError::malformed(format!(
                    "doc_len for ordinal {idx} is {stored}, but postings imply {computed}"
                )));
            }
            total_doc_len = total_doc_len
                .checked_add(stored)
                .ok_or_else(|| HybridError::limit("total document length overflow"))?;
        }
        let expected_avg = if self.header.doc_count == 0 {
            0.0
        } else {
            total_doc_len as f32 / self.header.doc_count as f32
        };
        if (self.header.avg_doc_len - expected_avg).abs() > f32::EPSILON * expected_avg.max(1.0) {
            return Err(HybridError::malformed(
                "avg_doc_len does not match document lengths",
            ));
        }
        Ok(RowIdTables {
            doc_to_row_id,
            row_id_to_doc,
        })
    }

    fn pinned_row_id(&self, doc_ordinal: u32) -> Result<u64> {
        self.doc_to_row_id
            .get(doc_ordinal as usize)
            .copied()
            .ok_or_else(|| HybridError::malformed("doc ordinal out of range"))
    }

    fn row_id(&self, doc_ordinal: usize) -> Result<u64> {
        if doc_ordinal >= self.header.doc_count {
            return Err(HybridError::malformed("doc ordinal out of range"));
        }
        let offset = checked_offset(self.header.row_ids_offset, doc_ordinal, ROW_ID_LEN)?;
        le_u64_at(&self.mmap, offset)
    }

    fn doc_len(&self, doc_ordinal: usize) -> Result<u32> {
        if doc_ordinal >= self.header.doc_count {
            return Err(HybridError::malformed("doc ordinal out of range"));
        }
        let offset = checked_offset(self.header.doc_len_offset, doc_ordinal, DOC_LEN_LEN)?;
        le_u32_at(&self.mmap, offset)
    }

    fn term_entry(&self, idx: usize) -> Result<TermEntry> {
        if idx >= self.header.term_count {
            return Err(HybridError::malformed("term index out of range"));
        }
        let offset = checked_offset(self.header.term_table_offset, idx, TERM_ENTRY_LEN)?;
        let reserved0 = le_u32_at(&self.mmap, offset + 12)?;
        let reserved1 = le_u32_at(&self.mmap, offset + 28)?;
        if reserved0 != 0 || reserved1 != 0 {
            return Err(HybridError::malformed(
                "term entry reserved bytes must be zero",
            ));
        }
        Ok(TermEntry {
            term_offset: usize_from_u64(le_u64_at(&self.mmap, offset)?, "term offset")?,
            term_len: usize_from_u32(le_u32_at(&self.mmap, offset + 8)?, "term length")?,
            postings_offset: usize_from_u64(
                le_u64_at(&self.mmap, offset + 16)?,
                "postings offset",
            )?,
            postings_len: usize_from_u32(le_u32_at(&self.mmap, offset + 24)?, "postings length")?,
        })
    }

    fn term_bytes(&self, entry: TermEntry) -> Result<&[u8]> {
        let end = entry
            .term_offset
            .checked_add(entry.term_len)
            .ok_or_else(|| HybridError::limit("term byte range overflow"))?;
        if entry.term_offset < self.header.term_bytes_offset || end > self.header.postings_offset {
            return Err(HybridError::malformed(
                "term byte range is outside term section",
            ));
        }
        Ok(&self.mmap[entry.term_offset..end])
    }

    fn posting_range(&self, entry: TermEntry) -> Result<Range<usize>> {
        let end = checked_offset(entry.postings_offset, entry.postings_len, POSTING_LEN)?;
        if entry.postings_offset < self.header.postings_offset || end > self.mmap.len() {
            return Err(HybridError::malformed(
                "posting range is outside postings section",
            ));
        }
        Ok(entry.postings_offset..end)
    }

    fn find_term(&self, term: &str) -> Result<Option<TermEntry>> {
        let target = term.as_bytes();
        let mut lo = 0usize;
        let mut hi = self.header.term_count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let entry = self.term_entry(mid)?;
            let bytes = self.term_bytes(entry)?;
            match bytes.cmp(target) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Ok(Some(entry)),
            }
        }
        Ok(None)
    }
}

#[derive(Clone, Copy, Debug)]
struct Header {
    doc_count: usize,
    term_count: usize,
    postings_count: usize,
    avg_doc_len: f32,
    k1: f32,
    b: f32,
    tokenizer: TokenizerKind,
    tokenizer_version: u16,
    row_ids_offset: usize,
    doc_len_offset: usize,
    term_table_offset: usize,
    term_bytes_offset: usize,
    postings_offset: usize,
    file_len: usize,
}

impl Header {
    fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_LEN {
            return Err(HybridError::malformed(format!(
                "file length {} is smaller than header length {HEADER_LEN}",
                bytes.len()
            )));
        }
        if &bytes[..MAGIC.len()] != MAGIC {
            return Err(HybridError::malformed("invalid sparse mmap magic"));
        }
        if le_u32_at(bytes, VERSION_OFFSET)? != VERSION {
            return Err(HybridError::malformed("unsupported sparse mmap version"));
        }
        if le_u32_at(bytes, HEADER_LEN_OFFSET)? != HEADER_LEN as u32 {
            return Err(HybridError::malformed("invalid sparse mmap header length"));
        }
        if le_u32_at(bytes, ENDIAN_OFFSET)? != ENDIAN_MARKER {
            return Err(HybridError::malformed("invalid endian marker"));
        }
        if le_u32_at(bytes, FLAGS_OFFSET)? != 0
            || le_u32_at(bytes, RESERVED0_OFFSET)? != 0
            || le_u32_at(bytes, RESERVED1_OFFSET)? != 0
            || le_u64_at(bytes, RESERVED2_OFFSET)? != 0
        {
            return Err(HybridError::malformed("reserved header bytes must be zero"));
        }
        let tokenizer = TokenizerKind::from_u16(le_u16_at(bytes, TOKENIZER_KIND_OFFSET)?)?;
        let tokenizer_version = le_u16_at(bytes, TOKENIZER_VERSION_OFFSET)?;
        if tokenizer_version != TOKENIZER_VERSION {
            return Err(HybridError::tokenizer(format!(
                "unsupported tokenizer version {tokenizer_version}"
            )));
        }

        Ok(Self {
            doc_count: usize_from_u64(le_u64_at(bytes, DOC_COUNT_OFFSET)?, "doc count")?,
            term_count: usize_from_u64(le_u64_at(bytes, TERM_COUNT_OFFSET)?, "term count")?,
            postings_count: usize_from_u64(
                le_u64_at(bytes, POSTINGS_COUNT_OFFSET)?,
                "postings count",
            )?,
            avg_doc_len: le_f32_at(bytes, AVG_DOC_LEN_OFFSET)?,
            k1: le_f32_at(bytes, K1_OFFSET)?,
            b: le_f32_at(bytes, B_OFFSET)?,
            tokenizer,
            tokenizer_version,
            row_ids_offset: usize_from_u64(
                le_u64_at(bytes, ROW_IDS_OFFSET_OFFSET)?,
                "row-id offset",
            )?,
            doc_len_offset: usize_from_u64(
                le_u64_at(bytes, DOC_LEN_OFFSET_OFFSET)?,
                "doc-length offset",
            )?,
            term_table_offset: usize_from_u64(
                le_u64_at(bytes, TERM_TABLE_OFFSET_OFFSET)?,
                "term table offset",
            )?,
            term_bytes_offset: usize_from_u64(
                le_u64_at(bytes, TERM_BYTES_OFFSET_OFFSET)?,
                "term bytes offset",
            )?,
            postings_offset: usize_from_u64(
                le_u64_at(bytes, POSTINGS_OFFSET_OFFSET)?,
                "postings offset",
            )?,
            file_len: usize_from_u64(le_u64_at(bytes, FILE_LEN_OFFSET)?, "file length")?,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct TermEntry {
    term_offset: usize,
    term_len: usize,
    postings_offset: usize,
    postings_len: usize,
}

fn tokenize_text(tokenizer: TokenizerKind, text: &str) -> Vec<String> {
    let mut terms = Vec::new();
    for token in ascii_alnum_tokens(text) {
        if let Some(term) = normalize_term(token) {
            terms.push(term);
        }
        if tokenizer == TokenizerKind::IdentifierSubtokens {
            for subtoken in identifier_subtokens(token) {
                if let Some(term) = normalize_term(subtoken) {
                    terms.push(term);
                }
            }
        }
    }
    terms
}

fn ascii_alnum_tokens(text: &str) -> Vec<&str> {
    let mut tokens = Vec::new();
    let mut start = None;
    for (idx, ch) in text.char_indices() {
        if ch.is_ascii_alphanumeric() {
            start.get_or_insert(idx);
        } else if let Some(token_start) = start.take() {
            tokens.push(&text[token_start..idx]);
        }
    }
    if let Some(token_start) = start {
        tokens.push(&text[token_start..]);
    }
    tokens
}

fn normalize_term(term: &str) -> Option<String> {
    if !term.bytes().all(|byte| byte.is_ascii_alphanumeric()) {
        return None;
    }
    let mut term = term.to_ascii_lowercase();
    if term.len() < 3 {
        return None;
    }
    // Strip trailing `s` bytes until the term is a fixed point of the
    // tokenizer v1 read invariant (`is_normalized_term_v1`): a normalized
    // term never both exceeds four bytes and ends in `s`. A single strip is
    // not enough — "access" would become "acces", which write_mmap accepts
    // but Bm25MmapIndex::open rejects.
    while term.len() > 4 && term.ends_with('s') {
        term.pop();
    }
    if term.len() < 3 {
        None
    } else {
        Some(term)
    }
}

fn is_normalized_term_v1(term: &str) -> bool {
    term.len() >= 3
        && term
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
        && !(term.len() > 4 && term.ends_with('s'))
}

fn identifier_subtokens(token: &str) -> Vec<&str> {
    let bytes = token.as_bytes();
    if bytes.len() < 2 {
        return Vec::new();
    }
    let mut parts = Vec::new();
    let mut start = 0usize;
    for idx in 1..bytes.len() {
        let prev = bytes[idx - 1];
        let cur = bytes[idx];
        let next = bytes.get(idx + 1).copied();
        let split = cur.is_ascii_uppercase()
            && (prev.is_ascii_lowercase()
                || prev.is_ascii_digit()
                || (prev.is_ascii_uppercase() && next.is_some_and(|n| n.is_ascii_lowercase())));
        if split {
            if start < idx {
                parts.push(&token[start..idx]);
            }
            start = idx;
        }
    }
    if start < token.len() {
        parts.push(&token[start..]);
    }
    if parts.len() <= 1 {
        Vec::new()
    } else {
        parts
    }
}

fn dedupe_preserving_order(terms: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::with_capacity(terms.len());
    let mut out = Vec::new();
    for term in terms {
        if seen.insert(term.clone()) {
            out.push(term);
        }
    }
    out
}

fn open_regular_file(path: &Path) -> Result<File> {
    ensure_regular_file(path)?;
    let file = open_no_follow(path)?;
    let metadata = file.metadata()?;
    if !metadata.file_type().is_file() {
        return Err(HybridError::malformed(format!(
            "refusing to open non-regular file {}",
            path.display()
        )));
    }
    Ok(file)
}

#[cfg(unix)]
fn open_no_follow(path: &Path) -> Result<File> {
    Ok(OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)?)
}

#[cfg(not(unix))]
fn open_no_follow(path: &Path) -> Result<File> {
    Ok(File::open(path)?)
}

fn ensure_regular_file(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    let file_type = metadata.file_type();
    if file_type.is_symlink() {
        return Err(HybridError::malformed(format!(
            "refusing to open symlink {}",
            path.display()
        )));
    }
    if !file_type.is_file() {
        return Err(HybridError::malformed(format!(
            "refusing to open non-regular file {}",
            path.display()
        )));
    }
    Ok(())
}

fn ensure_writable_regular_path(path: &Path) -> Result<()> {
    if let Ok(metadata) = fs::symlink_metadata(path) {
        let file_type = metadata.file_type();
        if file_type.is_symlink() {
            return Err(HybridError::malformed(format!(
                "refusing to replace symlink {}",
                path.display()
            )));
        }
        if !file_type.is_file() {
            return Err(HybridError::malformed(format!(
                "refusing to replace non-regular file {}",
                path.display()
            )));
        }
    }
    Ok(())
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let (temp, mut file) = create_temp_file(path)?;
    let write_result = file.write_all(bytes).and_then(|()| file.sync_all());
    if let Err(error) = write_result {
        let _ = fs::remove_file(&temp);
        return Err(error.into());
    }
    drop(file);
    if let Err(error) = replace_temp_file(&temp, path) {
        let _ = fs::remove_file(&temp);
        return Err(error.into());
    }
    Ok(())
}

fn create_temp_file(path: &Path) -> Result<(PathBuf, File)> {
    for _ in 0..TEMP_CREATE_ATTEMPTS {
        let temp = temp_path(path)?;
        match secure_temp_open_options().open(&temp) {
            Ok(file) => return Ok((temp, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }

    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        format!(
            "could not create unique temporary file for {} after {TEMP_CREATE_ATTEMPTS} attempts",
            path.display()
        ),
    )
    .into())
}

fn secure_temp_open_options() -> OpenOptions {
    let mut options = OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    {
        options.mode(0o600);
    }
    options
}

#[cfg(not(windows))]
fn replace_temp_file(temp: &Path, path: &Path) -> io::Result<()> {
    fs::rename(temp, path)
}

#[cfg(windows)]
fn replace_temp_file(temp: &Path, path: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;

    const MOVEFILE_REPLACE_EXISTING: u32 = 0x0000_0001;
    const MOVEFILE_WRITE_THROUGH: u32 = 0x0000_0008;

    extern "system" {
        fn MoveFileExW(existing: *const u16, new: *const u16, flags: u32) -> i32;
    }

    fn wide_null(path: &Path) -> Vec<u16> {
        path.as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect()
    }

    let temp = wide_null(temp);
    let path = wide_null(path);
    let ok = unsafe {
        MoveFileExW(
            temp.as_ptr(),
            path.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if ok == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn temp_path(path: &Path) -> io::Result<PathBuf> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path has no file name"))?
        .to_string_lossy();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let nonce = TEMP_COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
    Ok(parent.join(format!(
        ".{name}.tmp-{}-{nanos}-{nonce}",
        std::process::id()
    )))
}

fn sparse_row_order(a: &ScoredRow, b: &ScoredRow) -> std::cmp::Ordering {
    b.score
        .total_cmp(&a.score)
        .then_with(|| a.row_id.cmp(&b.row_id))
}

fn checked_offset(base: usize, count: usize, elem_len: usize) -> Result<usize> {
    count
        .checked_mul(elem_len)
        .and_then(|len| base.checked_add(len))
        .ok_or_else(|| HybridError::limit("sparse mmap offset overflow"))
}

fn checked_section(
    file_len: usize,
    base: usize,
    count: usize,
    elem_len: usize,
    label: &str,
) -> Result<()> {
    let end = checked_offset(base, count, elem_len)?;
    if base > file_len || end > file_len {
        return Err(HybridError::malformed(format!(
            "{label} section exceeds file length"
        )));
    }
    Ok(())
}

fn usize_from_u64(value: u64, label: &str) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| HybridError::limit(format!("{label} {value} exceeds usize::MAX")))
}

fn usize_from_u32(value: u32, _label: &str) -> Result<usize> {
    Ok(value as usize)
}

fn read_ordinaldb_ids(path: &Path, expected_len: usize) -> Result<Vec<u64>> {
    let mut file = File::open(path)?;
    let expected_rows = u64::try_from(expected_len)
        .map_err(|_| HybridError::limit("OrdinalDB ID sidecar row count exceeds u64::MAX"))?;
    let expected_file_size = expected_rows
        .checked_mul(ROW_ID_LEN as u64)
        .and_then(|bytes| bytes.checked_add(16))
        .ok_or_else(|| HybridError::limit("OrdinalDB ID sidecar expected size overflow"))?;
    let observed_file_size = file.metadata()?.len();
    if observed_file_size < expected_file_size {
        return Err(HybridError::row_id(
            "verified OrdinalDB ID sidecar is truncated",
        ));
    }
    if observed_file_size > expected_file_size {
        return Err(HybridError::row_id(
            "verified OrdinalDB ID sidecar has trailing bytes",
        ));
    }

    let mut magic = [0u8; 8];
    read_exact_invalid(&mut file, &mut magic)?;
    if &magic != ORDINALDB_IDS_MAGIC {
        return Err(HybridError::row_id(
            "verified OrdinalDB ID sidecar has invalid magic",
        ));
    }

    let count = read_u64(&mut file)?;
    if count != expected_len as u64 {
        return Err(HybridError::row_id(format!(
            "verified OrdinalDB ID sidecar count {count} does not match sparse doc count {expected_len}"
        )));
    }

    let mut ids = Vec::new();
    ids.try_reserve_exact(expected_len)
        .map_err(|_| HybridError::limit("OrdinalDB ID sidecar allocation too large"))?;
    for _ in 0..expected_len {
        ids.push(read_u64(&mut file)?);
    }

    let mut trailing = [0u8; 1];
    if file.read(&mut trailing)? != 0 {
        return Err(HybridError::row_id(
            "verified OrdinalDB ID sidecar has trailing bytes",
        ));
    }
    Ok(ids)
}

fn read_u64(reader: &mut impl Read) -> Result<u64> {
    let mut bytes = [0u8; 8];
    read_exact_invalid(reader, &mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_exact_invalid(reader: &mut impl Read, buf: &mut [u8]) -> Result<()> {
    reader.read_exact(buf).map_err(|err| {
        if err.kind() == io::ErrorKind::UnexpectedEof {
            HybridError::row_id("truncated verified OrdinalDB ID sidecar")
        } else {
            HybridError::Io(err)
        }
    })
}

fn le_u16_at(bytes: &[u8], offset: usize) -> Result<u16> {
    let slice = bytes
        .get(offset..offset + 2)
        .ok_or_else(|| HybridError::malformed("unexpected EOF reading u16"))?;
    Ok(u16::from_le_bytes(slice.try_into().unwrap()))
}

fn le_u32_at(bytes: &[u8], offset: usize) -> Result<u32> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| HybridError::malformed("unexpected EOF reading u32"))?;
    Ok(u32::from_le_bytes(slice.try_into().unwrap()))
}

fn le_u64_at(bytes: &[u8], offset: usize) -> Result<u64> {
    let slice = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| HybridError::malformed("unexpected EOF reading u64"))?;
    Ok(u64::from_le_bytes(slice.try_into().unwrap()))
}

fn le_f32_at(bytes: &[u8], offset: usize) -> Result<f32> {
    Ok(f32::from_bits(le_u32_at(bytes, offset)?))
}

fn write_u16_vec(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn write_u32_vec(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn write_u64_vec(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn write_f32_vec(out: &mut Vec<u8>, value: f32) {
    write_u32_vec(out, value.to_bits());
}

#[cfg(test)]
mod tests {
    use super::*;
    use ordvec::RankQuant;
    use ordvec_manifest::{
        create_manifest_for_index_with_options, write_manifest_file, CreateAuxiliaryArtifact,
        CreateManifestOptions, CreateRowIdentity,
    };
    use tempfile::TempDir;

    fn write_sample(path: &Path, tokenizer: TokenizerKind) -> SparseBuildReport {
        let mut builder = SparseIndexBuilder::new(tokenizer);
        builder.add_text(10, "alpha alpha beta").unwrap();
        builder.add_text(20, "alpha beta").unwrap();
        builder.add_text(30, "gamma").unwrap();
        builder.write_mmap(path).unwrap()
    }

    fn write_identity_sample(path: &Path, tokenizer: TokenizerKind) -> SparseBuildReport {
        let mut builder = SparseIndexBuilder::new(tokenizer);
        builder.add_text(0, "alpha alpha beta").unwrap();
        builder.add_text(1, "alpha beta").unwrap();
        builder.add_text(2, "gamma").unwrap();
        builder.write_mmap(path).unwrap()
    }

    fn read_bytes(path: &Path) -> Vec<u8> {
        fs::read(path).unwrap()
    }

    fn write_bytes(path: &Path, bytes: &[u8]) {
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn ordinaldb_ids_reader_rejects_truncated_file_before_large_allocation() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("ids.bin");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(ORDINALDB_IDS_MAGIC);
        bytes.extend_from_slice(&1_000_000u64.to_le_bytes());
        write_bytes(&path, &bytes);

        let err = read_ordinaldb_ids(&path, 1_000_000).unwrap_err();
        assert!(err.to_string().contains("is truncated"), "{err}");
    }

    fn mutate(path: &Path, f: impl FnOnce(&mut Vec<u8>)) {
        let mut bytes = read_bytes(path);
        f(&mut bytes);
        write_bytes(path, &bytes);
    }

    fn write_test_manifest(root: &Path, sidecar: &Path, required: bool) -> PathBuf {
        let index_path = root.join("index.ovrq");
        let manifest_path = root.join("manifest.json");
        let mut rankquant = RankQuant::new(4, 2);
        rankquant.add(&[
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ]);
        rankquant.write(&index_path).unwrap();
        let mut options = CreateManifestOptions::default();
        options.auxiliary_artifacts.push(CreateAuxiliaryArtifact {
            name: DEFAULT_SPARSE_AUX_NAME.to_string(),
            path: sidecar.to_path_buf(),
            required,
        });
        let manifest = create_manifest_for_index_with_options(
            &index_path,
            CreateRowIdentity::RowIdIdentity,
            "test-model",
            &manifest_path,
            options,
        )
        .unwrap();
        write_manifest_file(&manifest, &manifest_path).unwrap();
        manifest_path
    }

    #[test]
    fn double_s_terms_round_trip_write_then_open() {
        // First-consumer blocker (cookbook/supportsearch): any 6+ letter word
        // ending in a double `s` ("access", "process", "address", ...) used to
        // normalize to a term still ending in `s`, which write_mmap accepted
        // but Bm25MmapIndex::open rejected as malformed.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("double-s.bm25");
        let mut builder = SparseIndexBuilder::new(TokenizerKind::Simple);
        builder
            .add_text(1, "users need access to the bucket")
            .unwrap();
        builder.write_mmap(&path).unwrap();
        let index = Bm25MmapIndex::open(&path).unwrap();
        // The same normalization runs on the query side, so the original
        // surface form must find its document again.
        assert_eq!(index.search("access", 10).unwrap()[0].row_id, 1);
        assert_eq!(index.search("users", 10).unwrap()[0].row_id, 1);
    }

    #[test]
    fn open_error_names_offending_unnormalized_term_bytes() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("sample.bm25");
        write_sample(&path, TokenizerKind::Simple);
        mutate(&path, |bytes| {
            let term_bytes_offset = le_u64_at(bytes, TERM_BYTES_OFFSET_OFFSET).unwrap() as usize;
            // "alpha" -> "Alpha": uppercase is not normalized for tokenizer v1.
            bytes[term_bytes_offset] = b'A';
        });
        let err = match Bm25MmapIndex::open(&path) {
            Ok(_) => panic!("unnormalized term bytes must be rejected"),
            Err(err) => err,
        };
        let message = err.to_string();
        assert!(
            message.contains("\"Alpha\""),
            "error must name the offending term bytes: {message}"
        );
        assert!(
            message.contains("len=5"),
            "error must report the term length: {message}"
        );
    }

    #[test]
    fn open_error_bounds_preview_for_pathological_long_term() {
        // term_len is attacker-controlled (bounded only by the mmap term
        // section), so the malformed-term error must never allocate a
        // message proportional to the term: a corrupted or malicious index
        // would otherwise OOM the reader merely by being rejected.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("long-term.bm25");
        let long_term = "a".repeat(1 << 20);
        let mut builder = SparseIndexBuilder::new(TokenizerKind::Simple);
        builder.add_text(1, &long_term).unwrap();
        builder.write_mmap(&path).unwrap();
        mutate(&path, |bytes| {
            let term_bytes_offset = le_u64_at(bytes, TERM_BYTES_OFFSET_OFFSET).unwrap() as usize;
            // "aaaa..." -> "Aaaa...": uppercase is not normalized for
            // tokenizer v1, so open must reject the term.
            bytes[term_bytes_offset] = b'A';
        });
        let err = match Bm25MmapIndex::open(&path) {
            Ok(_) => panic!("unnormalized term bytes must be rejected"),
            Err(err) => err,
        };
        let message = err.to_string();
        assert!(
            message.len() < 1024,
            "error message must stay bounded for huge terms, got {} bytes",
            message.len()
        );
        assert!(
            message.contains(&format!("len={}", 1usize << 20)),
            "error must report the full term length: {message}"
        );
        assert!(
            message.contains('…'),
            "error must mark the preview as truncated: {message}"
        );
    }

    /// Deterministic corpus for the write→open round-trip property test:
    /// English-like words (including the trailing-`s` family), identifier
    /// tokens, digit mixes, all-`s` runs, Unicode, and LCG-generated
    /// alphanumeric noise with appended `s` runs.
    fn round_trip_property_corpus() -> Vec<String> {
        let bases: &[&str] = &[
            "access",
            "process",
            "address",
            "success",
            "business",
            "progress",
            "express",
            "compress",
            "witness",
            "harness",
            "possess",
            "actress",
            "mattress",
            "congress",
            "darkness",
            "kindness",
            "illness",
            "wellness",
            "fitness",
            "wilderness",
            "class",
            "glass",
            "grass",
            "cross",
            "press",
            "dress",
            "stress",
            "chess",
            "bless",
            "guess",
            "boss",
            "mess",
            "loss",
            "kiss",
            "pass",
            "miss",
            "toss",
            "moss",
            "user",
            "vector",
            "index",
            "search",
            "query",
            "token",
            "bucket",
            "manifest",
            "sidecar",
            "cache",
            "batch",
            "score",
            "fusion",
            "sparse",
            "dense",
            "rank",
            "quant",
            "row",
            "term",
            "doc",
            "posting",
            "hybrid",
        ];
        let suffixes: &[&str] = &["", "s", "es", "ss", "sss", "ing", "ed", "0", "123", "s3"];
        let identifiers: &[&str] = &[
            "RankQuantFastScan",
            "HTTPServer",
            "parseJSONResponse",
            "userAccessTokens",
            "ProcessClassBoss",
            "getMessagesss",
            "ERR_POOL_EXHAUSTED_5432",
            "snake_case_access",
            "APIIds",
            "TLSAccess",
            "v0_2_0_rc1",
            "sha256sums",
        ];
        let unicode: &[&str] = &[
            "naïve",
            "café",
            "Übermaß",
            "мировые",
            "データベース",
            "🎉🎉🎉",
            "señors",
            "straße",
            "masses\u{0301}",
        ];

        let mut corpus = Vec::new();
        for base in bases {
            for suffix in suffixes {
                corpus.push(format!("{base}{suffix}"));
            }
        }
        corpus.extend(identifiers.iter().map(|token| token.to_string()));
        corpus.extend(unicode.iter().map(|token| token.to_string()));
        for len in 1..=12 {
            corpus.push("s".repeat(len));
            corpus.push(format!("a{}", "s".repeat(len)));
        }

        // Deterministic LCG noise over [a-z0-9], with trailing-`s` variants.
        const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state >> 33
        };
        for _ in 0..1000 {
            let len = 1 + (next() as usize % 16);
            let token: String = (0..len)
                .map(|_| ALPHABET[next() as usize % ALPHABET.len()] as char)
                .collect();
            let s_run = "s".repeat(next() as usize % 4);
            corpus.push(format!("{token}{s_run}"));
            corpus.push(token);
        }
        corpus
    }

    #[test]
    fn round_trip_property_write_then_open_for_both_tokenizers() {
        let corpus = round_trip_property_corpus();
        assert!(corpus.len() > 2500, "corpus size {}", corpus.len());

        // Invariant: whatever normalize_term emits must satisfy the read-side
        // validator, for every token in the corpus.
        for token in &corpus {
            for candidate in ascii_alnum_tokens(token) {
                if let Some(term) = normalize_term(candidate) {
                    assert!(
                        is_normalized_term_v1(&term),
                        "normalize_term({candidate:?}) produced read-invalid term {term:?}"
                    );
                }
            }
        }

        let dir = TempDir::new().unwrap();
        for tokenizer in [TokenizerKind::Simple, TokenizerKind::IdentifierSubtokens] {
            let path = dir.path().join(format!("property-{:?}.bm25", tokenizer));
            let mut builder = SparseIndexBuilder::new(tokenizer);
            for (row_id, chunk) in corpus.chunks(16).enumerate() {
                builder.add_text(row_id as u64, &chunk.join(" ")).unwrap();
            }
            builder.write_mmap(&path).unwrap();
            Bm25MmapIndex::open(&path).unwrap_or_else(|err| {
                panic!("write→open round trip failed for {tokenizer:?}: {err}")
            });
        }
    }

    #[test]
    fn mmap_roundtrip_and_bm25_known_answer() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("sparse.bm25");
        let report = write_sample(&path, TokenizerKind::Simple);
        assert_eq!(report.row_count, 3);
        let index = Bm25MmapIndex::open(&path).unwrap();
        let inspect = index.inspect();
        assert_eq!(inspect.tokenizer, TokenizerKind::Simple);
        assert_eq!(inspect.tokenizer_version, TOKENIZER_VERSION);
        assert_eq!(inspect.row_count, 3);
        let hits = index.search("alpha", 10).unwrap();
        assert_eq!(hits[0].row_id, 10);
        assert_eq!(hits[1].row_id, 20);
        assert!(hits[0].score > hits[1].score);
        assert!(hits[0].score > 0.0);
    }

    #[test]
    fn write_mmap_replaces_existing_sidecar() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("sparse.bm25");

        let mut first = SparseIndexBuilder::new(TokenizerKind::Simple);
        first.add_text(1, "alpha").unwrap();
        first.write_mmap(&path).unwrap();

        let mut second = SparseIndexBuilder::new(TokenizerKind::Simple);
        second.add_text(2, "beta beta").unwrap();
        second.write_mmap(&path).unwrap();

        let index = Bm25MmapIndex::open(&path).unwrap();
        assert!(index.search("alpha", 10).unwrap().is_empty());
        assert_eq!(index.search("beta", 10).unwrap()[0].row_id, 2);
    }

    #[test]
    fn temp_paths_are_unique_for_same_destination() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("sparse.bm25");
        assert_ne!(temp_path(&path).unwrap(), temp_path(&path).unwrap());
    }

    #[cfg(unix)]
    #[test]
    fn temp_files_are_owner_only_on_unix() {
        use std::os::unix::fs::PermissionsExt;

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("sparse.bm25");
        let (temp, file) = create_temp_file(&path).unwrap();
        drop(file);

        let mode = fs::metadata(&temp).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
        fs::remove_file(temp).unwrap();
    }

    #[cfg(windows)]
    #[test]
    fn windows_replace_preserves_destination_on_failed_move() {
        let dir = TempDir::new().unwrap();
        let target = dir.path().join("target.bm25");
        let missing_temp = dir.path().join("missing.tmp");
        fs::write(&target, b"old").unwrap();

        assert!(replace_temp_file(&missing_temp, &target).is_err());
        assert_eq!(fs::read(&target).unwrap(), b"old");
    }

    #[test]
    fn empty_corpus_and_query_are_safe() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("empty.bm25");
        SparseIndexBuilder::new(TokenizerKind::Simple)
            .write_mmap(&path)
            .unwrap();
        let index = Bm25MmapIndex::open(&path).unwrap();
        assert!(index.search("alpha", 10).unwrap().is_empty());

        let path = dir.path().join("sample.bm25");
        write_sample(&path, TokenizerKind::Simple);
        let index = Bm25MmapIndex::open(&path).unwrap();
        assert!(index.search("++", 10).unwrap().is_empty());
    }

    #[test]
    fn tokenizer_persistence_controls_query_behavior() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("ident.bm25");
        let mut builder = SparseIndexBuilder::new(TokenizerKind::IdentifierSubtokens);
        builder.add_text(7, "RankQuantFastScan").unwrap();
        builder.write_mmap(&path).unwrap();
        let index = Bm25MmapIndex::open(&path).unwrap();
        assert_eq!(
            index.inspect().tokenizer,
            TokenizerKind::IdentifierSubtokens
        );
        assert_eq!(index.search("fast scan", 10).unwrap()[0].row_id, 7);
    }

    #[test]
    fn row_ids_are_unique_and_public_results_are_row_ids() {
        let mut builder = SparseIndexBuilder::new(TokenizerKind::Simple);
        builder.add_text(42, "alpha").unwrap();
        assert!(builder.add_text(42, "beta").is_err());

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("sample.bm25");
        write_sample(&path, TokenizerKind::Simple);
        mutate(&path, |bytes| {
            let first = bytes[HEADER_LEN..HEADER_LEN + 8].to_vec();
            bytes[HEADER_LEN + 8..HEADER_LEN + 16].copy_from_slice(&first);
        });
        assert!(Bm25MmapIndex::open(&path).is_err());
    }

    #[test]
    fn allowlist_filters_before_sparse_ranking_and_batch_matches_single_query() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("sample.bm25");
        write_sample(&path, TokenizerKind::Simple);
        let index = Bm25MmapIndex::open(&path).unwrap();
        let hits = index
            .search_with_allowlist("alpha", 10, Some(&[20]))
            .unwrap();
        assert_eq!(
            hits,
            vec![ScoredRow {
                row_id: 20,
                score: hits[0].score
            }]
        );

        let single_alpha = index.search("alpha", 10).unwrap();
        let single_gamma = index.search("gamma", 10).unwrap();
        let batch = index.search_batch(&["alpha", "gamma"], 10).unwrap();
        assert_eq!(batch.hits_for_query(0).unwrap(), single_alpha.as_slice());
        assert_eq!(batch.hits_for_query(1).unwrap(), single_gamma.as_slice());

        let allowlists = [Some(&[20_u64][..]), Some(&[30_u64][..])];
        let filtered = index
            .search_batch_with_allowlists(&["alpha", "gamma"], 10, &allowlists)
            .unwrap();
        assert_eq!(filtered.hits_for_query(0).unwrap()[0].row_id, 20);
        assert_eq!(filtered.hits_for_query(1).unwrap()[0].row_id, 30);
    }

    #[test]
    fn prepared_allowlist_matches_raw_allowlist() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("sample.bm25");
        write_sample(&path, TokenizerKind::Simple);
        let index = Bm25MmapIndex::open(&path).unwrap();

        let raw = index
            .search_with_allowlist("alpha", 10, Some(&[20]))
            .unwrap();
        let prepared = index.prepare_allowlist(&[20]).unwrap();
        let prepared_hits = index
            .search_with_prepared_allowlist("alpha", 10, Some(&prepared))
            .unwrap();
        assert_eq!(prepared_hits, raw);

        let allowlists = [Some(&prepared), Some(&prepared)];
        let batch = index
            .search_batch_with_prepared_allowlists(&["alpha", "beta"], 10, &allowlists)
            .unwrap();
        assert_eq!(batch.hits_for_query(0).unwrap(), raw.as_slice());
        assert!(batch
            .hits_for_query(1)
            .unwrap()
            .iter()
            .all(|hit| hit.row_id == 20));
    }

    #[test]
    fn search_uses_partial_topk_order() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("many.bm25");
        let mut builder = SparseIndexBuilder::new(TokenizerKind::Simple);
        for row_id in 1..=32 {
            let mut text = String::new();
            for _ in 0..row_id {
                text.push_str("alpha ");
            }
            builder.add_text(row_id, &text).unwrap();
        }
        builder.write_mmap(&path).unwrap();
        let index = Bm25MmapIndex::open(&path).unwrap();

        let full = index.search("alpha", 64).unwrap();
        let top3 = index.search("alpha", 3).unwrap();
        assert_eq!(top3, full[..3]);
    }

    #[test]
    fn sparse_allowlist_rejects_unknown_row_ids() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("sample.bm25");
        write_sample(&path, TokenizerKind::Simple);
        let index = Bm25MmapIndex::open(&path).unwrap();
        let err = index
            .search_with_allowlist("alpha", 10, Some(&[999]))
            .unwrap_err();
        assert!(err.to_string().contains("allowlist row_id 999"));

        assert!(index
            .search_with_allowlist("++", 10, Some(&[999]))
            .unwrap_err()
            .to_string()
            .contains("allowlist row_id 999"));
        assert!(index
            .search_with_allowlist("alpha", 0, Some(&[999]))
            .unwrap_err()
            .to_string()
            .contains("allowlist row_id 999"));

        let empty_path = dir.path().join("empty.bm25");
        SparseIndexBuilder::new(TokenizerKind::Simple)
            .write_mmap(&empty_path)
            .unwrap();
        let empty = Bm25MmapIndex::open(&empty_path).unwrap();
        assert!(empty
            .search_with_allowlist("alpha", 10, Some(&[999]))
            .unwrap_err()
            .to_string()
            .contains("allowlist row_id 999"));
    }

    #[test]
    fn large_declared_doc_count_returns_error_not_abort() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("sample.bm25");
        write_sample(&path, TokenizerKind::Simple);
        mutate(&path, |bytes| {
            bytes[DOC_COUNT_OFFSET..DOC_COUNT_OFFSET + 8]
                .copy_from_slice(&u64::from(u32::MAX).saturating_add(1).to_le_bytes());
        });
        let err = match Bm25MmapIndex::open(&path) {
            Ok(_) => panic!("large declared doc count should fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("doc count exceeds u32::MAX"));
    }

    #[test]
    fn manifest_verified_sidecar_paths_are_used_without_custom_hashing() {
        let dir = TempDir::new().unwrap();
        let sidecar = dir.path().join("sparse.bm25");
        write_identity_sample(&sidecar, TokenizerKind::Simple);
        let manifest = write_test_manifest(dir.path(), &sidecar, true);

        let index = Bm25MmapIndex::open_verified_sidecar(
            &manifest,
            DEFAULT_SPARSE_AUX_NAME,
            VerifyOptions::default(),
        )
        .unwrap();
        assert_eq!(index.inspect().row_count, 3);

        let plan = verify_for_load(&manifest, VerifyOptions::default()).unwrap();
        let index = Bm25MmapIndex::open_from_verified_plan_unchecked_freshness(
            &plan,
            DEFAULT_SPARSE_AUX_NAME,
        )
        .unwrap();
        assert_eq!(index.inspect().term_count, 3);
    }

    #[test]
    fn verified_sidecar_rejects_row_id_identity_mismatch() {
        let dir = TempDir::new().unwrap();
        let sidecar = dir.path().join("sparse.bm25");
        write_sample(&sidecar, TokenizerKind::Simple);
        let manifest = write_test_manifest(dir.path(), &sidecar, true);

        let err = match Bm25MmapIndex::open_verified_sidecar(
            &manifest,
            DEFAULT_SPARSE_AUX_NAME,
            VerifyOptions::default(),
        ) {
            Ok(_) => panic!("row_id_identity must reject sparse row IDs that are not doc ordinals"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("row_id_identity expected 0"));
    }

    #[test]
    fn verified_sidecar_rejects_jsonl_row_identity_until_u64_binding_exists() {
        let dir = TempDir::new().unwrap();
        let sidecar = dir.path().join("sparse.bm25");
        write_identity_sample(&sidecar, TokenizerKind::Simple);

        let index_path = dir.path().join("index.ovrq");
        let manifest_path = dir.path().join("manifest.json");
        let row_map = dir.path().join("rows.jsonl");
        fs::write(
            &row_map,
            concat!(
                "{\"row_id\":0,\"db_id\":\"00000000-0000-4000-8000-000000000000\"}\n",
                "{\"row_id\":1,\"db_id\":\"00000000-0000-4000-8000-000000000001\"}\n",
                "{\"row_id\":2,\"db_id\":\"00000000-0000-4000-8000-000000000002\"}\n",
            ),
        )
        .unwrap();

        let mut rankquant = RankQuant::new(4, 2);
        rankquant.add(&[
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ]);
        rankquant.write(&index_path).unwrap();
        let mut options = CreateManifestOptions::default();
        options.auxiliary_artifacts.push(CreateAuxiliaryArtifact {
            name: DEFAULT_SPARSE_AUX_NAME.to_string(),
            path: sidecar,
            required: true,
        });
        let manifest = create_manifest_for_index_with_options(
            &index_path,
            CreateRowIdentity::Jsonl(row_map),
            "test-model",
            &manifest_path,
            options,
        )
        .unwrap();
        write_manifest_file(&manifest, &manifest_path).unwrap();

        let err = match Bm25MmapIndex::open_verified_sidecar(
            &manifest_path,
            DEFAULT_SPARSE_AUX_NAME,
            VerifyOptions::default(),
        ) {
            Ok(_) => panic!("jsonl row identity must be rejected until sparse u64 binding exists"),
            Err(err) => err,
        };
        assert!(err
            .to_string()
            .contains("jsonl row identity sidecars are not supported"));
    }

    #[test]
    fn verified_sidecar_rejects_manifest_row_count_mismatch() {
        let dir = TempDir::new().unwrap();
        let sidecar = dir.path().join("sparse.bm25");
        write_sample(&sidecar, TokenizerKind::Simple);

        let index_path = dir.path().join("index.ovrq");
        let manifest_path = dir.path().join("manifest.json");
        let mut rankquant = RankQuant::new(4, 2);
        rankquant.add(&[
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0,
        ]);
        rankquant.write(&index_path).unwrap();
        let mut options = CreateManifestOptions::default();
        options.auxiliary_artifacts.push(CreateAuxiliaryArtifact {
            name: DEFAULT_SPARSE_AUX_NAME.to_string(),
            path: sidecar,
            required: true,
        });
        let manifest = create_manifest_for_index_with_options(
            &index_path,
            CreateRowIdentity::RowIdIdentity,
            "test-model",
            &manifest_path,
            options,
        )
        .unwrap();
        write_manifest_file(&manifest, &manifest_path).unwrap();

        let err = match Bm25MmapIndex::open_verified_sidecar(
            &manifest_path,
            DEFAULT_SPARSE_AUX_NAME,
            VerifyOptions::default(),
        ) {
            Ok(_) => panic!("verified sidecar with mismatched row count should fail"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("sparse row count 3"));
    }

    #[test]
    fn missing_and_tampered_required_sidecars_are_rejected_by_manifest_layer() {
        let dir = TempDir::new().unwrap();
        let sidecar = dir.path().join("sparse.bm25");
        write_sample(&sidecar, TokenizerKind::Simple);
        let manifest = write_test_manifest(dir.path(), &sidecar, true);

        fs::remove_file(&sidecar).unwrap();
        assert!(Bm25MmapIndex::open_verified_sidecar(
            &manifest,
            DEFAULT_SPARSE_AUX_NAME,
            VerifyOptions::default(),
        )
        .is_err());

        write_sample(&sidecar, TokenizerKind::Simple);
        let manifest = write_test_manifest(dir.path(), &sidecar, true);
        let mut file = OpenOptions::new().append(true).open(&sidecar).unwrap();
        file.write_all(&[0]).unwrap();
        assert!(Bm25MmapIndex::open_verified_sidecar(
            &manifest,
            DEFAULT_SPARSE_AUX_NAME,
            VerifyOptions::default(),
        )
        .is_err());
    }

    #[test]
    fn optional_absent_sidecar_is_manifest_state_not_layout_validation() {
        let dir = TempDir::new().unwrap();
        let sidecar = dir.path().join("sparse.bm25");
        write_sample(&sidecar, TokenizerKind::Simple);
        let manifest = write_test_manifest(dir.path(), &sidecar, false);
        fs::remove_file(&sidecar).unwrap();
        let plan = verify_for_load(&manifest, VerifyOptions::default()).unwrap();
        assert!(plan.auxiliary_by_name(DEFAULT_SPARSE_AUX_NAME).is_some());
        assert!(Bm25MmapIndex::open_from_verified_plan_unchecked_freshness(
            &plan,
            DEFAULT_SPARSE_AUX_NAME,
        )
        .is_err());
    }

    #[test]
    fn auxiliary_size_limit_can_be_raised_for_large_sidecars() {
        let dir = TempDir::new().unwrap();
        let sidecar = dir.path().join("large-sidecar.bin");
        File::create(&sidecar)
            .unwrap()
            .set_len(64 * 1024 * 1024 + 1)
            .unwrap();
        let index_path = dir.path().join("index.ovrq");
        let manifest_path = dir.path().join("manifest.json");
        let mut rankquant = RankQuant::new(4, 2);
        rankquant.add(&[
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0,
        ]);
        rankquant.write(&index_path).unwrap();
        let mut create_options = CreateManifestOptions::default();
        create_options.limits.max_auxiliary_artifact_bytes = 2 * 1024 * 1024 * 1024;
        create_options
            .auxiliary_artifacts
            .push(CreateAuxiliaryArtifact {
                name: DEFAULT_SPARSE_AUX_NAME.to_string(),
                path: sidecar.clone(),
                required: true,
            });
        let manifest = create_manifest_for_index_with_options(
            &index_path,
            CreateRowIdentity::RowIdIdentity,
            "test-model",
            &manifest_path,
            create_options,
        )
        .unwrap();
        write_manifest_file(&manifest, &manifest_path).unwrap();
        // Derived limits (ordvec-manifest >= 0.6): default options bound
        // reads by the manifest-declared size, so a large declared+pinned
        // sidecar verifies without raising any knob...
        assert!(verify_for_load(&manifest_path, VerifyOptions::default()).is_ok());
        // ...while an explicitly configured tight cap remains an
        // enforceable ceiling.
        let mut verify_options = VerifyOptions::default();
        verify_options.limits.max_auxiliary_artifact_bytes = 1024;
        assert!(verify_for_load(&manifest_path, verify_options).is_err());
    }

    #[test]
    fn malformed_header_and_layout_cases_are_rejected() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("sample.bm25");

        for (offset, bytes) in [
            (0, b"BADMAGIC".as_slice()),
            (VERSION_OFFSET, &2_u32.to_le_bytes()[..]),
            (HEADER_LEN_OFFSET, &64_u32.to_le_bytes()[..]),
            (ENDIAN_OFFSET, &0_u32.to_le_bytes()[..]),
            (FLAGS_OFFSET, &1_u32.to_le_bytes()[..]),
            (ROW_IDS_OFFSET_OFFSET, &999_u64.to_le_bytes()[..]),
        ] {
            write_sample(&path, TokenizerKind::Simple);
            mutate(&path, |data| {
                data[offset..offset + bytes.len()].copy_from_slice(bytes)
            });
            assert!(
                Bm25MmapIndex::open(&path).is_err(),
                "offset {offset} should fail"
            );
        }

        write_sample(&path, TokenizerKind::Simple);
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        file.write_all(&[0]).unwrap();
        assert!(Bm25MmapIndex::open(&path).is_err());
    }

    #[test]
    fn malformed_terms_and_postings_are_rejected() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("sample.bm25");

        write_sample(&path, TokenizerKind::Simple);
        mutate(&path, |bytes| {
            let term_bytes_offset = le_u64_at(bytes, TERM_BYTES_OFFSET_OFFSET).unwrap() as usize;
            bytes[term_bytes_offset] = 0xff;
        });
        assert!(Bm25MmapIndex::open(&path).is_err());

        write_sample(&path, TokenizerKind::Simple);
        mutate(&path, |bytes| {
            let term_bytes_offset = le_u64_at(bytes, TERM_BYTES_OFFSET_OFFSET).unwrap() as usize;
            // "beta" follows "alpha"; making it "aeta" violates strict sorting.
            bytes[term_bytes_offset + "alpha".len()] = b'a';
        });
        assert!(Bm25MmapIndex::open(&path).is_err());

        write_sample(&path, TokenizerKind::Simple);
        mutate(&path, |bytes| {
            let postings_offset = le_u64_at(bytes, POSTINGS_OFFSET_OFFSET).unwrap() as usize;
            bytes[postings_offset + POSTING_LEN..postings_offset + POSTING_LEN + 4]
                .copy_from_slice(&0_u32.to_le_bytes());
        });
        assert!(Bm25MmapIndex::open(&path).is_err());

        write_sample(&path, TokenizerKind::Simple);
        mutate(&path, |bytes| {
            let postings_offset = le_u64_at(bytes, POSTINGS_OFFSET_OFFSET).unwrap() as usize;
            bytes[postings_offset + 4..postings_offset + 6].copy_from_slice(&0_u16.to_le_bytes());
        });
        assert!(Bm25MmapIndex::open(&path).is_err());

        write_sample(&path, TokenizerKind::Simple);
        mutate(&path, |bytes| {
            let postings_offset = le_u64_at(bytes, POSTINGS_OFFSET_OFFSET).unwrap() as usize;
            bytes[postings_offset + 6..postings_offset + 8].copy_from_slice(&1_u16.to_le_bytes());
        });
        assert!(Bm25MmapIndex::open(&path).is_err());
    }

    #[test]
    fn bm25_search_returns_ranked_hits_excluding_fusion() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("sample.bm25");
        write_sample(&path, TokenizerKind::Simple);
        let index = Bm25MmapIndex::open(&path).unwrap();
        let hits = index.search("alpha", 2).unwrap();
        assert_eq!(
            hits.iter().map(|hit| hit.row_id).collect::<Vec<_>>(),
            vec![10, 20]
        );
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn tokenizer_covers_identifier_and_subtoken_edges() {
        assert_eq!(
            tokenize_text(TokenizerKind::Simple, "foo_bar vectors x yz cars"),
            vec!["foo", "bar", "vector", "cars"]
        );
        assert_eq!(
            tokenize_text(
                TokenizerKind::IdentifierSubtokens,
                "RankQuantFastScan HTTPServer"
            ),
            vec![
                "rankquantfastscan",
                "rank",
                "quant",
                "fast",
                "scan",
                "httpserver",
                "http",
                "server"
            ]
        );
        assert_eq!(
            tokenize_text(TokenizerKind::IdentifierSubtokens, "IO DB ab APIIds"),
            vec!["apiid", "api", "ids"]
        );
    }
}
