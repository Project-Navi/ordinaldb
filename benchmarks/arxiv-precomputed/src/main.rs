// Battle-test harness for OrdinalDB core (OrdinalIndex, bits=2, sign sidecar
// default-on) against a precomputed-embedding corpus: 2-D little-endian fp32
// `.npy` embeddings plus one or more self-retrieval query sets with ground
// truth. Originally built for the arXiv-1M corpus (1.26M rows x 1024 dims,
// four 2,048-query sets); all paths are CLI arguments and no corpus data
// ships in this repository (see README.md).
//
// Measures: ingest wall time + RSS, verified-bundle write/size, cold
// open_verified, per-query sequential latency (Auto -> SignTwoStage),
// batched throughput, ExactRankQuant subset latency, and self-retrieval
// recall@1/@10 per query set. Dumps top-10 row ids per query for the
// external cosine-baseline comparison.

use clap::Parser;
use ordinaldb::manifest::{CreateManifestOptions, VerifyOptions};
use ordinaldb::{DenseLoadOptions, DenseSearchOptions, OrdinalIndex};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const BITS: u8 = 2;
const K: usize = 10;
const EXACT_SUBSET: usize = 256;
const INGEST_CHUNK_ROWS: usize = 16384;

/// Battle-test OrdinalDB core against a precomputed-embedding corpus.
#[derive(Parser)]
#[command(about)]
struct Args {
    /// Corpus embeddings: 2-D little-endian fp32 `.npy`, C order, rows x dim.
    #[arg(long)]
    corpus_npy: PathBuf,
    /// Query embeddings (repeatable): 2-D fp32 `.npy` with the corpus dim.
    /// Paired positionally with `--qids-jsonl`.
    #[arg(long, required = true)]
    queries_npy: Vec<PathBuf>,
    /// Ground-truth ids (repeatable): JSONL with one line per query carrying
    /// a `"paper_id": <row>` field naming the ground-truth corpus row.
    #[arg(long, required = true)]
    qids_jsonl: Vec<PathBuf>,
    /// Output directory for the bundle, report, and top-10 dumps.
    #[arg(long)]
    out_dir: PathBuf,
}

fn npy_open(path: &Path) -> (BufReader<File>, usize, usize) {
    let f = File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let mut r = BufReader::with_capacity(1 << 20, f);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic).expect("npy magic");
    assert_eq!(&magic[..6], b"\x93NUMPY", "not an npy file: {}", path.display());
    let header_len: usize = if magic[6] == 1 {
        let mut l = [0u8; 2];
        r.read_exact(&mut l).expect("npy header len");
        u16::from_le_bytes(l) as usize
    } else {
        let mut l = [0u8; 4];
        r.read_exact(&mut l).expect("npy header len");
        u32::from_le_bytes(l) as usize
    };
    let mut header = vec![0u8; header_len];
    r.read_exact(&mut header).expect("npy header");
    let header = String::from_utf8_lossy(&header).to_string();
    assert!(header.contains("'<f4'"), "expected little-endian f32: {header}");
    assert!(header.contains("'fortran_order': False"), "expected C order: {header}");
    let shape_start = header.find("'shape': (").expect("shape key") + "'shape': (".len();
    let shape_end = header[shape_start..].find(')').expect("shape close") + shape_start;
    let dims: Vec<usize> = header[shape_start..shape_end]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().expect("shape dim"))
        .collect();
    assert_eq!(dims.len(), 2, "expected 2-d array: {header}");
    (r, dims[0], dims[1])
}

fn read_rows(r: &mut BufReader<File>, rows: usize, cols: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; rows * cols * 4];
    r.read_exact(&mut bytes).expect("read npy rows");
    let mut out = Vec::with_capacity(rows * cols);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    out
}

fn proc_status_kb(key: &str) -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").expect("proc status");
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix(key) {
            return rest
                .trim_start_matches(':')
                .trim()
                .trim_end_matches(" kB")
                .parse()
                .expect("kB value");
        }
    }
    0
}

fn dir_size_bytes(path: &Path) -> u64 {
    let mut total = 0;
    for entry in std::fs::read_dir(path).expect("read_dir") {
        let entry = entry.expect("dir entry");
        let meta = entry.metadata().expect("metadata");
        total += if meta.is_dir() { dir_size_bytes(&entry.path()) } else { meta.len() };
    }
    total
}

fn load_qids(path: &Path) -> Vec<i64> {
    let f = File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    BufReader::new(f)
        .lines()
        .map(|line| {
            let line = line.expect("qid line");
            let start = line.find("\"paper_id\":").expect("paper_id key") + "\"paper_id\":".len();
            let rest = line[start..].trim_start();
            let end = rest
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(rest.len());
            rest[..end].parse().expect("paper_id value")
        })
        .collect()
}

/// Query-set label from the queries file stem, e.g. `title_queries.npy` -> `title`.
fn set_name(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("query_set");
    stem.strip_suffix("_queries").unwrap_or(stem).to_string()
}

fn percentile(sorted_us: &[u128], p: f64) -> u128 {
    let idx = ((sorted_us.len() as f64 - 1.0) * p).round() as usize;
    sorted_us[idx]
}

struct QueryStats {
    mean_us: f64,
    p50_us: u128,
    p95_us: u128,
    p99_us: u128,
    recall_at_1: f64,
    recall_at_10: f64,
}

fn recall(indices: &[i64], qids: &[i64], k: usize, at: usize) -> f64 {
    let hits = qids
        .iter()
        .enumerate()
        .filter(|(q, qid)| indices[q * k..q * k + at].contains(qid))
        .count();
    hits as f64 / qids.len() as f64
}

fn run_queries(
    index: &OrdinalIndex,
    queries: &[f32],
    qids: &[i64],
    dim: usize,
    options: DenseSearchOptions,
    limit: usize,
) -> (QueryStats, Vec<i64>) {
    let nq = (queries.len() / dim).min(limit);
    let qids = &qids[..nq];
    // Warm-up pass over the first 32 queries.
    for q in 0..nq.min(32) {
        let _ = index.search_with_options(&queries[q * dim..(q + 1) * dim], K, options);
    }
    let mut latencies_us = Vec::with_capacity(nq);
    let mut all_indices = Vec::with_capacity(nq * K);
    for q in 0..nq {
        let t = Instant::now();
        let res = index.search_with_options(&queries[q * dim..(q + 1) * dim], K, options);
        latencies_us.push(t.elapsed().as_micros());
        all_indices.extend_from_slice(&res.indices);
    }
    let mut sorted = latencies_us.clone();
    sorted.sort_unstable();
    let stats = QueryStats {
        mean_us: latencies_us.iter().sum::<u128>() as f64 / nq as f64,
        p50_us: percentile(&sorted, 0.50),
        p95_us: percentile(&sorted, 0.95),
        p99_us: percentile(&sorted, 0.99),
        recall_at_1: recall(&all_indices, qids, K, 1),
        recall_at_10: recall(&all_indices, qids, K, 10),
    };
    (stats, all_indices)
}

fn main() {
    let args = Args::parse();
    assert_eq!(
        args.queries_npy.len(),
        args.qids_jsonl.len(),
        "--queries-npy and --qids-jsonl must be repeated the same number of times \
         (they pair positionally)"
    );
    std::fs::create_dir_all(&args.out_dir).expect("create out dir");
    let out_dir = args.out_dir.as_path();
    let mut report = String::from("{\n");

    // ---- Ingest ----
    let (mut reader, rows, dim) = npy_open(&args.corpus_npy);
    eprintln!("corpus: {rows} rows x {dim} dims");
    let mut index = OrdinalIndex::new(dim, BITS).expect("construct index");
    assert!(index.has_sign_sidecar(), "sign sidecar must be on by default");
    let t_ingest = Instant::now();
    let mut done = 0usize;
    let mut read_secs = 0f64;
    while done < rows {
        let n = INGEST_CHUNK_ROWS.min(rows - done);
        let t_read = Instant::now();
        let chunk = read_rows(&mut reader, n, dim);
        read_secs += t_read.elapsed().as_secs_f64();
        index.add_2d(&chunk, dim).expect("add_2d");
        done += n;
        if done % (INGEST_CHUNK_ROWS * 16) == 0 {
            eprintln!("  ingested {done}/{rows}");
        }
    }
    let ingest_secs = t_ingest.elapsed().as_secs_f64();
    let encode_secs = ingest_secs - read_secs;
    let rss_after_ingest_kb = proc_status_kb("VmRSS");
    eprintln!(
        "ingest: {ingest_secs:.1}s total ({read_secs:.1}s file read, {encode_secs:.1}s encode), \
         rss {} MB",
        rss_after_ingest_kb / 1024
    );
    report.push_str(&format!(
        "  \"rows\": {rows},\n  \"dim\": {dim},\n  \"bits\": {BITS},\n  \
         \"ingest_total_secs\": {ingest_secs:.3},\n  \"ingest_read_secs\": {read_secs:.3},\n  \
         \"ingest_encode_secs\": {encode_secs:.3},\n  \"rss_after_ingest_kb\": {rss_after_ingest_kb},\n"
    ));

    // ---- Verified bundle write ----
    let bundle_path = out_dir.join("corpus.odb");
    if bundle_path.exists() {
        std::fs::remove_dir_all(&bundle_path).expect("clean old bundle");
    }
    // Resource limits are manifest-derived: each auxiliary artifact read is
    // bounded by its manifest-declared, SHA-256-pinned size, so the sign
    // sidecar (`sign.ovsb`, rows x dim / 8 bytes; ~161 MB at 1.26M x 1024)
    // needs no manual limit raise. ordinaldb 0.2.0 against ordvec 0.5.0
    // shipped a flat 64 MB default that had to be raised here (finding #1
    // of the original battle test).
    let manifest_options = CreateManifestOptions::default();
    let t_write = Instant::now();
    let write_report = index
        .write_verified_bundle(&bundle_path, manifest_options, Vec::new())
        .expect("write verified bundle");
    let write_secs = t_write.elapsed().as_secs_f64();
    assert!(write_report.has_sign, "bundle must carry sign sidecar");
    let bundle_bytes = dir_size_bytes(&bundle_path);
    let raw_bytes = rows as u64 * dim as u64 * 4;
    eprintln!(
        "bundle write: {write_secs:.2}s, {bundle_bytes} bytes ({:.1}x smaller than raw fp32)",
        raw_bytes as f64 / bundle_bytes as f64
    );
    report.push_str(&format!(
        "  \"bundle_write_secs\": {write_secs:.3},\n  \"bundle_bytes\": {bundle_bytes},\n  \
         \"raw_fp32_bytes\": {raw_bytes},\n"
    ));

    // ---- Cold verified open ----
    drop(index);
    let manifest_path = bundle_path.join("manifest.json");
    let t_open = Instant::now();
    let index = OrdinalIndex::open_verified(
        &manifest_path,
        VerifyOptions::default(),
        DenseLoadOptions {
            require_sign: true,
            expected_dim: Some(dim),
            expected_bits: Some(BITS),
        },
    )
    .expect("open verified bundle");
    let open_secs = t_open.elapsed().as_secs_f64();
    assert_eq!(index.len(), rows);
    let rss_after_open_kb = proc_status_kb("VmRSS");
    eprintln!("verified open: {open_secs:.2}s, rss {} MB", rss_after_open_kb / 1024);
    report.push_str(&format!(
        "  \"verified_open_secs\": {open_secs:.3},\n  \"rss_after_open_kb\": {rss_after_open_kb},\n"
    ));

    // ---- Query sets ----
    report.push_str("  \"query_sets\": {\n");
    let sets = args.queries_npy.len();
    for (si, (qpath, qids_path)) in args.queries_npy.iter().zip(&args.qids_jsonl).enumerate() {
        let set = set_name(qpath);
        let (mut qr, nq, qcols) = npy_open(qpath);
        assert_eq!(qcols, dim, "query dim mismatch: {}", qpath.display());
        let queries = read_rows(&mut qr, nq, qcols);
        let qids = load_qids(qids_path);
        assert_eq!(qids.len(), nq, "qids/queries count mismatch: {}", qids_path.display());

        // Sequential single-query latency + recall, default Auto (two-stage).
        let (auto_stats, auto_indices) =
            run_queries(&index, &queries, &qids, dim, DenseSearchOptions::default(), nq);

        // Batched throughput: one call, all queries.
        let t_batch = Instant::now();
        let batch_res = index.search_with_options(&queries, K, DenseSearchOptions::default());
        let batch_secs = t_batch.elapsed().as_secs_f64();
        assert_eq!(batch_res.indices.len(), nq * K);

        // Exact RankQuant scan on a subset.
        let (exact_stats, _) = run_queries(
            &index,
            &queries,
            &qids,
            dim,
            DenseSearchOptions::exact_rankquant(),
            EXACT_SUBSET,
        );

        // Dump Auto top-10 ids for external cosine-baseline comparison.
        let mut dump =
            File::create(out_dir.join(format!("{set}_ordinal_top10.i64"))).expect("dump file");
        let bytes: Vec<u8> = auto_indices.iter().flat_map(|i| i.to_le_bytes()).collect();
        dump.write_all(&bytes).expect("write dump");

        eprintln!(
            "{set}: auto p50 {}us p95 {}us r@1 {:.3} r@10 {:.3} | batch {:.0} q/s | \
             exact({EXACT_SUBSET}) p50 {}us r@10 {:.3}",
            auto_stats.p50_us,
            auto_stats.p95_us,
            auto_stats.recall_at_1,
            auto_stats.recall_at_10,
            nq as f64 / batch_secs,
            exact_stats.p50_us,
            exact_stats.recall_at_10,
        );
        report.push_str(&format!(
            "    \"{set}\": {{\n      \"queries\": {nq},\n      \
             \"auto_mean_us\": {:.1},\n      \"auto_p50_us\": {},\n      \
             \"auto_p95_us\": {},\n      \"auto_p99_us\": {},\n      \
             \"auto_recall_at_1\": {:.4},\n      \"auto_recall_at_10\": {:.4},\n      \
             \"batch_secs\": {batch_secs:.3},\n      \"batch_qps\": {:.1},\n      \
             \"exact_subset\": {EXACT_SUBSET},\n      \"exact_mean_us\": {:.1},\n      \
             \"exact_p50_us\": {},\n      \"exact_p95_us\": {},\n      \
             \"exact_recall_at_1\": {:.4},\n      \"exact_recall_at_10\": {:.4}\n    }}{}\n",
            auto_stats.mean_us,
            auto_stats.p50_us,
            auto_stats.p95_us,
            auto_stats.p99_us,
            auto_stats.recall_at_1,
            auto_stats.recall_at_10,
            nq as f64 / batch_secs,
            exact_stats.mean_us,
            exact_stats.p50_us,
            exact_stats.p95_us,
            exact_stats.recall_at_1,
            exact_stats.recall_at_10,
            if si + 1 == sets { "" } else { "," },
        ));
    }
    report.push_str("  },\n");

    let vm_hwm_kb = proc_status_kb("VmHWM");
    report.push_str(&format!("  \"vm_hwm_kb\": {vm_hwm_kb}\n}}\n"));
    std::fs::write(out_dir.join("bench-results.json"), &report).expect("write report");
    eprintln!("peak rss {} MB; results at {}", vm_hwm_kb / 1024, out_dir.display());
}
