use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, ValueEnum};
use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{params::LlamaModelParams, AddBos, LlamaModel};
use ordinaldb::{
    DenseSearchExecution, DenseSearchOptions, DenseSearchPlan, DenseSearchTimings, IdMapIndex,
};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};
use tempfile::{tempdir, Builder};
#[cfg(feature = "turbovec-compare")]
use turbovec::TurboQuantIndex;
use zip::ZipArchive;

const BEIR_BASE_URL: &str = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets";
const HARRIER_QUERY_PREFIX: &str =
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ";
const EMBEDDING_CACHE_MAGIC: &[u8; 8] = b"ODBEMB1\0";
const DOWNLOAD_TIMEOUTS: DownloadTimeouts = DownloadTimeouts {
    connect: Duration::from_secs(15),
    read: Duration::from_secs(60),
    overall: Duration::from_secs(60 * 60),
};

#[derive(Parser, Debug)]
#[command(about = "Rust BEIR retrieval benchmark for OrdinalDB")]
struct Args {
    #[arg(long, default_value = "scifact")]
    dataset: String,
    #[arg(long, default_value = "test")]
    split: String,
    #[arg(long, default_value = "2")]
    bits: String,
    #[arg(long, default_value_t = 10)]
    k: usize,
    #[arg(long, default_value_t = 1)]
    query_repetitions: usize,
    #[arg(long, default_value = "benchmark-results/beir-scifact-test.json")]
    output: PathBuf,
    #[arg(long, default_value = ".ordinaldb-benchmark-cache")]
    cache_dir: PathBuf,
    #[arg(long)]
    max_docs: Option<usize>,
    #[arg(long)]
    max_queries: Option<usize>,
    #[arg(long, default_value_t = 1)]
    embed_batch_size: usize,
    #[arg(long, value_enum, default_value_t = HarrierQuant::Q8)]
    harrier_quant: HarrierQuant,
    #[arg(long, default_value_t = 999)]
    llama_gpu_layers: u32,
    #[arg(long, default_value_t = 32768)]
    llama_context_tokens: u32,
    #[arg(long, default_value_t = 512)]
    llama_batch_tokens: u32,
    #[arg(long, default_value_t = 512)]
    llama_ubatch_tokens: u32,
    #[arg(long)]
    llama_threads: Option<i32>,
    #[arg(long, default_value_t = 512)]
    max_text_tokens: usize,
    #[arg(long)]
    show_download_progress: bool,
    #[arg(long)]
    refresh_dataset: bool,
    #[arg(long)]
    refresh_model: bool,
    #[arg(long)]
    refresh_embeddings: bool,
    #[arg(long)]
    include_turbovec_b2: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum HarrierQuant {
    #[value(name = "q8_0", alias = "q8")]
    Q8,
    #[value(name = "q6_k", alias = "q6")]
    Q6,
}

impl HarrierQuant {
    fn spec(self) -> HarrierModelSpec {
        match self {
            HarrierQuant::Q8 => HarrierModelSpec {
                quantization: "Q8_0",
                repo: "SuperPauly/harrier-oss-v1-0.6b-gguf",
                filename: "harrier-oss-v1-0.6B-Q8_0.gguf",
                url: "https://huggingface.co/SuperPauly/harrier-oss-v1-0.6b-gguf/resolve/main/harrier-oss-v1-0.6B-Q8_0.gguf",
                expected_sha256: Some(
                    "f97092bd73f6814b8b1170ca855071bc468b5fa2cf61a6de7e1d2c4a8a6a50b0",
                ),
            },
            HarrierQuant::Q6 => HarrierModelSpec {
                quantization: "Q6_K",
                repo: "mradermacher/harrier-oss-v1-0.6b-GGUF",
                filename: "harrier-oss-v1-0.6b.Q6_K.gguf",
                url: "https://huggingface.co/mradermacher/harrier-oss-v1-0.6b-GGUF/resolve/main/harrier-oss-v1-0.6b.Q6_K.gguf",
                expected_sha256: Some(
                    "22ab6a9447624ede20e04f6d2586eeeceba5acb33751e437f682dc5e2d571dcc",
                ),
            },
        }
    }
}

#[derive(Debug)]
struct HarrierModelSpec {
    quantization: &'static str,
    repo: &'static str,
    filename: &'static str,
    url: &'static str,
    expected_sha256: Option<&'static str>,
}

#[derive(Debug)]
struct HarrierModelFile {
    path: PathBuf,
    spec: HarrierModelSpec,
    sha256: String,
    sha256_verified: bool,
    bytes: u64,
}

#[derive(Debug)]
struct BenchmarkDataset {
    doc_ids: Vec<String>,
    doc_texts: Vec<String>,
    query_ids: Vec<String>,
    query_texts: Vec<String>,
    qrels: HashMap<String, HashMap<String, f64>>,
    archive_verification: BeirArchiveVerification,
}

#[derive(Debug)]
struct BeirArchiveVerification {
    md5: String,
    md5_verified: bool,
    sha256: String,
    sha256_verified: bool,
}

#[derive(Clone, Copy, Debug)]
enum EmbeddingKind {
    Document,
    Query,
}

impl EmbeddingKind {
    fn label(self) -> &'static str {
        match self {
            EmbeddingKind::Document => "documents",
            EmbeddingKind::Query => "queries",
        }
    }
}

#[derive(Debug)]
struct CachedEmbeddingRead {
    embeddings: Vec<Vec<f32>>,
    seconds: f64,
}

#[derive(Clone, Copy, Debug)]
struct BeirArchiveSpec {
    md5: &'static str,
    sha256: &'static str,
}

#[derive(Clone, Copy, Debug)]
struct DownloadTimeouts {
    connect: Duration,
    read: Duration,
    overall: Duration,
}

#[derive(Deserialize)]
struct BeirTextRow {
    #[serde(rename = "_id")]
    id: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    text: String,
}

struct HarrierEmbedder<'model, 'context> {
    model: &'model LlamaModel,
    context: &'context mut LlamaContext<'model>,
    embed_batch_size: usize,
    batch_tokens: usize,
    max_text_tokens: usize,
}

impl<'model, 'context> HarrierEmbedder<'model, 'context> {
    fn new(
        model: &'model LlamaModel,
        context: &'context mut LlamaContext<'model>,
        embed_batch_size: usize,
        batch_tokens: usize,
        max_text_tokens: usize,
    ) -> Result<Self> {
        if embed_batch_size == 0 {
            bail!("--embed-batch-size must be greater than zero");
        }
        if batch_tokens == 0 {
            bail!("--llama-batch-tokens must be greater than zero");
        }
        Ok(Self {
            model,
            context,
            embed_batch_size,
            batch_tokens,
            max_text_tokens,
        })
    }

    fn embed_documents(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed_texts(texts, InputKind::Document)
    }

    fn embed_queries(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed_texts(texts, InputKind::Query)
    }

    fn embed_texts(&mut self, texts: &[String], kind: InputKind) -> Result<Vec<Vec<f32>>> {
        let mut outputs = Vec::with_capacity(texts.len());
        let mut cursor = 0;
        let mut next_progress = Instant::now() + Duration::from_secs(30);

        while cursor < texts.len() {
            let mut token_rows = Vec::new();
            let mut batch_token_count = 0usize;

            while cursor + token_rows.len() < texts.len()
                && token_rows.len() < self.embed_batch_size
            {
                let index = cursor + token_rows.len();
                let input = kind.prepare(&texts[index]);
                let tokens = self
                    .tokenize(&input)
                    .with_context(|| format!("tokenizing {kind:?} row {index}"))?;
                let token_count = tokens.len();
                if token_count > self.batch_tokens {
                    bail!(
                        "{kind:?} row {index} has {token_count} tokens, above --llama-batch-tokens {}; increase --llama-batch-tokens or set --max-text-tokens",
                        self.batch_tokens
                    );
                }
                if !token_rows.is_empty() && batch_token_count + token_count > self.batch_tokens {
                    break;
                }
                batch_token_count += token_count;
                token_rows.push(tokens);
            }

            if token_rows.is_empty() {
                bail!("internal error: empty Harrier embedding batch");
            }

            self.context.clear_kv_cache();
            let sequence_count =
                checked_i32_len(token_rows.len(), "embedding batch sequence count")?;
            let mut batch = LlamaBatch::new(batch_token_count, sequence_count);
            for (seq_id, tokens) in token_rows.iter().enumerate() {
                let seq_id = checked_i32_len(seq_id, "embedding sequence id")?;
                batch.add_sequence(tokens, seq_id, true).with_context(|| {
                    format!("building llama.cpp batch for {kind:?} seq {seq_id}")
                })?;
            }
            self.context
                .decode(&mut batch)
                .with_context(|| format!("decoding {kind:?} batch at row {cursor}"))?;

            for seq_id in 0..token_rows.len() {
                let seq_id = checked_i32_len(seq_id, "embedding sequence id")?;
                let embedding = self
                    .context
                    .embeddings_seq_ith(seq_id)
                    .with_context(|| format!("reading {kind:?} embedding seq {seq_id}"))?;
                outputs.push(embedding.to_vec());
            }

            cursor += token_rows.len();
            if Instant::now() >= next_progress || cursor == texts.len() {
                eprintln!(
                    "embedded {cursor}/{} {kind:?} rows with Harrier",
                    texts.len()
                );
                next_progress = Instant::now() + Duration::from_secs(30);
            }
        }

        Ok(outputs)
    }

    fn tokenize(&self, text: &str) -> Result<Vec<llama_cpp_2::token::LlamaToken>> {
        let mut tokens = self
            .model
            .str_to_token(text, AddBos::Never)
            .map_err(|err| anyhow!("llama.cpp tokenization failed: {err}"))?;
        if self.max_text_tokens > 0 {
            let max_text_tokens = self.max_text_tokens;
            tokens.truncate(max_text_tokens);
        }
        if tokens.is_empty() {
            bail!("Harrier tokenizer produced zero tokens");
        }
        Ok(tokens)
    }
}

#[derive(Clone, Copy, Debug)]
enum InputKind {
    Document,
    Query,
}

impl InputKind {
    fn prepare(self, text: &str) -> String {
        match self {
            InputKind::Document => text.to_string(),
            InputKind::Query => format!("{HARRIER_QUERY_PREFIX}{text}"),
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    validate_dataset(&args.dataset)?;
    validate_split(&args.split)?;
    let bits = parse_bits(&args.bits)?;
    if args.k == 0 {
        bail!("--k must be greater than zero");
    }
    if args.query_repetitions == 0 {
        bail!("--query-repetitions must be greater than zero");
    }
    ensure_requested_gpu_available(args.llama_gpu_layers)?;
    let dataset = load_dataset(&args)?;
    let model_file = ensure_harrier_model(&args)?;

    let document_cache_path =
        embedding_cache_path(&args, &dataset, &model_file, EmbeddingKind::Document);
    let query_cache_path = embedding_cache_path(&args, &dataset, &model_file, EmbeddingKind::Query);
    let cached_documents = if args.refresh_embeddings {
        None
    } else {
        read_embedding_cache(
            &document_cache_path,
            EmbeddingKind::Document,
            dataset.doc_ids.len(),
        )?
    };
    let cached_queries = if args.refresh_embeddings {
        None
    } else {
        read_embedding_cache(
            &query_cache_path,
            EmbeddingKind::Query,
            dataset.query_ids.len(),
        )?
    };

    let mut doc_embedding_cache_hit = cached_documents.is_some();
    let mut query_embedding_cache_hit = cached_queries.is_some();
    let mut doc_embedding_seconds = cached_documents
        .as_ref()
        .map_or(0.0, |cached| cached.seconds);
    let mut query_embedding_seconds = cached_queries.as_ref().map_or(0.0, |cached| cached.seconds);
    let mut doc_embeddings = cached_documents.map(|cached| cached.embeddings);
    let mut query_embeddings = cached_queries.map(|cached| cached.embeddings);

    if doc_embeddings.is_none() || query_embeddings.is_none() {
        let backend = LlamaBackend::init().context("initializing llama.cpp backend")?;
        let model_params = LlamaModelParams::default().with_n_gpu_layers(args.llama_gpu_layers);
        let model = LlamaModel::load_from_file(&backend, &model_file.path, &model_params)
            .with_context(|| format!("loading Harrier GGUF {}", model_file.path.display()))?;
        let n_ctx = NonZeroU32::new(args.llama_context_tokens)
            .context("--llama-context-tokens must be greater than zero")?;
        let n_seq_max = checked_u32_len(args.embed_batch_size, "--embed-batch-size")?;
        let mut context_params = LlamaContextParams::default()
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::Last)
            .with_n_ctx(Some(n_ctx))
            .with_n_batch(args.llama_batch_tokens)
            .with_n_ubatch(args.llama_ubatch_tokens)
            .with_n_seq_max(n_seq_max);
        if let Some(threads) = args.llama_threads {
            context_params = context_params
                .with_n_threads(threads)
                .with_n_threads_batch(threads);
        }
        let mut context = model
            .new_context(&backend, context_params)
            .context("creating llama.cpp embedding context")?;

        let mut embedder = HarrierEmbedder::new(
            &model,
            &mut context,
            args.embed_batch_size,
            args.llama_batch_tokens.min(args.llama_ubatch_tokens) as usize,
            args.max_text_tokens,
        )?;

        if doc_embeddings.is_none() {
            let started = Instant::now();
            let mut embedded = embedder
                .embed_documents(&dataset.doc_texts)
                .context("embedding BEIR documents with Harrier")?;
            normalize_rows(&mut embedded);
            write_embedding_cache(&document_cache_path, EmbeddingKind::Document, &embedded)?;
            doc_embedding_seconds = seconds(started.elapsed());
            doc_embedding_cache_hit = false;
            doc_embeddings = Some(embedded);
        }

        if query_embeddings.is_none() {
            let started = Instant::now();
            let mut embedded = embedder
                .embed_queries(&dataset.query_texts)
                .context("embedding BEIR queries with Harrier")?;
            normalize_rows(&mut embedded);
            write_embedding_cache(&query_cache_path, EmbeddingKind::Query, &embedded)?;
            query_embedding_seconds = seconds(started.elapsed());
            query_embedding_cache_hit = false;
            query_embeddings = Some(embedded);
        }
    }

    let doc_embeddings = doc_embeddings.expect("document embeddings initialized");
    let query_embeddings = query_embeddings.expect("query embeddings initialized");

    let dim = embedding_dim(&doc_embeddings, &query_embeddings)?;
    let (exact_indices, exact_query_samples) = repeat_timed(args.query_repetitions, || {
        Ok(exact_float_topk(&query_embeddings, &doc_embeddings, args.k))
    })?;
    let exact_query_seconds = mean(&exact_query_samples);
    let exact_run = indices_to_doc_ids(&exact_indices, &dataset.doc_ids);
    let exact_metrics = evaluate_run(&exact_run, &dataset.query_ids, &dataset.qrels, args.k);

    let doc_flat = flatten(&doc_embeddings);
    let query_flat = flatten(&query_embeddings);
    let ids: Vec<u64> = (1..=dataset.doc_ids.len() as u64).collect();

    let turbovec_b2 = if args.include_turbovec_b2 {
        Some(run_turbovec_b2(
            dim,
            &doc_flat,
            &query_flat,
            &dataset,
            args.k,
            args.query_repetitions,
            exact_query_seconds,
            &exact_run,
        )?)
    } else {
        None
    };

    let mut ordinal_rows = Vec::new();
    for bits in bits {
        let mut index =
            IdMapIndex::new(dim, bits).with_context(|| format!("creating b={bits} index"))?;

        let started = Instant::now();
        index
            .add_with_ids(&doc_flat, &ids)
            .with_context(|| format!("adding documents to b={bits} index"))?;
        let ingest_seconds = seconds(started.elapsed());

        let (search_report, query_samples) = repeat_timed(args.query_repetitions, || {
            Ok(index.search_with_report(&query_flat, args.k, DenseSearchOptions::default())?)
        })?;
        let query_seconds = mean(&query_samples);

        let effective_k = search_report.dense_plan.effective_k;
        if effective_k == 0 {
            bail!(
                "OrdinalDB returned effective_k=0 for --k {}; cannot chunk benchmark results",
                args.k
            );
        }
        let ordinal_run = mapped_ids_to_doc_ids(
            &search_report.ids,
            &dataset.doc_ids,
            dataset.query_ids.len(),
            effective_k,
        )?;
        let metrics = evaluate_run(&ordinal_run, &dataset.query_ids, &dataset.qrels, args.k);
        let overlap = mean_topk_overlap(&ordinal_run, &exact_run, effective_k);

        let temp = tempdir().context("creating temp directory for persistence benchmark")?;
        let bundle_path = temp.path().join(format!("ordinaldb-b{bits}.odb"));
        let started = Instant::now();
        index
            .write(&bundle_path)
            .with_context(|| format!("writing b={bits} bundle"))?;
        let write_seconds = seconds(started.elapsed());
        let bundle_bytes = directory_size(&bundle_path)?;

        let started = Instant::now();
        IdMapIndex::load(&bundle_path).with_context(|| format!("loading b={bits} bundle"))?;
        let load_seconds = seconds(started.elapsed());

        ordinal_rows.push(json!({
            "bits": bits,
            "ingest_seconds": ingest_seconds,
            "query_seconds": query_seconds,
            "milliseconds_per_query": 1000.0 * query_seconds / dataset.query_ids.len() as f64,
            "query_latency": latency_report(&query_samples, dataset.query_ids.len()),
            "queries_per_second": safe_div(dataset.query_ids.len() as f64, query_seconds),
            "speedup_vs_exact_float32_scan": safe_div(exact_query_seconds, query_seconds),
            "dense_search_plan": dense_search_plan_json(&search_report.dense_plan),
            "dense_phase_seconds_last_run": dense_search_timings_json(&search_report.dense_timings),
            "id_mapping_seconds_last_run": seconds(search_report.id_mapping),
            "reported_total_seconds_last_run": seconds(search_report.total),
            "metrics": metrics,
            "mean_topk_overlap_with_exact_float32": overlap,
            "write_seconds": write_seconds,
            "load_seconds": load_seconds,
            "bundle_bytes": bundle_bytes,
            "bundle_bytes_per_vector": safe_div(bundle_bytes as f64, dataset.doc_ids.len() as f64),
        }));
    }

    let report = json!({
        "benchmark": "ordinaldb_beir_retrieval_rust",
        "dataset": {
            "name": args.dataset,
            "split": args.split,
            "source": "BEIR public zip mirror",
            "source_url": format!("{BEIR_BASE_URL}/{}.zip", args.dataset),
            "archive_md5": dataset.archive_verification.md5,
            "archive_md5_verified": dataset.archive_verification.md5_verified,
            "archive_sha256": dataset.archive_verification.sha256,
            "archive_sha256_verified": dataset.archive_verification.sha256_verified,
            "corpus_packaged_with_ordinaldb": false,
            "documents": dataset.doc_ids.len(),
            "queries": dataset.query_ids.len(),
            "qrels_queries": dataset.qrels.len(),
            "max_docs": args.max_docs,
            "max_queries": args.max_queries,
        },
        "embedding": {
            "runtime": "llama-cpp-2",
            "base_model": "microsoft/harrier-oss-v1-0.6b",
            "model_repo": model_file.spec.repo,
            "model_file": model_file.spec.filename,
            "model_url": model_file.spec.url,
            "quantization": model_file.spec.quantization,
            "model_bytes": model_file.bytes,
            "sha256": model_file.sha256,
            "sha256_verified": model_file.sha256_verified,
            "dim": dim,
            "pooling": "last-token",
            "add_bos": false,
            "query_prefix": HARRIER_QUERY_PREFIX,
            "normalized_embeddings": true,
            "document_embedding_seconds": doc_embedding_seconds,
            "query_embedding_seconds": query_embedding_seconds,
            "document_embedding_cache_hit": doc_embedding_cache_hit,
            "query_embedding_cache_hit": query_embedding_cache_hit,
            "embedding_seconds_excluded_from_retrieval_timing": true,
            "llama_gpu_layers_requested": args.llama_gpu_layers,
            "llama_context_tokens": args.llama_context_tokens,
            "llama_batch_tokens": args.llama_batch_tokens,
            "llama_ubatch_tokens": args.llama_ubatch_tokens,
            "llama_seq_max": args.embed_batch_size,
            "embed_batch_size": args.embed_batch_size,
            "max_text_tokens": args.max_text_tokens,
        },
        "hardware": hardware_report(),
        "source": source_report(),
        "k": args.k,
        "query_repetitions": args.query_repetitions,
        "turbovec_b2": turbovec_b2,
        "exact_float32_baseline": {
            "description": "Rust scalar exact float32 cosine top-k over normalized embeddings",
            "query_seconds": exact_query_seconds,
            "milliseconds_per_query": 1000.0 * exact_query_seconds / dataset.query_ids.len() as f64,
            "query_latency": latency_report(&exact_query_samples, dataset.query_ids.len()),
            "queries_per_second": safe_div(dataset.query_ids.len() as f64, exact_query_seconds),
            "metrics": exact_metrics,
        },
        "ordinaldb": ordinal_rows,
        "claim_boundaries": [
            "The BEIR corpus is downloaded on the runner's machine and is not packaged with OrdinalDB.",
            "The Harrier GGUF model is downloaded on the runner's machine and is not packaged with OrdinalDB.",
            "Embedding time is measured separately and excluded from retrieval timing.",
            "The baseline is an exact Rust float32 scan over the same normalized embeddings, not an external vector database.",
            "BEIR metrics reflect this dataset, model, quantization, k, bit width, hardware, and thread configuration only.",
            "OrdinalDB scores are internal similarity scores and are not normalized cosine scores."
        ],
    });

    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }
    let body = serde_json::to_string_pretty(&report)? + "\n";
    fs::write(&args.output, body).with_context(|| format!("writing {}", args.output.display()))?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    eprintln!("Wrote benchmark report to {}", args.output.display());
    Ok(())
}

fn dense_search_plan_json(plan: &DenseSearchPlan) -> serde_json::Value {
    json!({
        "execution": match plan.execution {
            DenseSearchExecution::ExactRankQuant => "exact_rankquant",
            DenseSearchExecution::SignTwoStage => "sign_two_stage",
        },
        "dim": plan.dim,
        "query_count": plan.query_count,
        "requested_k": plan.requested_k,
        "effective_k": plan.effective_k,
        "search_space": plan.search_space,
        "candidate_count": plan.candidate_count,
    })
}

fn dense_search_timings_json(timings: &DenseSearchTimings) -> serde_json::Value {
    json!({
        "validation": seconds(timings.validation),
        "candidate_generation": seconds(timings.candidate_generation),
        "rerank": seconds(timings.rerank),
        "exact_search": seconds(timings.exact_search),
        "total": seconds(timings.total),
        "note": "candidate_generation and rerank are summed per worker for parallel two-stage search; total is wall time."
    })
}

#[cfg(feature = "turbovec-compare")]
fn run_turbovec_b2(
    dim: usize,
    doc_flat: &[f32],
    query_flat: &[f32],
    dataset: &BenchmarkDataset,
    k: usize,
    query_repetitions: usize,
    exact_query_seconds: f64,
    exact_run: &[Vec<String>],
) -> Result<serde_json::Value> {
    let mut index = TurboQuantIndex::new(dim, 2).context("creating upstream turbovec b=2 index")?;

    let started = Instant::now();
    index.add(doc_flat);
    let ingest_seconds = seconds(started.elapsed());

    let started = Instant::now();
    index.prepare();
    let prepare_seconds = seconds(started.elapsed());

    let (results, query_samples) =
        repeat_timed(query_repetitions, || Ok(index.search(query_flat, k)))?;
    let query_seconds = mean(&query_samples);
    let effective_k = results.k;
    let turbovec_run =
        turbovec_indices_to_doc_ids(&results.indices, &dataset.doc_ids, effective_k)?;
    let metrics = evaluate_run(&turbovec_run, &dataset.query_ids, &dataset.qrels, k);
    let overlap = mean_topk_overlap(&turbovec_run, exact_run, effective_k);

    Ok(json!({
        "implementation": "turbovec::TurboQuantIndex",
        "bits": 2,
        "source": {
            "git_url": "https://github.com/RyanCodrai/turbovec.git",
            "git_rev": "1e7200cfd8f26c92ce2855652db64bc7f85bc039",
            "crate_version": "0.9.0"
        },
        "ingest_seconds": ingest_seconds,
        "prepare_seconds": prepare_seconds,
        "prepare_seconds_excluded_from_query_timing": true,
        "query_seconds": query_seconds,
        "milliseconds_per_query": 1000.0 * query_seconds / dataset.query_ids.len() as f64,
        "query_latency": latency_report(&query_samples, dataset.query_ids.len()),
        "queries_per_second": safe_div(dataset.query_ids.len() as f64, query_seconds),
        "speedup_vs_exact_float32_scan": safe_div(exact_query_seconds, query_seconds),
        "metrics": metrics,
        "mean_topk_overlap_with_exact_float32": overlap,
        "claim_boundary": "This is upstream turbovec b=2 on the same normalized embeddings, corpus, queries, k, and qrels as the OrdinalDB rows; it does not use OrdinalDB persistence or sign-bitmap two-stage reranking."
    }))
}

#[cfg(not(feature = "turbovec-compare"))]
fn run_turbovec_b2(
    _dim: usize,
    _doc_flat: &[f32],
    _query_flat: &[f32],
    _dataset: &BenchmarkDataset,
    _k: usize,
    _query_repetitions: usize,
    _exact_query_seconds: f64,
    _exact_run: &[Vec<String>],
) -> Result<serde_json::Value> {
    bail!("--include-turbovec-b2 requires building the benchmark with --features turbovec-compare")
}

#[cfg(feature = "turbovec-compare")]
fn turbovec_indices_to_doc_ids(
    indices: &[i64],
    doc_ids: &[String],
    effective_k: usize,
) -> Result<Vec<Vec<String>>> {
    if effective_k == 0 {
        return Ok(Vec::new());
    }
    indices
        .chunks(effective_k)
        .map(|row| {
            row.iter()
                .map(|idx| {
                    let idx = usize::try_from(*idx).with_context(|| {
                        format!("upstream turbovec returned negative index {idx}")
                    })?;
                    doc_ids.get(idx).cloned().with_context(|| {
                        format!("upstream turbovec returned out-of-range index {idx}")
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()
}

fn ensure_harrier_model(args: &Args) -> Result<HarrierModelFile> {
    let spec = args.harrier_quant.spec();
    let model_dir = args
        .cache_dir
        .join("models")
        .join(spec.repo.replace('/', "__"));
    fs::create_dir_all(&model_dir)?;
    let model_path = model_dir.join(spec.filename);
    if args.refresh_model && model_path.exists() {
        fs::remove_file(&model_path)?;
    }
    if !model_path.exists() {
        eprintln!(
            "downloading Harrier {} GGUF from {}",
            spec.quantization, spec.url
        );
        download_to_path(spec.url, &model_path, args.show_download_progress)
            .with_context(|| format!("downloading Harrier model {}", spec.url))?;
    }

    let sha256 = sha256_file(&model_path)?;
    let sha256_verified = if let Some(expected) = spec.expected_sha256 {
        if sha256 != expected {
            remove_corrupt_cache_file(&model_path);
            bail!(
                "Harrier model checksum mismatch for {}: expected {}, got {}",
                spec.filename,
                expected,
                sha256
            );
        }
        true
    } else {
        false
    };
    let bytes = fs::metadata(&model_path)?.len();
    Ok(HarrierModelFile {
        path: model_path,
        spec,
        sha256,
        sha256_verified,
        bytes,
    })
}

fn embedding_cache_path(
    args: &Args,
    dataset: &BenchmarkDataset,
    model_file: &HarrierModelFile,
    kind: EmbeddingKind,
) -> PathBuf {
    let (ids, texts) = match kind {
        EmbeddingKind::Document => (&dataset.doc_ids, &dataset.doc_texts),
        EmbeddingKind::Query => (&dataset.query_ids, &dataset.query_texts),
    };
    let input_hash = embedding_input_hash(kind, ids, texts);
    let model_hash = short_hash(&model_file.sha256);
    args.cache_dir
        .join("embeddings")
        .join(cache_component(&args.dataset))
        .join(format!(
            "{}-split-{}-model-{}-quant-{}-tokens-{}-input-{}.f32bin",
            kind.label(),
            cache_component(&args.split),
            model_hash,
            cache_component(model_file.spec.quantization),
            args.max_text_tokens,
            input_hash
        ))
}

fn read_embedding_cache(
    path: &Path,
    kind: EmbeddingKind,
    expected_rows: usize,
) -> Result<Option<CachedEmbeddingRead>> {
    if !path.exists() {
        return Ok(None);
    }
    let started = Instant::now();
    let metadata = fs::metadata(path).with_context(|| format!("reading {}", path.display()))?;
    if !metadata.is_file() {
        bail!("{} embedding cache is not a regular file", kind.label());
    }
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic)
        .with_context(|| format!("reading {}", path.display()))?;
    if &magic != EMBEDDING_CACHE_MAGIC {
        bail!("{} embedding cache has invalid magic", kind.label());
    }
    let rows_raw = read_u64_le(&mut file, path)?;
    let rows = usize::try_from(rows_raw).with_context(|| {
        format!(
            "{} embedding cache row count {rows_raw} does not fit this platform",
            kind.label()
        )
    })?;
    let dim_raw = read_u64_le(&mut file, path)?;
    let dim = usize::try_from(dim_raw).with_context(|| {
        format!(
            "{} embedding cache dimension {dim_raw} does not fit this platform",
            kind.label()
        )
    })?;
    if rows != expected_rows {
        bail!(
            "{} embedding cache row count {rows} does not match expected {expected_rows}",
            kind.label()
        );
    }
    if dim == 0 {
        bail!("{} embedding cache has zero dimension", kind.label());
    }
    let cells = rows
        .checked_mul(dim)
        .context("embedding cache cell count overflow")?;
    let payload_bytes = cells
        .checked_mul(std::mem::size_of::<f32>())
        .context("embedding cache byte count overflow")?;
    let payload_bytes_u64 =
        u64::try_from(payload_bytes).context("embedding cache byte count too large")?;
    let expected_len = 24u64
        .checked_add(payload_bytes_u64)
        .context("embedding cache file length overflow")?;
    if metadata.len() != expected_len {
        bail!(
            "{} embedding cache size mismatch: expected {expected_len} bytes, got {}",
            kind.label(),
            metadata.len()
        );
    }
    let mut bytes = vec![0u8; payload_bytes];
    file.read_exact(&mut bytes)
        .with_context(|| format!("reading {}", path.display()))?;
    let mut embeddings = Vec::with_capacity(rows);
    for row_bytes in bytes.chunks_exact(dim * std::mem::size_of::<f32>()) {
        let mut row = Vec::with_capacity(dim);
        for value in row_bytes.chunks_exact(std::mem::size_of::<f32>()) {
            row.push(f32::from_le_bytes([value[0], value[1], value[2], value[3]]));
        }
        embeddings.push(row);
    }
    eprintln!(
        "loaded cached normalized {} embeddings from {}",
        kind.label(),
        path.display()
    );
    Ok(Some(CachedEmbeddingRead {
        embeddings,
        seconds: seconds(started.elapsed()),
    }))
}

fn write_embedding_cache(path: &Path, kind: EmbeddingKind, embeddings: &[Vec<f32>]) -> Result<()> {
    let (rows, dim) = embedding_matrix_shape(kind, embeddings)?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    let mut temp_file = Builder::new()
        .prefix(".embedding-")
        .suffix(".tmp")
        .tempfile_in(parent)
        .with_context(|| format!("creating temporary embedding cache in {}", parent.display()))?;
    let temp_path = temp_file.path().to_path_buf();
    temp_file.write_all(EMBEDDING_CACHE_MAGIC)?;
    temp_file.write_all(&(rows as u64).to_le_bytes())?;
    temp_file.write_all(&(dim as u64).to_le_bytes())?;
    for row in embeddings {
        for value in row {
            temp_file.write_all(&value.to_le_bytes())?;
        }
    }
    temp_file
        .as_file_mut()
        .sync_all()
        .with_context(|| format!("syncing {}", temp_path.display()))?;
    let temp_path_guard = temp_file.into_temp_path();
    temp_path_guard.persist(path).map_err(|error| {
        anyhow!(
            "persisting {} to {}: {error}",
            temp_path.display(),
            path.display()
        )
    })?;
    eprintln!(
        "wrote normalized {} embedding cache to {}",
        kind.label(),
        path.display()
    );
    Ok(())
}

fn embedding_matrix_shape(kind: EmbeddingKind, embeddings: &[Vec<f32>]) -> Result<(usize, usize)> {
    let rows = embeddings.len();
    let first = embeddings
        .first()
        .with_context(|| format!("{} embeddings are empty", kind.label()))?;
    let dim = first.len();
    if dim == 0 {
        bail!("{} embeddings have zero dimension", kind.label());
    }
    if !embeddings.iter().all(|row| row.len() == dim) {
        bail!("{} embeddings have inconsistent dimensions", kind.label());
    }
    Ok((rows, dim))
}

fn read_u64_le(file: &mut File, path: &Path) -> Result<u64> {
    let mut bytes = [0u8; 8];
    file.read_exact(&mut bytes)
        .with_context(|| format!("reading {}", path.display()))?;
    Ok(u64::from_le_bytes(bytes))
}

fn embedding_input_hash(kind: EmbeddingKind, ids: &[String], texts: &[String]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(kind.label().as_bytes());
    hasher.update([0]);
    if matches!(kind, EmbeddingKind::Query) {
        hasher.update(HARRIER_QUERY_PREFIX.as_bytes());
        hasher.update([0]);
    }
    for (id, text) in ids.iter().zip(texts.iter()) {
        hasher.update(id.as_bytes());
        hasher.update([0]);
        hasher.update(text.as_bytes());
        hasher.update([0]);
    }
    short_hash(&format!("{:x}", hasher.finalize()))
}

fn short_hash(hash: &str) -> String {
    hash.chars().take(16).collect()
}

fn cache_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn parse_bits(value: &str) -> Result<Vec<u8>> {
    let bits = value
        .split(|ch: char| ch == ',' || ch.is_whitespace())
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<u8>()
                .with_context(|| format!("invalid bit width {part:?}"))
        })
        .collect::<Result<Vec<_>>>()?;
    if bits.is_empty() {
        bail!("--bits must contain at least one bit width");
    }
    for bit_width in &bits {
        if !matches!(*bit_width, 1 | 2 | 4) {
            bail!("unsupported bit width {bit_width}; expected one of 1, 2, or 4");
        }
    }
    Ok(bits)
}

fn load_dataset(args: &Args) -> Result<BenchmarkDataset> {
    let (dataset_dir, archive_verification) = ensure_beir_dataset(args)?;
    let corpus_path = dataset_dir.join("corpus.jsonl");
    let queries_path = dataset_dir.join("queries.jsonl");
    let qrels_path = dataset_dir
        .join("qrels")
        .join(format!("{}.tsv", args.split));

    let (doc_ids, doc_texts) = read_corpus(&corpus_path, args.max_docs)?;
    let selected_docs = doc_ids.iter().cloned().collect::<HashSet<_>>();
    let mut qrels = read_qrels(&qrels_path, &selected_docs)?;
    let (query_ids, query_texts) = read_queries(&queries_path, &qrels, args.max_queries)?;
    let selected_queries = query_ids.iter().map(String::as_str).collect::<HashSet<_>>();
    qrels.retain(|query_id, _| selected_queries.contains(query_id.as_str()));

    if doc_ids.is_empty() {
        bail!("{} produced no benchmark documents", args.dataset);
    }
    if query_ids.is_empty() {
        bail!(
            "{} produced no qrels-bearing benchmark queries",
            args.dataset
        );
    }

    Ok(BenchmarkDataset {
        doc_ids,
        doc_texts,
        query_ids,
        query_texts,
        qrels,
        archive_verification,
    })
}

fn ensure_beir_dataset(args: &Args) -> Result<(PathBuf, BeirArchiveVerification)> {
    let datasets_root = args.cache_dir.join("datasets");
    let downloads_root = args.cache_dir.join("downloads");
    let dataset_dir = datasets_root.join(&args.dataset);

    fs::create_dir_all(&datasets_root)?;
    fs::create_dir_all(&downloads_root)?;
    let zip_path = downloads_root.join(format!("{}.zip", args.dataset));
    if !zip_path.exists() || args.refresh_dataset {
        download_dataset_zip(&args.dataset, &zip_path, args.show_download_progress)?;
    }
    let archive_verification = match verify_beir_archive(&args.dataset, &zip_path) {
        Ok(verification) => verification,
        Err(error) => {
            remove_corrupt_cache_file(&zip_path);
            return Err(error).with_context(|| {
                format!(
                    "removing corrupt cached BEIR archive {}",
                    zip_path.display()
                )
            });
        }
    };
    if !args.refresh_dataset && beir_dataset_is_extracted(&dataset_dir, &args.split) {
        return Ok((dataset_dir, archive_verification));
    }
    extract_dataset_zip(&zip_path, &datasets_root, &args.dataset, &args.split)?;
    Ok((dataset_dir, archive_verification))
}

fn download_dataset_zip(dataset: &str, zip_path: &Path, show_progress: bool) -> Result<()> {
    let url = format!("{BEIR_BASE_URL}/{dataset}.zip");
    eprintln!("downloading BEIR dataset from {url}");
    download_to_path(&url, zip_path, show_progress)
}

fn download_to_path(url: &str, path: &Path, show_progress: bool) -> Result<()> {
    download_to_path_with_timeouts(url, path, show_progress, DOWNLOAD_TIMEOUTS)
}

fn download_to_path_with_timeouts(
    url: &str,
    path: &Path,
    show_progress: bool,
    timeouts: DownloadTimeouts,
) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let mut temp_file = Builder::new()
        .prefix(".download-")
        .suffix(".tmp")
        .tempfile_in(parent)?;
    let temp_path = temp_file.path().to_path_buf();
    let started = Instant::now();
    let response = download_agent(timeouts)
        .get(url)
        .call()
        .map_err(|err| anyhow!("downloading {url}: {err}"))?;
    let total_bytes = response
        .header("Content-Length")
        .and_then(|value| value.parse::<u64>().ok());
    let mut reader = response.into_reader();
    let mut buffer = [0u8; 1024 * 1024];
    let mut copied = 0u64;
    let mut next_progress = 64 * 1024 * 1024;
    loop {
        ensure_download_deadline(started, timeouts.overall, url)?;
        let read = reader
            .read(&mut buffer)
            .with_context(|| format!("reading {url}"))?;
        if read == 0 {
            break;
        }
        temp_file
            .as_file_mut()
            .write_all(&buffer[..read])
            .with_context(|| format!("writing {}", temp_path.display()))?;
        copied += read as u64;
        if show_progress && copied >= next_progress {
            match total_bytes {
                Some(total) => eprintln!("downloaded {copied}/{total} bytes from {url}"),
                None => eprintln!("downloaded {copied} bytes from {url}"),
            }
            next_progress += 64 * 1024 * 1024;
        }
    }
    temp_file.as_file_mut().flush()?;
    temp_file.as_file_mut().sync_all()?;
    if show_progress {
        match total_bytes {
            Some(total) => eprintln!("downloaded {copied}/{total} bytes from {url}"),
            None => eprintln!("downloaded {copied} bytes from {url}"),
        }
    }
    let temp_path_guard = temp_file.into_temp_path();
    temp_path_guard.persist(path).map_err(|error| {
        anyhow!(
            "persisting {} to {}: {error}",
            temp_path.display(),
            path.display()
        )
    })?;
    Ok(())
}

fn download_agent(timeouts: DownloadTimeouts) -> ureq::Agent {
    ureq::builder()
        .timeout_connect(timeouts.connect)
        .timeout_read(timeouts.read)
        .timeout(timeouts.overall)
        .build()
}

fn ensure_download_deadline(started: Instant, timeout: Duration, url: &str) -> Result<()> {
    if started.elapsed() > timeout {
        bail!(
            "downloading {url} exceeded overall timeout of {:.0} seconds",
            timeout.as_secs_f64()
        );
    }
    Ok(())
}

fn verify_beir_archive(dataset: &str, zip_path: &Path) -> Result<BeirArchiveVerification> {
    let expected = known_beir_archive(dataset).with_context(|| {
        format!("unsupported BEIR dataset {dataset:?}; cannot verify downloaded archive")
    })?;
    let actual_md5 = md5_file(zip_path)?;
    if actual_md5 != expected.md5 {
        bail!(
            "BEIR zip MD5 mismatch for {dataset}: expected {}, got {}",
            expected.md5,
            actual_md5
        );
    }
    let actual_sha256 = sha256_file(zip_path)?;
    if actual_sha256 != expected.sha256 {
        bail!(
            "BEIR zip SHA-256 mismatch for {dataset}: expected {}, got {}",
            expected.sha256,
            actual_sha256
        );
    }
    Ok(BeirArchiveVerification {
        md5: actual_md5,
        md5_verified: true,
        sha256: actual_sha256,
        sha256_verified: true,
    })
}

fn known_beir_archive(dataset: &str) -> Option<BeirArchiveSpec> {
    match dataset {
        "scifact" => Some(BeirArchiveSpec {
            md5: "5f7d1de60b170fc8027bb7898e2efca1",
            sha256: "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165",
        }),
        "fiqa" => Some(BeirArchiveSpec {
            md5: "17918ed23cd04fb15047f73e6c3bd9d9",
            sha256: "32c7df99ed21252fdfb2cf3f5673502a8d245ee0c44c4a133570d92ce2b3ad02",
        }),
        "scidocs" => Some(BeirArchiveSpec {
            md5: "38121350fc3a4d2f48850f6aff52e4a9",
            sha256: "96640201687767c9b1fcc5af7a80b90fb325b37fa25329c2586c25edcfa17ef1",
        }),
        "arguana" => Some(BeirArchiveSpec {
            md5: "8ad3e3c2a5867cdced806d6503f29b99",
            sha256: "cfdf79adce27a401b3cd3ea267903134dbfab2c6afeb95d7fe5724a00bf7557b",
        }),
        "nfcorpus" => Some(BeirArchiveSpec {
            md5: "a89dba18a62ef92f7d323ec890a0d38d",
            sha256: "efe5be03f8c5b86a5870102d0599d227c8c6e2484328e68c6522560385671b0b",
        }),
        "quora" => Some(BeirArchiveSpec {
            md5: "18fb154900ba42a600f84b839c173167",
            sha256: "56aacd9dcc4d9c093b63f175afdda0e21cbc8442ecf6c717d09de7b358d77531",
        }),
        _ => None,
    }
}

fn validate_dataset(dataset: &str) -> Result<()> {
    if known_beir_archive(dataset).is_none() {
        bail!(
            "unsupported BEIR dataset {dataset:?}; supported SHA-256-pinned BEIR datasets are scifact, fiqa, scidocs, arguana, nfcorpus, and quora"
        );
    }
    Ok(())
}

fn validate_split(split: &str) -> Result<()> {
    if split.is_empty()
        || split == "."
        || split == ".."
        || !split
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        bail!(
            "unsupported BEIR split {split:?}; split must be a safe filename stem using only ASCII letters, digits, '_' or '-'"
        );
    }
    Ok(())
}

fn checked_u32_len(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).with_context(|| format!("{label} value {value} exceeds u32::MAX"))
}

fn checked_i32_len(value: usize, label: &str) -> Result<i32> {
    i32::try_from(value).with_context(|| format!("{label} value {value} exceeds i32::MAX"))
}

fn ensure_requested_gpu_available(llama_gpu_layers: u32) -> Result<()> {
    if llama_gpu_layers == 0 {
        return Ok(());
    }
    if command_output("nvidia-smi", &["-L"]).is_none() {
        eprintln!(
            "warning: --llama-gpu-layers was set above zero, but nvidia-smi is unavailable or reported no NVIDIA GPU; continuing so llama.cpp can use CUDA if the runtime exposes it"
        );
    }
    Ok(())
}

fn extract_dataset_zip(
    zip_path: &Path,
    datasets_root: &Path,
    dataset: &str,
    split: &str,
) -> Result<()> {
    fs::create_dir_all(datasets_root)?;
    let temp_dir = Builder::new()
        .prefix(&format!(".{dataset}."))
        .tempdir_in(datasets_root)?;
    let temp_root = temp_dir.path().to_path_buf();
    let target = datasets_root.join(dataset);

    let result = (|| -> Result<()> {
        let file = File::open(zip_path)?;
        let mut archive = ZipArchive::new(file)?;
        for index in 0..archive.len() {
            let mut entry = archive.by_index(index)?;
            let Some(name) = entry.enclosed_name() else {
                continue;
            };
            let out_path = temp_root.join(name);
            if entry.is_dir() {
                fs::create_dir_all(&out_path)?;
            } else {
                if let Some(parent) = out_path.parent() {
                    fs::create_dir_all(parent)?;
                }
                let mut out = File::create(&out_path)?;
                std::io::copy(&mut entry, &mut out)?;
            }
        }

        let extracted = temp_root.join(dataset);
        if !extracted.join("corpus.jsonl").exists() {
            bail!("BEIR zip did not contain expected {dataset}/corpus.jsonl");
        }
        let split_qrels = extracted.join("qrels").join(format!("{split}.tsv"));
        if !split_qrels.exists() {
            bail!(
                "BEIR zip did not contain requested qrels split {}",
                split_qrels.display()
            );
        }
        replace_dataset_dir(&target, &extracted, datasets_root, dataset)?;
        Ok(())
    })();
    result
}

fn beir_dataset_is_extracted(dataset_dir: &Path, split: &str) -> bool {
    dataset_dir.join("corpus.jsonl").is_file()
        && dataset_dir.join("queries.jsonl").is_file()
        && dataset_dir
            .join("qrels")
            .join(format!("{split}.tsv"))
            .is_file()
}

fn remove_corrupt_cache_file(path: &Path) {
    if let Err(error) = fs::remove_file(path) {
        if error.kind() != std::io::ErrorKind::NotFound {
            eprintln!(
                "warning: failed to remove corrupt cached artifact {}: {error}",
                path.display()
            );
        }
    }
}

fn replace_dataset_dir(
    target: &Path,
    extracted: &Path,
    datasets_root: &Path,
    dataset: &str,
) -> Result<()> {
    let mut backup_dir = None;
    if target.exists() {
        if !target.is_dir() {
            bail!(
                "cannot replace non-directory dataset cache {}",
                target.display()
            );
        }
        let backup_guard = Builder::new()
            .prefix(&format!(".{dataset}.backup."))
            .tempdir_in(datasets_root)?;
        let backup_path = backup_guard.path().to_path_buf();
        fs::remove_dir_all(&backup_path)?;
        fs::rename(target, &backup_path).with_context(|| {
            format!(
                "moving existing dataset cache {} to {}",
                target.display(),
                backup_path.display()
            )
        })?;
        backup_dir = Some((backup_guard, backup_path));
    }

    match fs::rename(extracted, target) {
        Ok(()) => Ok(()),
        Err(error) => {
            if let Some((_guard, backup_path)) = &backup_dir {
                if !target.exists() {
                    let _ = fs::rename(backup_path, target);
                }
            }
            Err(error).with_context(|| {
                format!(
                    "moving extracted dataset {} to {}",
                    extracted.display(),
                    target.display()
                )
            })
        }
    }
}

fn read_corpus(path: &Path, max_docs: Option<usize>) -> Result<(Vec<String>, Vec<String>)> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut ids = Vec::new();
    let mut texts = Vec::new();
    for line in reader.lines() {
        if max_docs.is_some_and(|limit| ids.len() >= limit) {
            break;
        }
        let row: BeirTextRow = serde_json::from_str(&line?)?;
        let text = joined_text(&row.title, &row.text);
        if text.is_empty() {
            continue;
        }
        ids.push(row.id);
        texts.push(text);
    }
    Ok((ids, texts))
}

fn read_queries(
    path: &Path,
    qrels: &HashMap<String, HashMap<String, f64>>,
    max_queries: Option<usize>,
) -> Result<(Vec<String>, Vec<String>)> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut ids = Vec::new();
    let mut texts = Vec::new();
    for line in reader.lines() {
        if max_queries.is_some_and(|limit| ids.len() >= limit) {
            break;
        }
        let row: BeirTextRow = serde_json::from_str(&line?)?;
        if !qrels.contains_key(&row.id) {
            continue;
        }
        let text = row.text.trim().to_string();
        if text.is_empty() {
            continue;
        }
        ids.push(row.id);
        texts.push(text);
    }
    Ok((ids, texts))
}

fn read_qrels(
    path: &Path,
    selected_docs: &HashSet<String>,
) -> Result<HashMap<String, HashMap<String, f64>>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut qrels: HashMap<String, HashMap<String, f64>> = HashMap::new();
    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        if line_no == 0 && line.contains("query-id") {
            continue;
        }
        let parts = line.split_whitespace().collect::<Vec<_>>();
        if parts.len() < 3 {
            continue;
        }
        let query_id = parts[0].to_string();
        let doc_id = parts[1].to_string();
        let relevance = parts[2].parse::<f64>().with_context(|| {
            format!(
                "parsing relevance score on {} line {}",
                path.display(),
                line_no + 1
            )
        })?;
        if relevance <= 0.0 || !selected_docs.contains(&doc_id) {
            continue;
        }
        qrels.entry(query_id).or_default().insert(doc_id, relevance);
    }
    Ok(qrels)
}

fn joined_text(title: &str, text: &str) -> String {
    let title = title.trim();
    let text = text.trim();
    match (title.is_empty(), text.is_empty()) {
        (false, false) => format!("{title}\n{text}"),
        (false, true) => title.to_string(),
        (true, false) => text.to_string(),
        (true, true) => String::new(),
    }
}

fn normalize_rows(rows: &mut [Vec<f32>]) {
    for row in rows {
        let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in row {
                *value /= norm;
            }
        }
    }
}

fn embedding_dim(docs: &[Vec<f32>], queries: &[Vec<f32>]) -> Result<usize> {
    let dim = docs
        .first()
        .map(|row| row.len())
        .context("empty document embeddings")?;
    if dim == 0 {
        bail!("embedding dimension is zero");
    }
    if docs.iter().any(|row| row.len() != dim) || queries.iter().any(|row| row.len() != dim) {
        bail!("embedding rows have inconsistent dimensions");
    }
    Ok(dim)
}

fn flatten(rows: &[Vec<f32>]) -> Vec<f32> {
    let total = rows.iter().map(Vec::len).sum();
    let mut out = Vec::with_capacity(total);
    for row in rows {
        out.extend_from_slice(row);
    }
    out
}

fn exact_float_topk(queries: &[Vec<f32>], docs: &[Vec<f32>], k: usize) -> Vec<Vec<usize>> {
    let effective_k = k.min(docs.len());
    queries
        .iter()
        .map(|query| {
            let mut scores = docs
                .iter()
                .enumerate()
                .map(|(idx, doc)| (dot(query, doc), idx))
                .collect::<Vec<_>>();
            if effective_k < scores.len() {
                scores
                    .select_nth_unstable_by(effective_k, |left, right| right.0.total_cmp(&left.0));
                scores.truncate(effective_k);
            }
            scores.sort_unstable_by(|left, right| right.0.total_cmp(&left.0));
            scores.into_iter().map(|(_, idx)| idx).collect()
        })
        .collect()
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn indices_to_doc_ids(rows: &[Vec<usize>], doc_ids: &[String]) -> Vec<Vec<String>> {
    rows.iter()
        .map(|row| row.iter().map(|idx| doc_ids[*idx].clone()).collect())
        .collect()
}

fn mapped_ids_to_doc_ids(
    ids: &[u64],
    doc_ids: &[String],
    query_count: usize,
    effective_k: usize,
) -> Result<Vec<Vec<String>>> {
    if effective_k == 0 {
        if ids.is_empty() {
            return Ok(vec![Vec::new(); query_count]);
        }
        bail!(
            "OrdinalDB returned {} mapped ids for effective_k=0; cannot preserve query boundaries",
            ids.len()
        );
    }
    let expected = query_count
        .checked_mul(effective_k)
        .context("benchmark result shape overflow")?;
    if ids.len() != expected {
        bail!(
            "OrdinalDB returned {} mapped ids, expected {query_count} queries * effective_k {effective_k} = {expected}; refusing to chunk ambiguous results",
            ids.len()
        );
    }
    ids.chunks_exact(effective_k)
        .map(|row| {
            row.iter()
                .map(|id| {
                    let idx = id
                        .checked_sub(1)
                        .and_then(|id| usize::try_from(id).ok())
                        .with_context(|| format!("OrdinalDB returned invalid row id {id}"))?;
                    doc_ids
                        .get(idx)
                        .cloned()
                        .with_context(|| format!("OrdinalDB returned out-of-range row id {id}"))
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()
}

fn evaluate_run(
    run: &[Vec<String>],
    query_ids: &[String],
    qrels: &HashMap<String, HashMap<String, f64>>,
    k: usize,
) -> serde_json::Value {
    let mut recalls = Vec::new();
    let mut ndcgs = Vec::new();
    let mut mrrs = Vec::new();
    for (query_id, retrieved) in query_ids.iter().zip(run) {
        let Some(rels) = qrels.get(query_id) else {
            continue;
        };
        let relevant = rels
            .iter()
            .filter_map(|(doc_id, rel)| (*rel > 0.0).then_some(doc_id.clone()))
            .collect::<HashSet<_>>();
        if relevant.is_empty() {
            continue;
        }
        let top = retrieved.iter().take(k).cloned().collect::<Vec<_>>();
        let hits = top
            .iter()
            .filter(|doc_id| relevant.contains(*doc_id))
            .count();
        recalls.push(hits as f64 / relevant.len() as f64);
        ndcgs.push(ndcg_at_k(&top, rels, k));
        mrrs.push(reciprocal_rank(&top, &relevant));
    }
    json!({
        format!("recall@{k}"): mean(&recalls),
        format!("ndcg@{k}"): mean(&ndcgs),
        format!("mrr@{k}"): mean(&mrrs),
        "evaluated_queries": recalls.len(),
    })
}

fn ndcg_at_k(retrieved: &[String], rels: &HashMap<String, f64>, k: usize) -> f64 {
    let dcg = retrieved
        .iter()
        .take(k)
        .enumerate()
        .map(|(rank, doc_id)| {
            let rel = rels.get(doc_id).copied().unwrap_or(0.0);
            (2.0_f64.powf(rel) - 1.0) / ((rank + 2) as f64).log2()
        })
        .sum::<f64>();
    let mut ideal_rels = rels
        .values()
        .copied()
        .filter(|rel| *rel > 0.0)
        .collect::<Vec<_>>();
    ideal_rels.sort_by(|left, right| right.total_cmp(left));
    let ideal = ideal_rels
        .into_iter()
        .take(k)
        .enumerate()
        .map(|(rank, rel)| (2.0_f64.powf(rel) - 1.0) / ((rank + 2) as f64).log2())
        .sum::<f64>();
    safe_div(dcg, ideal)
}

fn reciprocal_rank(retrieved: &[String], relevant: &HashSet<String>) -> f64 {
    retrieved
        .iter()
        .position(|doc_id| relevant.contains(doc_id))
        .map(|idx| 1.0 / (idx + 1) as f64)
        .unwrap_or(0.0)
}

fn mean_topk_overlap(left: &[Vec<String>], right: &[Vec<String>], k: usize) -> f64 {
    let values = left
        .iter()
        .zip(right)
        .map(|(left_row, right_row)| {
            let right_set = right_row.iter().take(k).collect::<HashSet<_>>();
            let overlap = left_row
                .iter()
                .take(k)
                .filter(|doc_id| right_set.contains(*doc_id))
                .count();
            overlap as f64 / k.max(1) as f64
        })
        .collect::<Vec<_>>();
    mean(&values)
}

fn directory_size(path: &Path) -> Result<u64> {
    let metadata = fs::metadata(path)?;
    if metadata.is_file() {
        return Ok(metadata.len());
    }
    if !metadata.is_dir() {
        return Ok(0);
    }
    let mut total = 0;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if metadata.is_dir() {
            total += directory_size(&entry.path())?;
        } else {
            total += metadata.len();
        }
    }
    Ok(total)
}

fn md5_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut context = md5::Context::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        context.consume(&buffer[..read]);
    }
    Ok(format!("{:x}", context.compute()))
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn hardware_report() -> serde_json::Value {
    json!({
        "nvidia_smi": command_output("nvidia-smi", &[
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader"
        ]),
        "cuda_compiler": command_output("nvcc", &["--version"]),
        "rustc": command_output("rustc", &["--version"]),
    })
}

fn source_report() -> serde_json::Value {
    json!({
        "git_commit": command_output("git", &["rev-parse", "HEAD"]),
        "git_branch": command_output("git", &["branch", "--show-current"]),
        "git_dirty": git_dirty(),
    })
}

fn git_dirty() -> Option<bool> {
    let output = Command::new("git")
        .args(["status", "--short"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    Some(!output.stdout.is_empty())
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{}{}", stdout.trim(), stderr.trim());
    (!combined.is_empty()).then_some(combined)
}

fn repeat_timed<T, F>(repetitions: usize, mut run: F) -> Result<(T, Vec<f64>)>
where
    F: FnMut() -> Result<T>,
{
    if repetitions == 0 {
        bail!("repetitions must be greater than zero");
    }
    let mut samples = Vec::with_capacity(repetitions);
    let mut last = None;
    for _ in 0..repetitions {
        let started = Instant::now();
        let value = run()?;
        samples.push(seconds(started.elapsed()));
        last = Some(value);
    }
    Ok((last.context("timed run produced no result")?, samples))
}

fn latency_report(samples: &[f64], query_count: usize) -> serde_json::Value {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mean_seconds = mean(samples);
    json!({
        "samples": samples.len(),
        "batch_seconds": {
            "mean": mean_seconds,
            "min": *sorted.first().unwrap_or(&0.0),
            "p50": percentile_sorted(&sorted, 0.50),
            "p95": percentile_sorted(&sorted, 0.95),
            "max": *sorted.last().unwrap_or(&0.0),
        },
        "milliseconds_per_query": {
            "mean": 1000.0 * safe_div(mean_seconds, query_count as f64),
            "min": 1000.0 * safe_div(*sorted.first().unwrap_or(&0.0), query_count as f64),
            "p50": 1000.0 * safe_div(percentile_sorted(&sorted, 0.50), query_count as f64),
            "p95": 1000.0 * safe_div(percentile_sorted(&sorted, 0.95), query_count as f64),
            "max": 1000.0 * safe_div(*sorted.last().unwrap_or(&0.0), query_count as f64),
        },
    })
}

fn percentile_sorted(sorted: &[f64], percentile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = percentile.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let fraction = rank - lower as f64;
        sorted[lower] + (sorted[upper] - sorted[lower]) * fraction
    }
}

fn seconds(duration: Duration) -> f64 {
    duration.as_secs_f64()
}

fn safe_div(numerator: f64, denominator: f64) -> f64 {
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn mean(values: &[f64]) -> f64 {
    safe_div(values.iter().sum(), values.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread;

    #[test]
    fn exact_float_topk_zero_k_returns_empty_rows() {
        let queries = vec![vec![1.0, 0.0]];
        let docs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let rows = exact_float_topk(&queries, &docs, 0);

        assert_eq!(rows, vec![Vec::<usize>::new()]);
    }

    #[test]
    fn zero_max_docs_and_queries_read_no_rows() -> Result<()> {
        let temp = tempdir()?;
        let corpus_path = temp.path().join("corpus.jsonl");
        fs::write(
            &corpus_path,
            r#"{"_id":"d1","title":"title","text":"body"}"#.to_string() + "\n",
        )?;

        let (doc_ids, doc_texts) = read_corpus(&corpus_path, Some(0))?;
        assert!(doc_ids.is_empty());
        assert!(doc_texts.is_empty());

        let queries_path = temp.path().join("queries.jsonl");
        fs::write(
            &queries_path,
            r#"{"_id":"q1","text":"query"}"#.to_string() + "\n",
        )?;
        let qrels = HashMap::from([("q1".to_string(), HashMap::from([("d1".to_string(), 1.0)]))]);

        let (query_ids, query_texts) = read_queries(&queries_path, &qrels, Some(0))?;
        assert!(query_ids.is_empty());
        assert!(query_texts.is_empty());

        Ok(())
    }

    #[test]
    fn read_qrels_keeps_only_positive_selected_docs() -> Result<()> {
        let temp = tempdir()?;
        let qrels_path = temp.path().join("qrels.tsv");
        fs::write(
            &qrels_path,
            "query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td2\t1\nq2\td1\t0\nq3\td1\t2\n",
        )?;
        let selected_docs = HashSet::from(["d1".to_string()]);

        let qrels = read_qrels(&qrels_path, &selected_docs)?;

        assert_eq!(qrels.len(), 2);
        assert_eq!(qrels["q1"].get("d1"), Some(&1.0));
        assert!(!qrels["q1"].contains_key("d2"));
        assert!(!qrels.contains_key("q2"));
        assert_eq!(qrels["q3"].get("d1"), Some(&2.0));

        Ok(())
    }

    #[test]
    fn read_qrels_errors_on_malformed_relevance() -> Result<()> {
        let temp = tempdir()?;
        let qrels_path = temp.path().join("qrels.tsv");
        fs::write(&qrels_path, "query-id\tcorpus-id\tscore\nq1\td1\tnope\n")?;
        let selected_docs = HashSet::from(["d1".to_string()]);

        let err = read_qrels(&qrels_path, &selected_docs).unwrap_err();
        let message = format!("{err:#}");
        assert!(message.contains("parsing relevance score"), "{message}");
        assert!(message.contains("line 2"), "{message}");

        Ok(())
    }

    #[test]
    fn mapped_ids_to_doc_ids_requires_complete_query_rows() {
        let doc_ids = vec!["d1".to_string(), "d2".to_string(), "d3".to_string()];
        let rows = mapped_ids_to_doc_ids(&[1, 2, 2, 3], &doc_ids, 2, 2).unwrap();
        assert_eq!(
            rows,
            vec![
                vec!["d1".to_string(), "d2".to_string()],
                vec!["d2".to_string(), "d3".to_string()],
            ]
        );

        let err = mapped_ids_to_doc_ids(&[1, 2, 3], &doc_ids, 2, 2).unwrap_err();
        assert!(err.to_string().contains("expected 2 queries"));

        let err = mapped_ids_to_doc_ids(&[4], &doc_ids, 1, 1).unwrap_err();
        assert!(err.to_string().contains("out-of-range row id"));
    }

    #[test]
    fn validate_split_rejects_path_components() {
        for split in ["test", "dev-1", "train_2026"] {
            validate_split(split).unwrap();
        }

        for split in ["", ".", "..", "../test", "qrels/test", "test.tsv", "te st"] {
            let err = validate_split(split).unwrap_err();
            assert!(err.to_string().contains("unsupported BEIR split"));
        }
    }

    #[test]
    fn checked_batch_size_conversions_reject_overflow() {
        assert_eq!(checked_u32_len(7, "--embed-batch-size").unwrap(), 7);
        assert_eq!(checked_i32_len(7, "seq").unwrap(), 7);

        if usize::BITS > 32 {
            let err = checked_u32_len(u32::MAX as usize + 1, "--embed-batch-size").unwrap_err();
            assert!(err.to_string().contains("u32::MAX"));
        }
        let err = checked_i32_len(i32::MAX as usize + 1, "seq").unwrap_err();
        assert!(err.to_string().contains("i32::MAX"));
    }

    #[test]
    fn extracted_dataset_check_requires_expected_beir_files() -> Result<()> {
        let temp = tempdir()?;
        let dataset = temp.path().join("scifact");

        assert!(!beir_dataset_is_extracted(&dataset, "test"));
        fs::create_dir_all(dataset.join("qrels"))?;
        fs::write(dataset.join("corpus.jsonl"), b"{}\n")?;
        fs::write(dataset.join("queries.jsonl"), b"{}\n")?;
        assert!(!beir_dataset_is_extracted(&dataset, "test"));
        fs::write(
            dataset.join("qrels").join("test.tsv"),
            b"query-id\tcorpus-id\tscore\n",
        )?;
        assert!(beir_dataset_is_extracted(&dataset, "test"));
        assert!(!beir_dataset_is_extracted(&dataset, "dev"));

        Ok(())
    }

    #[test]
    fn replace_dataset_dir_swaps_existing_cache() -> Result<()> {
        let temp = tempdir()?;
        let target = temp.path().join("scifact");
        let extracted = temp.path().join("new-scifact");
        fs::create_dir_all(&target)?;
        fs::write(target.join("old.txt"), b"old")?;
        fs::create_dir_all(&extracted)?;
        fs::write(extracted.join("new.txt"), b"new")?;

        replace_dataset_dir(&target, &extracted, temp.path(), "scifact")?;

        assert!(!extracted.exists());
        assert!(!target.join("old.txt").exists());
        assert_eq!(fs::read(target.join("new.txt"))?, b"new");
        Ok(())
    }

    #[test]
    fn embedding_cache_write_replaces_existing_file_without_leftover_temps() -> Result<()> {
        let temp = tempdir()?;
        let cache_path = temp.path().join("embeddings.f32bin");

        write_embedding_cache(
            &cache_path,
            EmbeddingKind::Document,
            &[vec![1.0, 0.0], vec![0.0, 1.0]],
        )?;
        write_embedding_cache(
            &cache_path,
            EmbeddingKind::Document,
            &[vec![0.5, 0.5], vec![0.25, 0.75]],
        )?;

        let cached = read_embedding_cache(&cache_path, EmbeddingKind::Document, 2)?
            .expect("cache should exist");
        assert_eq!(cached.embeddings, vec![vec![0.5, 0.5], vec![0.25, 0.75]]);

        let leftovers = fs::read_dir(temp.path())?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".embedding-")
            })
            .collect::<Vec<_>>();
        assert!(leftovers.is_empty(), "leftover temp cache files found");

        Ok(())
    }

    #[test]
    fn embedding_cache_rejects_unrepresentable_header_counts() -> Result<()> {
        let temp = tempdir()?;
        let cache_path = temp.path().join("bad-embeddings.f32bin");
        let mut bytes = Vec::new();
        bytes.extend_from_slice(EMBEDDING_CACHE_MAGIC);
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        fs::write(&cache_path, bytes)?;

        let err = read_embedding_cache(&cache_path, EmbeddingKind::Document, 1).unwrap_err();
        let message = format!("{err:#}");
        assert!(
            message.contains("does not fit this platform")
                || message.contains("row count")
                || message.contains("size mismatch"),
            "{message}"
        );

        Ok(())
    }

    #[test]
    fn download_to_path_times_out_on_stalled_body() -> Result<()> {
        let listener = TcpListener::bind(("127.0.0.1", 0))?;
        let url = format!("http://{}/stall", listener.local_addr()?);
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept test connection");
            let _ = stream.set_read_timeout(Some(Duration::from_secs(1)));
            let mut request = [0u8; 1024];
            let _ = stream.read(&mut request);
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 1024\r\n\r\n")
                .expect("write response headers");
            thread::sleep(Duration::from_millis(250));
        });

        let temp = tempdir()?;
        let output = temp.path().join("download.bin");
        let err = download_to_path_with_timeouts(
            &url,
            &output,
            false,
            DownloadTimeouts {
                connect: Duration::from_millis(50),
                read: Duration::from_millis(50),
                overall: Duration::from_millis(500),
            },
        )
        .expect_err("stalled response body should time out");

        handle.join().expect("test server should finish");
        assert!(!output.exists(), "partial download should not be persisted");
        let error = format!("{err:#}");
        assert!(
            error.contains("reading") || error.contains("timed out") || error.contains("timeout"),
            "unexpected error: {error}"
        );

        Ok(())
    }

    #[test]
    fn download_deadline_reports_overall_timeout() {
        let started = Instant::now();
        thread::sleep(Duration::from_millis(2));

        let err = ensure_download_deadline(
            started,
            Duration::from_nanos(1),
            "https://example.test/file",
        )
        .expect_err("expired deadline should fail");

        assert!(err.to_string().contains("overall timeout"));
    }
}
