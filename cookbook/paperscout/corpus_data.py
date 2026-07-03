"""Bundled default corpus for PaperScout.

This is the corpus `demo.py` indexes by default: 40 real, well-known papers
spanning cs.LG, cs.IR, and cs.DB, with short abstracts paraphrased from
memory (not verbatim quotes) and best-effort year/author-count metadata.
It's a deterministic, offline, no-network substitute for a live arXiv
fetch -- good enough to demonstrate discovery, metadata filtering, and
persistence without depending on arXiv being reachable. Pass `--live` to
`demo.py` to fetch fresh abstracts from the arXiv API instead.
"""

from __future__ import annotations

# Each row: (title, abstract, category, year, authors_count)
_RAW_PAPERS: list[tuple[str, str, str, int, int]] = [
    # ---------------------------------------------------------------- cs.LG
    ("Attention Is All You Need",
     "Introduces the Transformer, a sequence transduction architecture built "
     "entirely on self-attention, dispensing with recurrence and convolutions "
     "while achieving state-of-the-art machine translation quality with less "
     "training time.",
     "cs.LG", 2017, 8),
    ("Deep Residual Learning for Image Recognition",
     "Proposes residual connections that let very deep convolutional networks "
     "be trained by reformulating layers as learning residual functions, "
     "enabling networks over 100 layers deep to outperform shallower ones.",
     "cs.LG", 2015, 4),
    ("Generative Adversarial Networks",
     "Frames generative modeling as a two-player minimax game between a "
     "generator and a discriminator, showing the approach can produce sharp "
     "samples without explicit likelihood estimation.",
     "cs.LG", 2014, 8),
    ("Dropout: A Simple Way to Prevent Neural Networks from Overfitting",
     "Introduces dropout, randomly deactivating units during training as an "
     "implicit ensemble technique, substantially reducing overfitting across "
     "vision, speech, and text tasks.",
     "cs.LG", 2014, 5),
    ("LoRA: Low-Rank Adaptation of Large Language Models",
     "Freezes pretrained model weights and injects trainable low-rank "
     "decomposition matrices into attention layers, cutting the number of "
     "trainable parameters for fine-tuning by orders of magnitude.",
     "cs.LG", 2021, 6),
    ("Training Compute-Optimal Large Language Models",
     "Argues that most large language models of the era were "
     "undertrained relative to their parameter count, and derives scaling "
     "laws showing model size and training tokens should scale together.",
     "cs.LG", 2022, 20),
    ("Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
     "Shows that prompting a large language model with a few worked "
     "reasoning examples substantially improves its performance on "
     "arithmetic, commonsense, and symbolic reasoning tasks.",
     "cs.LG", 2022, 6),
    ("Direct Preference Optimization: Your Language Model Is Secretly a Reward Model",
     "Reformulates RLHF's reward modeling and RL step as a single "
     "classification-style loss over preference pairs, matching RLHF "
     "quality without a separate reward model or PPO loop.",
     "cs.LG", 2023, 6),
    ("QLoRA: Efficient Finetuning of Quantized LLMs",
     "Combines 4-bit quantization with low-rank adapters and paged "
     "optimizers to fine-tune very large language models on a single GPU "
     "while preserving full-precision fine-tuning quality.",
     "cs.LG", 2023, 4),
    ("Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
     "Introduces a selective state-space model with input-dependent "
     "parameters and a hardware-aware scan, matching Transformer quality "
     "on language modeling with linear-time sequence scaling.",
     "cs.LG", 2023, 2),
    ("DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
     "Introduces Group Relative Policy Optimization, a memory-efficient RL "
     "algorithm that forgoes a separate value network, and uses it to push "
     "open math reasoning performance close to closed frontier models.",
     "cs.LG", 2024, 13),
    ("Mixtral of Experts",
     "Presents a sparse mixture-of-experts language model that routes each "
     "token to 2 of 8 experts per layer, matching or beating a much larger "
     "dense model at a fraction of the active-parameter inference cost.",
     "cs.LG", 2024, 20),
    ("The Llama 3 Herd of Models",
     "Describes a family of open foundation language models trained on "
     "over 15 trillion tokens with multilingual, coding, reasoning, and "
     "tool-use capability, including a 405B-parameter flagship model.",
     "cs.LG", 2024, 40),
    ("DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning",
     "Shows that large-scale reinforcement learning alone, without an "
     "initial supervised fine-tuning stage, can elicit strong chain-of-"
     "thought reasoning in a base language model.",
     "cs.LG", 2025, 60),

    # ---------------------------------------------------------------- cs.IR
    ("The Probabilistic Relevance Framework: BM25 and Beyond",
     "Surveys the probabilistic relevance framework underlying the BM25 "
     "ranking function, tracing its theoretical motivation and its "
     "extensions for structured and multi-field document retrieval.",
     "cs.IR", 2009, 2),
    ("Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs",
     "Introduces HNSW, a graph-based approximate nearest-neighbor index "
     "using layered proximity graphs, offering strong recall-latency "
     "tradeoffs without a training phase.",
     "cs.IR", 2016, 2),
    ("Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
     "Fine-tunes BERT in a siamese/triplet network structure to produce "
     "semantically meaningful sentence embeddings that can be compared "
     "with cosine similarity, dramatically speeding up sentence-pair tasks.",
     "cs.IR", 2019, 2),
    ("Dense Passage Retrieval for Open-Domain Question Answering",
     "Shows that dense embeddings learned with a dual-encoder trained on a "
     "small number of question-passage pairs can substantially outperform "
     "traditional sparse retrieval like BM25 for open-domain QA.",
     "cs.IR", 2020, 6),
    ("ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT",
     "Introduces late interaction, encoding queries and documents "
     "independently into token-level vectors and deferring fine-grained "
     "matching to query time, balancing BERT quality with retrieval speed.",
     "cs.IR", 2020, 2),
    ("Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
     "Combines a pretrained parametric seq2seq model with a non-parametric "
     "dense retrieval index over Wikipedia, generating answers conditioned "
     "on retrieved passages for knowledge-intensive tasks.",
     "cs.IR", 2020, 12),
    ("ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction",
     "Improves late-interaction retrieval with denoised supervision and "
     "residual vector compression, cutting the index footprint while "
     "maintaining state-of-the-art retrieval quality.",
     "cs.IR", 2021, 4),
    ("SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking",
     "Learns sparse term-expansion representations regularized toward "
     "sparsity, combining the efficiency of inverted indexes with the "
     "effectiveness of learned term weighting and expansion.",
     "cs.IR", 2021, 4),
    ("Matryoshka Representation Learning",
     "Trains embeddings so that prefixes of the vector are themselves "
     "useful lower-dimensional representations, letting a single model "
     "serve multiple embedding sizes without retraining.",
     "cs.IR", 2022, 10),
    ("MTEB: Massive Text Embedding Benchmark",
     "Introduces a benchmark spanning eight embedding task categories "
     "across many languages, standardizing comparison of text embedding "
     "models on retrieval, clustering, and classification.",
     "cs.IR", 2022, 5),
    ("Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
     "Trains a language model to adaptively decide when to retrieve, and "
     "to generate special reflection tokens that critique its own "
     "generations against retrieved evidence.",
     "cs.IR", 2023, 6),
    ("RAGAS: Automated Evaluation of Retrieval Augmented Generation",
     "Proposes a reference-free evaluation framework for RAG pipelines "
     "that scores faithfulness, answer relevance, and context relevance "
     "without requiring human-annotated ground truth.",
     "cs.IR", 2023, 4),
    ("From Local to Global: A Graph RAG Approach to Query-Focused Summarization",
     "Builds a hierarchical community-graph index over a corpus using an "
     "LLM to extract entities and relations, enabling query-focused "
     "summarization over themes that span many documents.",
     "cs.IR", 2024, 9),
    ("RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval",
     "Builds a tree of recursive text summaries over a corpus and "
     "retrieves from multiple tree levels, letting retrieval span both "
     "fine-grained details and higher-level themes.",
     "cs.IR", 2024, 4),

    # ---------------------------------------------------------------- cs.DB
    ("The Log-Structured Merge-Tree",
     "Introduces a data structure that defers and batches index updates "
     "in memory before merging them into on-disk components, trading "
     "read amplification for much higher write throughput.",
     "cs.DB", 1996, 3),
    ("Bigtable: A Distributed Storage System for Structured Data",
     "Describes a distributed storage system for managing structured data "
     "designed to scale to petabytes across thousands of commodity "
     "servers, underlying many large production services.",
     "cs.DB", 2006, 8),
    ("Dynamo: Amazon's Highly Available Key-Value Store",
     "Describes a highly available key-value storage system that "
     "prioritizes availability over consistency, using consistent hashing, "
     "vector clocks, and quorum-based replication.",
     "cs.DB", 2007, 8),
    ("Cassandra: A Decentralized Structured Storage System",
     "Presents a decentralized, eventually consistent structured storage "
     "system combining a Dynamo-style distributed design with a "
     "Bigtable-style column-family data model.",
     "cs.DB", 2010, 2),
    ("Spanner: Google's Globally-Distributed Database",
     "Presents a globally distributed database that supports externally "
     "consistent distributed transactions using synchronized clocks via a "
     "TrueTime API, spanning datacenters worldwide.",
     "cs.DB", 2012, 22),
    ("Calvin: Fast Distributed Transactions for Partitioned Database Systems",
     "Proposes deterministically ordering transactions before execution "
     "across partitions, removing the need for expensive distributed "
     "commit protocols while still supporting full ACID transactions.",
     "cs.DB", 2012, 4),
    ("DiskANN: Fast Accurate Billion-Point Nearest Neighbor Search on a Single Node",
     "Introduces a disk-resident graph index that supports billion-scale "
     "approximate nearest-neighbor search on a single machine with SSDs, "
     "using a compressed in-memory index for candidate generation.",
     "cs.DB", 2019, 6),
    ("DuckDB: An Embeddable Analytical Database",
     "Presents an embeddable, in-process analytical database designed "
     "like SQLite but optimized for OLAP workloads with a vectorized "
     "columnar execution engine.",
     "cs.DB", 2019, 2),
    ("Delta Lake: High-Performance ACID Table Storage over Cloud Object Stores",
     "Adds a transaction log layer over cloud object storage to provide "
     "ACID guarantees, schema enforcement, and time travel for large "
     "analytical tables stored as data files in object stores.",
     "cs.DB", 2020, 8),
    ("Milvus: A Purpose-Built Vector Data Management System",
     "Presents a distributed system purpose-built for managing and "
     "searching large-scale vector embeddings, supporting multiple index "
     "types and hybrid scalar-vector filtering.",
     "cs.DB", 2021, 9),
    ("A Comprehensive Survey on Vector Database: Storage and Retrieval Technique, Challenge",
     "Surveys the storage layouts, indexing algorithms, and query "
     "processing techniques used by vector database systems, and "
     "discusses open challenges in scaling and hybrid search.",
     "cs.DB", 2023, 5),
    ("Starling: An I/O-Efficient Disk-Resident Graph Index Framework for High-Dimensional Vector Similarity Search",
     "Proposes a disk-resident graph index layout and search algorithm "
     "that reduces I/O amplification for high-dimensional approximate "
     "nearest-neighbor search on datasets larger than memory.",
     "cs.DB", 2024, 5),
]


def load_bundled_corpus() -> list[dict]:
    """Return the bundled corpus as a list of paper metadata dicts."""
    papers = []
    for idx, (title, abstract, category, year, authors_count) in enumerate(_RAW_PAPERS, start=1):
        papers.append(
            {
                "id": f"bundled-{idx:03d}",
                "title": title,
                "abstract": abstract,
                "category": category,
                "year": year,
                "authors_count": authors_count,
                "source": "bundled_corpus",
            }
        )
    return papers
