//! The three demo query classes, paired with the corpus rows in
//! `src/corpus.rs` that make each one a genuine (not cherry-picked-after-
//! the-fact) test of hybrid search.

use crate::corpus;

pub struct DemoQuery {
    pub label: &'static str,
    pub text: &'static str,
    pub gold_id: u64,
    pub claim: &'static str,
}

pub fn demo_queries() -> Vec<DemoQuery> {
    vec![
        DemoQuery {
            label: "class 1: exact identifier (BM25 should win, dense should miss)",
            text: "ERR_POOL_EXHAUSTED_5432",
            gold_id: corpus::POOL_GOLD_ID,
            claim: "the one document containing this exact code, among five other \
                    near-duplicate connection-pool documents",
        },
        DemoQuery {
            label: "class 2: paraphrase (dense should win, BM25 should miss)",
            text: "can't log in after the cert change",
            gold_id: corpus::TLS_AUTH_GOLD_ID,
            claim: "a document about mTLS-certificate-rotation login failures that \
                    shares zero exact BM25 terms with this phrasing",
        },
        DemoQuery {
            label: "class 3: mixed identifier + semantics (RRF should win or match)",
            text: "seeing ERR_RATE_LIMIT_EXCEEDED spikes after upgrading to v3.4.0, how do I stop the throttling",
            gold_id: corpus::RATELIMIT_GOLD_ID,
            claim: "a document with moderate evidence in both signals, versus a \
                    changelog decoy that only wins BM25 and a generic-advice decoy \
                    that only wins dense",
        },
    ]
}
