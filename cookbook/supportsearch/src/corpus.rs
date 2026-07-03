//! A ~120-document support knowledge base for the fictional "Meridian" API
//! gateway/data platform: KB articles, support tickets, changelog notes, and
//! troubleshooting runbooks, salted with realistic exact identifiers (error
//! codes, config keys, CLI flags, version strings).
//!
//! Three topics are deliberately engineered to demonstrate hybrid search's
//! value (see `src/queries.rs` for the paired demo queries):
//!
//! - `pool` (rows 100000-100005): six near-duplicate connection-pool
//!   documents. Only one contains the exact code `ERR_POOL_EXHAUSTED_5432`;
//!   the rest discuss the same general topic with different codes/prose, so
//!   a dense embedding of the bare code string cannot cleanly separate the
//!   one right answer from its five look-alikes, while BM25's inverse
//!   document frequency on the rare token `5432` can.
//! - `tls_auth` (rows 100006-100011): the gold document describes
//!   certificate-rotation login failures using *zero* of the query's exact
//!   words (see the comment on its `d(TLS_AUTH_GOLD_ID, ...)` call below),
//!   so BM25 cannot find it by term overlap, while a semantic embedding of
//!   the paraphrase can.
//! - `ratelimit` (rows 100012-100017): a gold document that in practice
//!   wins both dense and BM25 outright (not just "moderately" -- an
//!   earlier draft of this corpus aimed for a gold doc that needed RRF to
//!   rescue it from a middling rank in both lists, but repeated corpus
//!   edits couldn't force that shape without contrivance; the honest,
//!   measured result is reported instead). It still demonstrates the other
//!   legitimate RRF property: when both signals already agree, fusion
//!   preserves that agreement instead of introducing noise. A BM25-only
//!   decoy (a changelog entry that name-drops the exact code and version
//!   with no real guidance) and a dense-leaning decoy (generic throttling
//!   advice) are included for contrast.
//!
//! The remaining ~17 topics exist to give the corpus real bulk and BM25/
//! dense-search competition beyond the three curated queries.

pub struct Doc {
    pub id: u64,
    pub title: &'static str,
    pub body: String,
    pub category: &'static str,
}

/// Row id of the one document that contains `ERR_POOL_EXHAUSTED_5432`.
pub const POOL_GOLD_ID: u64 = 100_000;
/// Row id of the certificate-rotation login-failure document (paraphrase gold).
pub const TLS_AUTH_GOLD_ID: u64 = 100_006;
/// Row id of the rate-limit doc with moderate dense+sparse evidence (RRF gold).
pub const RATELIMIT_GOLD_ID: u64 = 100_012;
/// Row id of the changelog decoy that only a BM25 exact-term match favors.
pub const RATELIMIT_BM25_DECOY_ID: u64 = 100_013;
/// Row id of the generic throttling article that only dense similarity favors.
pub const RATELIMIT_DENSE_DECOY_ID: u64 = 100_014;

pub(crate) fn d(id: u64, title: &'static str, body: impl Into<String>, category: &'static str) -> Doc {
    Doc {
        id,
        title,
        body: body.into(),
        category,
    }
}

pub fn corpus() -> Vec<Doc> {
    use crate::corpus_filler::*;

    let mut docs = Vec::new();
    docs.extend(pool_topic());
    docs.extend(tls_auth_topic());
    docs.extend(ratelimit_topic());
    docs.extend(webhook_topic());
    docs.extend(apikey_topic());
    docs.extend(migration_topic());
    docs.extend(backup_topic());
    docs.extend(cli_topic());
    docs.extend(config_topic());
    docs.extend(changelog_topic());
    docs.extend(sso_topic());
    docs.extend(cache_topic());
    docs.extend(billing_topic());
    docs.extend(logging_topic());
    docs.extend(grpc_topic());
    docs.extend(s3_topic());
    docs.extend(lb_topic());
    docs.extend(dns_topic());
    docs.extend(diskquota_topic());
    docs.extend(oom_topic());
    docs
}

// ---------------------------------------------------------------------
// Hero topic 1: connection pool exhaustion (exact-identifier query class)
// ---------------------------------------------------------------------

fn pool_topic() -> Vec<Doc> {
    vec![
        d(
            100_000,
            "Connection Pool Exhaustion: ERR_POOL_EXHAUSTED_5432",
            "When every connection in the Postgres pool on port 5432 is checked out \
             and a new request cannot acquire one before the deadline, Meridian raises \
             ERR_POOL_EXHAUSTED_5432. Raise pool.max_size in the gateway config or pass \
             --pool-timeout-ms 8000 on the CLI to give bursts more room. Check for \
             connection leaks first: a climbing checked-out count that never drops back \
             down after traffic subsides usually means a handler is not releasing \
             connections back to the pool.",
            "pool",
        ),
        d(
            100_001,
            "Tuning pool.max_size And pool.acquire_timeout_ms",
            "The two knobs that matter most for pool sizing are pool.max_size (the \
             ceiling on concurrent database connections) and pool.acquire_timeout_ms \
             (how long a caller waits for a free slot before failing). Undersized pools \
             fail fast under burst traffic; oversized pools can overwhelm the database \
             server itself. Start with max_size equal to (cpu_cores * 4) and adjust from \
             observed p99 checkout wait time.",
            "pool",
        ),
        d(
            100_002,
            "Ticket #4821: pool keeps timing out during nightly batch job",
            "Customer reports the nightly reconciliation batch fails around 2am with \
             repeated timeouts acquiring a database connection. Root cause: the batch \
             job and the live API traffic share one pool, and the batch opens hundreds \
             of long-running connections. Recommended fix: give the batch job its own \
             pool with a separate --max-conns setting instead of sharing the API's pool.",
            "pool",
        ),
        d(
            100_003,
            "Runbook: diagnosing a stuck connection pool",
            "1) Run `meridianctl pool stats` to see checked-out vs idle counts. 2) If \
             checked-out is pinned at pool.max_size with zero idle connections for more \
             than a minute, suspect a leak rather than legitimate load. 3) Enable \
             pool.leak_detection_ms to have the gateway log a stack trace for any \
             connection held longer than the threshold. 4) Restart the affected \
             replica only as a last resort; it does not fix the underlying leak.",
            "pool",
        ),
        d(
            100_004,
            "FAQ: what does WARN_POOL_NEAR_LIMIT_80 mean?",
            "WARN_POOL_NEAR_LIMIT_80 fires when checked-out connections cross 80% of \
             pool.max_size, well before the pool is actually full. It is an early \
             warning, not an outage: use it to scale pool.max_size or add read \
             replicas ahead of a real exhaustion event instead of waiting for one.",
            "pool",
        ),
        d(
            100_005,
            "Release note: pool acquire retries in v3.2.0",
            "v3.2.0 adds automatic retry-with-backoff when a connection checkout hits \
             ERR_POOL_TIMEOUT_5433, up to pool.acquire_retry_limit attempts. This \
             smooths over brief spikes but does not substitute for correctly sizing \
             pool.max_size for sustained load; a pool that is chronically undersized \
             will still exhaust and start rejecting checkouts.",
            "pool",
        ),
    ]
}

// ---------------------------------------------------------------------
// Hero topic 2: certificate rotation breaks sign-in (paraphrase query class)
// ---------------------------------------------------------------------

fn tls_auth_topic() -> Vec<Doc> {
    vec![
        // TLS_AUTH_GOLD_ID: deliberately avoids the standalone tokens
        // "log", "cert", "change", and "can" so it shares zero normalized
        // BM25 terms with the query "can't log in after the cert change".
        // ("certificate" and "rotation" are distinct tokens from "cert" and
        // "change" under the tokenizer's ASCII-alnum-run splitting.)
        d(
            TLS_AUTH_GOLD_ID,
            "Authentication Failures Following mTLS Certificate Rotation",
            "After rotating the client-side TLS certificate used for service-to-service \
             authentication, some users are unable to sign in until the new certificate \
             is trusted by the identity provider. This usually shows up as a wave of 401 \
             responses from the authentication gateway immediately following a rotation \
             window. Confirm the new certificate's fingerprint is registered with the \
             identity provider before the previous one is retired, and stagger rotation \
             over multiple regions so a single provider outage cannot lock out every \
             user at once.",
            "tls_auth",
        ),
        d(
            100_007,
            "Ticket #5190: users locked out right after we rotated our client cert",
            "Support ticket from a customer: their signed-in sessions were fine, but \
             every new sign-in attempt started failing right after they rotated the \
             mTLS client certificate on their side. Diagnosis: their identity provider \
             had not yet trusted the new certificate's fingerprint, so the handshake \
             fell back and the gateway rejected the session as unauthenticated.",
            "tls_auth",
        ),
        d(
            100_008,
            "ERR_MTLS_HANDSHAKE_FAILED reference",
            "ERR_MTLS_HANDSHAKE_FAILED is raised when the TLS handshake between a \
             client and the gateway cannot agree on a mutually trusted certificate \
             chain. Common causes: an expired leaf certificate, a certificate signed by \
             an untrusted intermediate, or a client presenting the old certificate after \
             the gateway has already rotated to a new certificate authority.",
            "tls_auth",
        ),
        d(
            100_009,
            "Console: web login page shows an outdated security certificate warning",
            "Unrelated to the API gateway: this is about the operator console's own web \
             login page occasionally showing a browser certificate-authority warning \
             banner right after a routine renewal, purely cosmetic and cleared by a hard \
             refresh. Does not affect API sign-in or the authentication gateway.",
            "tls_auth",
        ),
        d(
            100_010,
            "Audit log entries for authentication events",
            "Every authentication attempt against the gateway is written to the audit \
             log with the outcome, the client identifier, and (on failure) a reason \
             code such as ERR_AUTH_CERT_EXPIRED_401. Query the audit log by time window \
             around a suspected incident to see whether failures cluster right after a \
             deploy or a certificate change on either side of the connection.",
            "tls_auth",
        ),
        d(
            100_011,
            "Runbook: rotating the gateway's own TLS certificate with zero downtime",
            "Load the new certificate alongside the old one via tls.additional_trust, \
             wait for all long-lived client connections to naturally cycle, then remove \
             the old certificate from the trust bundle. Skipping the overlap window is \
             the single most common cause of a rotation-triggered outage.",
            "tls_auth",
        ),
    ]
}

// ---------------------------------------------------------------------
// Hero topic 3: rate limiting (RRF fusion query class)
// ---------------------------------------------------------------------

fn ratelimit_topic() -> Vec<Doc> {
    vec![
        d(
            RATELIMIT_GOLD_ID,
            "Rate Limit Spikes After Upgrading To v3.4.0",
            "Several operators saw a rise in ERR_RATE_LIMIT_EXCEEDED responses right \
             after upgrading to v3.4.0. v3.4.0 tightened the default token-bucket \
             refill rate to close a burst-abuse loophole. If your traffic is bursty by \
             design, raise ratelimit.burst_size and enable client-side backoff so \
             throttled requests retry with jitter instead of retrying immediately and \
             making the spike worse.",
            "ratelimit",
        ),
        d(
            RATELIMIT_BM25_DECOY_ID,
            "v3.4.0 changelog",
            "v3.4.0: fixed webhook retry double-delivery; ERR_RATE_LIMIT_EXCEEDED now \
             includes a Retry-After header; reduced idle memory footprint; upgraded \
             tokenizer dependency; fixed a race in ERR_RATE_LIMIT_EXCEEDED counter \
             resets; corrected a typo in the CLI --output-format help text; v3.4.0 also \
             bumps the bundled TLS library.",
            "ratelimit",
        ),
        d(
            RATELIMIT_DENSE_DECOY_ID,
            "How Do I Stop My Client From Being Throttled?",
            "If your client keeps getting throttled, the fix is almost never on the \
             server side: slow your own send rate, add exponential backoff with \
             jitter on every retry, and spread bursts out over time instead of \
             sending them all at once. Throttling is the service protecting itself \
             from a sudden spike, not a bug to work around; well-behaved clients that \
             back off gracefully stop seeing it almost entirely.",
            "ratelimit",
        ),
        d(
            100_015,
            "Ticket #6003: sudden 429s after our weekend deploy",
            "Customer reports a sudden increase in 429 responses coinciding with their \
             own weekend deploy, which happened to land the same week as the platform's \
             v3.4.0 rollout. Advised them to check whether their retry logic was already \
             tuned for the old, looser rate limit defaults and to add jitter.",
            "ratelimit",
        ),
        d(
            100_016,
            "Config reference: ratelimit.burst_size and ratelimit.refill_per_sec",
            "ratelimit.burst_size controls how many requests can arrive back-to-back \
             before throttling kicks in; ratelimit.refill_per_sec controls the steady-\
             state sustained rate afterward. Raising burst_size absorbs short spikes; \
             raising refill_per_sec raises the sustained ceiling and should be changed \
             more cautiously.",
            "ratelimit",
        ),
        d(
            100_017,
            "FAQ: does the rate limit apply per API key or per IP?",
            "Rate limiting is applied per API key by default. A shared IP (for \
             example, behind a corporate NAT) does not share a limit between different \
             API keys, and a single API key's limit follows it between source IPs.",
            "ratelimit",
        ),
    ]
}
