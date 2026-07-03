//! The ~17 non-hero topics from `src/corpus.rs`: realistic bulk KB content
//! (error codes, config keys, CLI flags, version strings) that gives BM25
//! and dense search real competition beyond the three curated demo
//! queries, split into its own file to keep `corpus.rs` itself focused on
//! the three hero topics the demo queries are engineered against.

use crate::corpus::{d, Doc};

pub(crate) fn webhook_topic() -> Vec<Doc> {
    vec![
        d(100_018, "Webhook Delivery Failures: ERR_WEBHOOK_DELIVERY_FAILED_502",
            "ERR_WEBHOOK_DELIVERY_FAILED_502 means the receiving endpoint returned a \
             502 during a webhook delivery attempt. Meridian retries with \
             webhook.retry_backoff_ms between attempts, up to webhook.max_retries.",
            "webhook"),
        d(100_019, "Runbook: replaying a dropped webhook",
            "Use `meridianctl webhook replay --event-id <id>` to resend a single event \
             that exhausted its retries. Replays are idempotent as long as the receiver \
             deduplicates on the event id header.",
            "webhook"),
        d(100_020, "Ticket #3312: webhooks arriving out of order",
            "Customer noticed events sometimes arrive out of the order they were \
             produced. Webhook delivery does not guarantee ordering between parallel \
             delivery workers; consumers needing strict order should sequence by the \
             event's own timestamp field, not arrival order.",
            "webhook"),
        d(100_021, "Config reference: webhook.retry_backoff_ms and webhook.max_retries",
            "webhook.retry_backoff_ms sets the base delay between delivery attempts \
             (doubled each retry); webhook.max_retries caps total attempts before an \
             event is marked failed and surfaced in the dashboard.",
            "webhook"),
        d(100_022, "FAQ: how do I verify a webhook payload's signature?",
            "Every webhook request includes an HMAC signature header computed over the \
             raw request body with your endpoint secret. Recompute it on receipt and \
             reject the request if the signatures do not match in constant time.",
            "webhook"),
        d(100_023, "Release note: webhook ordering headers in v3.1.0",
            "v3.1.0 adds an optional sequence header per webhook topic so consumers \
             that need ordering can detect and buffer out-of-order deliveries instead \
             of assuming arrival order is delivery order.",
            "webhook"),
    ]
}

pub(crate) fn apikey_topic() -> Vec<Doc> {
    vec![
        d(100_024, "API Key Revoked: ERR_APIKEY_REVOKED_403",
            "ERR_APIKEY_REVOKED_403 is returned for any request using a key that has \
             been explicitly revoked from the dashboard. Revocation takes effect within \
             a few seconds and is not reversible; issue a new key instead.",
            "apikey"),
        d(100_025, "Runbook: rotating an API key with zero downtime",
            "Issue a second key, deploy it alongside the first, confirm traffic is \
             flowing on the new key via the usage dashboard, then revoke the old key. \
             Do not revoke the old key until the new one shows nonzero traffic.",
            "apikey"),
        d(100_026, "Ticket #2207: key stopped working with no changes on our end",
            "Customer's key stopped authenticating with no code change on their side. \
             Cause: an automated policy revoked the key after it appeared in a public \
             GitHub commit; issued a replacement and pointed them at secret-scanning \
             docs.",
            "apikey"),
        d(100_027, "FAQ: do API keys expire automatically?",
            "Keys do not expire on a timer by default, but an organization admin can \
             set apikey.max_age_days to force periodic rotation for every key in the \
             account.",
            "apikey"),
        d(100_028, "Config reference: apikey.max_age_days and apikey.scopes",
            "apikey.max_age_days forces rotation after N days; apikey.scopes restricts \
             a key to a subset of endpoints. Scoping keys narrowly limits the blast \
             radius of an accidental leak.",
            "apikey"),
        d(100_029, "Release note: scoped keys in v2.9.12",
            "v2.9.12 introduces apikey.scopes so a single account can issue read-only \
             keys for reporting tools separately from write-capable keys used by \
             production services.",
            "apikey"),
    ]
}

pub(crate) fn migration_topic() -> Vec<Doc> {
    vec![
        d(100_030, "Migration Lock Timeout: ERR_MIGRATION_LOCK_TIMEOUT_55P03",
            "ERR_MIGRATION_LOCK_TIMEOUT_55P03 mirrors Postgres's own lock_not_available \
             error and means a schema migration could not acquire the table lock \
             within migration.lock_timeout_ms, usually because a long-running query is \
             still holding it.",
            "migration"),
        d(100_031, "Runbook: finding what is blocking a migration",
            "Query pg_locks joined against pg_stat_activity for the target table to \
             find the blocking session, then decide whether to wait it out or terminate \
             it. Never terminate a session you do not recognize without checking first.",
            "migration"),
        d(100_032, "Ticket #7741: migration hung for twenty minutes then failed",
            "Customer's migration hung, then failed with a lock timeout. A long \
             analytics query against the same table had been running since before the \
             migration started and never released its lock in time.",
            "migration"),
        d(100_033, "Config reference: migration.lock_timeout_ms",
            "migration.lock_timeout_ms bounds how long a migration waits for a \
             conflicting lock before giving up and rolling back cleanly instead of \
             blocking indefinitely.",
            "migration"),
        d(100_034, "FAQ: are migrations transactional?",
            "Yes, each migration runs inside a single transaction and rolls back \
             entirely on any failure, including a lock timeout; there is no partially \
             applied migration state to clean up by hand.",
            "migration"),
        d(100_035, "Release note: online index builds in v3.0.0",
            "v3.0.0 adds support for CONCURRENTLY index builds in migrations so large \
             tables no longer need a maintenance window just to add an index.",
            "migration"),
    ]
}

pub(crate) fn backup_topic() -> Vec<Doc> {
    vec![
        d(100_036, "Stale Backup Snapshot: ERR_BACKUP_SNAPSHOT_STALE",
            "ERR_BACKUP_SNAPSHOT_STALE fires when a restore is attempted from a \
             snapshot older than backup.max_snapshot_age_hours, a guardrail against \
             accidentally restoring very old data without an explicit override flag.",
            "backup"),
        d(100_037, "Runbook: restoring from a snapshot",
            "1) List available snapshots with `meridianctl backup list`. 2) Verify the \
             snapshot's checksum. 3) Restore to a scratch environment first and smoke-\
             test before pointing production traffic at the restored data.",
            "backup"),
        d(100_038, "Ticket #1180: restore rejected as too old",
            "Customer tried to restore a six-month-old snapshot and hit \
             ERR_BACKUP_SNAPSHOT_STALE. Confirmed they actually wanted a recent \
             snapshot and had grabbed the wrong one from the list.",
            "backup"),
        d(100_039, "Config reference: backup.max_snapshot_age_hours",
            "backup.max_snapshot_age_hours defaults to 720 (30 days); lower it if your \
             compliance policy requires tighter recency guarantees on any restore.",
            "backup"),
        d(100_040, "FAQ: how often are automatic backups taken?",
            "Automatic snapshots run hourly by default, controlled by \
             backup.snapshot_interval_minutes, with retention governed separately by \
             backup.retention_days.",
            "backup"),
        d(100_041, "Release note: cross-region backup replication in v3.3.0",
            "v3.3.0 adds optional replication of snapshots to a second region for \
             disaster recovery, configured via backup.replica_region.",
            "backup"),
    ]
}

pub(crate) fn cli_topic() -> Vec<Doc> {
    vec![
        d(100_042, "CLI Reference: meridianctl global flags",
            "Global flags available on every meridianctl subcommand: --config-path, \
             --output-format (text or json), --dry-run, and --verbose. --dry-run \
             applies to any command that would mutate state.",
            "cli"),
        d(100_043, "Runbook: scripting meridianctl with --output-format json",
            "Pass --output-format json to any read command to get stable, parseable \
             output suitable for piping into jq in CI pipelines instead of scraping \
             the human-readable text format.",
            "cli"),
        d(100_044, "Ticket #900: --dry-run did nothing visible",
            "Customer expected --dry-run to print a diff. It only suppresses the \
             mutation and prints a short summary line; pair it with --verbose for a \
             fuller preview of what would change.",
            "cli"),
        d(100_045, "FAQ: where does meridianctl read its config from?",
            "By default from ./meridian.toml in the current directory, overridable \
             with --config-path or the MERIDIAN_CONFIG environment variable.",
            "cli"),
        d(100_046, "Config reference: precedence between flags, env vars, and file",
            "Command-line flags win over environment variables, which win over the \
             config file, which wins over built-in defaults. This precedence order \
             applies uniformly to every setting.",
            "cli"),
        d(100_047, "Release note: shell completion in v2.9.0",
            "v2.9.0 adds `meridianctl completion` for bash, zsh, and fish, generating a \
             completion script you can source from your shell profile.",
            "cli"),
    ]
}

pub(crate) fn config_topic() -> Vec<Doc> {
    vec![
        d(100_048, "Config Reference: server.max_body_bytes",
            "server.max_body_bytes caps the size of an incoming request body; requests \
             over the limit are rejected before the handler runs, protecting downstream \
             services from oversized payloads.",
            "config"),
        d(100_049, "Config Reference: tls.min_version",
            "tls.min_version sets the minimum accepted TLS protocol version for \
             inbound connections; raising it to 1.3 rejects older clients that cannot \
             negotiate the newer handshake.",
            "config"),
        d(100_050, "Runbook: validating a config file before deploying it",
            "Run `meridianctl config validate --config-path staged.toml` to catch \
             typos and out-of-range values before rolling a config change out to \
             production traffic.",
            "config"),
        d(100_051, "Ticket #3390: config change had no effect",
            "Customer edited meridian.toml but the running worker kept the old \
             behavior. Config is loaded once at startup; a running worker must be \
             reloaded or restarted to pick up file changes.",
            "config"),
        d(100_052, "FAQ: can config be reloaded without a restart?",
            "Sending SIGHUP triggers a hot reload of most settings without dropping \
             connections; a small set of settings (listen ports, TLS version) still \
             require a full restart.",
            "config"),
        d(100_053, "Release note: config hot reload in v2.8.0",
            "v2.8.0 introduces SIGHUP-triggered hot reload for the majority of runtime \
             settings, reducing how often a restart is needed for routine tuning.",
            "config"),
    ]
}

pub(crate) fn changelog_topic() -> Vec<Doc> {
    vec![
        d(100_054, "v3.4.1 changelog", "v3.4.1: patch release. Fixed a crash when \
             ratelimit.burst_size was set to zero. No other behavior changes.", "changelog"),
        d(100_055, "v3.3.0 changelog", "v3.3.0: cross-region backup replication; \
             minor performance improvements to the connection pool checkout path.", "changelog"),
        d(100_056, "v3.2.0 changelog", "v3.2.0: pool acquire retries with backoff; \
             deprecated the old --legacy-pool flag, to be removed in v4.0.0.", "changelog"),
        d(100_057, "v3.1.0 changelog", "v3.1.0: webhook ordering sequence headers; \
             fixed a memory leak in long-lived gRPC streams.", "changelog"),
        d(100_058, "v3.0.0 changelog", "v3.0.0: major version. Online CONCURRENTLY \
             index builds in migrations; breaking change to the CLI's default output \
             format, now json instead of text.", "changelog"),
        d(100_059, "v2.9.12 changelog", "v2.9.12: scoped API keys; security fix for a \
             timing side-channel in webhook signature verification.", "changelog"),
    ]
}

pub(crate) fn sso_topic() -> Vec<Doc> {
    vec![
        d(100_060, "SAML Assertion Expired: ERR_SSO_ASSERTION_EXPIRED_SAML",
            "ERR_SSO_ASSERTION_EXPIRED_SAML means the identity provider's SAML \
             assertion arrived after its NotOnOrAfter deadline, usually caused by clock \
             drift between the identity provider and the gateway.",
            "sso"),
        d(100_061, "Runbook: diagnosing SSO clock drift",
            "Compare NTP sync status on both the identity provider and the gateway \
             host; even a few minutes of drift is enough to expire short-lived SAML \
             assertions before they are validated.",
            "sso"),
        d(100_062, "Ticket #5560: SSO works for some users but not others",
            "Turned out to be a per-user clock skew from users whose SSO client ran on \
             a machine with an unsynced clock, not a gateway-side bug at all.",
            "sso"),
        d(100_063, "Config reference: sso.clock_skew_tolerance_sec",
            "sso.clock_skew_tolerance_sec widens the acceptance window for assertion \
             timestamps to absorb small amounts of drift without weakening security \
             meaningfully.",
            "sso"),
        d(100_064, "FAQ: does Meridian support OAuth in addition to SAML?",
            "Yes, OIDC/OAuth2 is supported alongside SAML; both can be enabled \
             simultaneously for different groups of users under sso.providers.",
            "sso"),
        d(100_065, "Release note: OIDC support in v2.7.0",
            "v2.7.0 adds OIDC as a second SSO protocol alongside the existing SAML \
             integration, configured under the same sso.providers list.",
            "sso"),
    ]
}

pub(crate) fn cache_topic() -> Vec<Doc> {
    vec![
        d(100_066, "Cache Stampede Detected: ERR_CACHE_STAMPEDE_DETECTED",
            "ERR_CACHE_STAMPEDE_DETECTED is logged (not returned to clients) when many \
             concurrent requests miss the cache for the same key at once; the gateway \
             coalesces them into a single backend fetch instead of forwarding all of \
             them.",
            "cache"),
        d(100_067, "Runbook: forcing a cache invalidation",
            "Use `meridianctl cache purge --key <key>` for a single key or --prefix for \
             a whole namespace; prefix purges are more expensive and should be scoped \
             as narrowly as possible.",
            "cache"),
        d(100_068, "Ticket #4410: stale data served for several minutes after an update",
            "Cause: cache.ttl_seconds was set far higher than the data's actual change \
             frequency. Lowered the TTL for that namespace and added an explicit purge \
             to the update path.",
            "cache"),
        d(100_069, "Config reference: cache.ttl_seconds and cache.stampede_lock_ms",
            "cache.ttl_seconds controls how long an entry is served before revalidation; \
             cache.stampede_lock_ms controls how long the coalescing lock is held during \
             a stampede-protected refetch.",
            "cache"),
        d(100_070, "FAQ: is the cache shared between regions?",
            "No, each region maintains its own cache; a purge in one region does not \
             automatically propagate to others, by design, to avoid cross-region purge \
             storms.",
            "cache"),
        d(100_071, "Release note: stampede protection in v2.6.0",
            "v2.6.0 introduces automatic request coalescing for concurrent cache misses \
             on the same key, cutting backend load during traffic spikes on hot keys.",
            "cache"),
    ]
}

pub(crate) fn billing_topic() -> Vec<Doc> {
    vec![
        d(100_072, "Plan Quota Exceeded: ERR_QUOTA_EXCEEDED_PLAN",
            "ERR_QUOTA_EXCEEDED_PLAN is returned once an account's monthly request \
             count crosses its plan limit; upgrading the plan or purchasing overage \
             credits both raise the ceiling immediately.",
            "billing"),
        d(100_073, "Runbook: checking current usage against plan quota",
            "The dashboard's Usage tab shows real-time consumption against the plan \
             limit; `meridianctl usage show` gives the same numbers from the CLI for \
             scripting a usage alert.",
            "billing"),
        d(100_074, "Ticket #2299: hit quota mid-month unexpectedly",
            "A misconfigured retry loop on the customer's side was resending failed \
             requests without backoff, multiplying their real usage several times over \
             and burning through quota early.",
            "billing"),
        d(100_075, "FAQ: does quota reset on the calendar month or billing anniversary?",
            "Quota resets on the account's billing anniversary date, not the calendar \
             month, which can surprise customers who assume a calendar-month reset.",
            "billing"),
        d(100_076, "Config reference: billing.overage_alert_threshold_pct",
            "billing.overage_alert_threshold_pct sends a proactive email once usage \
             crosses the configured percentage of plan quota, well before the hard \
             limit is hit.",
            "billing"),
        d(100_077, "Release note: usage alerts in v2.5.0",
            "v2.5.0 adds configurable proactive usage alerts so accounts are not \
             surprised by ERR_QUOTA_EXCEEDED_PLAN with no warning beforehand.",
            "billing"),
    ]
}

pub(crate) fn logging_topic() -> Vec<Doc> {
    vec![
        d(100_078, "Config Reference: log.level and log.sampling_rate",
            "log.level sets the minimum severity emitted (debug, info, warn, error); \
             log.sampling_rate thins high-volume debug/info logs to a fraction of \
             events to control log storage cost.",
            "logging"),
        d(100_079, "Runbook: correlating a request between services with a trace id",
            "Every request is tagged with an x-request-id header; grep logs from every \
             service for that id to reconstruct the full path of a single request \
             through the system.",
            "logging"),
        d(100_080, "Ticket #6612: cannot find logs for a specific failed request",
            "Customer had log.sampling_rate set low enough that most info-level logs, \
             including the one they wanted, were sampled out; raised sampling \
             temporarily to reproduce and capture the issue.",
            "logging"),
        d(100_081, "FAQ: how long are logs retained?",
            "Logs are retained for log.retention_days (30 by default) before being \
             deleted; export to your own long-term storage if you need longer \
             retention.",
            "logging"),
        d(100_082, "Release note: structured JSON logging in v2.4.0",
            "v2.4.0 switches the default log format to structured JSON, making logs \
             easier to ingest into external log pipelines without custom parsing.",
            "logging"),
        d(100_083, "Config reference: log.format (text or json)",
            "log.format toggles between human-readable text and structured JSON; JSON \
             is recommended whenever logs feed an external aggregation pipeline.",
            "logging"),
    ]
}

pub(crate) fn grpc_topic() -> Vec<Doc> {
    vec![
        d(100_084, "gRPC Deadline Exceeded: ERR_GRPC_DEADLINE_EXCEEDED_504",
            "ERR_GRPC_DEADLINE_EXCEEDED_504 means the client's deadline elapsed before \
             the server finished handling the call; raise the client deadline or \
             investigate why the handler is slower than expected.",
            "grpc"),
        d(100_085, "Runbook: profiling a slow gRPC handler",
            "Enable grpc.server_timing_headers to get per-stage latency breakdowns and \
             narrow down whether the delay is in the handler itself or a downstream \
             dependency it calls.",
            "grpc"),
        d(100_086, "Ticket #7020: streaming call disconnects after exactly sixty seconds",
            "Cause: a load balancer's idle-timeout was shorter than the intended \
             streaming call duration; raised the load balancer's timeout to match the \
             gRPC stream's expected lifetime.",
            "grpc"),
        d(100_087, "Config reference: grpc.default_deadline_ms",
            "grpc.default_deadline_ms applies to any call that does not set its own \
             deadline explicitly; leaving it unset lets a single stuck call hang \
             indefinitely.",
            "grpc"),
        d(100_088, "FAQ: does gRPC support the same rate limiting as REST?",
            "Yes, ratelimit.burst_size and ratelimit.refill_per_sec apply uniformly \
             to both the REST and gRPC surfaces of the same API key.",
            "grpc"),
        d(100_089, "Release note: gRPC streaming leak fix in v3.1.0",
            "v3.1.0 fixed a memory leak where long-lived gRPC streams were not fully \
             releasing buffers on disconnect.",
            "grpc"),
    ]
}

pub(crate) fn s3_topic() -> Vec<Doc> {
    vec![
        d(100_090, "Object Storage Permission Denied: ERR_S3_PERMISSION_DENIED_403",
            "ERR_S3_PERMISSION_DENIED_403 usually means the configured storage \
             credentials lack permission on the target bucket, not that the bucket \
             does not exist; check the IAM policy attached to the credentials first.",
            "s3"),
        d(100_091, "Runbook: verifying object storage credentials",
            "Run `meridianctl storage check` to attempt a small read/write round trip \
             against the configured bucket and surface the exact permission that is \
             missing.",
            "s3"),
        d(100_092, "Ticket #8801: exports suddenly failing to write to the bucket",
            "The bucket's policy had been tightened during a security review, removing \
             the write grant the export job depended on. Restored a narrowly scoped \
             write-only grant for the export job's role.",
            "s3"),
        d(100_093, "Config reference: storage.bucket and storage.region",
            "storage.bucket and storage.region must match exactly; a region mismatch \
             produces a redirect that some SDKs surface as an opaque permission-denied \
             error instead of a clear region error.",
            "s3"),
        d(100_094, "FAQ: can exports write to a bucket in a different account?",
            "Yes, with a cross-account bucket policy granting the export role's ARN \
             an explicit grant; a same-account role alone is not required.",
            "s3"),
        d(100_095, "Release note: export retry-on-permission-error in v2.3.0",
            "v2.3.0 adds a bounded retry for transient storage permission errors that \
             clear themselves within a few seconds, without retrying persistent \
             ERR_S3_PERMISSION_DENIED_403 failures indefinitely.",
            "s3"),
    ]
}

pub(crate) fn lb_topic() -> Vec<Doc> {
    vec![
        d(100_096, "Upstream Bad Gateway: ERR_UPSTREAM_502_BAD_GATEWAY",
            "ERR_UPSTREAM_502_BAD_GATEWAY means the load balancer could not get a \
             valid response from any healthy upstream instance, often during a rolling \
             deploy when instances are cycling in and out of the pool.",
            "loadbalancer"),
        d(100_097, "Runbook: reducing 502s during a rolling deploy",
            "Increase lb.drain_timeout_ms so instances finish in-flight requests \
             before being removed from rotation, instead of being pulled abruptly \
             mid-request.",
            "loadbalancer"),
        d(100_098, "Ticket #9012: brief spike of 502s during every deploy",
            "The old instance was being terminated before its in-flight requests \
             drained. Raising lb.drain_timeout_ms to cover the slowest observed request \
             eliminated the spike.",
            "loadbalancer"),
        d(100_099, "Config reference: lb.health_check_interval_ms",
            "lb.health_check_interval_ms controls how often the load balancer probes \
             each upstream; too long an interval delays detecting a newly unhealthy \
             instance and routing around it.",
            "loadbalancer"),
        d(100_100, "FAQ: how many consecutive failed health checks before an instance is pulled?",
            "Three consecutive failures by default, controlled by \
             lb.unhealthy_threshold, to avoid pulling an instance over a single \
             transient blip.",
            "loadbalancer"),
        d(100_101, "Release note: graceful drain in v2.2.0",
            "v2.2.0 adds lb.drain_timeout_ms so outgoing instances can finish in-flight \
             work instead of dropping connections abruptly during deploys.",
            "loadbalancer"),
    ]
}

pub(crate) fn dns_topic() -> Vec<Doc> {
    vec![
        d(100_102, "DNS Resolution Timeout: ERR_DNS_RESOLUTION_TIMEOUT",
            "ERR_DNS_RESOLUTION_TIMEOUT means an upstream hostname could not be \
             resolved within dns.resolve_timeout_ms, often caused by an internal \
             resolver being overloaded rather than the target host being genuinely \
             unreachable.",
            "dns"),
        d(100_103, "Runbook: diagnosing intermittent DNS timeouts",
            "Compare resolution latency from the affected host against a known-good \
             host in the same region; a resolver-side problem shows up on every host, \
             a network-path problem shows up on only some.",
            "dns"),
        d(100_104, "Ticket #3305: outbound calls to one partner intermittently time out",
            "Traced to that partner's DNS record having a very short TTL and \
             occasionally resolving to a decommissioned IP for a few seconds during \
             their own DNS cutover.",
            "dns"),
        d(100_105, "Config reference: dns.resolve_timeout_ms and dns.cache_ttl_floor_sec",
            "dns.cache_ttl_floor_sec enforces a minimum cache time even if the record's \
             own TTL is set unreasonably low, protecting against resolver overload from \
             a misconfigured upstream.",
            "dns"),
        d(100_106, "FAQ: does Meridian cache negative DNS lookups?",
            "Yes, briefly, to avoid hammering the resolver with repeated lookups for a \
             hostname that is currently failing to resolve at all.",
            "dns"),
        d(100_107, "Release note: DNS cache floor in v2.1.0",
            "v2.1.0 introduces dns.cache_ttl_floor_sec after repeated incidents caused \
             by upstream partners publishing unreasonably short DNS TTLs.",
            "dns"),
    ]
}

pub(crate) fn diskquota_topic() -> Vec<Doc> {
    vec![
        d(100_108, "Disk Quota Exceeded: ERR_DISK_QUOTA_EXCEEDED_ENOSPC",
            "ERR_DISK_QUOTA_EXCEEDED_ENOSPC surfaces when a write-ahead log or local \
             cache directory fills the volume; clearing old segments or growing the \
             volume both resolve it, but growing the volume is safer under active \
             write load.",
            "diskquota"),
        d(100_109, "Runbook: freeing disk space safely under load",
            "Never delete write-ahead log segments that have not yet been checkpointed; \
             use `meridianctl storage gc --dry-run` first to see exactly what would be \
             removed before running it for real.",
            "diskquota"),
        d(100_110, "Ticket #4470: disk filled up overnight with no traffic change",
            "A stuck checkpointer had stopped advancing, so write-ahead log \
             segments accumulated indefinitely instead of being reclaimed on schedule. \
             Restarting the checkpointer resumed normal reclamation.",
            "diskquota"),
        d(100_111, "Config reference: storage.wal_retention_mb",
            "storage.wal_retention_mb caps how much write-ahead log is kept before \
             older segments are eligible for reclamation, bounding worst-case disk \
             usage.",
            "diskquota"),
        d(100_112, "FAQ: does Meridian alert before disk actually fills up?",
            "Yes, a warning fires at 85% volume usage by default via \
             diskquota.warn_threshold_pct, well ahead of ERR_DISK_QUOTA_EXCEEDED_ENOSPC.",
            "diskquota"),
        d(100_113, "Release note: proactive disk warnings in v2.0.0",
            "v2.0.0 adds diskquota.warn_threshold_pct so operators get a warning well \
             before a volume actually fills and writes start failing outright.",
            "diskquota"),
    ]
}

pub(crate) fn oom_topic() -> Vec<Doc> {
    vec![
        d(100_114, "Worker Killed By OOM: ERR_OOM_KILLED_137",
            "Exit code 137 (128 + SIGKILL) after ERR_OOM_KILLED_137 means the kernel's \
             out-of-memory killer terminated the worker; check container memory \
             limits against actual observed resident set size before assuming a \
             genuine leak.",
            "oom"),
        d(100_115, "Runbook: telling a leak apart from a legitimate memory ceiling",
            "Graph resident set size over several restarts: a leak climbs steadily \
             within each run and never plateaus, while a legitimate ceiling issue \
             plateaus near the same value every time under similar load.",
            "oom"),
        d(100_116, "Ticket #5581: worker restarts every few hours under heavy load",
            "Memory climbed steadily and never plateaued within a single worker \
             lifetime, consistent with a genuine leak rather than an undersized memory \
             limit; a heap profile pinpointed an unbounded internal cache.",
            "oom"),
        d(100_117, "Config reference: cache.max_entries",
            "cache.max_entries bounds the in-memory cache's entry count; leaving it \
             unset allows unbounded growth under a sufficiently diverse key space, \
             which can itself look like a memory leak.",
            "oom"),
        d(100_118, "FAQ: how do I get a heap profile from a running worker?",
            "Send SIGUSR2 to trigger a one-shot heap profile dump to the configured \
             profile directory without restarting the worker.",
            "oom"),
        d(100_119, "Release note: bounded internal caches in v3.4.0",
            "v3.4.0 adds a default cache.max_entries ceiling to several previously \
             unbounded internal caches identified as sources of slow memory growth \
             under long uptime.",
            "oom"),
    ]
}
