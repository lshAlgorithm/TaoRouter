//! Session-sticky least-load routing over data-parallel ranks
//!
//! A port of ATOM's DP load balancer (`--dp-load-balance least_tokens` with
//! `ATOM_DP_SESSION_AFFINITY=1`, atom/model_engine/engine_core_mgr.py) to the router. With
//! `--intra-node-data-parallel-size N` every worker is one DP rank, and the ranks of one engine step
//! in lockstep, so the busiest rank sets every step's time.
//!
//! Each worker carries `load = outstanding prefill tokens + req_equiv x in-flight requests`.
//! - A new session (an unseen `X-Session-ID`, `X-Dynamo-Session-ID` or `X-Correlation-ID`) goes to
//!   the least-loaded worker, ties broken by a rendezvous hash of (session, worker).
//! - A known session always goes to its owner; load never moves it. Only an owner that left the
//!   healthy set makes the session be placed again.
//! - A sub-agent with its own id is a session of its own, placed by load. A session id used with
//!   several models is a session per model.
//! - A request without a session id goes to the least-loaded worker (rotating start on ties).
//!
//! The router does not tokenize: a prompt counts `len(request text) / 4` tokens. The policy asks
//! for the prompt text (`needs_prompt_text`), so a chat completion hands it the conversation text
//! that Program scheduling uses. A request of a known session is charged only the growth over
//! that session's last prefilled prompt; a request that fails before any output does not count as
//! prefilled. The token charge is released when the worker returns output (the prefill is done)
//! or when the request ends; the request count when it ends. Only `select_worker_tracked`
//! charges; `select_worker_with_headers` places the same way without charging.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use tracing::{debug, info};

use super::{get_healthy_worker_indices, normalize_model_key};
use super::{LoadBalancingPolicy, LoadGuard, RequestHeaders, RequestLoadGuard};
use crate::core::Worker;
use crate::metrics::RouterMetrics;
use crate::policies::ConsistentHashPolicy;

/// Session headers, in priority order (lowercase). Per-request ids (x-request-id) are not sessions.
const SESSION_HEADERS: &[&str] = &["x-session-id", "x-dynamo-session-id", "x-correlation-id"];
/// Bytes of request text per prompt token.
const BYTES_PER_TOKEN: u64 = 4;
const STATS_INTERVAL: Duration = Duration::from_secs(30);
const PRUNE_INTERVAL: Duration = Duration::from_secs(60);

/// Configuration for the session least-load policy
#[derive(Debug, Clone)]
pub struct SessionLeastLoadConfig {
    /// Tokens one in-flight request weighs (ATOM_DP_LB_REQ_EQUIV)
    pub req_equiv: u64,
    /// A session unseen for this long is forgotten
    pub session_ttl_secs: u64,
}

impl Default for SessionLeastLoadConfig {
    fn default() -> Self {
        Self {
            req_equiv: 512,
            session_ttl_secs: 4 * 3600,
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
struct WorkerLoad {
    reqs: u64,
    tokens: u64,
    routed: u64,
}

#[derive(Debug)]
struct Session {
    owner: String,
    /// Tokens of the last prompt the owner prefilled for this session
    prompt_tokens: u64,
    last_seen: Instant,
}

#[derive(Debug, Default)]
struct Counters {
    new_sessions: u64,
    owner_hits: u64,
    owner_lost: u64,
    no_session: u64,
}

/// (model, session id): a session id used with several models is one session per model
type SessionKey = (String, String);

#[derive(Debug)]
struct State {
    /// Keyed by worker URL (`http://host:port@rank` for DP-aware workers)
    loads: HashMap<String, WorkerLoad>,
    sessions: HashMap<SessionKey, Session>,
    counters: Counters,
    cursor: usize,
    last_prune: Instant,
    last_stats: Instant,
}

impl State {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            loads: HashMap::new(),
            sessions: HashMap::new(),
            counters: Counters::default(),
            cursor: 0,
            last_prune: now,
            last_stats: now,
        }
    }

    fn load_of(&self, url: &str, req_equiv: u64) -> u64 {
        self.loads
            .get(url)
            .map_or(0, |l| l.tokens + req_equiv * l.reqs)
    }

    /// Index of the least-loaded healthy worker. A session breaks ties by the highest
    /// rendezvous hash of (session, worker); a request without one by a rotating start.
    fn least_loaded(
        &mut self,
        workers: &[Arc<dyn Worker>],
        healthy: &[usize],
        session: Option<&str>,
        req_equiv: u64,
    ) -> usize {
        let n = healthy.len();
        let start = match session {
            Some(_) => 0,
            None => {
                self.cursor = self.cursor.wrapping_add(1);
                self.cursor % n
            }
        };
        let mut best = healthy[start];
        let mut best_load = self.load_of(workers[best].url(), req_equiv);
        let mut best_hash: Option<u64> = None;
        for k in 1..n {
            let idx = healthy[(start + k) % n];
            let load = self.load_of(workers[idx].url(), req_equiv);
            if load < best_load {
                best = idx;
                best_load = load;
                best_hash = None;
            } else if load == best_load {
                if let Some(sid) = session {
                    let hash = rendezvous(sid, workers[idx].url());
                    let current =
                        *best_hash.get_or_insert_with(|| rendezvous(sid, workers[best].url()));
                    if hash > current {
                        best = idx;
                        best_hash = Some(hash);
                    }
                }
            }
        }
        best
    }

    fn prune(&mut self, now: Instant, ttl: Duration) {
        if now.duration_since(self.last_prune) < PRUNE_INTERVAL {
            return;
        }
        self.last_prune = now;
        let before = self.sessions.len();
        self.sessions
            .retain(|_, s| now.duration_since(s.last_seen) < ttl);
        let dropped = before - self.sessions.len();
        if dropped > 0 {
            debug!("session_least_load: forgot {} idle sessions", dropped);
        }
    }

    fn log_stats(&mut self, now: Instant, workers: &[Arc<dyn Worker>]) {
        if now.duration_since(self.last_stats) < STATS_INTERVAL {
            return;
        }
        self.last_stats = now;
        let mut owned: HashMap<&str, u64> = HashMap::new();
        for s in self.sessions.values() {
            *owned.entry(s.owner.as_str()).or_default() += 1;
        }
        // the registry hands workers over in map order: list them by (base URL, DP rank)
        let mut sorted: Vec<&Arc<dyn Worker>> = workers.iter().collect();
        sorted.sort_by_key(|w| (w.base_url().to_string(), w.dp_rank()));
        let per_worker = |f: &dyn Fn(&str) -> u64| {
            sorted
                .iter()
                .map(|w| f(w.url()).to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        let load = |url: &str| self.loads.get(url).copied().unwrap_or_default();
        let c = &self.counters;
        info!(
            "session_least_load: sessions={} new={} owner_hits={} owner_lost={} no_session={} | \
             in-flight reqs=[{}] prefill_tokens=[{}] sessions=[{}] routed=[{}]",
            self.sessions.len(),
            c.new_sessions,
            c.owner_hits,
            c.owner_lost,
            c.no_session,
            per_worker(&|u| load(u).reqs),
            per_worker(&|u| load(u).tokens),
            per_worker(&|u| owned.get(u).copied().unwrap_or(0)),
            per_worker(&|u| load(u).routed),
        );
    }
}

fn rendezvous(session: &str, worker_url: &str) -> u64 {
    ConsistentHashPolicy::fbi_hash(&format!("{}:{}", session, worker_url))
}

fn session_id(headers: Option<&RequestHeaders>) -> Option<&str> {
    let headers = headers?;
    SESSION_HEADERS
        .iter()
        .filter_map(|name| headers.get(*name))
        .map(|v| v.as_str())
        .find(|v| !v.is_empty())
}

fn estimate_prompt_tokens(request_text: Option<&str>) -> u64 {
    request_text.map_or(0, |t| t.len() as u64 / BYTES_PER_TOKEN)
}

/// Session-sticky least-load policy (see the module docs)
#[derive(Debug)]
pub struct SessionLeastLoadPolicy {
    config: SessionLeastLoadConfig,
    state: Arc<Mutex<State>>,
}

impl SessionLeastLoadPolicy {
    pub fn new() -> Self {
        Self::with_config(SessionLeastLoadConfig::default())
    }

    pub fn with_config(config: SessionLeastLoadConfig) -> Self {
        info!(
            "session_least_load: req_equiv={} session_ttl_secs={}",
            config.req_equiv, config.session_ttl_secs
        );
        Self {
            config,
            state: Arc::new(Mutex::new(State::new())),
        }
    }

    fn place(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
        charge: bool,
    ) -> Option<(usize, Option<LoadGuard>)> {
        let healthy = get_healthy_worker_indices(workers);
        if healthy.is_empty() {
            return None;
        }
        let session = session_id(headers);
        // Sessions are kept per model: one policy instance can serve several models (the
        // registry hands its default policy to every model without a policy of its own).
        let key = session.map(|sid| {
            let model = normalize_model_key(workers[healthy[0]].model_id());
            (model.to_string(), sid.to_string())
        });
        let prompt_tokens = estimate_prompt_tokens(request_text);
        let req_equiv = self.config.req_equiv;
        let now = Instant::now();

        let mut st = self.state.lock();
        st.prune(now, Duration::from_secs(self.config.session_ttl_secs));
        // the state this decision sees, before its own charge
        st.log_stats(now, workers);

        let (idx, tokens, decision) = match &key {
            None => {
                st.counters.no_session += 1;
                let idx = st.least_loaded(workers, &healthy, None, req_equiv);
                (idx, prompt_tokens, "no_session")
            }
            Some(key) => {
                let owner = st.sessions.get(key).map(|s| {
                    let idx = healthy
                        .iter()
                        .copied()
                        .find(|&i| workers[i].url() == s.owner);
                    (idx, s.prompt_tokens)
                });
                match owner {
                    Some((Some(idx), previous)) => {
                        st.counters.owner_hits += 1;
                        if let Some(s) = st.sessions.get_mut(key) {
                            // A tracked request records its prompt once it is prefilled.
                            if !charge {
                                s.prompt_tokens = prompt_tokens;
                            }
                            s.last_seen = now;
                        }
                        // A later turn extends the previous prompt: charge only the growth.
                        (idx, prompt_tokens.saturating_sub(previous), "owner")
                    }
                    known => {
                        let decision = if known.is_some() {
                            st.counters.owner_lost += 1;
                            "owner_lost"
                        } else {
                            st.counters.new_sessions += 1;
                            "new"
                        };
                        let idx = st.least_loaded(workers, &healthy, Some(&key.1), req_equiv);
                        st.sessions.insert(
                            key.clone(),
                            Session {
                                owner: workers[idx].url().to_string(),
                                prompt_tokens: if charge { 0 } else { prompt_tokens },
                                last_seen: now,
                            },
                        );
                        (idx, prompt_tokens, decision)
                    }
                }
            }
        };

        let url = workers[idx].url();
        let guard = if charge {
            let load = st.loads.entry(url.to_string()).or_default();
            load.reqs += 1;
            load.tokens += tokens;
            load.routed += 1;
            RouterMetrics::set_running_requests(url, load.reqs as usize);
            Some(Box::new(SessionLoadGuard {
                state: Arc::clone(&self.state),
                worker_url: url.to_string(),
                tokens,
                session_prompt: key.map(|key| (key, prompt_tokens)),
            }) as LoadGuard)
        } else {
            None
        };
        debug!(
            "session_least_load: {} session={:?} -> {} prompt_tokens={} charged={} load={}",
            decision,
            session,
            url,
            prompt_tokens,
            if charge { tokens } else { 0 },
            st.load_of(url, req_equiv)
        );
        drop(st);

        workers[idx].increment_processed();
        RouterMetrics::record_processed_request(url);
        RouterMetrics::record_policy_decision(self.name(), url);
        Some((idx, guard))
    }
}

impl Default for SessionLeastLoadPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl LoadBalancingPolicy for SessionLeastLoadPolicy {
    fn select_worker_with_headers(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<usize> {
        self.place(workers, request_text, headers, false)
            .map(|(idx, _)| idx)
    }

    fn select_worker_tracked(
        &self,
        workers: &[Arc<dyn Worker>],
        request_text: Option<&str>,
        headers: Option<&RequestHeaders>,
    ) -> Option<(usize, Option<LoadGuard>)> {
        self.place(workers, request_text, headers, true)
    }

    fn name(&self) -> &'static str {
        "session_least_load"
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    fn needs_headers(&self) -> bool {
        true
    }

    fn needs_prompt_text(&self) -> bool {
        true
    }

    fn reset(&self) {
        let mut st = self.state.lock();
        st.sessions.clear();
        st.counters = Counters::default();
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// One request's charge on one worker: the prompt tokens until the prefill is done, one request
/// until the guard is dropped.
struct SessionLoadGuard {
    state: Arc<Mutex<State>>,
    worker_url: String,
    tokens: u64,
    /// (session key, prompt tokens) to record on the session once the prompt is prefilled
    session_prompt: Option<(SessionKey, u64)>,
}

impl RequestLoadGuard for SessionLoadGuard {
    fn prefill_done(&mut self) {
        if self.tokens == 0 && self.session_prompt.is_none() {
            return;
        }
        let mut st = self.state.lock();
        if let Some(load) = st.loads.get_mut(&self.worker_url) {
            load.tokens = load.tokens.saturating_sub(self.tokens);
        }
        self.tokens = 0;
        // The owner now caches this prompt: the session's next turn is charged only its growth.
        if let Some((key, prompt_tokens)) = self.session_prompt.take() {
            if let Some(s) = st.sessions.get_mut(&key) {
                if s.owner == self.worker_url {
                    s.prompt_tokens = prompt_tokens;
                }
            }
        }
    }
}

impl Drop for SessionLoadGuard {
    fn drop(&mut self) {
        let mut st = self.state.lock();
        if let Some(load) = st.loads.get_mut(&self.worker_url) {
            load.reqs = load.reqs.saturating_sub(1);
            load.tokens = load.tokens.saturating_sub(self.tokens);
            RouterMetrics::set_running_requests(&self.worker_url, load.reqs as usize);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorker, DPAwareWorker, WorkerType};

    fn dp_workers(n: usize) -> Vec<Arc<dyn Worker>> {
        (0..n)
            .map(|r| {
                Arc::new(DPAwareWorker::new(
                    "http://127.0.0.1:8000".to_string(),
                    r,
                    n,
                    WorkerType::Regular,
                )) as Arc<dyn Worker>
            })
            .collect()
    }

    fn session(id: &str) -> RequestHeaders {
        let mut h = RequestHeaders::new();
        h.insert("x-session-id".to_string(), id.to_string());
        h
    }

    /// A routing text of `tokens` estimated prompt tokens
    fn text(tokens: usize) -> String {
        "a".repeat(tokens * BYTES_PER_TOKEN as usize)
    }

    fn policy(req_equiv: u64) -> SessionLeastLoadPolicy {
        SessionLeastLoadPolicy::with_config(SessionLeastLoadConfig {
            req_equiv,
            session_ttl_secs: 3600,
        })
    }

    fn route(
        p: &SessionLeastLoadPolicy,
        workers: &[Arc<dyn Worker>],
        sid: Option<&str>,
        tokens: usize,
    ) -> (usize, LoadGuard) {
        let h = sid.map(session);
        let (idx, guard) = p
            .select_worker_tracked(workers, Some(&text(tokens)), h.as_ref())
            .unwrap();
        (idx, guard.unwrap())
    }

    fn loads(p: &SessionLeastLoadPolicy, workers: &[Arc<dyn Worker>]) -> Vec<(u64, u64)> {
        let st = p.state.lock();
        workers
            .iter()
            .map(|w| {
                let l = st.loads.get(w.url()).copied().unwrap_or_default();
                (l.reqs, l.tokens)
            })
            .collect()
    }

    #[test]
    fn test_new_sessions_spread_by_load() {
        let workers = dp_workers(4);
        let p = policy(512);
        let mut guards = Vec::new();
        let mut used = std::collections::HashSet::new();
        for i in 0..4 {
            let (idx, g) = route(&p, &workers, Some(&format!("s{}", i)), 1000);
            used.insert(idx);
            guards.push(g);
        }
        // Each new session lands on an idle rank while one is left.
        assert_eq!(used.len(), 4);
        assert!(loads(&p, &workers).iter().all(|&l| l == (1, 1000)));
    }

    #[test]
    fn test_new_session_goes_to_least_tokens_plus_requests() {
        let workers = dp_workers(3);
        let p = policy(512);
        // rank a: one request with 10000 prompt tokens -> load 10512
        let (a, _ga) = route(&p, &workers, Some("big"), 10000);
        // rank b: one request with 100 tokens -> load 612
        let (b, _gb) = route(&p, &workers, Some("small"), 100);
        assert_ne!(a, b);
        // the third rank is idle: a new session goes there
        let (c, _gc) = route(&p, &workers, Some("third"), 100);
        assert!(c != a && c != b);
        // now c (612) ties b (612), a (10512) is heaviest: the next lands on b or c, never a
        let (d, _gd) = route(&p, &workers, Some("fourth"), 100);
        assert!(d == b || d == c);
    }

    #[test]
    fn test_requests_outweigh_tokens_with_large_req_equiv() {
        let workers = dp_workers(2);
        let p = policy(512);
        // rank x: one 10-token request (load 522); rank y: one 900-token request (load 1412)
        let (x, _g1) = route(&p, &workers, Some("x1"), 10);
        let (y, _g2) = route(&p, &workers, Some("y1"), 900);
        assert_ne!(x, y);
        // x has 522 < 1412: the next new session goes to x
        let (x2, _g3) = route(&p, &workers, Some("x2"), 10);
        assert_eq!(x2, x);
        // x: 1044 < 1412 still
        let (x3, _g4) = route(&p, &workers, Some("x3"), 10);
        assert_eq!(x3, x);
        // x: 1566 > 1412: now y
        let (y2, _g5) = route(&p, &workers, Some("y2"), 10);
        assert_eq!(y2, y);
    }

    #[test]
    fn test_session_is_sticky_under_load() {
        let workers = dp_workers(4);
        let p = policy(512);
        let (home, g) = route(&p, &workers, Some("sticky"), 100);
        drop(g);
        // Pile load on the owner; later turns stay anyway.
        let mut guards = Vec::new();
        for _ in 0..10 {
            let (idx, g) = route(&p, &workers, Some("sticky"), 5000);
            assert_eq!(idx, home);
            guards.push(g);
        }
        let st = p.state.lock();
        assert_eq!(st.counters.new_sessions, 1);
        assert_eq!(st.counters.owner_hits, 10);
    }

    #[test]
    fn test_child_session_placed_by_load() {
        let workers = dp_workers(2);
        let p = policy(512);
        let (parent, _g) = route(&p, &workers, Some("root"), 1000);
        // A sub-agent with its own correlation id is a new session: it avoids the busy parent rank.
        let mut h = RequestHeaders::new();
        h.insert("x-correlation-id".to_string(), "root-child-1".to_string());
        let (child, _gc) = p
            .select_worker_tracked(&workers, Some(&text(10)), Some(&h))
            .unwrap();
        assert_ne!(child, parent);
    }

    #[test]
    fn test_extension_only_token_charge() {
        let workers = dp_workers(2);
        let p = policy(512);
        let (idx, mut g1) = route(&p, &workers, Some("s"), 1000);
        assert_eq!(loads(&p, &workers)[idx], (1, 1000));
        g1.prefill_done();
        drop(g1);
        assert_eq!(loads(&p, &workers)[idx], (0, 0));
        // second turn: 1000 -> 1300 tokens, charged 300
        let (idx2, mut g2) = route(&p, &workers, Some("s"), 1300);
        assert_eq!(idx2, idx);
        assert_eq!(loads(&p, &workers)[idx], (1, 300));
        g2.prefill_done();
        drop(g2);
        // a shorter prompt (context compaction) is charged nothing
        let (_, _g3) = route(&p, &workers, Some("s"), 200);
        assert_eq!(loads(&p, &workers)[idx], (1, 0));
    }

    #[test]
    fn test_failed_request_does_not_record_prompt() {
        let workers = dp_workers(2);
        let p = policy(512);
        // The first try fails before any output: its retry is charged the full prompt again.
        let (idx, g1) = route(&p, &workers, Some("s"), 1000);
        drop(g1);
        let (idx2, mut g2) = route(&p, &workers, Some("s"), 1000);
        assert_eq!(idx2, idx);
        assert_eq!(loads(&p, &workers)[idx], (1, 1000));
        g2.prefill_done();
        drop(g2);
        // Prefilled now: the next turn is charged only its growth.
        let (_, _g3) = route(&p, &workers, Some("s"), 1100);
        assert_eq!(loads(&p, &workers)[idx], (1, 100));
    }

    fn model_workers(model: &str, n: usize) -> Vec<Arc<dyn Worker>> {
        (0..n)
            .map(|i| {
                let labels = HashMap::from([("model_id".to_string(), model.to_string())]);
                Arc::new(
                    BasicWorker::new(format!("http://{}-{}:8000", model, i), WorkerType::Regular)
                        .with_labels(labels),
                ) as Arc<dyn Worker>
            })
            .collect()
    }

    #[test]
    fn test_session_scoped_per_model() {
        let a = model_workers("model-a", 2);
        let b = model_workers("model-b", 2);
        let p = policy(512);
        // One session id used with two models: each model keeps its own owner and prompt.
        let (owner_a, mut g) = route(&p, &a, Some("s"), 1000);
        g.prefill_done();
        drop(g);
        let (owner_b, mut g) = route(&p, &b, Some("s"), 2000);
        g.prefill_done();
        drop(g);
        let (idx, g) = route(&p, &a, Some("s"), 1100);
        assert_eq!(idx, owner_a);
        assert_eq!(loads(&p, &a)[owner_a], (1, 100));
        drop(g);
        let (idx, _g) = route(&p, &b, Some("s"), 2200);
        assert_eq!(idx, owner_b);
        assert_eq!(loads(&p, &b)[owner_b], (1, 200));
        let st = p.state.lock();
        assert_eq!(st.counters.owner_lost, 0);
        assert_eq!(st.sessions.len(), 2);
    }

    #[test]
    fn test_release_on_first_chunk_then_finish() {
        let workers = dp_workers(2);
        let p = policy(512);
        let (idx, mut g) = route(&p, &workers, Some("s"), 800);
        assert_eq!(loads(&p, &workers)[idx], (1, 800));
        g.prefill_done();
        assert_eq!(loads(&p, &workers)[idx], (1, 0));
        // every later chunk calls it again: a no-op
        g.prefill_done();
        assert_eq!(loads(&p, &workers)[idx], (1, 0));
        drop(g);
        assert_eq!(loads(&p, &workers)[idx], (0, 0));
    }

    #[test]
    fn test_release_on_abort_before_first_chunk() {
        let workers = dp_workers(2);
        let p = policy(512);
        let (idx, g) = route(&p, &workers, None, 700);
        assert_eq!(loads(&p, &workers)[idx], (1, 700));
        drop(g);
        assert_eq!(loads(&p, &workers), vec![(0, 0), (0, 0)]);
    }

    #[test]
    fn test_untracked_selection_does_not_charge() {
        let workers = dp_workers(2);
        let p = policy(512);
        let h = session("s");
        let idx = p
            .select_worker_with_headers(&workers, Some(&text(500)), Some(&h))
            .unwrap();
        assert_eq!(loads(&p, &workers), vec![(0, 0), (0, 0)]);
        // the session still owns its rank
        let (idx2, _g) = route(&p, &workers, Some("s"), 600);
        assert_eq!(idx2, idx);
    }

    #[test]
    fn test_tie_break_is_deterministic_per_session() {
        let workers = dp_workers(8);
        // On idle ranks the rendezvous hash decides: the same id lands on the same rank in a
        // fresh policy, and different ids spread.
        let mut ranks = std::collections::HashSet::new();
        for i in 0..64 {
            let sid = format!("session-{}", i);
            let a = policy(512)
                .select_worker_with_headers(&workers, Some("x"), Some(&session(&sid)))
                .unwrap();
            let b = policy(512)
                .select_worker_with_headers(&workers, Some("x"), Some(&session(&sid)))
                .unwrap();
            assert_eq!(a, b);
            ranks.insert(a);
        }
        assert!(ranks.len() >= 6, "ties spread over {} ranks", ranks.len());
    }

    #[test]
    fn test_no_session_requests_rotate_on_ties() {
        let workers = dp_workers(4);
        let p = policy(512);
        // Released at once, so every request sees all ranks tied.
        let mut seen = std::collections::HashSet::new();
        for _ in 0..4 {
            let (idx, g) = route(&p, &workers, None, 10);
            drop(g);
            seen.insert(idx);
        }
        assert_eq!(seen.len(), 4);
    }

    #[test]
    fn test_unhealthy_owner_replaced() {
        let workers = dp_workers(2);
        let p = policy(512);
        let (home, g) = route(&p, &workers, Some("s"), 1000);
        drop(g);
        workers[home].set_healthy(false);
        let (idx, g2) = route(&p, &workers, Some("s"), 1200);
        assert_ne!(idx, home);
        // the new owner has no cache of this session: the full prompt is charged
        assert_eq!(loads(&p, &workers)[idx], (1, 1200));
        drop(g2);
        workers[home].set_healthy(true);
        // and the session stays with its new owner
        let (idx3, _g3) = route(&p, &workers, Some("s"), 1300);
        assert_eq!(idx3, idx);
        assert_eq!(p.state.lock().counters.owner_lost, 1);
    }

    #[test]
    fn test_idle_sessions_forgotten_after_ttl() {
        let workers = dp_workers(2);
        let p = SessionLeastLoadPolicy::with_config(SessionLeastLoadConfig {
            req_equiv: 512,
            session_ttl_secs: 0,
        });
        let (_, g) = route(&p, &workers, Some("s"), 10);
        drop(g);
        {
            let mut st = p.state.lock();
            assert_eq!(st.sessions.len(), 1);
            st.last_prune = Instant::now() - 2 * PRUNE_INTERVAL;
        }
        let (_, _g) = route(&p, &workers, Some("t"), 10);
        let st = p.state.lock();
        assert!(!st.sessions.keys().any(|(_, sid)| sid == "s"));
        assert_eq!(st.counters.new_sessions, 2);
    }

    #[test]
    fn test_session_header_priority() {
        let mut h = RequestHeaders::new();
        h.insert("x-correlation-id".to_string(), "corr".to_string());
        assert_eq!(session_id(Some(&h)), Some("corr"));
        h.insert("x-session-id".to_string(), "sess".to_string());
        assert_eq!(session_id(Some(&h)), Some("sess"));
        let mut r = RequestHeaders::new();
        r.insert("x-request-id".to_string(), "req".to_string());
        assert_eq!(session_id(Some(&r)), None);
        r.insert("x-session-id".to_string(), String::new());
        assert_eq!(session_id(Some(&r)), None);
    }
}
