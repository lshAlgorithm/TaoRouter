"""KV-aware process STs: pytest -> router binary -> gRPC workers + KV events.

Run after `cargo build --bin vllm-router` and `pip install -e '.[dev]'`:
    pytest py_test/integration/load_balancing/test_kv_aware.py -v

CPU only, with a local tokenizer. Assertions observe selected workers through
the public chat API; no policy/scorer/index hooks or pinned token IDs.
"""

import concurrent.futures
import collections
import json
import time

import pytest
import requests

pytestmark = pytest.mark.integration
PROMPT = " ".join(f"t{i}" for i in range(17))


class KvScenario:
    def __init__(self, workers, url):
        self.workers = workers
        self.url = url
        self.marker = 64
        self.session = requests.Session()
        self.session.trust_env = False
        self.stream_session = requests.Session()
        self.stream_session.trust_env = False

    def request(self, text=PROMPT, **extra):
        client = self.stream_session if extra.get("stream") else self.session
        return client.post(
            f"{self.url}/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": text}],
                "max_tokens": 2,
                **extra,
            },
            timeout=10,
            stream=extra.get("stream", False),
        )

    def choose(self, text=PROMPT, **extra):
        with self.request(text, **extra) as response:
            assert response.status_code == 200, response.text
            worker_id = response.json()["choices"][0]["message"]["content"]
        return next(w for w in self.workers if w.worker_id == worker_id)

    def wait_choice(self, expected, text=PROMPT, **extra):
        deadline = time.monotonic() + 10
        seen = []
        while time.monotonic() < deadline:
            actual = self.choose(text, **extra)
            seen.append(actual.worker_id)
            if actual is expected:
                return
            time.sleep(0.02)
        pytest.fail(f"Expected {expected.worker_id}, last observations: {seen[-20:]}")

    def tokens(self, text=PROMPT):
        worker = self.choose(text)
        request = worker.captured[-1]
        assert request.WhichOneof("prompt") == "token_ids"
        tokens = list(request.token_ids.ids)
        assert len(tokens) == 17, tokens
        return tokens

    @staticmethod
    def first_output(response):
        assert response.status_code == 200, response.text
        lines = response.iter_lines()
        for line in lines:
            if line.startswith(b"data: ") and line != b"data: [DONE]":
                chunk = json.loads(line[6:])
                if chunk.get("choices", [{}])[0].get("delta", {}).get("content"):
                    # Keep the iterator alive while the request is held. Its
                    # destruction can close urllib3's underlying HTTP stream.
                    return lines
        pytest.fail("No first model output")

    def publish_applied(self, worker, events, sequence=None):
        """A fresh sentinel in the SAME batch must change its selected worker.

        If cold routing already chooses the producer, first cache a shorter
        sentinel on its rival. Thus waiting cannot succeed on an unchanged
        cold tie, and negative assertions run only after the batch was applied.
        """
        marker = " ".join(f"t{i}" for i in range(self.marker, self.marker + 17))
        self.marker += 17
        baseline = self.choose(marker)
        tokens = list(baseline.captured[-1].token_ids.ids)
        if baseline is worker:
            rival = next(w for w in self.workers if w is not worker)
            rival.publish([rival.store(tokens[:8])])
            self.wait_choice(rival, marker)
        worker.publish(events + [worker.store(tokens[:16])], sequence=sequence)
        self.wait_choice(worker, marker)


@pytest.fixture
def kv_scenario(kv_workers, kv_st_runtime, kv_tokenizer, router_manager):
    scenarios = []

    def start(extra=None, worker_options=None):
        workers = [kv_workers(**options) for options in (worker_options or [{}, {}])]
        router = router_manager.start_router(
            worker_urls=[w.url for w in workers],
            policy="kv_aware",
            router_bin=str(kv_st_runtime[1]),
            extra={
                "router_temperature": 0,
                "disable_retries": True,
                "disable_circuit_breaker": True,
                **(extra or {}),
            },
            env={
                "VLLM_ROUTER_TOKENIZER": str(kv_tokenizer),
                "NO_PROXY": "127.0.0.1,localhost",
            },
        )
        for worker in workers:
            if worker.block_size and worker.model and worker.events_enabled:
                worker.wait_subscribed()
        scenario = KvScenario(workers, router.url)
        scenarios.append(scenario)
        return scenario

    yield start
    for scenario in scenarios:
        scenario.session.close()
        scenario.stream_session.close()


@pytest.mark.parametrize(
    "invalidation", ["remove", "clear", "gap", "restart", "malformed"]
)
def test_kv_aware_cache_events_change_routing(kv_scenario, invalidation):
    st = kv_scenario()
    tokens = st.tokens()
    a, b = st.workers
    # N=17, B=2: A reuses 16 -> q=1; B reuses 14 -> q=3.
    cached = a.store(tokens[:16])
    st.publish_applied(b, [b.store(tokens[:14])])
    st.publish_applied(a, [cached])
    for _ in range(5):
        assert st.choose() is a

    if invalidation == "restart":
        # A lower sequence signals restart; repeating the initial 0 is a
        # duplicate, so first advance this publisher beyond 0.
        st.publish_applied(a, [cached])

    if invalidation == "remove":
        events = [
            {
                "type": "BlockRemoved",
                "medium": "device",
                "block_hashes": cached["block_hashes"],
            }
        ]
        sequence = None
    elif invalidation == "clear":
        events = [{"type": "AllBlocksCleared"}]
        sequence = None
    elif invalidation == "malformed":
        a.publish([{"type": "UnknownEvent"}])
        # Observe the invalidation before sending a fresh valid batch, so a
        # subsequent sequence gap cannot mask a broken malformed decoder.
        st.wait_choice(b)
        events, sequence = [], None
    else:
        events = []
        sequence = a.sequence + 1 if invalidation == "gap" else 0
    st.publish_applied(a, events, sequence=sequence)
    # B still has valid cache: there is no cold-tie expectation after eviction.
    for _ in range(5):
        assert st.choose() is b


@pytest.mark.parametrize("tier", ["host", "external", "disk"])
@pytest.mark.parametrize("credit,expected", [(0.8, "b"), (0.2, "a")])
def test_kv_aware_transfer_credit(kv_scenario, tier, credit, expected):
    st = kv_scenario(extra={f"kv_penalty_{tier}": credit})
    tokens = st.tokens()
    a, b = st.workers
    st.publish_applied(a, [a.store(tokens[:8])])
    st.publish_applied(
        b,
        [
            b.store(
                tokens[:16],
                medium=tier,
                ownership="shared-pool" if tier == "external" else None,
            )
        ],
    )
    # GPU A: q=9. Transfer B: ceil(17-16*.8)=5, or ceil(17-16*.2)=14.
    winner = a if expected == "a" else b
    for _ in range(5):
        assert st.choose() is winner


def test_kv_aware_block_boundary_keeps_last_token_work(kv_scenario):
    st = kv_scenario(worker_options=[{"block_size": 4}, {"block_size": 2}])
    tokens = st.tokens()
    a, b = st.workers
    st.publish_applied(a, [a.store(tokens[:16])])
    st.publish_applied(b, [b.store(tokens[:14])])
    shorter = " ".join(PROMPT.split()[:16])
    # N=16: A can reuse only 12 (q=4); B reuses 14 (q=2).
    assert st.choose(shorter) is b
    # N=17: A reuses 16 (q=1); B reuses 14 (q=3).
    assert st.choose() is a


@pytest.mark.parametrize("connected", [True, False])
def test_kv_aware_only_credits_connected_tier_continuation(kv_scenario, connected):
    st = kv_scenario()
    tokens = st.tokens()
    a, b = st.workers
    st.publish_applied(a, [a.store(tokens[:8])])
    device = b.store(tokens[:4])
    # A missing parent leaves the host chain orphaned; it cannot contribute
    # to the request's continuous prefix even though its tokens are present.
    host = b.store(
        tokens[4:12],
        medium="host",
        parent=(device["block_hashes"][-1] if connected else b"missing-parent"),
    )
    st.publish_applied(b, [device, host])
    # A q=9; connected B q=ceil(17-4-8*.8)=7; orphaned B q=13.
    for _ in range(5):
        assert st.choose() is (b if connected else a)


@pytest.mark.parametrize("discovery", ["no_events", "empty_model", "zero_block"])
def test_kv_aware_missing_kv_metadata_uses_load(kv_scenario, discovery):
    options = {
        "no_events": {"events_enabled": False},
        "empty_model": {"model": ""},
        "zero_block": {"block_size": 0},
    }[discovery]
    st = kv_scenario(worker_options=[options, options])
    first = st.choose()
    other = next(w for w in st.workers if w is not first)
    gate = first.hold_next()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(st.request, stream=True)
        response = None
        try:
            assert gate.entered.wait(5)
            for _ in range(5):
                assert st.choose() is other
        finally:
            gate.output.set()
            gate.finish.set()
            response = pending.result(timeout=5)
            response.close()


@pytest.mark.parametrize("termination", ["error", "cancel", "retry"])
def test_kv_aware_failure_and_cancel_release_prefill(kv_scenario, termination):
    st = kv_scenario(
        extra=(
            {
                "disable_retries": False,
                "retry_max_retries": 2,
                "retry_initial_backoff_ms": 1,
                "retry_max_backoff_ms": 1,
                "retry_jitter_factor": 0,
            }
            if termination == "retry"
            else None
        )
    )
    tokens = st.tokens()
    a, b = st.workers
    st.publish_applied(b, [b.store(tokens[:14])])
    st.publish_applied(a, [a.store(tokens[:16])])
    assert st.choose() is a
    if termination in ("error", "retry"):
        before = len(a.captured)
        a.fail_next()
        with st.request() as response:
            if termination == "retry":
                assert response.status_code == 200, response.text
                assert (
                    response.json()["choices"][0]["message"]["content"] == a.worker_id
                )
                assert len(a.captured) == before + 2
            else:
                assert response.status_code == 503, response.text
    else:
        gate = a.hold_next()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(st.request, stream=True)
            try:
                assert gate.entered.wait(5)
                response = pending.result(timeout=5)
                response.close()
                assert gate.done.wait(5), "Cancelled gRPC request stayed active"
            finally:
                gate.output.set()
                gate.finish.set()
    # A leaked prefill slot costs 4, so B (3) would win; clean A costs 1.
    st.wait_choice(a)
    for _ in range(5):
        assert st.choose() is a


def test_kv_aware_prefill_decode_and_finish_change_routing(kv_scenario):
    st = kv_scenario()
    tokens = st.tokens()
    a, b = st.workers
    st.publish_applied(b, [b.store(tokens[:14])])
    st.publish_applied(a, [a.store(tokens[:16])])
    assert st.choose() is a
    gate = a.hold_next()
    # Use a separate client: the held streaming request overlaps probe requests.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(st.request, stream=True)
        try:
            assert gate.entered.wait(5), "Cached worker did not receive request"
            # A prefill: (1+1)*2=4; B idle: 3 -> B.
            st.wait_choice(b)
            gate.output.set()
            response = pending.result(timeout=5)
            lines = st.first_output(response)
            # A decode: 1*2=2; B idle: 3 -> A.
            st.wait_choice(a)
            gate.finish.set()
            assert b"data: [DONE]" in list(lines)
            response.close()
            assert gate.done.wait(5)
            # Reduce B's prefix to 12 -> q=5. A second held prefill costs
            # clean A (1+1)*2=4, but 6 if the first decode slot leaked.
            # The winners differ strictly; no equal-cost tie is assumed.
            st.publish_applied(b, [{"type": "AllBlocksCleared"}, b.store(tokens[:12])])
            second = a.hold_next()
            second_pending = executor.submit(st.request, stream=True)
            assert second.entered.wait(5)
            second_response = second_pending.result(timeout=5)
            try:
                st.wait_choice(a)
                second.output.set()
                second.finish.set()
                assert b"data: [DONE]" in list(second_response.iter_lines())
            finally:
                second.finish.set()
                second_response.close()
            assert second.done.wait(5)
            st.wait_choice(a)
        finally:
            gate.output.set()
            gate.finish.set()


@pytest.mark.parametrize("namespace", ["cache_salt", "lora"])
def test_kv_aware_cache_namespace_isolation(kv_scenario, namespace):
    st = kv_scenario()
    tokens = st.tokens()
    a, b = st.workers
    st.publish_applied(b, [b.store(tokens[:14])])
    st.publish_applied(a, [a.store(tokens[:16])])
    assert st.choose() is a
    if namespace == "lora":
        st.publish_applied(b, [b.store(tokens[:14], lora_name="adapter-b")])
        # A's longer base-model cache must not match a LoRA request.
        for _ in range(5):
            assert st.choose(lora_name="adapter-b") is b
    else:
        # With salt neither worker can claim placement credit. Hold an A
        # decode slot: load-only chooses B (17 vs 34), whereas wrongly using
        # cache credit still chooses A (2 vs 3).
        gate = a.hold_next()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(st.request, stream=True)
            response = None
            try:
                assert gate.entered.wait(5)
                gate.output.set()
                response = pending.result(timeout=5)
                lines = st.first_output(response)
                st.wait_choice(a)
                for _ in range(5):
                    assert st.choose(cache_salt="tenant-b") is b
                gate.finish.set()
                assert b"data: [DONE]" in list(lines)
            finally:
                gate.finish.set()
                if response is not None:
                    response.close()


def test_kv_aware_positive_temperature_samples_equal_cost_workers(kv_scenario):
    st = kv_scenario(extra={"router_temperature": 100})
    # Equal costs give equal softmax weights. A broad bound makes accidental
    # statistical failure negligible while rejecting deterministic picking.
    counts = collections.Counter(st.choose().worker_id for _ in range(200))
    for worker in st.workers:
        assert counts[worker.worker_id] >= 20, counts
