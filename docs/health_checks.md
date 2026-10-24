# Worker health checks and inference probes

The router runs two independent, optional mechanisms against workers. They look
similar and answer different questions, so they are documented together but
kept strictly separate:

| | Worker health check | Inference probe |
|---|---|---|
| What it asks | is the process serving? | can it actually generate a token? |
| Signal | `GET {--health-check-endpoint}` | `POST /v1/completions`, `max_tokens: 1` |
| Effect | **decides routing**: healthy workers get traffic | **reports only** |
| Flag | `--health-check-endpoint` (+ thresholds) | `--inference-probe` |
| Default | enabled | disabled |

The distinction is deliberate. Worker health drives node selection and is
therefore conservative, threshold-based, and operator-tuned. The inference probe
is a diagnostic: it produces metrics and log lines, and nothing else.

## Worker health check (drives routing)

The periodic checker probes every registered worker — including workers added
later by Kubernetes service discovery. A worker that accumulates
`--health-failure-threshold` consecutive failures is marked unhealthy and
excluded from routing until it passes `--health-success-threshold` consecutive
probes.

```bash
vllm-router \
    --worker-urls http://127.0.0.1:8000 http://127.0.0.1:8001 \
    --health-check-endpoint /health \
    --health-check-interval-secs 30 \
    --health-failure-threshold 3 \
    --health-success-threshold 2
```

A `GET /health` probe validates that the worker process is alive. It says
nothing about whether the worker can serve a request, which is the gap
[issue #158](https://github.com/vllm-project/router/issues/158) reports: the
router marks workers healthy while chat completions fail.

Probe requests never touch worker load counters, so the periodic checker does
not perturb load-based routing (see PR #216), and they are excluded from
OpenTelemetry server spans.

## Inference probe (reports only)

The inference probe asks the question `/health` cannot: it sends a real
one-token generation to every registered worker on each sweep.

```bash
vllm-router \
    --worker-urls http://127.0.0.1:8000 http://127.0.0.1:8001 \
    --inference-probe \
    --inference-probe-interval-secs 60 \
    --inference-probe-timeout-secs 30 \
    --inference-probe-concurrency 32
```

```bash
# Python launcher
vllm-router --inference-probe ...
python -m vllm_router.launch_router --inference-probe ...
```

In the YAML config the probe is a **top-level** key, a sibling of
`health_check` rather than a child of it:

```yaml
inference_probe:
  enabled: true
  interval_secs: 60
  timeout_secs: 30
  concurrency: 32
```

The placement is deliberate: the probe never affects worker health, so it must
not be reachable through a health key.

The request is `POST {worker}/v1/completions` with
`{"prompt": "hi", "max_tokens": 1}` — the same endpoint the router forwards user
traffic to. A worker is reported as serving only when it **completes the
generation and returns a well-formed completion choice**. A 2xx status alone is
not enough: an engine that answers without producing a token is reported as
failing.

### What it does not do

The probe is a diagnostic, and this is enforced by construction:

- it never marks a worker unhealthy or healthy;
- it never feeds the circuit breaker;
- it never changes routing, pair selection, or drain state;
- it never touches worker load counters.

Worker health remains exclusively the business of the health check above. A
defect in the probe can therefore produce a wrong *report*, never an outage,
which is what makes it safe to enable on a running cluster. Acting on a report
— draining a node, filing a ticket — is a decision for an operator or an
external controller consuming the metrics.

### Authentication

When `--api-key` (the worker authorization key) is set, probes carry it as
`Authorization: Bearer <key>`, the same credential the proxy sends. Without it,
workers behind an API key would reject every probe and be reported as failing
for the wrong reason.

### Coverage

The sweep iterates over **all registered HTTP workers**, so every node of the
P/D topology is covered, including nodes that never receive traffic. Workers
currently being verified by their own real traffic are skipped rather than
probed (see *Continuous traffic* below), and gRPC workers are excluded (see
*Other modes*). This is deliberately different from the router-level
`GET /health_generate` endpoint, which samples a single P/D pair per call.

### Cost and tuning

Each sweep costs one real generation per worker per interval. Size the interval
and timeout for inference latency, not for a socket round trip: the default
30s timeout reflects that, and a timeout tuned for a liveness probe will report
false failures on a busy or still-loading worker.

A sweep probes at most `--inference-probe-concurrency` workers at a time, so one
sweep lasts about `ceil(workers / concurrency) * timeout` in the worst case,
when every probe has to time out. Raise the concurrency on large fleets if a
sweep can outlast the interval: workers are probed in fixed batches, so a single
stalled batch delays the batches behind it and their gauges go stale. Being
bounded also keeps a sweep from firing thousands of simultaneous generations at
a cluster that is already struggling.

The first sweep happens one full interval after startup, not immediately, so a
rollout does not stamp `inference_ok = 0` on workers that are still loading.

### Continuous traffic: a worker may be skipped

A worker that has demonstrably served a real proxied generation within the last
interval already answered the probe's question, so the sweep **skips** it
instead of spending another real one-token generation on it. Under continuous
traffic the same busy workers can therefore go many intervals without ever being
probed, which is intended, not a gap.

A skip is reported, never silent: the sweep increments
`vllm_router_worker_inference_probe_total{result="fresh_traffic"}`, sets
`vllm_router_worker_inference_ok` to `1`, and refreshes
`vllm_router_worker_inference_last_probe_ts` exactly as a passing probe would.
The `fresh_traffic` label is what keeps "skipped because it is serving" apart
from "probed and passed" (`result="ok"`), so a dashboard can show how much of
the fleet is being verified by real traffic rather than by the probe.

Skipping is self-healing. Evidence is only credited for a response that
actually carries generated tokens (the same bar a passing probe uses: a non-empty
choice or a positive `completion_tokens`, never a bare 2xx). A worker that stops
generating — a hung stream, a stalled engine — stops being credited, its
evidence ages out within one interval, and the next sweep probes it again.

### Prefill-role workers

The sweep covers every registered worker, including P/D **prefill**-role
instances, because a prefill instance is a full vLLM engine and the failure
class in #158 is engine-local. A healthy prefill instance is expected to serve
the standalone one-token completion and be reported `ok`.

The standalone request deliberately carries **no `kv_transfer_params`**. In a
real P/D pair that envelope is built from the *decode* peer's coordinates, and
for the Mooncake connector it also writes the connector's `transfer_id`
bookkeeping on the prefill side. Sending the envelope from a worker-level probe
would overwrite that bookkeeping with an id no decode peer is waiting on, and
would reserve KV blocks that are never released by a decode handshake. A probe
is not a P/D pair, so it must not impersonate one.

The consequence is that a prefill-role instance which cannot serve a standalone
one-token completion is reported as **failing**, even though it may still be
able to serve prefill on a real paired request. Read a failing prefill worker as
"this engine cannot complete a generation on its own" — an engine-level defect
worth investigating — rather than as proof that paired P/D traffic is broken.
Conversely, a passing prefill worker does not prove the paired
prefill→decode path works: see *What it cannot see* below.

### What it cannot see

The probe is **worker-local**. It validates that a node can run inference and
return a token, and it does **not** validate prefill→decode coordination — KV
transfer or the bootstrap socket.

That boundary is deliberate. The P-side `kv_transfer_params` envelope is
connector-specific and is built from the *decode* peer's coordinates
(`VllmPDRouter::build_prefill_kv_transfer_params` takes `decode_base` and
`decode_dp_rank`; NIXL pull needs `remote_block_ids`/`remote_engine_id`/
`remote_host`/`remote_port`, Mooncake needs `transfer_id`, MoRIIO WRITE needs
the remote TP/DP sizes). A worker-level probe has no decode peer to target.
Verifying a full paired P/D path is separate work.

### Metrics

| Metric | Meaning |
|---|---|
| `vllm_router_worker_inference_probe_total{worker,result}` | probe outcomes; `result` is `ok`, `fresh_traffic`, `request_failed`, `http_error`, `unreadable_body`, `no_choice`, `abnormal_finish`, or `zero_tokens` |
| `vllm_router_worker_inference_ok{worker}` | last sweep's result (`1` served or was skipped as `fresh_traffic`, `0` failed) |
| `vllm_router_worker_inference_probe_latency_ms{worker}` | latency of successful probes |
| `vllm_router_worker_inference_last_probe_ts{worker}` | Unix seconds of the last probe — or, for a skipped sweep, of the real generation that qualified it. `0` once the worker is removed from the registry |

`vllm_router_worker_inference_ok` is the metric to alert on, qualified by the
recency of `vllm_router_worker_inference_last_probe_ts` so a series orphaned by
worker removal cannot keep firing: it is `0` exactly for the workers that are
up, that the health checker considers healthy, and that cannot serve. Failures
also log a warning naming the worker, the classifier, the error detail, and
whether the health checker currently considers the worker healthy — which makes
"healthy but cannot infer" immediately greppable.

### Other modes

- gRPC workers (`grpc://`) are excluded from the sweep by design: the probe is
  HTTP-only (it POSTs to `/v1/completions`), so probing a gRPC-only worker would
  report `request_failed` and pin `inference_ok = 0` on a healthy serving
  worker. gRPC workers keep using the gRPC health service.
- `--health-check-endpoint` is unrelated to the inference probe and does not
  affect it.
