# SLM Router

**Bounded inference plans. Measured evidence. Targeted recovery. Replayable decisions.**

SLM Router chooses an execution plan under quality, token-price, latency, privacy and reliability constraints. It can reject unsafe or insufficiently supported plans, execute the selected plan, learn from trusted outcomes, and use a bounded measurement budget to refresh uncertain decisions.

The current application also includes a local operator **Test Bench** at `/dashboard/`. It can authenticate to the router, build OpenAI-compatible test requests, inspect the selected plan and rejected alternatives, and verify deterministic decision replay without storing the router credential, prompt or response in browser storage.

The current public pre-release is **2.0.0a2 — adaptive evidence preview**. It integrates the most important foundations from the proposed feature set without claiming that the full v2 production boundary has been met. The legacy `slm_router` package remains available; the default CLI uses `inference_control`.

> **Benchmark status:** the included experiments are synthetic. Real RouteLLM and vLLM Semantic Router adapters are implemented, but those competitors have **not** been executed for this release. No public-dataset, live-provider or “beats all benchmarks” claim is made. The generated reports explicitly block that claim.

> **Known failed stress case:** the included hidden-latency-spike run had 8 latency violations in 40 decisions and no detection within that horizon. Certificates depend on stable, applicable evidence; this preview does not guarantee safety through arbitrary drift. See [validation](docs/release/VALIDATION.md).

## Install and try it

Python 3.11–3.13 is supported. The final local release cycle ran on Python 3.12.14; an earlier validation cycle ran on Python 3.13. The core install does not require PyTorch, sentence-transformers, a GPU, a model account or Redis.

```bash
git clone https://github.com/sonusharma26/slm-router.git
cd slm-router
python -m pip install -e ".[dev]"
slm --help
slm policy validate examples/v2/policy.yaml --endpoints examples/v2/endpoints.json
slm demo --out results/my-demo
slm benchmark static --synthetic --requests 1200 --out results/my-static
```

The demo uses no network. It issues a fixture certificate, executes a mock provider, detects an endpoint revision change, routes to a supported fallback, spends at most 100 synthetic probe calls, re-certifies the affected endpoint, and verifies decisions after a SQLite restart. Choose a new output directory: the demo refuses to overwrite existing audit state.

To run from a checkout without an editable install, put `src` on `PYTHONPATH` and use `python -m inference_control.cli` instead of `slm`.

## What is integrated

| Area | Implemented release behavior |
|---|---|
| Plan search | Direct, two-call error cascade, two-call verify/escalate, two-branch parallel/select and abstention. Compound plans require their own measured joint evidence. |
| Conditional evidence | Frozen query features, exact task/governance/schema slices, length buckets, kNN estimates, distinct train/calibration requests, hashes, sample counts, staleness and uncertainty. |
| Enforcement | Bound-aware eligibility, cumulative call/spend reservation, exact-request certificates, dependency invalidation, pre-dispatch rechecks, no automatic provider retries and explicit unknown accounting. |
| Adaptive recovery | Decision-boundary acquisition, durable UTC daily budgets, snapshot drift, windowed quality/latency/failure monitoring, targeted metric invalidation, and recovery acknowledgement. |
| Outcome authority | Delayed decision/execution correlation; deterministic/application/human sources; calibrated judge expiry; weak-label exclusion; supersession and disputes. |
| Safe rollout | Durable draft → offline-validated → shadow → canary → active transitions, sticky bounded canary allocation and stop/rollback rules. Candidate creation remains operator-driven. |
| Persistence | Hash-chained SQLite events, durable execution claims, policy/snapshot/map history, outcome learning recovery, detector windows, lifecycle state and deterministic decision replay. |
| Adoption | OpenAI-compatible provider APIs, a thin non-streaming `/v1/chat/completions` route, structured-output/tool admission, Prometheus text output and a dependency-free local test dashboard. |
| Operator Test Bench | In-memory bearer-key connection, model discovery, request presets, metadata and schema/tool controls, payload preview, decision evidence, rejected-alternative inspection, run history and replay verification. |
| Benchmark Lab | Disjoint data splits, trivial/kNN/threshold baselines, real competitor seams, policy diff, matched operating-point curves, dynamic faults and sparse-refresh experiments. |

The selected scope is intentionally narrower than a general LLM gateway. The dashboard is a local testing and decision-inspection surface, not a multi-tenant administration or billing console. There is no agent orchestration, arbitrary production policy code, RAG stack, vector database, billing system or Kubernetes operator.

## A bounded policy

```yaml
policy_id: support
version: "17"
minimum_quality: 0.90
cost:
  expected_max: 0.01
  absolute_max: 0.03
latency:
  p95_max_ms: 2500
privacy:
  providers: [local]
  boundary: local
plans:
  allowed: [direct, cascade, verify_escalate, abstain]
  max_calls: 2
  verifier: nonempty
  max_candidates: 512
evidence:
  require_certificate: true
  min_samples: 30
  max_age_seconds: 86400
  certificate_ttl_seconds: 3600
on_infeasible: abstain
```

`nonempty` is only a response sanity check, **not a correctness verifier**. Its behavior must be represented in observed end-to-end plan results. A YAML plan declaration does not manufacture evidence. Unknown fields, contradictory limits and missing verifier/selector names fail validation. A structurally valid policy can still abstain on a particular request because its statistical requirements are infeasible.

## Run the service

The example service configuration uses placeholder local endpoints. Replace model IDs, prices, capability declarations and revision metadata with values you have verified. Provide an application-specific capability-map export or collect trusted observations through the Python control-plane API; an empty map safely abstains.

```bash
# Bash; use the equivalent environment-variable syntax in your shell.
export SLM_ROUTER_API_KEY="replace-with-an-independent-long-random-credential"
slm serve --config examples/v2/service.yaml --host 127.0.0.1 --port 8000
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $SLM_ROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"local-policy","messages":[{"role":"user","content":"Explain this function."}],"max_completion_tokens":256}'
```

Use one process/one worker. This is a single trusted-operator control plane, not a hardened multi-tenant public service. The same credential protects operator and inference routes. See [runtime and safety boundaries](docs/release/SAFETY.md) before connecting real providers.

## Use the Test Bench dashboard

With the service running, open:

```text
http://127.0.0.1:8000/dashboard/
```

Enter the value of `SLM_ROUTER_API_KEY` and choose **Connect**. The browser uses that credential for authenticated `/health`, `/v1/models`, `/v1/chat/completions` and decision-inspection calls. The page keeps the key and run history in memory only; it does not use `localStorage`, `sessionStorage` or cookies.

The Test Bench provides:

- request presets plus editable system and user messages;
- application, session, task, traffic-slice, privacy and token metadata;
- optional JSON-schema and tool-call request controls;
- the exact outbound payload before execution;
- provider response, latency, usage, spend, decision ID and execution ID;
- selected-plan estimates and eligible or rejected alternatives;
- raw decision JSON and deterministic replay verification.

The **Exact answer - route** preset matches the included NVIDIA pilot evidence. The Reasoning, JSON, Tool and Privacy gate presets intentionally exercise eligibility or safe-abstention behavior and are not expected to route unless matching evidence and capabilities exist.

## NVIDIA live functional pilot

The current NVIDIA profile connects the router to two OpenAI-compatible NVIDIA Build endpoints. It is a small functional-routing pilot, not a quality benchmark or production calibration. See [`examples/nvidia-live/README.md`](examples/nvidia-live/README.md) for the provider models, pilot generation procedure and safety boundary.

Create a local `.env.local` file containing two separate credentials:

```dotenv
NVIDIA_API_KEY="replace-with-your-provider-credential"
SLM_ROUTER_API_KEY="replace-with-an-independent-random-value-of-at-least-24-characters"
```

The NVIDIA key authenticates upstream provider calls. `SLM_ROUTER_API_KEY` protects this router's local API and must contain at least 24 characters. Never commit `.env.local`, print either secret in logs or reuse the provider key as the router key.

Run a fresh three-call pilot before starting a newly configured service:

```powershell
python scripts/run_nvidia_live_pilot.py --out results/nvidia-live-pilot
Get-Content results/nvidia-live-pilot/pilot-report.json
```

The pilot refuses to overwrite an existing output directory. Point `examples/nvidia-live/service.yaml` at the newly generated `endpoints.json`, `capability-map.json` and ledger location before serving it. A successful pilot is expected to remain `uncertified`: one observation per endpoint is enough for this functional route check, but not enough for a meaningful production certificate.

### Run the current profile with Docker

Generate `results/nvidia-live-pilot` first. The checked-in service profile reads those generated artifacts through the `/app/results` bind mount:

```powershell
docker build --file Dockerfile.sandbox --tag slm-router-sandbox:local .

docker run --rm --name slm-router-api `
  --env-file .env.local `
  --publish 127.0.0.1:8000:8000 `
  --mount "type=bind,source=$((Resolve-Path .\results).Path),target=/app/results" `
  slm-router-sandbox:local `
  -m inference_control.cli serve `
  --config examples/nvidia-live/service.yaml `
  --host 0.0.0.0 `
  --port 8000
```

The bind-mounted SQLite ledger and pilot artifacts remain on the host. Rebuild the image after changing application or dashboard source. Recreate the container after changing `.env.local`, because Docker reads `--env-file` when the container starts.

### Why `ESTIMATE_MISSING` occurs

Evidence is request-conditioned. A capability-map observation applies only when relevant request attributes match, including application, task, traffic slice, privacy class, feature version, input/output length buckets, endpoint revision and any structured-output or tool-schema hashes.

For the included NVIDIA pilot, use these dashboard values:

| Field | Pilot value |
|---|---|
| Application | `nvidia-live-pilot` |
| Task | `exact-number` |
| Traffic slice | `pilot` |
| Privacy | `public` |
| Maximum output tokens | `512` |
| System message | empty |
| User message | `Return 9.` |

`ESTIMATE_MISSING` means no current observation matches the request context. It is a safe abstention, not a provider failure. First use **Exact answer - route** to eliminate metadata mismatch. If that exact preset also fails, regenerate the live pilot evidence and update the service configuration. The included policy sets `evidence.max_age_seconds: 3600`, so its observations expire one hour after collection.

## API surfaces

All API routes, including health and inspection routes, require the configured bearer credential. The dashboard shell itself contains no credential or control-plane data.

| Route | Purpose |
|---|---|
| `GET /dashboard/` | Load the local Test Bench shell. |
| `GET /health` | Check service status, ledger verification and capability-map version. |
| `GET /v1/models` | List the active router policy as an OpenAI-compatible model. |
| `POST /v1/chat/completions` | Decide, execute and return a non-streaming OpenAI-compatible response. |
| `POST /v2/decide` | Create a decision without executing it. |
| `POST /v2/execute` | Execute an existing or supplied decision request. |
| `GET /v2/decisions/{decision_id}` | Inspect the persisted decision record. |
| `GET /v2/decisions/{decision_id}/replay` | Rebuild the historical decision and compare fingerprints. |
| `POST /v2/outcomes` | Add a delayed trusted outcome. |
| `POST /v2/outcomes/{outcome_id}/dispute` | Dispute a previously recorded outcome. |
| `GET /metrics` | Return Prometheus-compatible control-plane metrics. |

Streaming is rejected before provider inference. The `/v1/chat/completions` implementation is intentionally a narrow text-oriented compatibility surface rather than a complete OpenAI API replacement.

## Validated synthetic benchmark snapshot

The final offline run used seed 42, 1,200 generated requests, 207 held-out test requests and 400 bootstrap resamples. Values below are synthetic fixture measurements, not live-model results.

| Router | Mean quality | Mean simulated cost/request | Latency p95 |
|---|---:|---:|---:|
| SLM Router | 0.963344 | $0.00040348 | 207.83 ms |
| Best single | 0.965087 | $0.00057600 | 195.22 ms |
| kNN baseline | 0.883971 | $0.00021797 | 193.02 ms |
| Cheapest | 0.660502 | $0.00007680 | 86.95 ms |

At this operating point, SLM Router used **29.95% less simulated serving spend** than the best-single baseline with slightly lower mean quality. Certificate coverage was 100% and no static fixture constraint violation occurred. This does not establish superiority: the data is synthetic, external competitors were not run, and the generated claim gate remains blocked.

The dynamic suite also retained its unfavorable result: under a hidden 3× latency shift, the impact strategy incurred **8 violations in 40 decisions** and did not detect the shift within the run. Sparse-refresh strategies tied on the small fixture, so no adaptive advantage or ≤25%-of-exhaustive claim is made.

Release validation: **85 offline tests passed** on Python 3.12.14 with current declared dependencies. No GitHub Actions workflow was run for this preparation cycle. See [validation details](docs/release/VALIDATION.md) and the [benchmark protocol](docs/release/BENCHMARK_LAB.md).

## Benchmark commands

```bash
slm benchmark dynamic --scenario all --steps 40 --initial-requests 2400 --out results/my-dynamic
slm benchmark sparse --steps 40 --out results/my-sparse
slm benchmark import-llmrouterbench /path/to/curated/results/bench \
  --endpoints /path/to/pinned-model-pool.json --out data/my-benchmark.json
slm benchmark static --dataset data/my-benchmark.json --out results/my-real-benchmark
slm benchmark policy-diff --dataset data/my-benchmark.json \
  --current configs/current.yaml --candidate configs/candidate.yaml
slm replay --ledger results/my-demo/demo.sqlite3 --decision-id DECISION_ID
```

The same public model pool and held-out requests are used for each participant. RouteLLM's two-model comparison requires restricting **the entire dataset and all routers** to that pair. Missing router overhead cost remains unknown. Missing request-level latency remains missing. A full-information response matrix cannot score an unobserved verification or parallel-selection outcome.

See [Benchmark Lab](docs/release/BENCHMARK_LAB.md) for competitor setup, interpretation and evidence requirements. Exact validation commands and result boundaries are in [VALIDATION.md](docs/release/VALIDATION.md).

## What a certificate means

A certificate is an auditable statistical statement under explicit sampling and stability assumptions—not a guarantee that an individual answer is correct. Quality uses a bounded-score local-mean lower confidence bound; latency uses an exact-binomial order-statistic confidence bound on a local population p95. Candidate comparisons consume the declared risk budget. Sparse or stale calibration results in no certificate.

Static token-price reservations are different from statistical claims. They depend on correctly pinned prices, verified input/token overhead, a provider that honors the output cap and no hidden billing. Missing usage, provider errors or revision mismatch cannot be silently converted into free or safe execution.

## Repository guide

- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md): selected scope and checkpoints.
- [`docs/release/IMPLEMENTATION.md`](docs/release/IMPLEMENTATION.md): code paths, learning, certificates, persistence and rollout details.
- [`docs/release/BENCHMARK_LAB.md`](docs/release/BENCHMARK_LAB.md): datasets, competitors, experiment boundaries and reproduction.
- [`docs/release/SAFETY.md`](docs/release/SAFETY.md): operational assumptions and unsupported behavior.
- [`docs/release/VALIDATION.md`](docs/release/VALIDATION.md): what actually ran.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): development setup and contribution expectations.
- [`SECURITY.md`](SECURITY.md): private vulnerability reporting and deployment boundaries.
- [`CHANGELOG.md`](CHANGELOG.md): release changes and compatibility notes.
- [`LICENSE`](LICENSE): MIT license.

`slm-legacy` exposes the previous CLI after installing `.[legacy]`. Historical roadmap/checklist documents are retained for context, not as current release-verification evidence. The obsolete v0.1 `uv.lock` was removed because it described a different dependency graph; do not use old `uv sync --frozen` instructions. A fresh lock can be generated for your deployment with `uv lock`.
