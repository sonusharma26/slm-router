# Self-Improving Model Router for SLM–LLM Systems

A research platform that learns to route queries between Small Language Models
(Phi, Gemma, Qwen, Llama 8B) and frontier LLMs to keep LLM-level quality at a
fraction of the cost — and improves its policy from its own logged traces.

```
User Query → Meta-Router (learned policy) → {SLM | small LLM | frontier LLM}
                                          → Confidence Engine (calibrated)
                                          → Final Response
                                          → Trace logged → feedback loop re-trains
```

Objective: `Reward = Quality − λ·Cost − β·Latency`.

## Layout

```
src/slm_router/
  models/      OpenRouter client, registry, cache, rate-limit, async batch
  ml_core/     difficulty estimator · confidence + calibration (ECE) · routing (LinUCB / Thompson / offline-RL / cascade)
  eval/        datasets (MMLU/GPQA/HotpotQA/GSM8K/HumanEval) · scoring · metrics (ECE/regret/Pareto) · workflows
  trace/       SQLite trace store (metrics source of truth) + oracle-matrix queries
  feedback/    self-improvement loop with FQE promotion gate
  obs/         vendor-neutral tracing seam (OTel → Langfuse/Phoenix), noop by default
  serve/       FastAPI POST /route
  cli.py       `slm` entrypoint
configs/       config.yaml (reward weights, datasets) · models.yaml (registry + pricing)
```

## Setup

```bash
uv sync                       # install deps (defined in pyproject.toml)
cp .env.example .env          # add OPENROUTER_API_KEY
```

## Workflow

```bash
# 1. Build the oracle matrix: every candidate model × every eval item, scored once.
uv run slm build-oracle --limit 50

# 2. Evaluate routing policies against the frozen matrix (free, no API calls):
#    oracle vs always-SLM vs always-frontier vs random vs learned.
uv run slm evaluate          # -> results/router_eval.json + results/pareto.png

# 3. Serve live routing.
uv run slm serve             # POST /route {"query": "..."}
```

The oracle build is idempotent (re-running resumes) and gated behind a cost
estimate; the response cache means repeated dev runs don't re-pay.

## Tests

Offline tests cover the trace store, registry cost math, and the ECE/regret/Pareto
metrics on synthetic data:

```bash
uv run pytest
```
