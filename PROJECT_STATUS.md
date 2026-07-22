# Project Status Report — SLM Router

**Date:** 2026-07-22
**Assessed by:** Automated review (parallel code-analysis subagents)
**Scope:** `src/slm_router/`, `tests/`, `configs/`, `data/`, `results/`, packaging

---

## Verdict: **Not complete — ~90% built, but not end-to-end wired or ever run**

The project is a real, substantial implementation (62 Python files, ~5,900 LOC),
**not** a skeleton. Every module described in the README exists and contains genuine
algorithmic content. However, concrete gaps keep it from being a finished deliverable:
serving is disconnected from the learned router, the self-improvement loop is not
reachable from the CLI, and the eval pipeline has never been run.

---

## What's Done ✅

| Module | Status | Notes |
|--------|--------|-------|
| `models/` | Complete | Async OpenRouter client, registry, cache, rate-limiter, batch runner |
| `ml_core/` | Complete | LinUCB + Thompson bandits, offline RL (FQE), cascade, difficulty estimator, confidence + calibration (temperature scaling, conformal, ensemble) — real math |
| `eval/` | Complete* | Dataset loaders (MMLU/GSM8K/GPQA/HotpotQA/HumanEval), scorers (exact/numeric/code-exec/LLM-judge), metrics (ECE/regret/Pareto), workflows |
| `trace/` | Complete | SQLite store, idempotent insert, oracle-matrix + best-model queries, parquet export |
| `feedback/` | Complete | FQE-gated self-improvement loop (online update + batch retrain w/ promotion gate) |
| `obs/` | Complete | Noop/OTel tracing seam, degrades cleanly |
| `configs/` | Complete | Real reward weights, datasets, model registry + OpenRouter pricing — no placeholders |
| `pyproject.toml` / `.gitignore` | Complete | Production-shaped packaging, CLI entrypoint, dep groups |

\* one intentional stub in eval — see below.

---

## What's Incomplete ❌

1. **Broken serving integration** — `serve/app.py:27` imports `slm_router.routing`
   (does not exist; real path is `slm_router.ml_core.routing`) and a `get_policy`
   factory that exists nowhere. The `ImportError` is silently swallowed, so `/route`
   **always** falls back to a trivial cheapest-model heuristic — the learned policies
   are never used in serving.

2. **Two no-op CLI commands** — `cli.py:56` `collect_traces` and `cli.py:83`
   `train_router` just print a message and do nothing; the feedback loop is not wired
   to the CLI.

3. **One explicit eval stub** — `eval/metrics/judge_reliability.py:120`
   `position_bias_check()` returns placeholder values (documented as a stub).

4. **Pipeline never run** — `data/` and `results/` contain only `.gitkeep`. No oracle
   matrix, no `router_eval.json`, no `pareto.png`. The `build-oracle → evaluate → serve`
   workflow has never been executed in this checkout.

5. **Thin tests** — only 8 test functions covering config / registry / trace-store /
   metrics. Routing, scoring, serving, dataset loaders, and the feedback loop are
   untested.

---

## Remaining Work to Reach "Complete"

- [x] Fix `serve/app.py` import (`slm_router.ml_core.routing`) and add a `get_policy` factory — *done 2026-07-22*
- [x] Wire `train_router` / `collect_traces` CLI commands to `SelfImprovementLoop` — *done 2026-07-22*
- [x] Implement `position_bias_check()` in `judge_reliability.py` — *done 2026-07-22*
- [x] Fix `ModelSpec.id` vs `.model_id` mismatch in `bandit.py`/`cascade.py`/`offline_rl.py` (would `AttributeError` at runtime) — *done 2026-07-22*
- [ ] Run the eval pipeline (`build-oracle → evaluate`) to produce result artifacts — **requires live OpenRouter API key + network; not run**
- [ ] Broaden test coverage (routing policies, scoring, serving, dataset loaders, feedback loop) — **not done**

### Fixes applied 2026-07-22 (via Sonnet subagents)
- Added `CheapestFirstPolicy` + `get_policy(name, **kwargs)` factory to `ml_core/routing/__init__.py`
  (dispatches cheapest / linucb / thompson / offline_rl / cascade).
- `serve/app.py:27` import corrected; learned policy now reachable, cheapest-first kept only as a genuine error fallback.
- `cli.py` `collect_traces` queries the trace store; `train_router` builds incumbent+candidate
  policies and runs `SelfImprovementLoop.retrain_batch` with a promotion gate.
- `position_bias_check()` now measures judge flip-rate across swapped (A,B)/(B,A) orderings.
- Replaced all `.model_id` → `.id` (real `ModelSpec` field) in bandit/cascade/offline_rl.

---

## Notes

- No `TODO` / `FIXME` / `NotImplementedError` markers anywhere in source.
- `.env.example` present but unreadable in the review sandbox (protected `.env*` path);
  inferred to declare `OPENROUTER_API_KEY` as a template value.
- Numerous `try/except` blocks are legitimate optional-dependency / defensive patterns,
  not unfinished logic.
