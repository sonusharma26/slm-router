# Project Roadmap — SLM Router

**Date:** 2026-07-22
**Status:** Code wired end-to-end (see [PROJECT_STATUS.md](PROJECT_STATUS.md)); pipeline never executed, no data/results artifacts yet.
**Purpose:** Sequenced plan to take the project from "built" to "trained, evaluated, and hardened."

---

## Where we are

- **~90% built, 0% run.** Every module has real algorithmic content; serving and the
  self-improvement loop are now reachable after the 2026-07-22 fixes.
- **No data exists.** `data/` and `results/` contain only `.gitkeep`. No oracle matrix,
  no `router_eval.json`, no `pareto.png`.
- **Fixes verified by read + `ast.parse` only** — never executed. First real run will
  surface anything static checks missed.

The critical implication: **nothing in Phases 2–4 works until Phase 1 produces traces.**

---

## Phase 1 — Bring it to life *(prerequisite for everything)*

The learned router and feedback loop are reachable but have zero data to act on.

### 1.1 Environment
- [ ] Set `OPENROUTER_API_KEY` in `.env` (template is `.env.example`).
- [ ] Decide a hard budget cap for the first run. Treat `build-oracle` as the single
      most token/cost-hungry operation in the whole project.

### 1.2 Build the oracle matrix — **start tiny**
- [ ] `slm build-oracle` on a *small slice*: ~50 questions × 4 models, GSM8K + MMLU only.
- [ ] The oracle matrix is ground-truth quality / cost / latency per (model × question).
      It is what every policy trains and evaluates against.
- [ ] **Do NOT run the full dataset suite first.** Confirm scoring, tracing, and cost
      accounting work on the small slice, inspect the SQLite trace store, then scale.

### 1.3 First evaluation
- [ ] `slm evaluate` → produces `results/router_eval.json` + `results/pareto.png`.
- [ ] Establishes the baseline: cheapest-first heuristic vs. oracle-best.

**Gate:** complete 1.1–1.3 on ~50–100 items before spending real budget. This is the
single go/no-go checkpoint for the project.

---

## Phase 2 — Training the router

Once traces exist. Most of this is **offline and costs no tokens.**

### 2.1 Offline training (cheap, no API)
- [ ] Train LinUCB, Thompson Sampling, and offline-RL policies on collected oracle traces.
- [ ] Use **FQE (Fitted Q Evaluation)** to score candidate policies *without live calls* —
      this is the entire point of the offline-RL setup and the cheapest way to iterate.

### 2.2 Batch retrain with promotion gate
- [ ] `slm train-router` → runs `SelfImprovementLoop.retrain_batch(candidate)`.
- [ ] Candidate is promoted over the incumbent **only** if it beats it by `--margin`
      under FQE. Tune `--margin`, `--context-dim`, and the reward weights
      (`lambda_cost`, `beta_latency`, `correctness_weight`).

### 2.3 Online loop (later)
- [ ] Once serving collects live traffic, feed traces to `SelfImprovementLoop.step_online`
      for continuous updates. Requires Phase 1 + a running server.

---

## Phase 3 — Evaluations that matter

Run against the artifacts from Phases 1–2.

### 3.1 Routing quality
- [ ] **Regret** vs. oracle-best per query.
- [ ] **Pareto frontier**: quality vs. cost (and vs. latency).
- [ ] **SLM retention rate**: fraction of queries kept on cheap SLMs without quality loss.

### 3.2 Calibration
- [ ] **ECE** before/after temperature scaling.
- [ ] **Conformal coverage**: does the predicted set contain the truth at the target rate?
- [ ] Ensemble uncertainty sanity checks.

### 3.3 Judge reliability
- [ ] Run `position_bias_check` on LLM-judge pairs — confirm **flip-rate is low**
      (`position_bias_detected == False`) *before* trusting any judge-scored metric.
- [ ] If bias is detected, always score with swapped orderings and average.

---

## Phase 4 — Hardening

### 4.1 Tests (offline, no API — do this early)
- [ ] Routing policies (LinUCB / Thompson / offline-RL / cascade / cheapest).
- [ ] Scorers (exact-match / numeric / code-exec / LLM-judge).
- [ ] Dataset loaders (MMLU / GSM8K / GPQA / HotpotQA / HumanEval).
- [ ] Feedback loop (`step_online`, `retrain_batch`, promotion gate).
- [ ] Serving `/route` path with a stubbed model client.
- Current coverage: only ~8 test functions (config / registry / trace-store / metrics).

### 4.2 Serving validation
- [ ] Actually hit `/route` and confirm the **learned policy** fires, not the
      cheapest-first error fallback.

### 4.3 Observability
- [ ] Enable the OTel / Langfuse tracing seam for a live run and confirm spans emit.

---

## Recommended immediate next steps (both low-token)

Do the offline, near-free things first to de-risk before spending API budget:

1. **`uv run pytest`** — confirm the 2026-07-22 wiring holds under *execution*, not just
   syntax. No network.
2. **Expanded offline test suite** (Phase 4.1) — no network at all.
3. **Draft a small-scale `build-oracle` config** (~20 items, 2 models) so the first live
   run is cheap and controlled.

Then decide the real eval budget and run Phase 1.

---

## Cost/risk summary

| Activity | Token/API cost | Risk if skipped |
|----------|----------------|-----------------|
| `pytest` / offline tests | None | Wiring bugs surface only at runtime |
| Offline training + FQE | None | — |
| Small `build-oracle` (~20–50 items) | Low | — |
| Full `build-oracle` (all datasets × all models) | **High** | — |
| Online loop / live serving | Ongoing | — |

**Rule:** never run full `build-oracle` before a small slice has validated the pipeline.
