> **Legacy document (frozen 2026-07-27).** This v1 plan is superseded by [NEXT_PHASE_V2_ROADMAP.md](NEXT_PHASE_V2_ROADMAP.md). Its FQE/scalar-reward assumptions are not valid for v2.

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

## Phase 5 — Benchmark claim-readiness

**Gate before publishing *any* performance or cost-savings claim.** The current
results (~100 items, 2 datasets, all `:free` models) are a smoke test, not
evidence. Three things make them indefensible today: **coverage**, **the cost
axis**, and **statistical rigor**. All three must be closed.

### 5.1 Dataset coverage
Loaders already exist for all five ([eval/datasets/loaders/](src/slm_router/eval/datasets/loaders/)) — this is about *using* them across difficulty regimes and task types, so a math-only win can't masquerade as a general one.

| Dataset | Regime | Why required |
|---|---|---|
| MMLU | Broad knowledge, MCQ | Easy/medium band — proves we don't over-escalate |
| GSM8K | Grade-school math, CoT | Reasoning where mid models beat SLMs |
| **GPQA** | Graduate-level, hard | The escalation case — router must send these to the frontier |
| HotpotQA | Multi-hop QA | Non-math reasoning; guards against math-only overfit |
| HumanEval | Code generation | Exec-based scoring; stresses featurizer generality |

- [ ] Minimum bar: **4–5 datasets** including one genuinely hard (**GPQA**) and one code (**HumanEval**). Two datasets is a demo.
- [ ] Highest-leverage single addition: **GPQA** — without a hard set we can only show "picks cheap models when everything is easy," not that it escalates correctly.

### 5.2 Sample size & coverage
- [ ] **≥300–500 scored items per dataset** (currently ~50). Below ~300 the regret CI is wider than the effect being claimed.
- [ ] **Full oracle coverage** — every candidate × every item, no missing cells. The current ~45% rate-limit loss makes even small numbers suspect (paced free-tier collection over time, or paid oracle calls).

### 5.3 The cost axis *(non-negotiable)*
Today every candidate is `:free`, so `λ·Cost` is inert and the cost-savings thesis is **literally untested**.
- [ ] Put **≥3 real price tiers** in the candidate set: SLM (cheap), mid, and **one real paid frontier model** (GPT-4o / Claude / Gemini Pro class).
- [ ] This is the only way to compute the headline: *"X% of frontier quality at Y% of frontier cost."* It requires relaxing the "free models only" constraint for the oracle build — decide the budget explicitly.

### 5.4 Metrics & thresholds that constitute a defensible claim
- [ ] **Regret vs oracle** with a 95% bootstrap CI over items (not a point estimate).
- [ ] **Cost–quality Pareto**: router sits on/above the frontier of fixed baselines (always-SLM / always-mid / always-frontier) — "no fixed model dominates us."
- [ ] **Headline**: quality retained vs frontier alongside cost reduction vs frontier. Candidate target ≈ **≥95% frontier quality at ≤40% frontier cost** — pick from the data, don't reverse-engineer.
- [ ] **Held-out ECE ≤ ~0.05** with a reliability diagram (train ECE alone proves nothing; current train ECE = 0.041).
- [ ] **Decisively beat naive baselines** — random and confidence-threshold cascade must lose by more than the noise band. Current red flag: random ties always-SLM (0.931 vs 0.931) — must be cleared.

### 5.5 Methodology rigor (what a reviewer will attack)
- [ ] Evaluate the promoted policy **only on held-out items** it never trained on; holdout ≥100 items for stable numbers.
- [ ] **Validate the DR-FQE estimator**: the last run had only 8 holdout items. With 300+ items and real logged-vs-greedy disagreement, confirm the DR estimate tracks actual on-policy regret. Until then the promotion gate is unvalidated.
- [ ] Report **mean ± std over ≥3 seeds** for the split.

**Claim gate (all must hold):** (1) 4–5 datasets incl. GPQA + HumanEval; (2) ≥300 fully-covered items/dataset; (3) ≥3 real price tiers incl. one paid frontier; (4) held-out regret + Pareto + ECE with bootstrap CIs over ≥3 seeds; (5) decisively beats random and cascade. **Highest-leverage next step: GPQA + a paid frontier tier in the oracle** — that turns "SLM ties random on easy MCQs" into a real cost-savings story with an escalation signal.

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
