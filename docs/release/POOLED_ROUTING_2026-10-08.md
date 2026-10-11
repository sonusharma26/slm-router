# Pooled learned routing, 8 October 2026

SLM Router now forecasts every endpoint's quality from training evidence, certifies routes with one calibration over all full-information calibration requests, and serves uncertified requests with the best-forecast endpoint instead of abstaining. On synthetic fixtures it has the best cost-quality frontier among the routers we can run here, and its quality guarantee held in every certified check. Whether it is the best router on real data is **not established**: that requires running `scripts/run_llmrouterbench.py` against LLMRouterBench's data, which this environment cannot download.

## Why the previous router could not compete

The recorded replay abstained on all 29 held-out requests at the 0.65 target because each request's quality bound came from a local neighborhood of 4–12 calibration samples. The lexical neighborhood forecast was worse than a constant, and abstaining scored zero. Public router benchmarks (RouterArena, LLMRouterBench) require a model for every query, so the previous design could not be scored on them at all.

## What changed

| Area | Before | Now |
|---|---|---|
| Forecast | Local kNN mean on 64-d hashed words | Learned predictor: kNN plus a spherical k-means cluster table, shrunk toward slice and global means; hyperparameters chosen by validation Brier score |
| Features | `hash-text-v1` (64-d) | `hash-ngram-v2-256` (word unigrams and bigrams, sublinear tf, length channel) or sentence-transformer embeddings |
| Quality bound | Per-neighborhood KL/Hoeffding bound per candidate plan | Pooled learn-then-test over the planner's own routing rule (default `pooled-ltt`) |
| Uncertified request | Abstain (score 0) | `best_predicted_fallback`: highest-forecast endpoint, with a validation-tuned margin toward the best training endpoint |
| Comparison | One operating point vs simple baselines | Frontiers, AIQ, matched-cost/quality summaries and a guarantee check across resplits |

## The guarantee

For a policy with target `τ = minimum_quality` and risk `δ = quality_risk`, the planner builds, for each candidate threshold `t`, the rule *"among allowed endpoints whose forecast is at least `t`, take the first in the planner's selection order"*. Allowed means it passes every eligibility, spend, deadline, call and failure gate, replayed exactly as serving applies them. A request is certified at `t` when some allowed endpoint clears `t`.

For each `t` the null hypothesis is that the mean quality of certified requests is below `τ`. Certification depends only on request features and training-derived forecasts, so certified calibration requests form an i.i.d. sample of that population. Each hypothesis gets a Hoeffding–Bentkus p-value. Thresholds are tested strict to loose in a fixed sequence that stops at the first non-rejection, so the family-wise error is at most `δ`. The selected threshold is the loosest one rejected.

**Statement:** with probability at least `1 − δ`, the mean quality over all requests certified at the selected threshold is at least `τ`.

**Assumptions** (recorded in each pooled certificate):

- Scores are bounded in [0, 1].
- Future traffic is exchangeable with the calibration requests: same mix and same endpoint behavior.
- Each calibration request was scored on every endpoint the policy allows, and which requests were scored is unrelated to their quality. Benchmark imports and exhaustive probes provide this; sparse served-traffic outcomes do not, so the router then stays uncertified and falls back.
- Forecasts, the threshold grid and the sequence's starting point use only training and validation evidence.

**Scope:** this is a population guarantee, not a per-answer guarantee. Fallback requests are outside it. Pooled mode routes direct (single-call) plans only.

Two practical details:

- **Grid:** the thresholds extend below the point where every request is certified, because lowering the threshold further still lets cheaper endpoints qualify.
- **Anchor:** opening on a threshold with very few certified requests would end the sequence immediately. The sequence therefore starts at the strictest threshold whose pessimistic forecast, from leave-one-out training references, predicts a passing test with at least 40 expected samples. Starting later never weakens the guarantee, because only tested thresholds can be selected.

A Monte Carlo test (`tests/test_pooled_routing.py`) checks that the procedure's false-certification rate stays at or below `δ`.

## Behavior to expect

With a cost objective, pooled routing meets the quality target at the lowest cost it can certify. On the 3-model static fixture at the default 0.65 target it now serves every request with certified mean quality 0.73 at $91 per million requests. The previous router spent $403 per million to reach 0.96, because only expensive models could pass its per-neighborhood bounds. To buy more quality, raise `minimum_quality`; a target of 1.0 turns certification off and makes the router a pure maximum-quality (performance-only) router.

## Results

### Synthetic model zoo (code-path evidence only)

`make_zoo(4000)`: 12 simulated models spanning about two orders of magnitude in price, 8 domains with specialists, binary correctness, no task hints. Three grouped resplits (seeds 42, 43, 44). Latency and spend are reported but not enforced, so every router faces the same constraints.

| Router | AIQ (mean ± sd) | Best quality | Quality at best-single cost |
|---|---:|---:|---:|
| Oracle (reads test labels; reference) | 0.988 ± 0.003 | 0.999 | 0.999 |
| **slm-router (pooled)** | **0.883 ± 0.005** | **0.952** | **0.952** |
| kNN baseline | 0.878 ± 0.007 | 0.942 | 0.942 |
| Threshold baseline | 0.097 ± 0.008 | 0.944 | 0.944 |
| Best single model (`frontier-reason`) | 0.000 (one point at the top of the cost range) | 0.952 | 0.952 |
| slm-router-local (previous), target 0.7 | 0.317 ± 0.026 | 0.911 | 0.911 |
| Random | 0.487 ± 0.024 | 0.543 | 0.543 |
| Cheapest | 0.144 ± 0.010 | 0.144 | 0.144 |

Paired by split, slm-router's AIQ exceeds the kNN baseline's by +0.003, +0.003 and +0.008, and its best quality by +0.005, +0.012 and +0.012. The margins are consistent but small. The pooled router never beat the best single model's quality, because in this simulator that model is strongest everywhere; its gains are at lower spend. For example, on split 42 it reached 0.858 quality at 14 m$ per request and 0.924 at 41 m$, against 0.944 at 60 m$ for the best single model.

**Guarantee check:** 21 of 27 target/split combinations certified test traffic. One fell below its target: split 44 at target 0.5 had a test certified mean of 0.470 over about 600 requests. At 5% risk per check, about one miss in 21 is expected, and a 600-request test mean also carries roughly ±0.02 sampling error. Targets of 0.95 and 1.0 certified nothing on any split, so the router served those requests by fallback, at best-single quality.

Full report: [evidence/zoo-compare-2026-10-08.md](evidence/zoo-compare-2026-10-08.md). These numbers come from a simulator in which one frontier model is strongest in every domain. They show that the code paths work and the guarantee is respected. They say nothing about real models.

### LLMRouterBench integration check

`scripts/run_llmrouterbench.py` was run end to end against LLMRouterBench's own `BaselineDataLoader` and `split_by_dataset_then_prompt` (repository commit `c77cb05`), using synthetic result files written in its `results/bench` format, because its real data is hosted on Hugging Face and in Git LFS, which this environment cannot reach.

- **Paths exercised:** loading, the per-dataset 80/20 split, full-matrix construction, train/calibration/validation subdivision of its training prompts, and the comparison report.
- **Heavy dependencies avoided:** the script registers LLMRouterBench's `baselines.adaptors` package without running its `__init__`, so the split helper loads without torch or pandas.
- **Result on that synthetic stand-in:** slm-router had AIQ 0.875, above kNN's 0.868. Its maximum-quality mode matched the best single model (0.964), and all 7 certified targets held on test.

## How to decide whether it is the best

Run, on a machine with the data:

```bash
git clone https://github.com/ynulihao/LLMRouterBench
# fetch its result data into LLMRouterBench/results/bench per its README
pip install -r LLMRouterBench/requirements.txt
python scripts/run_llmrouterbench.py --lrb-root ../LLMRouterBench \
  --config config/baseline_config_performance_cost.yaml --seeds 42,43,44 --out results/llmrouterbench
# optional: --embedder st:all-MiniLM-L6-v2   (semantic features; needs sentence-transformers)
```

The report's test prompts are the ones LLMRouterBench's baselines are scored on, for the same seed and train ratio. Compare:

- **Per-dataset average** with the paper's AvgAcc (performance-only setting; the `target=1.0` point).
- **Frontier and AIQ** with its performance-cost results.
- **LLMRouterBench's own baselines** (Avengers-Pro, EmbedLLM, GraphRouter, RouteLLM, ...), run on the same seeds for a direct table.

Only after that comparison, and ideally a RouterArena submission (which needs live model calls with your API keys), would "best" be a supportable claim.

## Compatibility

- **Defaults:** new `ConditionalCapabilityMap()` and `SLMRouter()` use `pooled-ltt`. Saved maps keep their recorded method.
- **Pinned studies:** the dynamic drift and probing study, the drift-recovery demo and the earlier quality-improvement script pin `local-mean-kl` with abstention. Their outputs are byte-identical to before.
- **Policy compiler:** `evidence.require_certificate` now defaults to `false`, and `on_infeasible` defaults to `best_predicted_fallback`, or to `abstain` when certificates are required. Uncertified fallback cannot be combined with required certificates. Documents that omit these fields compile to a new version.
- **New dependency:** `numpy>=2.0,<3`.

## Verification

- **Tests:** the full offline suite passes: 156 passed, 2 skipped. The 2 skips are legacy tests that need the optional `joblib`; before this change they failed whenever numpy was installed.
- **New tests cover:**
  - Hoeffding–Bentkus validity at the null boundary.
  - The fixed sequence's false-certification rate.
  - Serving decisions matching the calibration replay on every calibration request tested.
  - Pooled certificates passing `validate_decision`.
  - The fallback margin.
  - Map config round-trip.
  - Policy rules and compiler defaults.
  - Frontier math.
  - LLMRouterBench conversion.
- **Smoke check:** `python -m inference_control.smoke` passes with the new defaults.
- **Lint:** new modules pass the project's full ruff configuration.
