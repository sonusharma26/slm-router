# Answer-checking cascades and the price-penalty rule, 9 October 2026

Question-only routing on the synthetic zoo is near its ceiling, and slm-router was only 0.003–0.008 AIQ ahead of kNN. A ceiling analysis showed why. A router that reads the question perfectly reaches AIQ 0.904 on average. The remaining 0.08 below the oracle (0.988) is per-question luck, and only looking at answers can capture it. This change adds both levers that analysis identified.

## What changed

- **Price-penalty rule (new pooled default, `PooledConfig.rule="utility"`).** Each request goes to `argmax(forecast − λ·expected cost)`. Penalties come from the switching points between models on training forecasts, ordered from λ = 0 (quality first) upward. A fixed-sequence Hoeffding–Bentkus test picks the cheapest passing λ. With probability at least `1 − quality_risk`, mean quality over **all routed traffic** is at least the target; the threshold rule only covered requests it certified. `rule="threshold"` remains available, and saved maps keep their rule.
- **Answer-agreement cascade (`inference_control.routing.cascade`).**
  - **Normal path:** serve the price-penalty choice. If it is not the quality-first model and its forecast is below `accept`, also run the next-best model as a checker.
  - **Agreement:** if the two extracted answers agree, serve the cheap answer. If they disagree, escalate to the quality-first model.
  - **Ordering:** configurations `(λ, accept)` are ordered by quality on leave-one-out training references.
  - **Selection:** a fixed-sequence test runs on full-information calibration requests. Every rejected hypothesis holds simultaneously, so the cheapest passing configuration is chosen without weakening the guarantee. Mean quality over all routed traffic is at least the target with probability `1 − quality_risk`.
- **Batch-invariant forecasts.** Ties in neighbor similarity were broken differently by single-row and batched matrix arithmetic, so one request could get a forecast about 0.01 different alone than in a batch. Serving and calibration replay could then disagree. Similarities are now rounded to 1e-9 and ties break by training index; a test asserts byte-identical forecasts.
- **Benchmarks.** `slm benchmark compare --zoo` adds simulated answers and reports the cascade. `scripts/run_llmrouterbench.py` uses each model's stored extracted answer (`prediction`), so cascades are evaluated offline on real benchmark outputs at no API cost (`--no-cascade` disables it).

## Results on the synthetic zoo (code-path evidence only)

12 simulated models, 4,000 requests, three grouped resplits. The zoo's simulated answers share mistakes: two wrong models agree about a third of the time. Free-form writing answers never agree.

| Router | AIQ (mean ± sd) | Best quality | Cost to match best single model |
|---|---:|---:|---:|
| Oracle (reads outcomes; reference) | 0.988 ± 0.003 | 0.999 | 1% |
| **slm-cascade** | **0.905 ± 0.011** | **0.960** | **58.5%** (2 of 3 splits; never matched on the third) |
| Best question-only router possible (knows true probabilities given the text) | 0.904 | — | — |
| slm-router (price-penalty rule) | 0.886 ± 0.014 | 0.952 | 100% |
| slm-router (previous threshold rule, 8 Oct) | 0.883 ± 0.005 | 0.952 | 100% |
| kNN baseline | 0.878 ± 0.007 | 0.942 | never |

- **Cascade vs kNN:** the cascade leads kNN by +0.021, +0.035 and +0.024 AIQ per split. That is three to eight times the earlier margin, and it matches the question-only ceiling.
- **Above the best single model:** on two of three splits its top quality exceeds the best single model (0.959 vs 0.944; 0.953 vs 0.942) at 55–62% of that model's cost. On the third it reached 0.967 vs 0.970.
- **Price-penalty rule, mixed:** it lifted AIQ from 0.883 to 0.886 on average, but split by split it was −0.001, +0.020 and +0.005 against kNN.
- **Guarantee:** 42 target/split checks across both routers; one fell below target, slm-router at target 0.5 on split 44 with 0.497. That is consistent with the 5% risk.

Full report: [evidence/zoo-compare-cascade-2026-10-09.md](evidence/zoo-compare-cascade-2026-10-09.md).

**How much to trust these numbers:** cascade gains depend entirely on how often wrong answers agree. The zoo fixes that rate at about one in three. Real models share mistakes in ways this simulator cannot know. The LLMRouterBench run measures it directly from stored outputs, and it is the number that matters.

## Limits

- **Offline only:** the cascade is evaluated offline on full-information matrices with stored answers. Live serving through the planner's `verify_escalate` plans is not wired yet.
- **Comparable answers only:** agreement only helps when answers can be compared, such as final answers, choices or numbers. On free-form tasks, checks mostly escalate and the calibration learns to avoid them.
- **Same assumptions as pooled routing:** calibration requests are exchangeable with future traffic and are scored, with answers, on every allowed endpoint.

## Verification

- **Tests:** 166 passed, 3 skipped (legacy tests needing the optional `joblib`).
- **New tests cover:**
  - False-certification control for rule sequences and cascades, by Monte Carlo against known population truth.
  - Cascade mechanics checked by hand.
  - Answer normalization.
  - Batch-invariant forecasts.
  - Serving matching the calibration replay for both rules.
  - Full-traffic certification under the utility rule.
  - Compare-harness output with answers.
  - LLMRouterBench answer conversion.
- **End to end:** `scripts/run_llmrouterbench.py` ran against LLMRouterBench's own loader on synthetic data in its format.
