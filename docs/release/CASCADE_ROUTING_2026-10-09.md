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

## Live serving

The planner now serves the cascade through a new plan type, `agree_escalate`. It is the same `(λ, accept)` rule the offline module evaluates, with the serving limits applied.

**Opt in.** Allow `agree_escalate` and `direct` in `plans.allowed`, set `max_calls: 2` or more, and name an answer-agreement function in `plans.verifier`. The default function, `answer_agreement`, is registered in `configuration.py`; it extracts and normalizes each completion's final answer the same way the stored calibration answers were. Without this opt-in nothing changes. The compiler rejects `agree_escalate` without a verifier or with fewer than two calls.

```yaml
plans:
  allowed: [direct, agree_escalate, abstain]
  max_calls: 2
  verifier: answer_agreement
```

**Evidence.** Record each endpoint's answer for every calibration and training request with `ConditionalCapabilityMap.add_answers` (or `SLMRouter.fit(..., answers=...)`). Answers are stored as 16-hex hashes of the normalized text, keyed by `sample_id`. They enter `export`, `restore`, `version` and `evidence_version` only when present, so a map without answers keeps its hashes. A map with answers produces the cascade; a map without them produces the direct rule as before.

**Plan.** `agree_escalate` is `Call(first)`, `Call(checker)`, `Verify(agree)`, then an optional `Call(final)`. `max_calls` is the number of Call steps and `max_spend` is the sum of their absolute bounds. First and checker run concurrently. The executor serves the first answer when the registered function says they agree, and calls the quality-first endpoint when they disagree. A failed call or an agreement function that raises counts as disagreement (fail closed), an unregistered function is rejected before any spend, and the escalation call goes through the same deadline, certificate and snapshot guard as any call. When the checker is itself the quality-first endpoint, disagreement serves its answer with no third call.

**Calibration.** Inside `pooled_calibration` the planner calibrates the cascade first, with one shared decision function used for ordering, the test and serving:

- **Gates.** A request whose compound plan fails a hard gate (call limit, absolute spend, expected spend, deadline) does not check. Its first endpoint answers alone, which is the plain price-penalty rule. Gates use training evidence only (training probability that two endpoints' answers differ, 1.0 below the sample minimum), so the calibration stays valid.
- **Guarantee.** With probability at least `1 − quality_risk`, mean quality over all routed requests (checked or not) is at least the target. The certificate names `hb-fixed-sequence-cascade-v1` and lists the extra assumptions.
- **Fallback.** If the cascade does not certify, the direct rule is tested. Each of the two tests then uses `quality_risk / 2` (union bound), so a certified direct plan is slightly harder to obtain with answers present than without.

**Estimates and metadata.** A compound plan's expected cost is the first endpoint's plus the checker's plus, when the final endpoint is a third model, its cost weighted by the training probability that the pair disagrees. Its latency bound is the larger of first and checker plus the final call, and its failure bound is the largest member bound. Decisions carry `plan_metadata.cascade` with the configuration, whether a check was planned, and the plan type served; it is omitted from the record when absent, so existing decision hashes do not move.

**Learning.** Executions flow through `ControlPlane.execute` and `learn` like any plan; compound rows are not used as pooled training outcomes, and drift recovery replays them from stored maps. `SLMRouter.choose` still returns a single endpoint for direct decisions, and `evaluate_live_cascade` in `benchmarks/routers.py` replays `agree_escalate` decisions offline against full-information matrices.

**Still not covered.** Real-model agreement rates (the LLMRouterBench run is the evidence to read), vote configurations, streaming, and any provider retry. Servers must produce answers the stored ones can be compared with; a different extraction at serving time breaks the calibration's assumption.

## Limits

- **Live serving needs opt-in and stored answers:** see [Live serving](#live-serving). Without both, the router serves exactly as before. Vote configurations stay offline.
- **Comparable answers only:** agreement only helps when answers can be compared, such as final answers, choices or numbers. On free-form tasks, checks mostly escalate and the calibration learns to avoid them.
- **Same assumptions as pooled routing:** calibration requests are exchangeable with future traffic and are scored, with answers, on every allowed endpoint.

## Verification

- **Tests:** 210 passed, 3 skipped (legacy tests needing the optional `joblib`); 44 of them are the live-serving tests in `tests/test_live_cascade.py`.
- **New tests cover:**
  - False-certification control for rule sequences and cascades, by Monte Carlo against known population truth.
  - Cascade mechanics checked by hand.
  - Answer normalization.
  - Batch-invariant forecasts.
  - Serving matching the calibration replay for both rules.
  - Full-traffic certification under the utility rule.
  - Compare-harness output with answers.
  - LLMRouterBench answer conversion.
  - Live serving: answers as map evidence (hashes unchanged when absent, restore round trip), the `agree_escalate` contract and compiler rules, executor concurrency, fail-closed agreement and the pre-escalation guard, serving decisions equal to the gated calibration replay, gated and ungated equality with the offline `cascade` functions, Monte Carlo false-certification control with a binding gate, `ControlPlane` execute/learn and drift recovery, and identical behaviour when the policy does not opt in or answers are absent.
- **End to end:** `scripts/run_llmrouterbench.py` ran against LLMRouterBench's own loader on synthetic data in its format.
