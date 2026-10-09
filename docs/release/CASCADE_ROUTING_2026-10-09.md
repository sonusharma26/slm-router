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

## Majority voting

Votes are a third kind of configuration in the same calibrated family, next to plain price-penalty routing and agreement cascades. Calibration picks whichever kind is cheapest among those that pass.

**Method** (`CascadeConfig(lam, accept, voters=k, quorum=q, sequential=...)`, `2 <= q <= k`):

- **Voters:** the top-k allowed endpoints by `forecast − λ·cost`, ties to the earlier position. Rank 1 is the price-penalty choice. Unlike the agreement check, a vote is also held when that choice is already the quality-first model. `accept` works as for cascades; the default family always votes (`accept > 1`).
- **Counting:** answers are normalized as for agreement checks. Missing or empty answers are not comparable and cast no vote. The plurality answer wins; a tie goes to the answer whose best-ranked voter ranks higher.
- **Serving:** if the winner has at least q votes it is served, with the realized quality of its best-ranked voter. Otherwise the request escalates to the quality-first model. If that model is already a voter, its own answer is served at no extra cost. With no comparable answers the count is zero, so the request escalates; a vote never guesses.
- **Cost:** the voters actually called, plus the quality-first model when it is called in addition. The default family uses `sequential=True`: voters are called in rank order and calling stops once an answer reaches q or none can. With `2q > k` it serves the same answer as calling all k at once and never costs more, but it adds latency. `sequential=False` (the dataclass default) calls everyone and is available for latency-bound settings.
- **Family:** `default_configs(grid, voters=(3, 5))` appends 32 vote configurations (8 evenly spaced penalties × quorums 2, 3 for three voters and 3, 4 for five) to the 150 plain and agreement ones. The compare harness uses voters (3, 5); `cascade_voters=()` or `--max-voters 0` gives back the agreement-only family, which reproduces the numbers above exactly. Everything else keeps its meaning for non-vote configurations.
- **Reporting:** `cascade.outcome` returns per-request quality, cost, calls and decision flags; `cascade.diagnostics` summarizes kind, voters, quorum, check/vote/escalation rates and mean calls. Each `slm-cascade` operating point in the compare report carries them, and `report.md` has a "Cascade operating points" table.

**Guarantee scope.** Unchanged. A vote is a deterministic function of the stored forecasts, costs, outcomes and answers, like every other configuration. The family is ordered on training references only. The fixed-sequence test then rejects a configuration whose true mean quality over all routed traffic is below target with probability at most the risk, and the cheapest rejected one is selected. Adding votes costs no multiplicity penalty, but it can shorten the tested sequence (less power) when a vote that ranks high on training references fails on calibration. In a Monte Carlo test with a 55-configuration mixed family (target 0.86, risk 0.1, 250 repetitions) there were no false certifications (bound 0.157). Votes were selected in 108 repetitions, agreement cascades in 61 and plain routing in 19. That check is conservative, not tight. As before, this certifies offline replay with stored answers; live voting would draw fresh answers.

### Results on the synthetic zoo (code-path evidence only)

12 simulated models, 4,000 requests, grouped resplits 42, 43 and 44, the same zoo and answers as above (wrong models agree about a third of the time). "Before" is the previous commit; the harness with `cascade_voters=()` gives identical numbers.

| `slm-cascade` | Agreement only | With votes |
|---|---:|---:|
| AIQ (mean ± sd) | 0.9049 ± 0.0113 | **0.9099 ± 0.0075** |
| AIQ per split (42 / 43 / 44) | 0.9026 / 0.9197 / 0.8923 | 0.9093 / 0.9194 / 0.9010 |
| AIQ from certified points only | 0.8928 | 0.9009 |
| Best quality (all points) | 0.9598 | 0.9653 |
| Quality at the best single model's cost | 0.9598 | 0.9627 |
| Cost to match the best single model | 58.5% (2 of 3 splits) | 58.1% ± 14.0% (3 of 3: 40%, 74%, 61%) |
| Certified target/split checks, violations | 21, 0 | 22, 0 |
| Kinds selected (27 target/split points) | 14 plain, 13 agree | 14 plain, 3 agree, 10 vote |

- **Gain:** +0.005 AIQ on average, from +0.0067, −0.0003 and +0.0087 on the three splits. That is about half the split-to-split spread (sd 0.01), so it is a small gain, not a significant one.
- **Where it comes from:** votes are chosen at targets 0.85–0.95. At target 0.90 the selected configuration cost 10%, 30% and 20% less than the agreement cascade, with test quality 0.946, 0.927 and 0.936 (all above target). On the cost-quality hull of all operating points, reaching quality 0.90 costs 13% less and 0.93 costs 16% less (means over the three splits); at 0.95 it costs 5% more, so there is no gain at the top.
- **Calls:** the five certified vote points all use three voters and call 2.5–3.1 endpoints per request; the vote carried 47–83% of requests and the rest escalated. The uncertified five-voter fallbacks call 3.5–4.6.
- **Uncertified points:** when no configuration passes (targets 0.95–1.0 here), the harness falls back to the configuration with the highest training-reference quality. It carries no guarantee and is marked "Calibrated: no". AIQ and "best quality" include those points, as before, so the certified-only row is the stricter comparison. It shows a larger mean gain (+0.008: +0.035, −0.017 and +0.006 by split) but a loss on split 43 (0.9163 → 0.8994): the vote chosen for target 0.90 there is cheaper and less accurate than the agreement cascade it replaced, which is allowed because both clear the target.
- **Ablations** (same splits, mean AIQ, differences of 0.003 are within noise): parallel calls 0.9078, sequential 0.9099, three voters only 0.9109, voters 3, 5 and 7 0.9099, extra `accept` gates 0.9, 0.8 on votes 0.9128. The default stays the simplest family.
- **Answer correlation** (the zoo's weakest assumption; mean AIQ over the same splits, agreement only → with votes):

| Two wrong models agree | Agreement only | With votes | Difference |
|---|---:|---:|---:|
| 5% (near-independent mistakes) | 0.9132 | 0.9289 | +0.0156 (all three splits) |
| 33% (default) | 0.9049 | 0.9099 | +0.0050 |
| 50% | 0.9013 | 0.9063 | +0.0050 |
| 81% (mostly shared mistakes) | 0.8981 | 0.8992 | +0.0012 (−0.0019 on one split) |

No guarantee violation occurred for either family in any of these runs.

Full report: [evidence/zoo-compare-votes-2026-10-09.md](evidence/zoo-compare-votes-2026-10-09.md). A run of `scripts/run_llmrouterbench.py` on a synthetic export in LLMRouterBench's format (stored predictions, one split, 12 models) also selected certified votes at targets 0.85 and 0.9; that is a wiring check only.

**Caveats.**

- **Synthetic answers:** correct models always give the identical answer and mistakes are shared at a fixed rate. Real correct answers differ in surface form (normalization only handles case, spaces and a trailing period), which lowers agreement and with it the value of votes. LLMRouterBench's stored predictions measure this directly and are the number that matters.
- **Votes need comparable answers:** the zoo's free-form writing domain (one request in eight) never agrees, so votes there always escalate.
- **One zoo draw:** three resplits of a single simulated population; the gains are small relative to their spread.
- **Cost, not latency:** sequential voting trades latency for cost, and the harness counts only model cost. Live serving is still not wired.

## Limits

- **Offline only:** the cascade is evaluated offline on full-information matrices with stored answers. Live serving through the planner's `verify_escalate` plans is not wired yet.
- **Comparable answers only:** agreement only helps when answers can be compared, such as final answers, choices or numbers. On free-form tasks, checks mostly escalate and the calibration learns to avoid them.
- **Same assumptions as pooled routing:** calibration requests are exchangeable with future traffic and are scored, with answers, on every allowed endpoint.

## Verification

- **Tests:** 180 passed, 3 skipped (legacy tests needing the optional `joblib`).
- **New tests cover:**
  - False-certification control for rule sequences and cascades, by Monte Carlo against known population truth.
  - Cascade mechanics checked by hand.
  - Answer normalization.
  - Batch-invariant forecasts.
  - Serving matching the calibration replay for both rules.
  - Full-traffic certification under the utility rule.
  - Compare-harness output with answers.
  - LLMRouterBench answer conversion.
  - Majority votes: plurality, quorum miss, final already a voter, no comparable answers, tie-breaking, cost accounting, validation, agreement-cascade equivalence for two voters, false-certification control for a mixed family, compare fields, zoo correlation knob, stored-prediction handling.
- **End to end:** `scripts/run_llmrouterbench.py` ran against LLMRouterBench's own loader on synthetic data in its format, with and without votes (`--max-voters 0`).
