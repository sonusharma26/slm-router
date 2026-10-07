# Recorded quality-bound diagnosis, 7 October 2026

This document records the initial diagnosis against the starting estimator. The subsequent implementation, tighter bounds, independent archival evidence and remaining limitations are in [Calibration and prediction improvements](CALIBRATION_IMPROVEMENT_2026-10-07.md).

The 29-request abstention failure is reproduced. **Sparse calibration neighborhoods are sufficient to force every candidate to fail the unchanged 0.65 target.** Prediction error is also substantial. No arithmetic error in the existing bound was found; this does not establish that its population assumptions hold on deployment traffic.

## Evidence and reproduction

The supplied evidence bundle tests commit `34480366e8aa9fab14d985590d276bde262adeed`, matching the starting checkout. The pinned public archive prefix and both selected run files were reconstructed using the bundle's bounded retrieval helper and verified against its recorded SHA-256 hashes. No live inference was needed.

The dataset contains 253 requests: 130 training, 53 calibration, 41 validation and 29 test requests, with two archived model outcomes per request. The split hash matches the supplied report. All **58 candidate lower bounds** match the original diagnosis to within `1e-12`; the router still abstains on every test request. The dataset fingerprint changes because the importer includes absolute source-run paths in provenance; the reconstructed paths are local to this checkout.

The scores are historical ArenaHard judge scores in `{0, 0.5, 1}`, not objective coding correctness or current NVIDIA measurements. Request latency is absent. The replay uses the original offline endpoint metadata assumptions and keeps benchmark-only evidence ineligible for production certificates.

Machine-readable results are under `results/quality-diagnosis-2026-10-07/`:

- `recorded-replay/report.json`: unchanged-policy replay, prediction errors and every candidate's bound decomposition.
- `latency-replay/report.json` and `latency-detector-trace.json`: reproduced drift failure and the detector's completed windows.
- `sparse-frozen-baseline/report.json`: zero-probe comparison against the supplied sparse benchmark.
- `recorded-input/imported.json`: reconstructed benchmark input, with raw third-party artifacts kept in the ignored results directory.

The recorded replay uses 40 bootstrap resamples for this diagnostic, versus 400 in the supplied evidence. No conclusion below depends on those confidence intervals. Routing times depend on the machine and are not compared as performance improvements.

## Why the lower bounds collapse

`ConditionalCapabilityMap.estimate()` currently computes a local calibration-mean bound, rather than a residual-calibrated prediction bound:

```text
alpha = policy.quality_risk / number_of_generated_plans
radius = sqrt(log(2 / alpha) / (2 * local_calibration_count))
calibration_lower = max(0, local_calibration_mean - radius)
quality_lower = min(calibration_lower, local_training_mean)
```

For this direct two-model search, `alpha = 0.05 / 2 = 0.025`, so the logarithm is `log(80)`. Hoeffding's inequality bounds deviations of averages of independent bounded variables; it does not turn a local mean into a guarantee of correctness for each new prompt. The implementation additionally assumes the recorded neighborhood represents the future local population. [Hoeffding's original paper](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf).

The exact stratum includes task, application, traffic slices, privacy/capabilities/schema and token-length buckets. In this dataset, the calibration evidence is fragmented into seven test-relevant strata by input-length bucket. Each candidate sees **4, 5, 10 or 12 calibration requests**, rather than all 53.

| Local calibration n | Radius | Highest possible lower bound, even with mean 1 |
|---:|---:|---:|
| 4 | 0.740104 | 0.259896 |
| 5 | 0.661969 | 0.338031 |
| 10 | 0.468083 | 0.531917 |
| 12 | 0.427299 | 0.572701 |

Every ceiling is below 0.65. The best Claude candidate has five perfect calibration scores: `1 - 0.661969 = 0.338031`. The best DeepSeek candidate has local calibration mean `2/3` and 12 observations: `2/3 - 0.427299 = 0.239368`. These reproduce the observed maxima.

The training-mean clamp contributes **zero additional penalty on all 58 candidates at the present sample sizes**. It is not the cause of this run's collapse. It can become a separate obstacle after collecting more calibration data: the training means already fall below 0.65 on 21 Claude and 27 DeepSeek candidate/request pairs.

Under the unchanged formula, reaching 0.65 requires at least 18 local calibration samples even at a perfect mean. If the local mean remains 0.8, the requirement is 98; at 0.7, it is 877. These are algebraic requirements at a fixed mean, not forecasts of future data. A local mean at or below the target cannot clear it simply by increasing n. The benchmark's `k=128` also prevents reaching the 877-sample case. Merely changing the two-sided `log(2/alpha)` factor to the one-sided `log(1/alpha)` would still leave the best n=12 ceiling below 0.65.

Pooling all 53 labels would not establish feasibility either: pooled calibration means are 0.509434 for Claude and 0.660377 for DeepSeek. Pooling or altering strata requires a new, justified population assumption and independent evaluation; it cannot be justified by selecting whichever test outcome looks better.

## Prediction quality is a second problem

A separate predictor was fitted to training rows only. It predicts each calibration request using the same feature/stratum rules and k limit. A constant baseline uses only the model's global training mean. Every calibration and test request has a training neighborhood in this reconstruction.

| Endpoint | Calibration prediction MAE | Constant baseline MAE | Calibration prediction bias | Test prediction MAE | Test constant MAE |
|---|---:|---:|---:|---:|---:|
| claude-sonnet-4 | 0.404422 | 0.382801 | +0.121559 | 0.397691 | 0.384218 |
| deepseek-v3.1-terminus | 0.399241 | 0.398549 | -0.131471 | 0.397074 | 0.375066 |

Neither neighborhood predictor improves MAE over the constant baseline here. The text features are a 64-dimensional signed lexical hash, not a trained task-difficulty or semantic-quality model. These errors do not prove a particular feature replacement will help. They establish that the current predictor has not earned an advantage on this sample.

The quality lower bound is computed from calibration scores, not from this predictor's residuals. Improving the training predictor alone therefore cannot remove the present sample-size ceiling. It could help a separately designed calibration method, which would need an explicit guarantee and fresh evaluation.

## Drift failure reproduced

The original `latency_x3`, seed 42, 40-step, 2,400-initial-request impact profile again produces **eight latency violations, all carrying current certificates**, with no detected event.

The first eight-observation window per endpoint/slice mixes pre-change and post-change calls:

| Slice | First observed window | Normalized shift | Detection threshold | Result |
|---|---|---:|---:|---|
| math | four 80 ms calls and four 240 ms calls | 0.226673 | 0.246147 | no event |
| code | five 80 ms calls and three 240 ms calls | 0.172948 | 0.236644 | no event |

The monitor clears each window after evaluating it. Too few subsequent cheap-endpoint observations arrive to complete another window before the run ends. Active probes update the capability map but do not call `DriftRecovery.observe()`. There is also no immediate latency-budget circuit breaker in this path. Raising probe count alone does not repair the missing detector integration.

A follow-up implementation should feed immediate measured latency/timeouts from both execution and probes into an operational safety rule, invalidate affected evidence and certificates when that rule trips, and require new post-change calibration before recertification. Its thresholds must distinguish expected tail observations from operational failures and be evaluated separately from the quality-bound change.

## Sparse probing and NVIDIA

The zero-probe frozen replay at seed 42, 40 steps and 1,200 initial requests gives mean fixture quality `0.9635414737002216` and regret `0.002443029189907789`, exactly matching every strategy in the supplied sparse report. This fixture's affected cheap endpoint does not change the useful serving choice at its operating point. Paid refresh has no observed quality/regret benefit here; the comparison needs a zero-refresh baseline and scenarios where drift changes the optimal feasible choice.

The NVIDIA failure in the supplied test bundle is in the legacy path: `configs/models.yaml` specifies `nvidia-build`, but `slm_router.types.Provider` accepts only `openrouter` or `google`. Registry loading fails before any HTTP request. This failure was reproduced directly. Adding a key cannot resolve it.

Simply extending the provider literal would be incomplete: the legacy `OpenRouterClient` dispatches every non-Google provider through the OpenRouter transport, and legacy settings do not expose an NVIDIA credential. Registry acceptance, credential wiring and provider-specific dispatch must be handled together. The newer `inference_control` path already has an OpenAI-compatible NVIDIA adapter. The legacy failure was diagnosed without live calls; the subsequent authorized control-plane check is recorded below.

## Authorized NVIDIA control-plane pilot

After the offline diagnosis, the user explicitly allowed the existing pilot to load only `NVIDIA_API_KEY` from `.env.local` without displaying it. The three-call pilot completed on October 7. Its report is `results/quality-diagnosis-2026-10-07/nvidia-live-pilot/pilot-report.json`.

| Call | Endpoint/model | Complete token usage | Exact-number calibration | Latency |
|---|---|---|---|---:|
| Calibration | nvidia-fast / openai/gpt-oss-20b | yes | passed | 3,874.0 ms |
| Calibration | nvidia-strong / meta/muse-glimmer-30b | yes | passed | 13,106.1 ms |
| Routed request | nvidia-fast / openai/gpt-oss-20b | yes | not scored by this pilot | 1,513.3 ms |

The routed execution completed with no recorded constraint violations, and its ledger verifies. It is **uncertified**. No immutable provider model revision was supplied. This confirms the newer adapter's functional path with the supplied credential; the legacy registry/transport blocker remains.

The existing pilot uses its separate prototype policy with `minimum_quality: 0.0` and one calibration observation per endpoint. That policy was not modified. It is deliberately a transport/accounting check and cannot demonstrate the 0.65 quality promise, production readiness, or an improvement in the recorded-data benchmark. The recorded benchmark's 0.65 target remains unchanged.

## Changes and next implementation decision

This patch adds auditable quality decomposition to evidence lineage and static benchmark reports. It records local sample counts, calibration means, the Hoeffding radius, the training clamp, comparison-adjusted risk, best-case sample ceilings, fixed-mean sample requirements, and train-only prediction errors against a constant baseline. It leaves the quality target, risk policy, estimator arithmetic and routing behavior unchanged.

Four diagnostic regression tests cover sparse abstention, held-out label isolation/map immutability, the actual eligible search family, and separation of sampling/clamping/missing-evidence effects. The diagnostic plus existing control-plane suites pass **69 tests** on Python 3.12.15. The legacy registry blocker remains; this is not a claim that the entire legacy test suite passes.

The next calibration implementation should first specify whether it promises a local population mean or per-response marginal coverage. Then choose a method for that promise and a calibration population with sufficient independent evidence. Compare the present mean bound against the proposed method on fresh calibration/test partitions, reporting prediction error, bound usefulness, coverage and abstention together. For the present method, sample planning must account for the mean-to-target gap and k cap; increasing data without checking those limits is not enough. The 0.65 target should remain unchanged during that comparison.
