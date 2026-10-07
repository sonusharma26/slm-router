# Calibration and prediction improvements, 7 October 2026

New capability maps now use a tighter bounded-score KL interval at the unchanged quality target and risk. Raw training predictions no longer cap that population confidence interval. The original coding replay still abstains on all 29 requests. Its local scores and sample counts remain insufficient. A larger independent archival replay shows a small improvement, with material generation-setting limitations.

## Implemented behavior

`ConditionalCapabilityMap` defaults to `local-mean-kl`. The lower and upper tails each receive half the comparison-adjusted quality risk. At quality risk 0.05 across two candidate plans, each plan receives 0.025 and each tail receives 0.0125. Fractional scores in [0,1] are supported by the bounded-variable Chernoff argument, rather than by assuming scores are binary. [Foong, Bruinsma and Burt, 2022](https://arxiv.org/abs/2205.07880).

The interval inverts binary relative entropy at `log(2/alpha)/n`. A small conservative numerical margin and outward endpoint rounding avoid inward floating-point roots. Independent review checked 256 settings with high-precision Decimal arithmetic and enumerated joint binomial failure probabilities. These are numerical checks of the implementation. They do not prove exchangeability of future traffic.

The ordered `Prediction` contract retains a center projected into the confidence interval. Diagnostics retain the separate raw training forecast and its projection. Prediction errors use the raw value. The training forecast cannot shrink a valid KL lower endpoint. Calibration strata, distinct request counting, minimum sample guards, evaluator trust, revisions, age, invalidations, cost limits and latency-tail guards remain enforced.

The explicit `local-mean-hoeffding` mode preserves the original arithmetic and training clamp. Checkpoints without a method restore that mode. All restored maps receive the v3 content identity, so certificates tied to the old map identity fail validation. Historical ledger records remain intact. New method and predictor settings survive export/restore and contribute to map, artifact and certificate identity. Existing saved maps do not silently switch to KL.

`quality_predictors` provides an opt-in `cohort-mean` forecast per target. It uses eligible training evidence from the same semantic and governance population while ignoring length buckets for the point forecast. The certifying calibration population retains those buckets. Validation selects between local and cohort forecasts; a half-shrink candidate is also measured as an experiment. No calibration or test quality label selects the forecast.

## Sample efficiency and original coding replay

These counts assume the observed mean stays fixed, target 0.65 and two eligible plans. Policy sample minimums still apply. They are conditional arithmetic, not forecasts of future labels.

| Fixed score mean | Hoeffding samples | Paired KL samples |
|---:|---:|---:|
| 1.0 | 18 | 11 |
| 0.9 | 36 | 27 |
| 0.8 | 98 | 81 |
| 0.7 | 877 | 779 |

At a perfect mean, KL needs 38.9% fewer samples. Increasing data alone cannot make a population with mean at or below 0.65 clear the target. Requirements above the neighborhood cap also need a separately justified estimator design.

The original 253 coding requests retain their split of 130 train, 53 calibration, 41 validation and 29 test requests. Local calibration neighborhoods still contain only 4 to 12 observations.

| Endpoint | Original maximum lower bound | KL maximum lower bound | Validation-selected forecast |
|---|---:|---:|---|
| Claude Sonnet 4 | 0.338031 | 0.416277 | Local |
| DeepSeek V3.1 Terminus | 0.239368 | 0.272636 | Cohort mean |

DeepSeek validation MAE falls from 0.462483 to 0.435084, a 5.9% decrease over 41 validation requests. Claude retains local prediction because its validation MAE is slightly better. This choice is specific to the benchmark and is not a global production default. The known test remains descriptive. All three interventions still abstain on every original test request at 0.65.

The report includes an exact-stratum collection plan, hypothetical total sample requirements, additional independent requests needed, populations whose current mean is too low, and `k=128` blockers. Repeating a prompt or scoring it on another endpoint does not increase independent request count.

## Additional independent archival evidence

The pinned public archive revision is `0e5af1b84bf73437a01a1849c0f1d2468baa93fc`. An additional 48 MiB range extended the existing 16 MiB prefix. The model and task selection manifest was written before reading scores. The first two model IDs in archive order and first two additional complete tasks produced DeepHermes 3 Llama 3 8B Preview and DeepSeek R1 Distill Qwen 7B on WinoGrande and ARC.

Four selected JSON members were verified by SHA-256 and parsed without extracting archive code. ARC contained two duplicate normalized prompts per model. The importer retains the first occurrence in source order, independent of the label. It refuses incomplete model matrices or any overlap with the old coding prompt groups.

The resulting dataset has 2,437 distinct prompt groups and 4,874 outcomes. Splits contain 1,203 training, 491 calibration, 373 validation and 370 held-out requests. The two tasks retain separate calibration populations. Their model pool differs from the original coding replay. Historical token prices were fitted to training cost rows only with residuals below `1e-8`; they are not current provider prices. All per-call latency values remain missing and all labels remain benchmark-only.

| Intervention | Answered / 370 | Correct archival answers | All-request quality | Mean historical cost/request |
|---|---:|---:|---:|---:|
| Hoeffding, local forecasts | 171 | 137 | 0.370270 | 0.0000825759 |
| KL, local forecasts | 173 | 139 | 0.375676 | 0.0000829859 |
| KL, validation-selected forecasts | 173 | 139 | 0.375676 | 0.0000829859 |

Both fresh predictors select local. The quality difference is two correct archival answers across 370 requests. The paired 40-resample bootstrap interval is [0, 0.013514], which includes zero. This does not establish broad superiority or repair the original coding population.

Independent review found 758 archived outcomes above the preselected 1,024-token output cap. The baseline serves 59 such responses and KL serves 60. The maximum archived output is 16,298 tokens. Those scores were obtained under unverified source generation settings and do not predict scores after imposing the router's cap. The runner now marks served output-cap mismatches as constraint violations. Their served-request rates are 34.5% and 34.7%, respectively. The claim gate includes `ARCHIVED_OUTPUT_CAP_MISMATCH`. No test outcomes were filtered or used to change the cap.

The original fresh-run report is preserved as `report-before-cap-check.json`. A documented measurement amendment re-scores its fixed stored choices for the missing output-cap predicate. It verifies unchanged router and estimator hashes, quality, cost, routing choices and intervals. It is not an additional routing run. Certificate coverage is zero throughout. These results are archival replay evidence, not budget-compliant live-provider performance.

## Reproduce and inspect

Use the existing environment. These commands install nothing and make no provider calls.

```powershell
.venv/Scripts/python.exe scripts/run_quality_improvement.py --dataset results/quality-diagnosis-2026-10-07/recorded-input/imported.json --out results/calibration-improvement-2026-10-07/recorded-report

.venv/Scripts/python.exe scripts/prepare_quality_evidence.py --archive-prefix results/calibration-improvement-2026-10-07/archive-prefix-64MiB.tar.gz.part --exclude-dataset results/quality-diagnosis-2026-10-07/recorded-input/imported.json --out results/calibration-improvement-2026-10-07/fresh-input

.venv/Scripts/python.exe scripts/run_quality_improvement.py --dataset results/calibration-improvement-2026-10-07/fresh-input/imported.json --out results/calibration-improvement-2026-10-07/fresh-report --test-status first-evaluation
```

Raw third-party data and reports remain in ignored `results/`. Tracked scripts provide the repeatable import and experiment. The original and fresh reports contain every candidate's bounds, raw forecast errors, frozen selection hashes, costs, serving choices and collection requirements.

Eight focused calibration, leakage, checkpoint and output-cap checks pass. Fifty-six adaptive release, probe-objective and evidence-time checks passed before the final scoring correction. Two performance benchmark checks pass after that correction. Full lint passes for the new scripts and calibration tests, critical-error source lint passes, and `git diff --check` passes. The broad legacy suite was not repeated. No dependency was installed and no secret file was read during this calibration work.

## Review attention

Reviewed by `gpt-6.1-sol` at high reasoning effort. Review verified numerical endpoints, paired risk, persistence, eligibility, source and archive hashes, duplicate handling and split separation. It found the output-cap measurement omission, now corrected. Review is based on source and artifacts; the actual session transcript was unavailable. The canonical decision trail is `.cache/calibration-improvement/decisions.tsv` and the experiment metric table remains `decision.tsv`.

Final independent audit closes the output-cap disclosure and trail conditions. Every canonical evidence path resolves. The amended fresh report changes only cap scoring annotations, violation rates, disclosure and scorer hashes. Frozen choices, scores, costs, intervals, policy and calibration-core hashes remain identical to the preserved report.

Explain the Number kept raw forecasts separate from interval centers and exposed the output-cap mismatch. Build the Lever produced the repeatable import, benchmark and collection-plan scripts. Production validation still requires independent labels and measured latency from the intended endpoint revisions and generation settings.
