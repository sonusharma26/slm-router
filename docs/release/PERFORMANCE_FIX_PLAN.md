# SLM Router performance fixes

The router needs useful quality evidence and reliable constraint protection. The recorded replay answered none of 29 held-out requests at the unchanged quality target of 0.65. The latency-shift simulation recorded eight violations among 40 decisions that still had current certificates. These results describe the supplied benchmark fixtures, not production performance.

## Priorities

| Priority | Evidence and required change | Acceptance check | Status |
| --- | --- | --- | --- |
| 1 | Local calibration neighborhoods contain only 4 to 12 samples. Even perfect scores cannot clear 0.65 with the current Hoeffding penalty. Define the guarantee and collect independent evidence in justified neighborhoods before changing estimator arithmetic. | Fresh held-out coverage, quality, regret, and bound validity by stratum. Keep the target at 0.65. | Open design and data work |
| 2 | The statistical detector missed the observed latency shift. Execution monitoring excluded incomplete accounting and joint plans. Probes bypassed monitoring. Add a bounded operational deadline rule and feed trustworthy measurements into recovery. | Repeated overruns invalidate affected evidence and certificates within a stated observation limit. Cover timeout, incomplete accounting, probe, restart, and compound attribution. | Fixed operational guard; broader statistical validation open |
| 3 | A response executed before drift could enter recovery evidence when its label arrived afterward. Evidence used evaluation time as observation time. | Delayed labels retain execution time through learning and restart. Pre-change responses cannot become fresh recovery samples. | Fixed and locally verified |
| 4 | Acquisition picked its incumbent by cost even when serving optimized quality or latency. A reproduced dominated candidate was marked decision-changing. | Acquisition shares serving feasibility and objective ordering, including tie-breaks. Cover all three objectives. | Fixed and locally verified |
| 5 | The current lexical neighborhood predictor has worse calibration MAE than the constant training-mean baseline for both recorded endpoints. | A replacement improves untouched-split prediction and routing outcomes at matched total spend and latency, then receives independent calibration. | Open experiment |
| 6 | Paid sparse probing matched frozen routing on quality and regret in the supplied fixture. The benchmark omitted the frozen strategy and a unified spend comparison. | Report a zero-probe control and total serving plus probe cost. Include scenarios that change the feasible winner and replicated seeds before claiming probing helps. | Controls fixed; benefit study open |
| 7 | CPU profiling finds repeated identity serialization and neighborhood lookup on the abstention path. A feedback row invalidates the index. Row fingerprints are already cached. | Preserve decisions, lineage, and replay identity. Measure realistic successful serving before claiming a speedup. | Deferred behind correctness |
| 8 | The legacy registry rejects NVIDIA and its client dispatches non-Google calls through OpenRouter. The newer NVIDIA adapter completed three functional pilot calls. | Either migrate maintained callers to the newer adapter or retire legacy serving. Validate registry, dispatch, usage, and revision behavior together. | Open compatibility decision |

The NVIDIA pilot was uncertified and used a prototype quality target of zero. It proves transport functionality only. It does not validate the 0.65 benchmark promise.

## First batch and verification

The first batch contains three reproduced correctness fixes and one benchmark control. Use the existing Python environment. Install nothing. Read no sensitive files and make no live provider calls. Preserve the existing quality diagnosis edits.

Each correctness fix gets a narrow regression check that fails before the production change and passes afterward. Initial worktrees isolate worker edits; the review repair has explicit file ownership in the integrated checkout. The parent reviews and integrates each patch, then runs the existing control-plane suites and the recorded replay. Do not equate passing regression tests with validated statistical coverage.

The throughput checkpoint is four independently checkable units. Evidence timing and objective selection are bounded changes. Operational drift spans execution, measurement, persistence, and recovery and receives the deepest design review. Sparse controls remain separate from runtime changes. Quality-method and predictor replacements require new data and are not part of this batch.

## Grounding

This section describes baseline source at `34480366e8aa9fab14d985590d276bde262adeed`, before this batch.

`ControlPlane.execute` persists an execution, then feeds only fully accounted single-endpoint latency into `DriftRecovery.observe`. `ControlPlane._learn_outcome` maps a trusted label to an evidence row and currently dates that row with the label's evaluation time. `ConditionalCapabilityMap` applies age and metric-specific invalidation against the row's observation time.

`Planner.decide` generates the candidate family, applies hard and statistical gates, and sorts feasible plans according to the policy objective. `ActiveMeasurementLoop.rank` repeats candidate estimation but chooses a cost-only incumbent. Probe completion appends an observation to the capability map and ledger without notifying recovery.

`DriftRecovery` persists fixed-window statistical detector state and metric-specific invalidations. New operational protection must remain distinct from its approximate statistical test. Compound execution latency must not be invented as an individual endpoint measurement.

## Evidence

- [Quality-bound diagnosis](QUALITY_BOUND_DIAGNOSIS.md) records the sample ceilings, prediction error, replay provenance, and functional NVIDIA pilot.
- `results/quality-diagnosis-2026-10-07/recorded-replay/report.json` contains the reconstructed recorded replay.
- `results/quality-diagnosis-2026-10-07/latency-detector-trace.json` contains the mixed detector windows that missed the shift.
- `results/quality-diagnosis-2026-10-07/sparse-frozen-baseline/report.json` contains the frozen comparison.

## Verified changes

Learned outcomes use the immutable `execution_started` ledger timestamp. Completion and label times cannot establish freshness. Startup corrects already checkpointed outcome rows, revokes dependent certificates, and preserves original ledger events. A historical execution without a durable start claim cannot establish fresh evidence. Outcome qualification remains separate from this rule.

Serving and acquisition use one planner assessment and selector. Probe projections retain nominal expected cost and use the same feasibility, objective, fallback, and tie rules. Incomplete enumeration produces no decision-impact claim. Hypothetical calibration remains a collection heuristic and never creates a certificate.

Sparse reports include frozen routing, realized serving cost, known probe cost, reserved probe cost, and combined totals. Unknown probe cost remains unknown and retains its reservation. The reserved total is an exposure bound. It is not actual spend. Initial evidence acquisition is a shared excluded cost.

Serving and probes share a recovery owner. Two consecutive overruns or typed provider transport timeouts invalidate affected latency evidence and certificates in the same target, slice, revision, policy, and deadline context. Healthy measurements reset the counter; generic errors do not. Billing completeness does not suppress an overrun. Durable receipts, counters, drift effects, and checkpoints commit in one existing ledger batch. Startup reconciles terminal facts once. A probe keeps its measurement start time through invalidation.

Compound total latency stays a plan measurement. A repeated compound failure conservatively invalidates all endpoint dependencies with an explicit compound scope; no aggregate duration enters endpoint statistical windows. This can discard valid marginal evidence. Recovery still requires the ordinary certificate gates and fresh independent evidence.

Before production edits, focused checks demonstrated false-fresh delayed evidence, wrong quality-objective acquisition, incomplete-search acquisition, the missing frozen comparator, and unmonitored repeated overruns. Before the final recovery-order repair, the integrated eight-file suite passed **95 tests**. It covers timestamp repair, objective and tie rules, incomplete enumeration, cost coverage, provider timeout classification, compound attribution, batch failure, and restart replay. The offline smoke check, new-test lint, undefined-name checks, and `git diff --check` passed. No dependencies were installed.

The matching latency replay uses `latency_x3`, 40 requests, seed 42, 2,400 initial requests, impact probing, and a 0.1 probe fraction.

| Result in the matching fixture | Baseline | Integrated batch |
| --- | ---: | ---: |
| Certified constraint violations | 8 / 40 | 2 / 40 |
| Detection delay after injected change | Undetected | 4 requests |
| Drift events before injected change | 0 | 0 |
| Quality across all requests | 0.9143 | 0.9455 |
| Mean routing regret | 0.05165 | 0.02046 |
| Mean serving cost | $0.00025824 | $0.00029472 |

The two remaining violations occur at steps 22 and 24. The second trips the guard. The protection is reactive and cannot prevent the first adverse observations. Later affected math requests use the strong endpoint. These are one-seed synthetic results, not a statistical coverage or false-positive guarantee. Serving cost rises because healthy routing uses the more expensive endpoint. CPU speed was not optimized or established by this batch.

The recorded replay after initial integration still abstains on **29 of 29** held-out requests at **0.65**. All quality diagnostic records equal the preceding replay. Useful quality bounds remain the first unresolved data and method problem. Passing regression checks does not establish statistical coverage or production readiness.

The sparse replay after initial integration uses `quality_minus20pct`, 40 requests, seed 42, 1,200 initial requests, and a 0.1 probe fraction. Frozen and impact routing have identical quality (0.96354), regret (0.002443), and zero abstentions or constraint violations. Impact makes 12 probes and raises known total cost from $0.01536 to $0.0162816, a 6% increase. Objective alignment fixes the reproduced ranking defect; it has not established that paid probing helps this fixture. A benefit study remains open.

The full suite can be reproduced with the existing environment. It was not repeated after the final repair because the user requested faster work and fewer tests:

```powershell
$env:PYTHONPATH=(Join-Path (Get-Location) 'src') + ';' + (Get-Location).Path
.\.venv\Scripts\python.exe -m pytest tests/test_quality_diagnostics.py tests/test_adaptive_release.py tests/test_v2_control_plane.py tests/test_v2_phases.py tests/test_evidence_time_regression.py tests/test_probe_objective_regression.py tests/test_performance_benchmarks.py tests/test_operational_drift_regression.py -q
.\.venv\Scripts\python.exe -m inference_control.smoke
```

## Poteto decisions

| Principle | Concrete effect on this batch |
| --- | --- |
| Model the Domain | Separate typed endpoint and compound latency; preserve unknown spend. |
| Fix Root Causes | Share the planner's selector and use durable sample time. |
| Explain the Number | Diagnose calibration ceilings and match the 2,400-request latency fixture. |
| Prove It Works | Reproduce failures before edits and run integrated regressions and replays. |
| Build the Lever | Add frozen routing and combined spend to sparse reports. |
| Sequence Work into Verifiable Units | Keep timing, selection, protection, and benchmark controls separately checkable. |
| Separate Before Serializing Shared State | Isolate worker edits, then review sequential integration and durable effects. |
| Laziness Protocol | Keep existing owners and defer data-dependent estimator replacement. |
| Make Operations Idempotent | Reconcile terminal order and require one-time repair after interrupted monitoring. |
| Test Behavior, Not Implementation | Assert routing, certificate revocation, and restart behavior rather than mock call counts. |

Local decision records and detailed before/after artifacts live in `.cache/performance-fixes`. The tracked regression checks are the reproducible verification artifact.

## Review checkpoint

Independent `gpt-6.1-sol` / `max` review confirmed the reported replay results and found a restart-order defect: an earlier terminal measurement without a receipt can replay onto a counter that already includes a later measurement. Both false trips and missed trips were reproduced. It also found an inherited recovery gap that accepts a certificate for a different slice. Three new behavior checks failed before the repair. The repair drains durable terminal facts in order before planning or further execution and probing. It uses a contiguous cursor for subsequent scans. Histories already processed out of order receive a one-time targeted latency invalidation; trusted quality data remains intact. Recovery acknowledgment now requires the affected traffic slice. Focused final validation passed **19 tests** across the ordering and operational drift files. It includes one-time historical repair, unaffected endpoint/slice certification, preserved quality rows, atomic repair rollback, and a second startup with no additional events. Final diff and undefined-name checks pass.

The reviewer also ran the legacy registry checks: two tests fail because the legacy model contract rejects `nvidia-build`, matching priority 8. The reviewed core and new regression subset passes. This is separate from the 95-case focused control-plane suite.

The comment review found no added-comment cleanup issue. An unrelated corrupted API title was restored. Review covered source, tests, and artifacts. The unavailable session transcript limits independent validation of historical delegation and test-run chronology. This is a review within the same model family.

The final repair retains the existing single recovery owner for the planner and probe loop. It does not establish distributed correctness for independent recovery owners or mixed-version processes. Startup scans history once; subsequent drains use the verified suffix cursor. The 95-case suite and benchmark replays above predate this last repair and were not repeated, as requested. No installation, live provider call, or sensitive-file read occurred during this fix batch.

## Requested performance rerun

The user subsequently requested performance tests. Current-source offline benchmarks and a bounded live NVIDIA run are recorded in [Performance results, 7 October 2026](PERFORMANCE_RESULTS_2026-10-07.md). The three offline conclusions reproduce. GPT-OSS-20B completes six measured calls; Muse Glimmer times out on warm-up. These live measurements do not resolve the archived-data quality bounds.
