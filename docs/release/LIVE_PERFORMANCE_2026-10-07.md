# Live coding performance, 7 October 2026

The quality bound clears the unchanged 0.65 target on a new, narrow live coding workload: 0.690129 with 128 calibration observations. The original 2.5-second service policy still admits no route because the latency upper bound is 8.100851 seconds. This diagnoses a provider latency problem after obtaining sufficient matched quality evidence; it does not establish that the original 29 recorded requests are fixed.

The separate 10-second diagnostic produced nine correct answers from 32 held-out requests. Eleven requests attempted a provider call, including two consecutive timeouts; the remaining 21 abstained after drift protection invalidated latency evidence. Useful live performance under the original deadline remains unproven.

## Changes made

- Added a bounded coding benchmark and deterministic hidden-case grader. Generated expressions run through a restricted AST interpreter, without executing arbitrary Python. All 16 hidden cases must pass for quality one.
- Added immutable, validated `chat_template_kwargs.enable_thinking` configuration to endpoint identities. The studied NVIDIA endpoint uses thinking disabled, temperature zero and a 1024-token output cap.
- Clarified that JSON objects become Python dictionaries. A public training-only syntax probe changed an unsupported attribute expression into a correct indexed expression. Compact prompts preserve the same specifications and hidden cases.
- Added durable attempt receipts, frozen source/configuration hashes, real routed execution, model identity checks and explicit unknown billing. Materialized evidence lists and a 144-row persistence check prevent the new harness from exhausting a generator during `add_many`.
- Cached exact squared feature distances in a bounded 8192-entry LRU. Evidence expiry, drift checks, neighbor ordering and map identity remain in the normal path.
- Completed legacy NVIDIA provider registration, key binding, dedicated NVIDIA transport and CLI/server wiring. NVIDIA, Google and OpenRouter credentials stay isolated. Runtime startup of the legacy CLI/server remains unverified because the current environment lacks `pydantic_settings`; nothing was installed.

## Fixed live design and results

Run `run-004` froze seed 43, 16 training tasks, 128 calibration tasks, 16 validation tasks and 32 held-out tasks. Specifications are unique across splits. Four task families cover numeric filtering, string normalization, record selection and modular reductions. The compact message admission range is 805 to 928 bytes, within one exact evidence stratum. This is bounded expression synthesis, not a general coding-agent evaluation.

The selected endpoint was `nvidia/nemotron-3.5-lightning-30b-a3b`. Preflight used training tasks only. Training and calibration enter the capability map; validation and test scores do not. Every failed calibration call counts as quality zero. The normal continuously running control plane executes the held-out requests; test scores never feed learning or recovery. Quality target 0.65, quality risk 0.05 and output cap 1024 stayed fixed.

| Split | Requests | Correct | Provider timeouts | Terminations above 2.5 seconds |
| --- | ---: | ---: | ---: | ---: |
| Training | 16 | 14 | 2 | 5 |
| Calibration | 128 | 102 | 19 | 38 |
| Validation | 16 | 15 | 0 | 1 |

The 160 non-test calls produced 131 correct answers, with 110 correct within 2.5 seconds. Their median request termination was 1.3596 seconds. Twenty-one provider timeouts and eight rejected expressions are retained. This 81.875% descriptive quality includes training and validation; the fitted calibration score alone is 102/128, or 79.6875%.

| Policy deadline | Quality lower bound | Latency p95 upper bound | Initial admission |
| --- | ---: | ---: | --- |
| 2500 ms, original | 0.690129 | 8100.851 ms | Abstain: `LATENCY_RISK` |
| 5000 ms, sensitivity | 0.690129 | 8100.851 ms | Abstain: `LATENCY_RISK` |
| 10000 ms, post hoc diagnostic | 0.690129 | 8100.851 ms | Selected NVIDIA endpoint |

The original collection stopped at a declared latency futility rule: with 128 planned calibration observations, tolerance rank 126 cannot admit the 5000 ms policy after three exceedances. Its partial failure report is preserved. A separate continuation completed the same fixed sample count to diagnose quality, retaining all original evidence and failures. It did not extend calibration until a favorable bound appeared.

The 10-second profile was chosen after observing failure of the two stricter profiles. It is exploratory. It attempted 11 of 32 held-out requests, delivered nine correct answers, timed out twice and abstained 21 times. Correct-answer coverage was 28.125%; accuracy among provider attempts was 81.818%. There were zero measured terminations above 10 seconds. These figures are not success under the original policy.

The provider wait cap is eight seconds, so measured latency includes terminated failures and cannot establish the latency of successful completion. Recovery currently treats `ProviderTimeout` as an operational deadline adverse event even when that transport cap expires before the 10-second policy deadline. Two consecutive timeouts triggered the later abstentions. This cap/recovery mismatch is a diagnostic confound; drift was kept enabled and was not reset to improve coverage.

There were 173 actual provider attempts: two preflight, 160 evidence and 11 diagnostic executions. The 194 completed receipts also include 21 abstention decisions. Run-004 has no uncertain attempt. Its execution ledger verifies. A stopped earlier discovery run preserves one uncertain attempt and never retries it automatically. Billing remains unknown, immutable provider revisions were not observed, and benchmark evidence is ineligible for production certification. There is one selected model, so no adaptive routing gain or fixed-baseline speedup is established.

## Router CPU measurement

Five interleaved comparisons exercised actual `ControlPlane.decide` against the archived implementation, with 160 total decisions and zero errors. For a synthetic 2400-row map repeating 112 feature vectors, the median of five warm-run medians changed from 127.13 ms to 75.95 ms; each paired run improved. The 224-row comparison was inconclusive: 26.73 ms versus 24.90 ms, with overlapping ranges and two slower optimized pairs. Machine load varied. The cache helps repeated feature distances on the larger measured workload; this is not a live end-to-end latency improvement claim.

## Verification and artifacts

Independent Sol high review found no material source blockers across 14 changed files. Eight focused offline/mocked tests passed, covering the real benchmark path, 144 persisted evidence rows, primary-policy abstention accounting, grading safety and provider credential isolation. Earlier reasoning-setting and calibration/time checks passed. No broad test suite was repeated for handoff and no dependency was installed.

Full scoped Ruff checks pass for the new scripts, grader, tests and legacy client/settings/types. Critical-error checks pass for the other modified source files. Broader lint still reports existing style and unused-import findings outside this narrow gate. Prove It Works guided checking real execution and persistence; Explain the Number guided separating correct answers, attempted calls, abstentions and unknown costs.

Final artifact checks verified all nine frozen runtime source hashes, the executed continuation source hash, complete receipt accounting and the SQLite ledger. The executed continuation source and original emitted report are archived. The current continuation differs afterward only in import formatting, explicit sanitized CLI exception types and reporting the transport/recovery limitations. A documented report amendment adds those limitations and audited counts without changing measured metrics.

- [Final live report](../../results/live-coding-2026-10-07/run-004/quality-diagnostic-report.json)
- [Quality assessments](../../results/live-coding-2026-10-07/run-004/quality-assessment.json)
- [Original strict-policy failure](../../results/live-coding-2026-10-07/run-004/strict-profile-report.json)
- [Frozen workload and source identities](../../results/live-coding-2026-10-07/run-004/manifest.json)
- [Final artifact audit](../../results/live-coding-2026-10-07/run-004/final-audit.json)
- [CPU comparison](../../.cache/live-usefulness/overhead/production-comparison.json)
- [Independent review](../../.cache/live-usefulness/final-review.md)
- [Decision trail](../../.cache/live-usefulness/decisions.tsv)

Raw results and the local decision trail are ignored by Git. This is an artifact audit; a session transcript link was unavailable. Baseline commit is `6f6a4eb`. The user subsequently authorized committing and pushing this follow-up; publication status is recorded in the local session record.

For a fresh authorized live run, choose a new output directory. Completed or uncertain calls cannot be replayed in place. The program may load only `NVIDIA_API_KEY` under the user's earlier explicit pilot authorization; no sensitive file was opened or displayed through a tool.

```powershell
.\.venv\Scripts\python.exe scripts/run_live_coding_benchmark.py prepare --out results/live-coding-new --candidate-ids nvidia-lightning-coding --seed 44 --prompt-style compact --calibration-requests 128
.\.venv\Scripts\python.exe scripts/run_live_coding_benchmark.py preflight --out results/live-coding-new
.\.venv\Scripts\python.exe scripts/run_live_coding_benchmark.py collect --out results/live-coding-new
.\.venv\Scripts\python.exe scripts/run_live_coding_benchmark.py holdout --out results/live-coding-new
```

Run `run_live_quality_continuation.py --out <directory>` only after a strict latency-futility stop if a separate fixed-count quality diagnosis is needed. Its 10-second diagnostic cannot replace the original results.

The next performance requirement is a provider whose tail latency fits 2.5 seconds on these matched prompts. Compact messages and reduced reasoning produced many fast correct responses, but did not remove the timeout tail. Before another diagnostic, align the transport cap, censoring semantics and recovery deadline attribution. The quality result should also be checked on representative recorded requests; this fresh task population cannot establish that the original coding population improved.
