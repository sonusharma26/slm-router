# Changelog

## Unreleased — answer-checking cascades and the price-penalty rule

- Pooled routing defaults to `rule="utility"`: route to argmax(forecast − λ·cost), with λ calibrated so mean quality over all routed traffic meets the target. `rule="threshold"` stays selectable; saved maps keep their rule.
- New `inference_control.routing.cascade`: answer-agreement cascades (cheap answer accepted when a checker agrees, otherwise escalate), calibrated with a fixed-sequence test that selects the cheapest passing configuration. Evaluated offline in `slm benchmark compare --zoo` and `scripts/run_llmrouterbench.py` (stored predictions).
- Live answer-agreement cascade: new `agree_escalate` plan type (first and checker concurrently, escalate on disagreement), a default `answer_agreement` function, answers as capability-map evidence (`add_answers`; hashes unchanged when absent), planner-side cascade calibration inside `pooled_calibration` with compound spend/call/deadline gates, and `SLMRouter.fit(..., answers=)`. Opt in with `plans.allowed: [direct, agree_escalate]`, `max_calls >= 2` and `plans.verifier: answer_agreement`; otherwise serving is unchanged. See the Live serving section of `docs/release/CASCADE_ROUTING_2026-10-09.md`.
- Majority voting joins the calibrated cascade family: `CascadeConfig(lam, accept, voters, quorum, sequential)` calls the top-k endpoints by `forecast − λ·cost` and serves the answer that reaches a quorum, otherwise escalates to the quality-first model (no comparable answers always escalates). Plain routing, agreement cascades and votes are calibrated in one fixed-sequence family, so the cheapest passing kind wins with the same guarantee. `default_configs`, `plan`, `run`, `order_configs` and `calibrate` behave as before for non-vote configs; new `cascade.outcome`/`diagnostics` report vote rate, escalation rate and calls per request, the compare report names the selected kind, and `scripts/run_llmrouterbench.py` gains `--max-voters`. On the synthetic zoo votes add about +0.005 AIQ (answer-correlation dependent); see `docs/release/CASCADE_ROUTING_2026-10-09.md`.
- Fix: forecasts are now identical for a request alone or in a batch (similarity ties broke differently under batched BLAS).
- Synthetic zoo: cascade AIQ 0.905 vs kNN 0.878 and best possible question-only 0.904. See `docs/release/CASCADE_ROUTING_2026-10-09.md`.

## Unreleased — pooled learned routing

- **Learned forecasts.** A numpy predictor combines kNN over request features with a spherical k-means cluster table, shrinking toward slice and global means; hyperparameters are chosen on validation only. New hashed word n-gram features (`hash-ngram-v2-256`) and optional sentence-transformer embeddings.
- **Pooled calibration (new default quality method `pooled-ltt`).** The planner calibrates its own routing rule over every full-information calibration request with a fixed-sequence learn-then-test of Hoeffding–Bentkus p-values. With probability at least `1 - quality_risk`, mean quality over all certified traffic is at least `minimum_quality`. Per-neighborhood `local-mean-kl` and `local-mean-hoeffding` remain selectable and restore unchanged.
- **Strong-model fallback.** New `best_predicted_fallback` infeasible behavior serves uncertified requests with the highest-forecast endpoint (validation-tuned margin toward the best training endpoint). Spend, deadline, privacy and call caps never relax. It is the policy-compiler default unless `evidence.require_certificate: true`, which now defaults to `false`.
- **Comparison harness.** `slm benchmark compare` reports cost-quality frontiers, AIQ, quality at the best single model's cost, cost to match it, and an empirical check of the pooled guarantee across resplits. `scripts/run_llmrouterbench.py` evaluates on LLMRouterBench with its own loader, split and test prompts.
- **Performance.** kNN baseline neighbors are computed once per request (≈13× faster); pooled decisions cache evidence, forecasts and signatures.

### Compatibility

- `ConditionalCapabilityMap()` and `SLMRouter()` now default to `pooled-ltt`. Saved maps keep their recorded method. The dynamic drift/probing study, the drift-recovery demo and earlier quality studies pin `local-mean-kl` with abstention, so their documented results reproduce exactly.
- Policy documents that omit `evidence.require_certificate` now compile to uncertified-fallback policies. Set `require_certificate: true` (or `on_infeasible: abstain`) to keep certified-only serving. Compiled versions of such documents change.
- Pooled certification needs calibration requests scored on every eligible endpoint (benchmark imports or exhaustive probes). With only sparse served-traffic evidence the router stays uncertified and falls back. Pooled mode routes direct plans only.
- `numpy>=2.0,<3` is a new core dependency. Two legacy tests now skip without `joblib` instead of failing when numpy is present.

## 2.0.0a2 — adaptive evidence preview

Implemented bounded direct/compound plan search; conditional kNN evidence with separate calibration; exact-request certificates and confidence bounds on local p95; targeted invalidation; durable SQLite read models, detector state and execution admission; trusted delayed outcomes; shadow/canary lifecycle; actual provider/runtime bridging; strict policies; a thin non-streaming Chat Completions API; and split-safe static/dynamic/sparse benchmark tooling.

Added focused adversarial and regression tests, synthetic reproduction reports, operator examples, primary interface references and explicit statistical/deployment limits.

### Compatibility

- The complete offline suite passes on Python 3.12 with current declared dependencies: 85 tests passed. The legacy temperature-scaling objective now extracts SciPy's one-element optimization array explicitly for compatibility with recent NumPy releases.
- `slm` now invokes the adaptive `inference_control.cli`. The old command is `slm-legacy` and its optional heavy dependencies are in `.[legacy]`.
- Core dependencies no longer require the model-training stack. The declared interpreter range is Python 3.11–3.13; local release validation has run on Python 3.12 and 3.13.
- The previous `uv.lock` was removed because it described a different project graph. Generate a new deployment lock; do not follow historical `uv sync --frozen` instructions.
- Frozen default values are validated before hashing, preventing `0` versus `0.0` identity mismatches after serialization. Configuration dictionaries are recursively frozen.
- Certificates now bind the exact request. Existing loosely bound certificates from the earlier skeleton are not valid execution evidence; re-measure/reissue them. Keep a database backup and do not claim an untested production migration from an old ledger.
- The legacy full-information replay helper no longer exposes realized held-out quality/cost to the policy. Supply independent predicted arms or use the new Benchmark Lab. Its former oracle-input behavior was not a valid learned-router benchmark.
- Audit streams now include configuration/map events and durable execution claims. A legacy test expecting only `decision, execution` was updated to verify the required ordering and hash chain instead.
- No full gateway parity, streaming support, live-provider validation or competitor win is implied by the preview version.
