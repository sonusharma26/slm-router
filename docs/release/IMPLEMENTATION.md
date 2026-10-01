# Adaptive evidence release implementation

## Scope and release boundary

This is an integrated preview, not a feature-complete, empirically superior public v2. Feature priorities were real bounded planning; request-conditioned evidence and certificate enforcement; durable execution; targeted measurement/recovery; evaluator authority and policy rollout; and reproducible benchmarking. Minimal policy diff and Prometheus output are also included. The original legacy package is retained.

| Proposed feature | Implementation and boundary |
|---|---|
| 1. Execution-plan planner | `planning/planner.py` enumerates direct plans, ordered two-endpoint cascades/verify plans and unordered parallel pairs. More than two model calls in a plan are not generated in this release. Search truncation abstains rather than pretending to have optimized the entire space. |
| 2. Conditional map | `capability/conditional.py`: auditable kNN, explicit train/calibration provenance, per-metric evidence and local bounds. No learned neural router, tree ensemble or ordinal regressor was added. |
| 3. Certificates | `policies/certificates.py`: exact request, policy, endpoint/config/price snapshots, plan topology, map, calibration and feature bindings. Execution verifies current validity before dispatch. |
| 4. Active measurement | `probes/active.py`: derives uncertainty/boundary/staleness scores and reserves daily dollars/calls before a measurement callback executes. This is a batch, decision-impact heuristic, not an exact Bayesian value-of-information estimator. |
| 5. Drift/recovery | `drift/recovery.py`: exact public snapshot changes plus windowed quality/latency/failure and supplied workload/evaluator scalar monitors. Statistical detection uses a normal approximation; its false-positive behavior is an empirical question. |
| 6. Outcome authority | `outcomes/trust.py`, `outcomes/store.py`, and the API learning path handle trusted sources, judge-vs-human calibration, expiry, disputes, supersession and delayed results. |
| 7. Lifecycle | `lifecycle/orchestrator.py` is invoked on chat routing and execution/outcome events. Candidate creation and independent offline evidence remain explicit operator inputs. |
| 8. Policy diff | `benchmarks/counterfactual.py`: identical held-out full-information replay of two frozen policies. No unobserved compound counterfactuals are invented. |
| 9. Durable state | `ledger/sqlite.py` and `ledger/state.py` reconstruct effective read models and execution claims; `ControlPlane.replay()` rebuilds the contemporaneous pool and evidence. |
| 10. Execution integration | Explicit bridge between provider APIs and the runtime, OpenAI-compatible transport, and a text-only non-streaming compatibility endpoint. |
| 11. Snapshots | `registry/snapshotter.py` preserves immutable content identities. Metadata refresh requires an adapter with an authoritative snapshot method; otherwise snapshots are manual. |
| 12. Policy compiler | Strict YAML/JSON into a bounded policy contract. No arbitrary Python imports or callbacks are accepted from policy documents. |
| 13. Benchmark Lab | Static runner, importer, baselines, actual competitor adapters, paired intervals and descriptive operating-point comparisons. External runs are explicitly opt-in. |
| 14. Dynamic pool | All eight named injections run in a deterministic synthetic world. Directly observable metadata and selected/probed outcomes are separated from hidden truth. |
| 15. Sparse refresh | 5%, 10%, 25%, and exhaustive budgets compare impact, uncertainty and random acquisition with actual reservations and outcomes. No retention result is assumed. |
| 16. Telemetry | Minimal Prometheus-compatible text, including decisions, certificates, execution errors, realized spend, drift and unresolved claims. No dashboard or complete OpenTelemetry instrumentation. |

## Request and plan contracts

Request conditioning uses application, task, traffic slice, tags, feature version, modality, required capabilities, privacy class, schema hashes and log2 input/output length buckets. The query vector is supplied before inference. The thin chat endpoint supplies a frozen signed lexical-hash vector, not a pretrained semantic embedding. Applications may supply their own frozen embeddings, with an immutable feature-version identifier and a documented pre-response feature protocol.

Each eligible plan has a static sum-of-calls price reservation using the input bound, operator-verified token overhead and maximum output tokens. This is deliberately conservative for a conditional escalation. A cascade executes a second endpoint only after an execution failure; verify/escalate executes it after verifier rejection/failure. Parallel branches run concurrently and the local selector can return only an actual branch result.

Compound quality, expected cost and latency require observed **joint end-to-end plan evidence** keyed by plan topology and operator identity. Marginal endpoint scores do not identify correlated errors, verifier errors or selection quality. Changing a local verifier/selector implementation requires a new registered identity and fresh joint evidence.

Eligibility rejects disallowed providers/endpoints, unsupported modalities or capabilities, governance mismatches, unavailable/unhealthy endpoints and context overflow. A certificate-required policy cannot use an uncertified fallback. Explicit least-shortfall fallback relaxes only the quality gate; it cannot bypass privacy, spend, latency or call caps. `safe_fallback` must pass the same constraints; it is not a privileged escape hatch.

## Estimates, confidence and evidence identity

Training neighbors estimate quality and token output expectations. Distinct calibration requests provide quality lower bounds and latency tolerance bounds. Repeated labels for a single request cannot inflate the calibration count; train/calibration request IDs are disjoint across the entire model pool.

For `n` independent bounded calibration scores and `M` searched plans, the quality radius is:

```
sqrt(log(2 / (quality_risk / M)) / (2*n))
```

The lower bound is the calibration mean minus that radius, conservatively expanded where needed to contain the reported point estimate. It concerns the selected local population mean under the stated assumptions. Neighborhood exchangeability and stability are assumptions, not facts proven by hashing.

For latency, an exact binomial CDF chooses the smallest calibration order statistic whose probability of lying below the population p95 is at most `latency_risk / M`. This distinguishes a confidence bound **on p95** from an ordinary 95% prediction interval. Even the observed maximum may be unsupported: with one plan and 5% risk, 58 IID observations are insufficient; 59 can support the maximum as the upper p95 tolerance bound. Larger searched families need more calibration. `k` must be large enough for the required order statistic. No certificate is issued when the tail cannot be supported.

Quality, failure and latency have separate bounds. This is not an anytime guarantee for arbitrary repeated searches over the same calibration set; adaptive policy tuning requires fresh calibration and controlled rollout. Whole-run “all constraints always hold” confidence is not the same as each component's risk budget.

Map and row fingerprints are content-addressed and cached. Hashes of immutable observations are reused rather than serializing every complete row on every decision. An estimate carries training/calibration hashes, neighborhood/order identity, estimator method, sample counts, revision applicability, age and assumptions. Exact request hashes prevent a certificate being transplanted to another prompt in the same coarse slice.

## Price and revision changes

Endpoint identity contains provider, model, revision, region/deployment, inference configuration and price version. Configuration dictionaries are recursively frozen and cannot contain provider credentials or transport overrides. Snapshot IDs are content hashes. A capability revision deliberately excludes price and availability fields, permitting quality evidence reuse across pure price changes.

A deterministic snapshot change identifies affected metric dimensions. Quality-only invalidation preserves latency evidence; latency-only invalidation preserves quality evidence. A revision/configuration change makes the old capability revision inapplicable. Certificate dependencies are invalidated before the updated planner is checkpointed, and durable invalidation events are replayed after an interrupted checkpoint.

Historical drift events invalidate only certificates issued at or before the event. They do not revoke newly issued post-recovery certificates on every restart. Recovery acknowledgement requires a current certificate that actually uses the affected endpoint for the requested slice. Completed recoveries and detector windows survive restart.

## Measurement and evaluator authority

`ActiveMeasurementLoop.run(tasks, policy, measure, ...)` closes the budget → acquisition → measurement → validated observation → map-update path. The application owns `measure`: it must execute the intended endpoint and provide an appropriate deterministic/application/human evaluation. The library cannot infer a general-purpose correctness oracle from raw generated text.

The impact heuristic asks whether quality/latency uncertainty can change an eligible decision, using nominal expected cost for cost-priority comparisons. It skips stable, dominated alternatives. This is not a guarantee of globally optimal information acquisition, and random/uncertainty refresh remain explicit comparison strategies. A run ranks a batch using current evidence; call it in bounded batches to re-evaluate acquisition after learning.

Budgets are UTC-day call counts and dollars. Reservations are transactional across SQLite connections. Failed or unknown-cost probes retain their entire reservation. The system never reports them as zero-cost successes. Supplied traffic-mass weights are application inputs, not automatically inferred workload frequencies.

`ControlPlane.add_outcome()` correlates a delayed outcome with the recorded decision/execution, evaluates source eligibility, persists it, creates a training/calibration observation and updates rollout evidence. A weak proxy cannot independently train this production map or promote a policy. A calibrated judge requires measured human agreement and an unexpired trust record; its trust expiry bounds derived evidence/certificate expiry. `invalidate_evaluator()` invalidates affected learned quality evidence and dependent certificates. Superseded/disputed labels are removed from effective learning rather than counted again.

Quality and latency observed for a compound plan are not falsely attributed as marginal quality/latency for each endpoint. Application-supplied source names are not cryptographic proof that evaluation happened: submitting outcomes is an authenticated operator capability.

## Durable execution and rollout

A decision is recorded with immutable policy, pool, feature, map and seed inputs. Before a provider side effect, SQLite records a unique execution claim. A completed execution is idempotently returned on repeat access. An interrupted/in-progress claim cannot be automatically re-executed: the external provider may already have charged or completed the call. This is conservative at-most-once dispatch admission, not an impossible claim of distributed exactly-once inference.

Endpoint snapshots and policy eligibility are rechecked before each dispatch, including uncertified execution. Certificates are additionally checked where required. The total plan reservation includes failed attempts. Unknown accounting is explicit; certified execution fails closed on incomplete accounting. Realized violations are recorded even though software cannot retroactively undo provider billing.

The lifecycle accepts independent full-information/live-holdout offline evidence, not synthetic or weak-proxy promotion evidence. The offline evidence object is an operator attestation; callers must retain and verify its source artifacts. Shadow decisions do not execute a candidate or affect the served route. Sticky canary allocation is a deterministic hash of application/session-or-request/candidate version. Outcome-count sequential bounds control quality stopping/promotion; malformed results, hard spend/accounting failures and excessive latency violations can abort earlier. An incumbent remains a rollback target. A bootstrap policy is explicitly marked operator-initialized and unvalidated.

There is no daemon or external scheduler hidden in these methods. The service invokes routing and outcome transitions on requests; applications schedule measurement and submit candidate policies through the Python orchestration API. Unbounded autonomous candidate generation/retraining was not added.
