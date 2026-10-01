# Benchmark Lab

## Claim boundary

The archive includes reproducible synthetic runs to validate the control-plane implementation and expose weaknesses. Synthetic rubric scores are **not** accuracy on real coding, reasoning or chat tasks. Frozen numeric difficulty/task features in those fixtures are generator inputs, not features a production router magically knows.

A static report's broad `claim_gate` is deliberately non-passing in this preview. It identifies synthetic evidence, missing competitors/accounting and unsupported quality-superiority comparisons. Even a completed single empirical run cannot justify “beats all benchmarks.” Establish a precisely scoped claim across preregistered datasets, seeds, cost/quality operating points and deployment conditions instead.

## Static protocol

A dataset contains `Query` records, public immutable endpoint snapshots, and a rectangular `Truth` matrix for every request/endpoint pair. Train/calibration/validation/test membership is frozen before fitting. Normalized prompt groups cannot cross splits. The runner rejects duplicate held-out groups rather than pretending correlated duplicates are independent samples.

The router receives only the query and public pool. Test answers, scores, realized cost and response latency remain in the evaluator until after selection. No model oracle is used to route. Calibration requests are disjoint from training, even across different endpoints. The threshold baseline is tuned using validation data, not the held-out test scores.

Participants include cheapest, seeded random, best single model selected on training data, a simple kNN baseline, validation-tuned threshold routing and SLM Router. The default SLM target is fixed by the policy. Predetermined target sweeps provide descriptive operating-point curves; selecting the best curve point after examining test results is not an independently validated win.

Reports include quality with abstentions scored zero, served quality, worst-slice quality, abstention/certificate coverage, known-cost coverage, mean cost, realized constraint violations, quality-shortfall diagnostics, p50/p95 routing overhead, available end-to-end latency, oracle regret and bootstrap/paired quality intervals. A low per-response score is reported separately from a hard spend/privacy/latency violation: a local-mean quality certificate is not a promise that every individual response exceeds the mean threshold.

Matched-cost/quality comparisons are computed from **observed operating points**, with no invented interpolation or oracle per-request mixture. These descriptive curves do not yet have simultaneous uncertainty bands. A real compound-plan comparison needs recorded verifier/selector outcomes; a marginal model-response matrix cannot identify those counterfactuals.

For measured total latency, route time is included in static evaluation. Missing latency stays null, with coverage disclosed. Missing router/classifier overhead cost stays unknown and blocks a complete cost claim. Served token costs and probe costs are separate in dynamic/sparse reports; compare their sum, not only the cheap serving route.

## Import LLMRouterBench

The importer supports the repository result-tree schema:

```
results/bench/<dataset>/<source-split>/<model>/<run>.json
    records: [origin_query or prompt, score, cost,
              prompt_tokens?, completion_tokens?, latency_ms?]
```

```bash
slm benchmark import-llmrouterbench /path/to/curated/results/bench \
  --endpoints /path/to/pinned-model-pool.json \
  --max-output-tokens 1024 --seed 42 --out data/benchmark.json
slm benchmark static --dataset data/benchmark.json --out results/real-run
```

Curate exactly one run per dataset/split/model **before** examining comparative results; repeated ambiguous runs are rejected. Scores must already be normalized to [0,1]. Only models in the provided manifest are imported, and an incomplete cross-model matrix is rejected. The maximum output budget is an explicit operator input, never inferred from held-out completion length. The common input count is a conservative maximum over supplied tokenizer counts; disclose that idealized metadata assumption when comparing deployment behavior.

The importer preserves absent request-level latency; it does not divide aggregate run duration into fictitious per-request timings. Unclassified dataset/judge labels can support a benchmark-relative score comparison but are tagged benchmark-only and cannot independently certify production routes. Do not re-label arbitrary dataset scores as deterministic application correctness.

The importer was tested against schema fixtures, **not** a downloaded full LLMRouterBench release in this session. Versioned real-data import remains part of external validation.

## Real competitor adapters

### RouteLLM

`RouteLLMAdapter` calls the actual public `Controller.route(prompt, router, threshold)` selection method. It does not fake the competitor with a random or cheapest implementation. Install and pin RouteLLM and its chosen checkpoint using its official setup instructions. The example manifest uses `mf` only as a configuration example.

RouteLLM's weak/strong comparison is pair-based. Restrict the **whole imported dataset and every participant** to the same chosen pair. Fix/tune thresholds on independent validation data and record router/checkpoint versions. Embedding or classifier costs are not assumed free; a null overhead value intentionally makes total cost unknown.

```bash
slm benchmark static --dataset data/two-model-matrix.json \
  --competitors examples/v2/competitors.routellm.json --allow-external \
  --out results/routellm-comparison
```

### vLLM Semantic Router

Use the real gateway deployment, pinned to a version/configuration, but point all candidate model backends at the same non-oracle selection stub:

```bash
slm benchmark replay-backend --endpoints /path/to/pinned-model-pool.json --port 8091
```

The stub always returns constant text and the selected model ID. It never receives or serves the benchmark answers. Configure the gateway's candidate model pool to match the entire dataset, then adapt `competitors.gateway.json` to the deployment.

vLLM Semantic Router must expose a model identifier mapped to the shared pool. The adapter rejects an unmapped router alias rather than guessing which model ran. Custom response-header schemes require an explicit adapter extension, not a silent assumption.

```bash
slm benchmark static --dataset data/benchmark.json \
  --competitors examples/v2/competitors.gateway.json --allow-external \
  --out results/gateway-comparison
```

Gateway adapters report **gateway-plus-stub** timing, not classifier-only CPU time. Configure real classifier/embedding overhead cost; it remains unknown otherwise. `replay_backend_confirmed` is an operator attestation, not an automated network-topology proof. Network execution may download checkpoints or charge classifier APIs; it is opt-in and was not performed for the included runs.

A static-only external router cannot automatically be credited with drift recovery. The dynamic runner currently compares the real SLM control-loop variants and trivial/frozen baselines; extending external gateways to receive equivalent dynamic metadata and refresh budgets is additional integration work.

## Dynamic pool

Eight deterministic injections are available: price ×2, endpoint disappearance, model revision change, latency ×3, quality ×0.8, workload shift, delayed rewards and a biased evaluator.

All participants receive the same public endpoint metadata. Hidden quality/latency changes are revealed only by outcomes of selected calls or charged probes. **Pre-decision probes use an independent request catalog**, not the current held-out request. Unselected oracle outcomes are used only to compute evaluation regret/map error after the decision.

Each strategy starts from the same initial evidence. Frozen means no outcome-driven map refresh; it still receives public metadata and follows ordinary availability/privacy gates. Cheapest, impact, random and exhaustive variants are explicitly named. The initial fitting cost is common and excluded from incremental refresh ratios; serving feedback is separate from probe calls.

Price/revision/availability detection is directly observable metadata detection. Quality/latency shifts use monitored windows. Workload-shift, reward-delay and biased-evaluator runs are stress tests; they are **not** labelled as successful automated detection of those faults. In the biased-evaluator stress case, untrusted feedback is logged but cannot train or promote the policy.

Reports include targeted detection delay, pre-change false-positive events, additional probe reservation/cost, bad decisions after the change, regret, error coverage, recovery time and violations during recovery. Null detection/recovery means not detected/not recovered. Recovery is a descriptive run of five consecutive served, nonviolating/non-shortfall requests, not proof of permanent restoration. Inspect abstention and slice composition too.

The inherited detector uses a variance-sensitive normal approximation with nominal alpha spending over non-overlapping looks. It is not a distribution-free certificate. Online/adaptive workload confidence intervals in these exploratory reports are descriptive; replicated seed-level analysis or an appropriate time-series method is required for inference.

## Sparse refresh

The permanent sparse runner compares impact, uncertainty and random acquisition at 5%, 10% and 25% of exhaustive refresh calls, plus a 100% exhaustive reference. Budgets are paced through the workload rather than automatically spent before the change.

Report actual calls, known/reserved probe cost, map error **and its observation coverage**, regret, serving cost, abstention and constraint violations. A method that produces no estimate must not be declared accurate simply because missing cells are excluded from its error average. A method that abstains on everything must not win solely by having zero realized violations.

The “retain performance at ≤25% refresh” hypothesis is **not automatically accepted**. Choose a quality/regret tolerance before running experiments, use replicated seeds and matched information, and evaluate total serving plus measurement cost. A favorable single fixture result is a lead for external validation, not a launch claim.

## Primary interface references

Reviewed for this implementation; pin exact revisions for a real benchmark run:

- LLMRouterBench repository/schema: https://github.com/ynulihao/LLMRouterBench
- RouteLLM repository and controller: https://github.com/lm-sys/RouteLLM and https://github.com/lm-sys/RouteLLM/blob/main/routellm/controller.py
- vLLM Semantic Router repository: https://github.com/vllm-project/semantic-router
- OpenAI Chat Completions API: https://platform.openai.com/docs/api-reference/chat/create

These references describe integration surfaces. They are not evidence that this release outperforms those systems.
