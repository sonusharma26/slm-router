# V2 Dataset Preparation and Benchmark Protocol

**Status:** Planned; no empirical gate is passed by this document.  
**Date:** 2026-07-27  
**Roadmap scope:** `V2-306`, `V2-700`, and `V2-705` through `V2-707`.

## Purpose

This protocol turns the dataset-independent V2 control plane into a reproducible
empirical evaluation. It intentionally replaces the legacy V1 oracle/FQE workflow:
V2 full-information data uses direct replay, while partial-feedback and sequential
data use their own declared evaluators.

## Data regime and first benchmark

Start with a public **full-information** benchmark matrix, using an
LLMRouterBench adapter (`V2-700`). Each request must have an observed outcome for
every endpoint or supported plan being compared. Do not build a paid live-provider
oracle before this adapter and a small pipeline-validation slice are working.

For full-information data:

- use direct held-out replay and paired comparisons;
- do not use FQE, IPS, SNIPS, DR, or SWITCH;
- retain missing endpoint coverage as missing, not as an imputed score.

Contextual-bandit and sequential-plan data must be stored and evaluated separately
under the requirements in ADR-001.

## Required normalized record

The adapter must preserve source fields and emit a versioned normalized record with:

- `request_id`, `dataset_id`, `source_dataset_version`, `domain`, and `traffic_slice`;
- prompt, reference, and response hashes (and protected/raw locations where allowed);
- immutable endpoint and pricing snapshot identifiers;
- plan type, quality dimensions, label provenance, and evaluator metadata;
- input/output token accounting, calculated cost, TTFT where available, total latency,
  provider status, and error fields;
- split identifier, source artifact hash, transformation version, and schema version.

Raw upstream data is immutable. Normalized data is a distinct derivative and must
retain its raw-source hash.

## Leakage-safe splits

Generate deterministic grouped splits before fitting any policy:

1. train;
2. calibration;
3. policy-selection validation;
4. blind test;
5. temporal or model-version holdout;
6. leave-domain-out test.

All endpoint rows, paraphrases, generated variants, shared templates, and cached
responses for one logical request belong to the same group. Blind-test labels must
not be available to training or tuning commands.

## Artifacts and manifests

Create and version these artifacts:

```text
benchmarks/
  manifests/
    llmrouterbench-v1.json
    splits-v1.json
    preregistration-v1.json
  raw/
    llmrouterbench/
  normalized/
    full_information.parquet
  reports/
```

The dataset manifest records source/version/license, raw and normalized hashes,
schema and transformation versions, endpoint and pricing snapshots, split hashes,
exclusions, counts, and creation time. The preregistration records hypotheses,
metrics, exclusions, thresholds, seeds, confidence level, and stopping rules; its
frozen hash is stored with every report.

## Execution order

1. Implement the LLMRouterBench adapter and normalized schema.
2. Build deterministic grouped splits and dataset/preregistration manifests.
3. Prepare a 100-500 request pipeline-validation slice across domains.
4. Validate completeness, label provenance, immutable identities, split isolation,
   missing-coverage handling, and direct-replay correctness.
5. Run static full-information replay on validation, then blind test.
6. Compare sparse refresh with full refresh and random/uncertainty/impact probing.
7. Run the Dynamic-Pool Gauntlet for each frozen profile and scenario.
8. Publish raw eligible observations, artifacts, exclusions, confidence intervals,
   and failure analysis.

## Mandatory static baselines

- every fixed endpoint;
- cheapest eligible endpoint;
- static best endpoint;
- random;
- tuned confidence cascade;
- kNN;
- logistic or gradient-boosted router;
- RouteLLM-compatible policy where the model pool permits;
- oracle/hindsight upper bound, explicitly labelled unattainable.

## Metrics

Report paired differences and bootstrap confidence intervals for:

- constraint-violation rate and upper confidence bound;
- quality at matched cost and cost at matched quality;
- regret against full-information or hindsight reference;
- worst required traffic-slice performance;
- active-probe spend and decision-relevant capability-map error;
- routing overhead, abstention rate, and uncertified-decision rate;
- for dynamic tests: detection delay, false-positive rate, recovery time, rollback
  frequency, and prevented violations.

## Sparse-refresh comparison

For identical requests and endpoint snapshots, compare exhaustive refresh against
random, uncertainty, and decision-impact schedules at fixed 5%, 10%, and 25%
budgets. The initial roadmap hypothesis is at most one percentage point held-out
routing regret at no more than 25% of exhaustive calls. Report failure if the
hypothesis does not pass; do not change the threshold after the blind test.

## Dynamic-Pool Gauntlet

Run all versioned scenarios against cost-capped, latency-critical, and
privacy-restricted profiles:

- price change;
- model replacement;
- endpoint loss;
- latency shift;
- quality regression;
- workload shift;
- reward delay;
- evaluator corruption.

Compare adaptive control with frozen/stale policies, random refresh, full refresh,
and a safe fixed-endpoint baseline where applicable.

## Current implementation gap

The repository already contains direct replay, baseline policies, capability-map
models, active-probe scheduling, and Gauntlet definitions. It does **not** yet
contain a V2 benchmark adapter, dataset/split preparation command, report runner,
or end-to-end Gauntlet runner. Those are the next implementation tasks; this
document is a protocol, not evidence that they have run.
