"""Rerunnable offline predictor selection, bound comparison and collection planning."""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference_control.benchmarks.data import BenchmarkDataset, Query, Truth
from inference_control.benchmarks.diagnostics import diagnose_quality, required_calibration_samples
from inference_control.benchmarks.metrics import paired_difference, summarize
from inference_control.benchmarks.routers import SLMRouter
from inference_control.benchmarks.runner import default_policy, evaluate_router, split_data
from inference_control.capability.conditional import stratum
from inference_control.contracts import EndpointSnapshot, RequestContext
from inference_control.util import canonical, digest

PREDICTORS = ("local", "half-shrink", "cohort-mean")
DEPLOYABLE = ("local", "cohort-mean")
AT = datetime(2026, 10, 7, tzinfo=UTC)
K = 128


def error_metrics(pairs):
    errors = [prediction - observed for prediction, observed in pairs]
    return {"requests": len(errors),
            "mae": mean(abs(error) for error in errors) if errors else None,
            "mse": mean(error * error for error in errors) if errors else None,
            "bias": mean(errors) if errors else None}


def select_predictors(dataset):
    train, outcomes = split_data(dataset, {"train"})
    validation, validation_rows = split_data(dataset, {"validation"})
    if not train or not validation:
        raise ValueError("train and validation requests are required for predictor selection")
    truth = {(row.request_id, row.endpoint_id): row.quality for row in validation_rows}
    fitted = {}
    for predictor in DEPLOYABLE:
        router = SLMRouter(dataset.endpoints, k=K, min_samples=8, at=AT,
                           quality_method="local-mean-kl",
                           quality_predictors={ep.endpoint_id: predictor for ep in dataset.endpoints})
        router.fit(train, outcomes)
        fitted[predictor] = router
    selections, endpoints = {}, {}
    for endpoint in dataset.endpoints:
        pairs = {name: [] for name in PREDICTORS}
        for query in validation:
            predictions = {}
            for predictor, router in fitted.items():
                estimate = router.map.estimate(query.request, endpoint.endpoint_id, (endpoint,), at=AT)
                predictions[predictor] = estimate.lineage.quality_diagnostics.raw_prediction if estimate else None
            if any(value is None for value in predictions.values()):
                continue
            predictions["half-shrink"] = (predictions["local"] + predictions["cohort-mean"]) / 2
            for predictor in PREDICTORS:
                pairs[predictor].append((predictions[predictor], truth[query.request.request_id, endpoint.endpoint_id]))
        metrics = {name: error_metrics(values) for name, values in pairs.items()}
        if not pairs["local"]:
            raise ValueError(f"no comparable validation predictions for {endpoint.endpoint_id}")
        ranking = lambda name, scores=metrics: (scores[name]["mae"], scores[name]["mse"], PREDICTORS.index(name))
        winner = min(PREDICTORS, key=ranking)
        deployed = min(DEPLOYABLE, key=ranking)
        selections[endpoint.endpoint_id] = deployed
        endpoints[endpoint.endpoint_id] = {"validation": metrics, "experiment_winner": winner,
            "selected_deployable": deployed, "excluded_validation_requests": len(validation) - len(pairs["local"]),
            "unsupported_half_shrink_won": winner == "half-shrink"}
    configuration = {"k": K, "min_samples": 8, "quality_method": "local-mean-kl",
                     "quality_predictors": selections, "selection_rule": "validation MAE, MSE tie-break, fixed order final tie"}
    return {"train_requests": len(train), "validation_requests": len(validation),
            "training_hash": digest({"queries": train, "outcomes": outcomes}),
            "validation_hash": digest({"queries": validation, "outcomes": validation_rows}),
            "selection_hash": digest(configuration), "configuration": configuration,
            "candidate_grid": {"local": 0., "half-shrink": .5, "cohort-mean": 1.},
            "endpoints": endpoints,
            "selection_labels": "Only train labels fit predictors; only validation labels select them. Actual counts are disclosed.",
            "frozen_before_calibration_and_test": True}


def compare(dataset, selected):
    fit, outcomes = split_data(dataset, {"train", "calibration"})
    missing_latency = any(row.latency_ms is None for row in outcomes)
    requests = {query.request.request_id: query.request for query in dataset.queries}
    archived_output_cap_mismatches = sum(row.output_tokens > requests[row.request_id].max_output_tokens
                                       for row in dataset.outcomes)
    policy = default_policy(minimum_quality=.65, missing_latency=missing_latency)
    assert policy.minimum_evidence_samples == 8 and policy.quality_risk == .05
    variants = (("baseline-hoeffding-local", "local-mean-hoeffding", {}),
                ("kl-local", "local-mean-kl", {}),
                ("kl-validation-selected", "local-mean-kl", selected))
    results, records = {}, {}
    for name, method, predictors in variants:
        router = SLMRouter(dataset.endpoints, k=K, min_samples=8, at=AT,
                           quality_method=method, quality_predictors=predictors)
        router.fit(fit, outcomes)
        rows = evaluate_router(router, dataset, policy, split="test")
        diagnostic = diagnose_quality(router, dataset, policy)
        maximum = {endpoint: values["quality_lower_range"][1] if values["quality_lower_range"] else None
                   for endpoint, values in diagnostic["endpoints"].items()}
        results[name] = {"configuration": {"method": method, "predictors": predictors, "k": K},
                         "map_hash": router.map.version, "summary": summarize(rows, seed=42, resamples=40),
                         "routing_counts": dict(Counter(row["endpoint_id"] or "abstain" for row in rows)),
                         "served_output_cap_mismatches":sum(row["output_budget_mismatch"] for row in rows),
                         "maximum_quality_lower_per_endpoint": maximum,
                         "raw_train_only_prediction_errors": {
                             endpoint: {"calibration": values["calibration_prediction_error"],
                                        "test": values["test_prediction_error"]}
                             for endpoint, values in diagnostic["endpoints"].items()},
                         "diagnostics": diagnostic, "routing_records": rows}
        records[name] = rows
    differences = {}
    for previous, current in pairwise(variants):
        differences[f"{current[0]} versus {previous[0]}"] = paired_difference(
            records[current[0]], records[previous[0]], seed=42, resamples=40)
    return {"policy": policy.model_dump(mode="json"), "policy_hash": digest(policy),
            "missing_latency": missing_latency, "interventions": results,
            "archived_output_cap_mismatches":archived_output_cap_mismatches,
            "successive_paired_quality_differences": differences}


def collection_plan(dataset, comparison):
    groups = defaultdict(list)
    for query in dataset.queries:
        groups[stratum(query.request)].append(query)
    truth = {(row.request_id, row.endpoint_id): row.quality for row in dataset.outcomes}
    candidates = comparison["interventions"]["kl-validation-selected"]["diagnostics"]["records"]
    local_requirements = defaultdict(list)
    request_strata = {query.request.request_id: stratum(query.request) for query in dataset.queries}
    for candidate in candidates:
        if not candidate["estimate_missing"]:
            local_requirements[request_strata[candidate["request_id"]], candidate["target_id"]].append({
                "request_id": candidate["request_id"], "local_calibration_n": candidate["calibration_size"],
                "fixed_observed_mean": candidate["bound"]["calibration_mean"],
                "required_total_n": candidate["required_samples_at_observed_mean"],
                "new_independent_requests_needed": max(0, candidate["required_samples_at_observed_mean"] - candidate["calibration_size"])
                    if candidate["required_samples_at_observed_mean"] is not None else None,
                "k_blocker": candidate["required_samples_exceed_k"],
                "comparison_family": candidate["comparisons"], "effective_risk": candidate["bound"]["effective_risk"]})
    entries = []
    family = len(dataset.endpoints)
    alpha = comparison["policy"]["quality_risk"] / max(1, family)
    for key, queries in sorted(groups.items()):
        counts = {split: len({query.request.request_id for query in queries if query.split == split})
                  for split in ("train", "calibration", "validation", "test")}
        calibration = [query for query in queries if query.split == "calibration"]
        endpoint_plans = {}
        for endpoint in dataset.endpoints:
            observed = mean(truth[query.request.request_id, endpoint.endpoint_id] for query in calibration) if calibration else None
            methods = {}
            for method in ("local-mean-hoeffding", "local-mean-kl"):
                needed = required_calibration_samples(observed, .65, alpha, method)
                methods[method] = {"required_total_n_at_fixed_mean": needed,
                    "new_independent_requests_needed": max(0, needed - counts["calibration"]) if needed is not None else None,
                    "impossible_at_fixed_mean": observed is not None and observed <= .65,
                    "missing_calibration": observed is None, "k_blocker": needed is not None and needed > K}
            endpoint_plans[endpoint.endpoint_id] = {"fixed_observed_calibration_mean": observed,
                "hypothetical_fixed_mean_only": True, "methods": methods,
                "actual_test_neighborhood_requirements": local_requirements[key, endpoint.endpoint_id]}
        representative = queries[0].request
        entries.append({"traffic_stratum": key, "request_counts": counts,
                        "group_counts": {split: len({query.group_id for query in queries if query.split == split}) for split in counts},
                        "minimum_training_coverage": 8,
                        "new_independent_training_requests_needed_for_coverage": max(0, 8 - counts["train"]),
                        "task_hint": representative.task_hint, "input_length_bucket": representative.input_tokens.bit_length() - 1,
                        "output_length_bucket": representative.max_output_tokens.bit_length() - 1,
                        "full_direct_comparison_family": family, "effective_risk": alpha,
                        "endpoints": endpoint_plans})
    return {"strata": entries,
            "instructions": ["Collect new independent requests in each exact stratum using the frozen feature, evaluator and endpoint versions.",
                "Score each new request on the full endpoint pool. Endpoint copies share one request and do not multiply independent n.",
                "Never duplicate existing calibration scores or treat train, validation or old test labels as new calibration.",
                "Counts assume the same fixed observed mean. Future scores can change the mean, so this is a collection plan, not a forecast.",
                "When required n exceeds k=128, collecting more alone cannot meet this neighborhood requirement. A separately approved pre-calibration k design is required.",
                "If the fixed mean is at or below .65, additional samples at that mean cannot certify the unchanged target.",
                "Satisfy independent/exchangeable request assumptions, train coverage, latency-tail, freshness and evaluator trust gates separately."]}


def synthetic_fixture():
    endpoints = [EndpointSnapshot(endpoint_id=name, provider="fixture", upstream_model=name,
        revision="r1", region="local", context_window=8192, config_hash="h", price_version="p1",
        input_price_per_million=1, output_price_per_million=1) for name in ("a", "b")]
    queries, outcomes = [], []
    for split, count in (("train", 10), ("calibration", 12), ("test", 2)):
        for index in range(count):
            request = RequestContext(request_id=f"{split}-{index}", application_id="app", tenant_policy_id="p",
                input_tokens=128, max_output_tokens=64, query_features=(index/count,), feature_version="fixed")
            queries.append(Query(request=request, text="fixture", split=split, group_id=request.request_id))
            for endpoint in endpoints:
                outcomes.append(Truth(request_id=request.request_id, endpoint_id=endpoint.endpoint_id,
                    quality=.1 if split == "train" else 1., cost=.0001, latency_ms=20,
                    input_tokens=128, output_tokens=32, evaluator_type="deterministic"))
    return BenchmarkDataset(name="contrasting-10-train-12-calibration-2-test", evidence_kind="synthetic",
                            queries=queries, outcomes=outcomes, endpoints=endpoints)


def write_markdown(report):
    lines = ["# Offline quality improvement evidence", "",
             f"Validation selects predictors. Test status is {report['test_status']}.", "",
             "| Endpoint | Selected predictor | Local validation MAE | Half shrink MAE | Cohort mean MAE |",
             "|---|---|---:|---:|---:|"]
    for endpoint, row in report["selection"]["endpoints"].items():
        values = row["validation"]
        lines.append(f"| {endpoint} | {row['selected_deployable']} | {values['local']['mae']:.6f} | {values['half-shrink']['mae']:.6f} | {values['cohort-mean']['mae']:.6f} |")
    lines += ["", "| Intervention | All request quality | Abstention | Certified | Maximum lower bounds |",
              "|---|---:|---:|---:|---|"]
    for name, row in report["comparison"]["interventions"].items():
        summary = row["summary"]
        lines.append(f"| {name} | {summary['quality_all_requests']:.4f} | {summary['abstention_rate']:.1%} | {summary['certificate_coverage']:.1%} | {row['maximum_quality_lower_per_endpoint']} |")
    lines += ["", "The synthetic fixture keeps train quality at .1 and calibration quality at 1. with 12 independent requests.",
              "KL must clear .65 while the original Hoeffding estimator fails. Latency-tail evidence can still prevent certification.",
              "", "Exact-stratum request counts, conditional fixed-mean collection requirements, raw train-only prediction errors, all candidate bounds and routing records are in report.json.",
              "", "Historical benchmark labels and missing per-call latency do not establish production certification.",
              f"Archived output-cap mismatches total {report['comparison']['archived_output_cap_mismatches']}. Served mismatches count as observed constraint violations.",
              "Historical scores are archival replay evidence under unverified generation configuration; they do not predict scores after imposing a different output cap.",
              "No provider calls, downloads or installations occur."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--test-status", choices=("previously-inspected", "first-evaluation"), default="previously-inspected")
    args = parser.parse_args()
    dataset = BenchmarkDataset.load(args.dataset)
    selection = select_predictors(dataset)
    selected = selection["configuration"]["quality_predictors"]
    recorded = compare(dataset, selected)
    synthetic = compare(synthetic_fixture(), {})
    old = synthetic["interventions"]["baseline-hoeffding-local"]
    new = synthetic["interventions"]["kl-local"]
    assert old["summary"]["abstention_rate"] == 1.
    assert new["summary"]["abstention_rate"] == 0.
    assert new["summary"]["quality_all_requests"] == 1.
    assert all(value >= .65 for value in new["maximum_quality_lower_per_endpoint"].values())
    assert all(value < .65 for value in old["maximum_quality_lower_per_endpoint"].values())
    source_paths = ("scripts/run_quality_improvement.py", "src/inference_control/capability/conditional.py",
                    "src/inference_control/benchmarks/routers.py", "src/inference_control/benchmarks/diagnostics.py",
                    "src/inference_control/benchmarks/runner.py", "src/inference_control/planning/planner.py")
    report = {"schema": "slm-offline-quality-improvement-v1", "dataset_hash": dataset.fingerprint,
              "source_hashes": {path: digest((ROOT / path).read_text(encoding="utf-8")) for path in source_paths},
              "split_hash": dataset.split_hash, "split_counts": dict(Counter(query.split for query in dataset.queries)),
              "selection": selection, "comparison": recorded, "test_status": args.test_status,
              "collection_plan": collection_plan(dataset, recorded), "synthetic_behavior_check": synthetic,
              "bootstrap_resamples": 40,
              "claim_gate": {"production_certification": False,
                  "blockers": ["HISTORICAL_BENCHMARK_LABELS", "LIVE_INDEPENDENT_EVIDENCE_REQUIRED"] +
                              (["PREVIOUSLY_INSPECTED_TEST"] if args.test_status == "previously-inspected" else []) +
                              (["ARCHIVED_OUTPUT_CAP_MISMATCH"] if recorded["archived_output_cap_mismatches"] else []) +
                              (["MISSING_PER_CALL_LATENCY"] if recorded["missing_latency"] else [])}}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(canonical(report) + "\n", encoding="utf-8")
    (args.out / "report.md").write_text(write_markdown(report), encoding="utf-8")
    print(canonical({"report": str(args.out / "report.json"), "selected": selected,
                     "recorded": {name: row["summary"]["quality_all_requests"] for name, row in recorded["interventions"].items()},
                     "synthetic_kl_quality": new["summary"]["quality_all_requests"]}))


if __name__ == "__main__":
    main()
