"""Offline quality diagnosis; held-out labels never enter the fitted router."""
from __future__ import annotations

import math
from collections import Counter
from statistics import mean

from inference_control.benchmarks.routers import SLMRouter
from inference_control.capability.conditional import binary_kl, quality_interval


def required_calibration_samples(score_mean, target, effective_risk, method="local-mean-hoeffding"):
    """Fixed-mean requirement for the declared interval; not a forecast."""
    if target == 0:
        return 0
    if score_mean is None or score_mean <= target:
        return None
    if method == "local-mean-hoeffding":
        return max(1, math.ceil(math.log(2/effective_risk)/(2*(score_mean-target)**2)))
    if method != "local-mean-kl":
        raise ValueError("unknown quality bound method")
    needed = max(1, math.ceil(math.log(2/effective_risk)/binary_kl(score_mean, target)))
    if quality_interval(score_mean, needed, effective_risk, method)[0] < target:
        needed += 1
    return needed


def _prediction_errors(rows):
    if not rows:
        return {"requests": 0, "mae": None, "mse": None, "bias": None,
                "constant_training_mean_mae": None}
    return {"requests": len(rows),
            "mae": mean(abs(r["prediction"]-r["observed"]) for r in rows),
            "mse": mean((r["prediction"]-r["observed"])**2 for r in rows),
            "bias": mean(r["prediction"]-r["observed"] for r in rows),
            "constant_training_mean_mae": mean(abs(r["baseline"]-r["observed"]) for r in rows)}


def _extent(rows, field):
    values = [r[field] for r in rows]
    return [min(values), max(values)] if values else None


def diagnose_quality(router, dataset, policy):
    """Use the actual searched family and estimator to explain rejected candidates.

    Calibration error uses a separate train-only predictor. Test errors and scores
    are descriptive evaluator outputs and never determine neighborhoods or policy.
    """
    train = [q for q in dataset.queries if q.split == "train"]
    train_ids = {q.request.request_id for q in train}
    train_rows = [r for r in dataset.outcomes if r.request_id in train_ids]
    predictor = SLMRouter(dataset.endpoints, k=router.map.k,
                          min_samples=router.map.min_samples, at=router.at,
                          quality_method=router.map.quality_method,
                          quality_predictors=dict(router.map.quality_predictors))
    predictor.fit(train, train_rows)
    baseline = {ep.endpoint_id: mean(r.quality for r in train_rows if r.endpoint_id == ep.endpoint_id)
                for ep in dataset.endpoints}
    truth = {(r.request_id, r.endpoint_id): r.quality for r in dataset.outcomes}
    errors = {ep.endpoint_id: {"calibration": [], "test": []} for ep in dataset.endpoints}
    for query in dataset.queries:
        if query.split not in {"calibration", "test"}:
            continue
        for ep in dataset.endpoints:
            estimate = predictor.map.estimate(query.request, ep.endpoint_id, (ep,), at=router.at)
            if estimate is not None:
                errors[ep.endpoint_id][query.split].append({
                    "prediction": estimate.lineage.quality_diagnostics.raw_prediction,
                    "observed": truth[(query.request.request_id, ep.endpoint_id)],
                    "baseline": baseline[ep.endpoint_id]})
    records = []
    rejection_counts = Counter()
    for query in dataset.queries:
        if query.split != "test":
            continue
        plans, rejected, _ = router.planner.generate(query.request, policy)
        rejection_counts.update(reason for r in rejected for reason in r.reason_codes)
        comparisons = max(1, len(plans))
        for plan in plans:
            estimate = router.planner.estimate_plan(query.request, policy, plan, router.at, comparisons)
            hard, statistical = router.planner.reasons(plan, estimate, policy)
            rejection_counts.update(hard+statistical)
            row = {"request_id": query.request.request_id, "target_id": plan.evidence_key,
                   "comparisons": comparisons, "reasons": hard+statistical}
            if estimate is None:
                row["estimate_missing"] = True
                records.append(row)
                continue
            lineage = estimate.lineage
            detail = lineage.quality_diagnostics
            effective_risk = detail.effective_risk
            needed = required_calibration_samples(detail.calibration_mean, policy.minimum_quality, effective_risk, detail.method)
            best_needed = required_calibration_samples(1., policy.minimum_quality, effective_risk, detail.method)
            n = lineage.calibration_size
            ceiling = quality_interval(1., n, effective_risk, detail.method)[0]
            row.update({"estimate_missing": False,
                        "training_size": lineage.sample_size, "calibration_size": n,
                        "quality_mean": estimate.quality.mean, "quality_lower": estimate.quality.lower,
                        "quality_upper": estimate.quality.upper,
                        "bound": detail.model_dump(mode="json"),
                        "lineage_reasons": list(lineage.reasons),
                        "best_case_lower_at_current_n": ceiling,
                        "sample_size_blocks_target_even_with_perfect_scores": ceiling < policy.minimum_quality,
                        "required_samples_at_perfect_mean": best_needed,
                        "required_samples_at_observed_mean": needed,
                        "calibration_mean_below_target": detail.calibration_mean is None or detail.calibration_mean <= policy.minimum_quality,
                        "training_mean_clamp_blocks_target": detail.method == "local-mean-hoeffding" and detail.training_mean is not None and detail.training_mean < policy.minimum_quality,
                        "required_samples_exceed_k": needed is not None and needed > router.map.k})
            records.append(row)
    endpoints = {}
    for ep in dataset.endpoints:
        candidates = [r for r in records if r["target_id"] == ep.endpoint_id and not r["estimate_missing"]]
        endpoints[ep.endpoint_id] = {
            "candidate_estimates": len(candidates), "calibration_size_range": _extent(candidates, "calibration_size"),
            "quality_lower_range": _extent(candidates, "quality_lower"),
            "sample_size_blocked_candidates": sum(r["sample_size_blocks_target_even_with_perfect_scores"] for r in candidates),
            "training_mean_clamp_blocked_candidates": sum(r["training_mean_clamp_blocks_target"] for r in candidates),
            "calibration_prediction_error": _prediction_errors(errors[ep.endpoint_id]["calibration"]),
            "test_prediction_error": _prediction_errors(errors[ep.endpoint_id]["test"])}
    return {"method": router.map.quality_method, "quality_target": policy.minimum_quality,
            "quality_risk": policy.quality_risk, "k": router.map.k,
            "split_counts": dict(Counter(q.split for q in dataset.queries)),
            "rejection_counts": dict(rejection_counts), "endpoints": endpoints, "records": records,
            "limitations": [
                "The declared estimator method is fixed before calibration; target and risk are unchanged.",
                "The bound concerns a local population mean, not individual response prediction coverage.",
                "Required sample counts assume a fixed mean and the current searched family; they are not forecasts.",
                "Prediction errors use train-only neighborhoods; missing neighborhoods are excluded and counts are disclosed.",
                "Held-out errors are descriptive and cannot justify tuning features, neighborhoods or risk on this test set."]}
