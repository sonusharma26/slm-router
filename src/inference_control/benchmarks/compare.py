"""Multi-split router comparison on full-information matrices (cost-quality frontiers).

Every router is fitted on training data (the pooled router additionally uses its
calibration split, and validation for model selection); test outcomes only score
decisions afterwards. Summaries per router:

* AIQ (RouterBench, Hu et al. 2024): area under the upper concave hull of
  (normalized cost, quality). Cost is normalized linearly between the cheapest and the
  most expensive single-model mean cost on the same split; mixing two operating points
  is realizable, which is why the concave hull is used.
* Quality at the best single model's cost and the cost needed to match its quality,
  both read from the router's hull.
* For the pooled router, the empirical check of its guarantee: the mean test quality
  of certified requests versus each target.

* With answers, the ``slm-cascade`` family (plain price-penalty routing, answer-agreement cascades
  and majority votes) is calibrated per target; each operating point reports which kind was
  selected, its voters/quorum, and the endpoints called per request.

Synthetic fixtures exercise code paths only; claims need external datasets.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

from inference_control.benchmarks.data import BenchmarkDataset, grouped_split
from inference_control.benchmarks.routers import Baseline, Selection, SLMRouter
from inference_control.benchmarks.runner import default_policy, environment, evaluate_router, split_data
from inference_control.contracts import InfeasibleBehavior
from inference_control.policies.validator import eligibility_reasons
from inference_control.util import canonical

DEFAULT_TARGETS = (.5, .6, .7, .75, .8, .85, .9, .95, 1.)
KNN_TARGETS = (.3, .5, .6, .7, .8, .9, .95)
THRESHOLDS = (.3, .5, .6, .7, .8, .9, .95)
LOCAL_TARGETS = (.5, .7, .9)
DEFAULT_VOTERS = (3, 5)  # majority-vote sizes added to the cascade family; () disables votes


class FixedRouter:
    """Always the same model: a single-model operating point."""

    def __init__(self, endpoint_id: str):
        self.endpoint_id = endpoint_id
        self.name = f"model:{endpoint_id}"

    def choose(self, query, endpoints, policy):
        endpoint = next((e for e in endpoints if e.endpoint_id == self.endpoint_id), None)
        if endpoint is None or eligibility_reasons(endpoint, query.request, policy):
            return Selection(None, reason="ineligible")
        return Selection(self.endpoint_id)


class OracleRouter:
    """Reference only: reads test outcomes (best quality, then lowest cost)."""
    name = "oracle"

    def __init__(self, dataset: BenchmarkDataset):
        self.lookup = {(r.request_id, r.endpoint_id): r for r in dataset.outcomes}

    def choose(self, query, endpoints, policy):
        rows = [self.lookup[(query.request.request_id, e.endpoint_id)] for e in endpoints
                if not eligibility_reasons(e, query.request, policy)]
        if not rows:
            return Selection(None, reason="ineligible")
        best = min(rows, key=lambda r: (-r.quality, r.cost, r.endpoint_id))
        return Selection(best.endpoint_id)


def resplit(dataset: BenchmarkDataset, seed: int) -> BenchmarkDataset:
    """Reassign grouped train/calibration/validation/test splits under a new seed."""
    queries = [q.model_copy(update={"split": grouped_split(q.group_id, seed)}) for q in dataset.queries]
    return dataset.model_copy(update={"queries": queries})


def summarize_rows(rows, *, slice_of=None) -> dict:
    costs = [r["cost"] for r in rows]
    certified = [r for r in rows if (r.get("metadata") or {}).get("quality_certified")]
    result = {"requests": len(rows), "quality": mean(r["quality"] for r in rows),
              "cost": mean(costs) if all(c is not None for c in costs) else None,
              "abstention": mean(float(r["abstained"]) for r in rows),
              "certified_share": len(certified) / len(rows),
              "certified_quality": mean(r["quality"] for r in certified) if certified else None}
    if slice_of:
        groups = {}
        for r in rows:
            groups.setdefault(slice_of.get(r["request_id"], "default"), []).append(r["quality"])
        result["slice_quality"] = {k: mean(v) for k, v in sorted(groups.items())}
        result["average_slice_quality"] = mean(mean(v) for v in groups.values())
    return result


def upper_hull(points):
    """Upper concave, non-decreasing envelope of (cost, quality) operating points."""
    pts = sorted({(float(c), float(q)) for c, q in points if c is not None and q is not None},
                 key=lambda p: (p[0], -p[1]))
    pareto = []
    for c, q in pts:
        if not pareto or q > pareto[-1][1] + 1e-15:
            if pareto and pareto[-1][0] == c:
                continue
            pareto.append((c, q))
    hull = []
    for p in pareto:
        while len(hull) >= 2:
            (ox, oy), (ax, ay) = hull[-2], hull[-1]
            if (ax - ox) * (p[1] - oy) - (ay - oy) * (p[0] - ox) >= 0:  # middle point on/below chord
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def hull_quality(hull, cost):
    if not hull or cost is None or cost < hull[0][0]:
        return None
    for (x1, y1), (x2, y2) in pairwise(hull):
        if x1 <= cost <= x2:
            return y1 + (y2 - y1) * (cost - x1) / (x2 - x1) if x2 > x1 else max(y1, y2)
    return hull[-1][1]


def hull_cost(hull, quality):
    if not hull or quality is None or quality > hull[-1][1] + 1e-12:
        return None
    if quality <= hull[0][1]:
        return hull[0][0]
    for (x1, y1), (x2, y2) in pairwise(hull):
        if y1 <= quality <= y2:
            return x1 + (x2 - x1) * (quality - y1) / (y2 - y1) if y2 > y1 else x1
    return hull[-1][0]


def aiq(hull, low, high) -> float | None:
    """Area under the hull over cost normalized to [0, 1]; zero left of the first point."""
    if not hull or high is None or low is None or high <= low:
        return None
    def x(c):
        return min(1., max(0., (c - low) / (high - low)))
    area, previous = 0., None
    for c, q in hull:
        if previous is not None:
            (pc, pq) = previous
            area += (x(c) - x(pc)) * (pq + q) / 2
        previous = (c, q)
    area += (1. - x(hull[-1][0])) * hull[-1][1]
    return area


@dataclass
class _Run:
    name: str
    points: list


def _point(label, rows, slice_of=None, **settings):
    summary = summarize_rows(rows, slice_of=slice_of)
    summary.pop("slice_quality", None)
    return {"label": label, **settings, **summary}


def cascade_runs(slm, dataset, policy, targets, answers, *, at, quality_risk, minimum_requests, slice_of=None,
                 voters=DEFAULT_VOTERS, vote_options=None):
    """Calibrate and evaluate answer-agreement cascades and majority votes on a full-information split.

    Uses the fitted pooled router's forecasts. The family mixes plain price-penalty routing,
    agreement cascades and (for ``voters`` sizes that fit the endpoint count) majority votes;
    ``voters=()`` gives the agreement-only family. Configurations are ordered on training
    references (leave-one-out forecasts), calibrated on the calibration split per target and
    scored on test. Every endpoint is assumed allowed (the harness policy sets no limits).
    ``vote_options`` forwards ``vote_*`` keywords to ``cascade.default_configs``.
    """
    from inference_control.routing import cascade
    from inference_control.routing.calibration import utility_grid
    lookup = {(r.request_id, r.endpoint_id): r for r in dataset.outcomes}
    feature_version = dataset.queries[0].request.feature_version
    evidence = slm.planner._pooled_evidence(policy(targets[0]), at, feature_version)
    ids = evidence.endpoint_ids
    train_cost = np.nanmean(np.where(np.isnan(evidence.train_cost), np.nan, evidence.train_cost), axis=0) \
        if len(evidence.train_requests) else np.zeros(len(ids))
    def matrices(requests, forecasts=None):
        F, C = forecasts if forecasts is not None else evidence.forecast(requests)
        C = np.where(np.isnan(C), np.broadcast_to(np.nan_to_num(train_cost), C.shape), C)
        Q = np.array([[lookup[(r.request_id, e)].quality for e in ids] for r in requests])
        R = np.array([[lookup[(r.request_id, e)].cost for e in ids] for r in requests])
        A = cascade.agreement_matrix([[answers.get((r.request_id, e)) for e in ids] for r in requests])
        return F, C, Q, R, A
    reference = matrices(evidence.train_requests, evidence.training_forecasts())
    grid = utility_grid(reference[0], reference[1], steps=30)
    sizes = tuple(v for v in voters if 2 <= v <= len(ids))
    family = cascade.default_configs(grid, voters=sizes, **(vote_options or {}))
    configs = cascade.order_configs(family, *reference)
    calibration_set = matrices(evidence.calibration_requests)
    test_requests = [q.request for q in dataset.queries if q.split == "test"]
    test = matrices(test_requests)
    points, checks = [], []
    for target in targets:
        result = cascade.calibrate(configs, *calibration_set, target=target, risk=quality_risk,
                                   minimum_requests=minimum_requests)
        config = result.selected or configs[0]
        realized = cascade.outcome(*test, config)
        q, c = realized.quality, realized.cost
        diagnostics = cascade.diagnostics(realized, config)
        rows = [{"request_id": r.request_id, "quality": float(qi), "cost": float(ci), "abstained": False,
                 "metadata": {"quality_certified": result.selected is not None}}
                for r, qi, ci in zip(test_requests, q, c)]
        point = _point(f"target={target}", rows, slice_of, target=target, calibrated=result.selected is not None,
                       **diagnostics)
        points.append(point)
        checks.append({"router": "slm-cascade", "target": target, "threshold": None, "price_penalty": config.lam,
                       "accept": config.accept, "kind": config.kind, "voters": config.voters, "quorum": config.quorum,
                       "sequential": config.sequential, "mean_calls": diagnostics["mean_calls"],
                       "calibration_certified": result.calibration_requests if result.selected else 0,
                       "calibration_requests": result.calibration_requests, "calibration_mean": result.certified_mean,
                       "test_certified_share": point["certified_share"], "test_certified_quality": point["certified_quality"],
                       "holds": point["certified_quality"] is None or point["certified_quality"] >= target})
    return points, checks


def compare_split(dataset: BenchmarkDataset, *, seed: int, targets=DEFAULT_TARGETS, knn_targets=KNN_TARGETS,
                  thresholds=THRESHOLDS, local_targets=LOCAL_TARGETS, quality_risk: float = .05,
                  min_samples: int = 8, slice_of=None, at=None, progress=None,
                  policy_overrides: dict | None = None, answers: dict | None = None,
                  cascade_voters=DEFAULT_VOTERS, cascade_vote_options: dict | None = None) -> dict:
    """Evaluate all routers on one fixed split; returns operating points and frontier summaries."""
    at = at or datetime(2026, 1, 1, tzinfo=UTC)
    train, train_rows = split_data(dataset, {"train"})
    fit, fit_rows = split_data(dataset, {"train", "calibration"})
    validation, validation_rows = split_data(dataset, {"validation"})
    if not train or not any(q.split == "test" for q in dataset.queries):
        raise ValueError("comparison needs nonempty train and test splits")
    # Baselines ignore latency and spend limits, so by default they are reported, not enforced,
    # for every router; pass policy_overrides to compare under real deadlines or budgets.
    base = default_policy(minimum_quality=.7, missing_latency=True).model_copy(
        update={"quality_risk": quality_risk, "max_expected_spend": 1e9, "max_absolute_spend": 1e9,
                **(policy_overrides or {})})
    abstain = base.model_copy(update={"infeasible_behavior": InfeasibleBehavior.ABSTAIN, "version": "local-abstain"})
    def policy(target, template=base):
        return template.model_copy(update={"minimum_quality": target, "version": f"{template.version}-{target}"})
    def note(message):
        if progress:
            progress(f"seed {seed}: {message}")
    runs: dict[str, list] = {}
    for endpoint in dataset.endpoints:
        rows = evaluate_router(FixedRouter(endpoint.endpoint_id), dataset, base)
        runs[f"model:{endpoint.endpoint_id}"] = [_point("always", rows, slice_of)]
    for mode in ("cheapest", "random", "best_single"):
        router = Baseline(mode, seed=seed)
        router.fit(train, train_rows)
        runs[mode] = [_point(mode, evaluate_router(router, dataset, base), slice_of)]
    note("single-model and simple baselines done")
    knn = Baseline("knn", seed=seed)
    knn.fit(train, train_rows)
    runs["knn"] = [_point(f"min_quality={t}", evaluate_router(knn, dataset, policy(t)), slice_of, min_quality=t) for t in knn_targets]
    threshold = Baseline("threshold", seed=seed)
    threshold.fit(train, train_rows)
    runs["threshold"] = []
    for value in thresholds:
        threshold.threshold = value
        runs["threshold"].append(_point(f"threshold={value}", evaluate_router(threshold, dataset, base), slice_of, threshold=value))
    note("kNN and threshold sweeps done")
    slm = SLMRouter(dataset.endpoints, min_samples=min_samples, at=at)
    slm.fit(fit, fit_rows)
    tuning = slm.tune(validation, validation_rows, base) if validation else None
    runs["slm-router"], guarantee = [], []
    for target in targets:
        rows = evaluate_router(slm, dataset, policy(target))
        point = _point(f"target={target}", rows, slice_of, target=target)
        runs["slm-router"].append(point)
        feature_version = fit[0].request.feature_version
        _, calibration = slm.planner.pooled_calibration(policy(target), at=at, feature_version=feature_version)
        guarantee.append({"router": "slm-router", "target": target, "threshold": calibration.threshold,
                          "price_penalty": getattr(calibration, "parameter", None),
                          "calibration_certified": calibration.certified_count,
                          "calibration_requests": calibration.calibration_requests,
                          "calibration_mean": calibration.certified_mean,
                          "test_certified_share": point["certified_share"],
                          "test_certified_quality": point["certified_quality"],
                          "holds": point["certified_quality"] is None or point["certified_quality"] >= target})
    note("pooled router sweep done")
    if answers:
        runs["slm-cascade"], cascade_checks = cascade_runs(
            slm, dataset, policy, tuple(targets), answers, at=at, quality_risk=quality_risk,
            minimum_requests=min_samples, slice_of=slice_of, voters=tuple(cascade_voters),
            vote_options=cascade_vote_options)
        guarantee.extend(cascade_checks)
        note("cascade sweep done")
    if local_targets:
        local = SLMRouter(dataset.endpoints, min_samples=min_samples, at=at, quality_method="local-mean-kl",
                          name="slm-router-local (previous)")
        local.fit(fit, fit_rows)
        runs[local.name] = [_point(f"target={t}", evaluate_router(local, dataset, policy(t, abstain)), slice_of, target=t)
                            for t in local_targets]
        note("previous local router done")
    oracle_rows = evaluate_router(OracleRouter(dataset), dataset, base)
    runs["oracle (reference)"] = [_point("oracle", oracle_rows, slice_of)]
    singles = [runs[f"model:{e.endpoint_id}"][0] for e in dataset.endpoints]
    costs = [p["cost"] for p in singles if p["cost"] is not None]
    low, high = (min(costs), max(costs)) if costs else (None, None)
    best = runs["best_single"][0]
    summary = {}
    for name, points in runs.items():
        hull = upper_hull((p["cost"], p["quality"]) for p in points)
        summary[name] = {
            "aiq": aiq(hull, low, high),
            "max_quality": max(p["quality"] for p in points),
            "max_average_slice_quality": max((p["average_slice_quality"] for p in points
                                              if p.get("average_slice_quality") is not None), default=None),
            "quality_at_best_single_cost": hull_quality(hull, best["cost"]),
            "cost_to_match_best_single": hull_cost(hull, best["quality"]),
            "hull": hull}
        match = summary[name]["cost_to_match_best_single"]
        summary[name]["cost_ratio_to_match_best_single"] = (match / best["cost"]) if match is not None and best["cost"] else None
    return {"seed": seed, "split_counts": {s: sum(q.split == s for q in dataset.queries)
                                           for s in ("train", "calibration", "validation", "test")},
            "cost_range": [low, high], "best_single": best, "runs": runs, "summary": summary,
            "pooled_guarantee": guarantee, "tuning": tuning}


def run_compare(dataset: BenchmarkDataset | None = None, *, seeds=(42, 43, 44), datasets=None,
                slice_of=None, progress=None, **options) -> dict:
    # ``slice_of`` maps request_id to a reporting group (e.g. source dataset); never a router input.
    """Compare routers across several splits.

    Either pass ``dataset`` (resplit per seed with the grouped split rule) or ``datasets``,
    a mapping of seed to an already split dataset (e.g. an external benchmark's own split).
    """
    if (dataset is None) == (datasets is None):
        raise ValueError("pass exactly one of dataset and datasets")
    splits = datasets or {seed: resplit(dataset, seed) for seed in seeds}
    reference = dataset or next(iter(splits.values()))
    results = [compare_split(data, seed=seed, slice_of=slice_of, progress=progress, **options)
               for seed, data in splits.items()]
    names = list(results[0]["summary"])
    def stat(name, field):
        values = [r["summary"][name][field] for r in results if r["summary"][name][field] is not None]
        return {"mean": mean(values), "sd": pstdev(values) if len(values) > 1 else 0., "n": len(values)} if values else None
    aggregate = {name: {field: stat(name, field) for field in
                        ("aiq", "max_quality", "max_average_slice_quality", "quality_at_best_single_cost",
                         "cost_ratio_to_match_best_single")}
                 for name in names}
    checks = [dict(g, seed=r["seed"]) for r in results for g in r["pooled_guarantee"]]
    certified = [c for c in checks if c["test_certified_quality"] is not None]
    blockers = []
    if reference.evidence_kind == "synthetic":
        blockers.append("SYNTHETIC_DATA_IS_NOT_EXTERNAL_BENCHMARK_EVIDENCE")
    blockers.append("EXTERNAL_ROUTERS_NOT_RUN_IN_THIS_HARNESS")
    if len(splits) < 3:
        blockers.append("FEWER_THAN_THREE_SPLITS")
    return {"schema": "slm-compare-v1", "generated_at": datetime.now(UTC).isoformat(),
            "environment": environment(), "dataset": reference.name, "evidence_kind": reference.evidence_kind,
            "models": [e.endpoint_id for e in reference.endpoints], "seeds": list(splits),
            "aggregate": aggregate,
            "pooled_guarantee": {"checks": checks, "certified_checks": len(certified),
                                 "violations": [c for c in certified if not c["holds"]]},
            "splits": results, "claim_gate": {"passed": False, "blockers": blockers},
            "notes": ["AIQ normalizes cost linearly between the cheapest and most expensive single model on each split.",
                      "Quality and cost read from each router's concave hull assume operating points can be mixed.",
                      "The oracle reads test outcomes and is a reference, not a router.",
                      ("The pooled guarantee is a 1-quality_risk statement per target and split; occasional "
                       "violations at about that rate are consistent with it.")]}


def _fmt(stat, scale=1., suffix="", digits=3):
    if not stat:
        return "n/a"
    if stat["n"] > 1:
        return f"{stat['mean']*scale:.{digits}f}{suffix} ± {stat['sd']*scale:.{digits}f}"
    return f"{stat['mean']*scale:.{digits}f}{suffix}"


def write_compare_report(report: dict, out) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(canonical(report) + "\n", encoding="utf-8")
    seeds = ", ".join(str(s) for s in report["seeds"])
    lines = ["# Router comparison", "", f"Dataset: **{report['dataset']}** ({report['evidence_kind']}); splits: {seeds}.", ""]
    if report["evidence_kind"] == "synthetic":
        lines += ["Synthetic data exercises code paths; it is not evidence about real models.", ""]
    sliced = any(row.get("max_average_slice_quality") for row in report["aggregate"].values())
    lines += ["| Router | AIQ | Quality at best-single cost | Cost to match best single | Best quality |"
              + (" Best per-dataset average |" if sliced else ""),
              "|---|---:|---:|---:|---:|" + ("---:|" if sliced else "")]
    order = sorted(report["aggregate"], key=lambda n: (n.startswith("model:"), -(report["aggregate"][n]["aiq"] or {"mean": -1})["mean"]))
    for name in order:
        row = report["aggregate"][name]
        lines.append(f"| {name} | {_fmt(row['aiq'])} | {_fmt(row['quality_at_best_single_cost'])} | "
                     f"{_fmt(row['cost_ratio_to_match_best_single'], 100, '%', 1)} | {_fmt(row['max_quality'])} |"
                     + (f" {_fmt(row['max_average_slice_quality'])} |" if sliced else ""))
    checks = report["pooled_guarantee"]
    lines += ["", "## Pooled quality guarantee", "",
              (f"{checks['certified_checks']} target/split checks had certified test traffic; "
               f"{len(checks['violations'])} fell below target."), "",
              "| Router | Split | Target | Certified share (test) | Certified mean (test) | Calibration mean | Holds |",
              "|---|---|---:|---:|---:|---:|---|"]
    for c in checks["checks"]:
        mean_test = f"{c['test_certified_quality']:.3f}" if c["test_certified_quality"] is not None else "—"
        mean_cal = f"{c['calibration_mean']:.3f}" if c["calibration_mean"] is not None else "—"
        lines.append(f"| {c.get('router', 'slm-router')} | {c['seed']} | {c['target']} | {c['test_certified_share']:.0%} | "
                     f"{mean_test} | {mean_cal} | {'yes' if c['holds'] else 'NO'} |")
    cascades = [(split["seed"], point) for split in report["splits"] for point in split["runs"].get("slm-cascade", [])]
    if cascades:
        lines += ["", "## Cascade operating points", "",
                  ("Kind: plain price-penalty routing, answer-agreement cascade, or majority vote (voters/quorum). "
                   "Rates are shares of test requests; calls are endpoints invoked per request."), "",
                  "| Split | Target | Kind | Voters/quorum | Penalty | Accept | Calls | Check | Vote | Escalated | Test quality | Test cost | Calibrated |",
                  "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for seed, point in cascades:
            voting = f"{point['voters']}/{point['quorum']}{' seq' if point.get('sequential') else ''}" if point.get("voters") else "—"
            cost = f"{point['cost']:.5f}" if point.get("cost") is not None else "—"
            lines.append(f"| {seed} | {point['target']} | {point.get('kind', 'agree')} | {voting} | {point['lam']:.4g} | "
                         f"{point['accept']:g} | {point.get('mean_calls', float('nan')):.2f} | {point['check_rate']:.0%} | "
                         f"{point.get('vote_rate', 0.):.0%} | {point['escalation_rate']:.0%} | {point['quality']:.3f} | {cost} | "
                         f"{'yes' if point.get('calibrated', True) else 'no'} |")
    lines += ["", "## Claim gate", "", *[f"- {b}" for b in report["claim_gate"]["blockers"]], "",
              *[f"- {n}" for n in report["notes"]]]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out / "report.json"
