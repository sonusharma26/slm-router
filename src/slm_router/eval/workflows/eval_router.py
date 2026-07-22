"""Workflow: evaluate routing policies against a frozen oracle matrix (no API calls).

Each policy picks a model per item; we look up that model's stored result in the
oracle matrix, so router evaluation is free and instant. Reports regret vs
oracle/random/single-model, ECE, and a cost-quality Pareto plot.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Protocol

from slm_router.eval.metrics import calibration as cal
from slm_router.eval.metrics import pareto as pf
from slm_router.eval.metrics import routing as rt


class MatrixPolicy(Protocol):
    """A policy that picks a model id given the per-model results for one item."""

    name: str

    def pick(self, item_id: str, by_model: dict[str, Any]) -> str: ...


# --- Inline baseline policies (replay against the matrix; no model calls) ----


class _Always:
    def __init__(self, tier_or_id: str, registry: Any | None, by_id: bool = False):
        self.name = f"always:{tier_or_id}"
        self._target = tier_or_id
        self._registry = registry
        self._by_id = by_id

    def pick(self, item_id: str, by_model: dict[str, Any]) -> str:
        if self._by_id and self._target in by_model:
            return self._target
        # pick cheapest model of the requested tier present for this item
        candidates = list(by_model)
        if self._registry is not None:
            tier_match = [
                m for m in candidates
                if _tier(self._registry, m) == self._target
            ]
            if tier_match:
                return min(tier_match, key=lambda m: by_model[m].cost_usd)
        return min(candidates, key=lambda m: by_model[m].cost_usd)


class _Oracle:
    name = "oracle"

    def pick(self, item_id: str, by_model: dict[str, Any]) -> str:
        return max(by_model, key=lambda m: by_model[m].score)


class _Random:
    def __init__(self, seed: int = 42):
        import random

        self.name = "random"
        self._rng = random.Random(seed)

    def pick(self, item_id: str, by_model: dict[str, Any]) -> str:
        return self._rng.choice(sorted(by_model))


def _tier(registry: Any, model_id: str) -> str:
    try:
        return registry.get(model_id).tier
    except Exception:
        return "unknown"


def default_policies(registry: Any) -> list[MatrixPolicy]:
    """The v0 baseline set: cheapest SLM, frontier, oracle, random."""
    return [
        _Always("SLM", registry),
        _Always("frontier", registry),
        _Oracle(),
        _Random(),
    ]


def _replay(policy: MatrixPolicy, matrix: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Run a policy over the matrix; return aggregate score, cost, and per-item picks."""
    scores: list[float] = []
    costs: list[float] = []
    confidences: list[float] = []
    correctness: list[bool] = []
    for item_id, by_model in matrix.items():
        if not by_model:
            continue
        chosen = policy.pick(item_id, by_model)
        t = by_model[chosen]
        scores.append(t.score)
        costs.append(t.cost_usd)
        if t.confidence is not None and t.correct is not None:
            confidences.append(t.confidence)
            correctness.append(bool(t.correct))
    n = max(len(scores), 1)
    return {
        "mean_score": sum(scores) / n,
        "mean_cost": sum(costs) / n,
        "confidences": confidences,
        "correctness": correctness,
    }


def eval_router(
    store: Any,
    registry: Any,
    config: Any,
    policies: list[MatrixPolicy] | None = None,
    dataset: str | None = None,
) -> dict[str, Any]:
    """Evaluate routing policies against the oracle matrix and write a report.

    Returns the report dict; also writes results/router_eval.json and
    results/pareto.png.
    """
    matrix = store.get_oracle_matrix(dataset=dataset)
    if not matrix:
        print("[eval_router] Oracle matrix is empty — run build-oracle first.")
        return {}

    policies = policies or default_policies(registry)

    oracle = rt.oracle_score(matrix)
    rand = rt.random_router_score(matrix, seed=getattr(config, "dataset_seed", 42))
    singles = rt.single_model_scores(matrix)

    cq_points: list[pf.CQPoint] = []
    rows: list[dict[str, Any]] = []
    for policy in policies:
        res = _replay(policy, matrix)
        ece = None
        if res["confidences"]:
            try:
                ece = cal.expected_calibration_error(res["confidences"], res["correctness"])
            except Exception:
                ece = None
        rows.append(
            {
                "policy": policy.name,
                "mean_score": res["mean_score"],
                "mean_cost": res["mean_cost"],
                "regret": rt.regret(oracle, res["mean_score"]),
                "ece": ece,
            }
        )
        cq_points.append(pf.CQPoint(policy.name, res["mean_cost"], res["mean_score"]))

    report = {
        "n_items": len(matrix),
        "oracle_score": oracle,
        "random_score": rand,
        "single_model_scores": singles,
        "policies": rows,
    }

    results_dir = getattr(getattr(config, "paths", None), "results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    out_json = os.path.join(results_dir, "router_eval.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    try:
        frontier = pf.pareto_frontier(cq_points)
        pf.plot_pareto(cq_points, frontier, os.path.join(results_dir, "pareto.png"))
    except Exception as exc:
        print(f"[eval_router] Pareto plot skipped: {exc}")

    print(f"[eval_router] oracle={oracle:.3f} | wrote {out_json}")
    for r in rows:
        ece_s = f"{r['ece']:.3f}" if r["ece"] is not None else "n/a"
        print(
            f"  {r['policy']:<18} score={r['mean_score']:.3f} "
            f"cost=${r['mean_cost']:.5f} regret={r['regret']:.3f} ece={ece_s}"
        )
    return report
