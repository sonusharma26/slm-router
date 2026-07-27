"""Replay routing policies against a held-out frozen oracle matrix."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol

from slm_router.eval.metrics import calibration as cal
from slm_router.eval.metrics import pareto as pf
from slm_router.eval.metrics import routing as rt


class MatrixPolicy(Protocol):
    """A policy that picks a model id for one oracle item."""

    name: str

    def pick(self, item_id: str, by_model: dict[str, Any]) -> str: ...


class _Always:
    def __init__(self, tier_or_id: str, registry: Any | None, by_id: bool = False):
        self.name = f"always:{tier_or_id}"
        self._target = tier_or_id
        self._registry = registry
        self._by_id = by_id

    def pick(self, item_id: str, by_model: dict[str, Any]) -> str:
        if self._by_id and self._target in by_model:
            return self._target
        candidates = list(by_model)
        if self._registry is not None:
            tier_match = [
                model_id
                for model_id in candidates
                if _tier(self._registry, model_id) == self._target
            ]
            if tier_match:
                return min(tier_match, key=lambda model_id: by_model[model_id].cost_usd)
        return min(candidates, key=lambda model_id: by_model[model_id].cost_usd)


class _Oracle:
    name = "oracle"

    def pick(self, item_id: str, by_model: dict[str, Any]) -> str:
        return max(by_model, key=lambda model_id: by_model[model_id].score)


class _Random:
    def __init__(self, seed: int = 42):
        import random

        self.name = "random"
        self._rng = random.Random(seed)

    def pick(self, item_id: str, by_model: dict[str, Any]) -> str:
        return self._rng.choice(sorted(by_model))


class _Learned:
    name = "trained:offline_rl"

    def __init__(
        self,
        policy: Any,
        featurizer: Any,
        item_texts: dict[str, str],
        candidates: list[Any],
    ) -> None:
        self._policy = policy
        self._featurizer = featurizer
        self._item_texts = item_texts
        self._candidates = candidates

    def pick(self, item_id: str, by_model: dict[str, Any]) -> str:
        from slm_router.types import Query

        context = self._featurizer.vectorize(
            Query(id=item_id, text=self._item_texts[item_id])
        )
        return self._policy.select(context, self._candidates).chosen_model


def _tier(registry: Any, model_id: str) -> str:
    try:
        return registry.get(model_id).tier
    except Exception:
        return "unknown"


def default_policies(
    registry: Any,
    candidate_models: list[str] | None = None,
) -> list[MatrixPolicy]:
    """Return baselines only for tiers represented by active candidates."""
    candidates = candidate_models or [spec.id for spec in registry.all()]
    tiers = {
        _tier(registry, model_id)
        for model_id in candidates
    }
    ordered_tiers = [
        tier for tier in ("SLM", "small", "frontier") if tier in tiers
    ]
    return [
        *(_Always(tier, registry) for tier in ordered_tiers),
        _Oracle(),
        _Random(),
    ]


def _replay(
    policy: MatrixPolicy,
    matrix: dict[str, dict[str, Any]],
    registry: Any,
    slm_quality_tolerance: float,
) -> dict[str, Any]:
    """Replay one policy and aggregate quality, cost, latency, and retention."""
    scores: list[float] = []
    costs: list[float] = []
    latencies: list[float] = []
    confidences: list[float] = []
    correctness: list[bool] = []
    slm_picks = 0
    retained_slm_picks = 0

    for item_id, by_model in matrix.items():
        chosen = policy.pick(item_id, by_model)
        trace = by_model[chosen]
        scores.append(trace.score)
        costs.append(trace.cost_usd)
        latencies.append(trace.latency_ms)
        if _tier(registry, chosen) == "SLM":
            slm_picks += 1
            oracle_item_score = max(item.score for item in by_model.values())
            if oracle_item_score - trace.score <= slm_quality_tolerance:
                retained_slm_picks += 1
        if trace.confidence is not None and trace.correct is not None:
            confidences.append(trace.confidence)
            correctness.append(bool(trace.correct))

    n = max(len(scores), 1)
    return {
        "mean_score": sum(scores) / n,
        "mean_cost": sum(costs) / n,
        "mean_latency_ms": sum(latencies) / n,
        "slm_pick_rate": slm_picks / n,
        "slm_retention_rate": retained_slm_picks / n,
        "confidences": confidences,
        "correctness": correctness,
    }


def _load_item_texts(config: Any) -> dict[str, str]:
    from slm_router.eval.datasets import load_dataset_items

    texts: dict[str, str] = {}
    for dataset in config.datasets:
        try:
            items = load_dataset_items(
                dataset,
                split="test",
                limit=config.dataset_limit,
                seed=config.dataset_seed,
            )
            for item in items:
                texts[item.item_id] = item.query
        except Exception as exc:
            print(f"[eval_router] Could not reload dataset '{dataset}': {exc}")
    return texts


def _load_learned_policy(
    registry: Any,
    config: Any,
    item_texts: dict[str, str],
) -> MatrixPolicy:
    from slm_router.ml_core.features import DifficultyFeaturizer, Embedder
    from slm_router.ml_core.routing import get_policy
    from slm_router.ml_core.routing.trace_adapter import SPLIT_VERSION

    candidate_models = list(config.candidate_models)
    policy_path = Path(config.paths.models_dir) / "policy.joblib"
    policy = get_policy(
        "offline_rl",
        n_arms=len(candidate_models),
        dim=9,
        lam=config.reward.lambda_cost,
        beta=config.reward.beta_latency,
        correctness_weight=config.reward.correctness_weight,
    )
    policy.load(policy_path)
    if getattr(policy, "candidate_models", None) != candidate_models:
        raise ValueError("trained policy candidate order does not match config")
    if (
        getattr(policy, "split_seed", None) != config.dataset_seed
        or getattr(policy, "holdout_fraction", None)
        != config.router_holdout_fraction
        or getattr(policy, "split_version", None) != SPLIT_VERSION
    ):
        raise ValueError("trained policy does not use the configured holdout split")
    candidates = [registry.get(model_id) for model_id in candidate_models]
    return _Learned(
        policy,
        DifficultyFeaturizer(Embedder()),
        item_texts,
        candidates,
    )


def _heldout_matrix(
    store: Any,
    config: Any,
    dataset: str | None,
) -> dict[str, dict[str, Any]]:
    from slm_router.eval.workflows.build_oracle import config_hash
    from slm_router.ml_core.routing.trace_adapter import select_holdout_items

    traces = store.query(run_kind="oracle", config_hash=config_hash(config))
    candidate_models = set(config.candidate_models)
    matrix: dict[str, dict[str, Any]] = {}
    for trace in traces:
        if trace.error or trace.model not in candidate_models:
            continue
        if dataset is not None and trace.dataset != dataset:
            continue
        matrix.setdefault(trace.item_id, {})[trace.model] = trace

    complete = {
        item_id: by_model
        for item_id, by_model in matrix.items()
        if set(by_model) == candidate_models
    }
    holdout_items = select_holdout_items(
        set(complete),
        config.dataset_seed,
        config.router_holdout_fraction,
    )
    return {
        item_id: by_model
        for item_id, by_model in complete.items()
        if item_id in holdout_items
    }


def _evidence_assessment(
    matrix: dict[str, dict[str, Any]],
    minimum_items: int,
) -> dict[str, Any]:
    quality_disagreements = sum(
        1
        for by_model in matrix.values()
        if max(trace.score for trace in by_model.values())
        - min(trace.score for trace in by_model.values())
        > 1e-9
    )
    cost_disagreements = sum(
        1
        for by_model in matrix.values()
        if max(trace.cost_usd for trace in by_model.values())
        - min(trace.cost_usd for trace in by_model.values())
        > 1e-12
    )
    positive_latency_items = sum(
        1
        for by_model in matrix.values()
        if all(trace.latency_ms > 0 for trace in by_model.values())
    )

    reasons: list[str] = []
    if len(matrix) < minimum_items:
        reasons.append(
            f"holdout has {len(matrix)} items; minimum is {minimum_items}"
        )
    if quality_disagreements == 0:
        reasons.append("no held-out item has model-quality disagreement")
    if cost_disagreements == 0:
        reasons.append("no held-out item has model-cost disagreement")
    if positive_latency_items < len(matrix):
        reasons.append(
            "one or more held-out items lack positive latency for every model"
        )

    return {
        "status": "insufficient_evidence" if reasons else "sufficient",
        "reasons": reasons,
        "quality_disagreement_items": quality_disagreements,
        "cost_disagreement_items": cost_disagreements,
        "complete_positive_latency_items": positive_latency_items,
    }


def eval_router(
    store: Any,
    registry: Any,
    config: Any,
    policies: list[MatrixPolicy] | None = None,
    dataset: str | None = None,
) -> dict[str, Any]:
    """Evaluate all policies on the current config's held-out oracle items."""
    matrix = _heldout_matrix(store, config, dataset)
    if not matrix:
        print(
            "[eval_router] Held-out oracle matrix is empty or incomplete "
            "for the current config."
        )
        return {}

    item_texts = _load_item_texts(config)
    missing_text = set(matrix) - set(item_texts)
    if missing_text:
        print(
            f"[eval_router] Dropping {len(missing_text)} held-out item(s) "
            "whose query text could not be reconstructed."
        )
        matrix = {
            item_id: by_model
            for item_id, by_model in matrix.items()
            if item_id in item_texts
        }
    if not matrix:
        print("[eval_router] No evaluable held-out items remain.")
        return {}

    if policies is None:
        policies = default_policies(registry, list(config.candidate_models))
        try:
            policies.append(_load_learned_policy(registry, config, item_texts))
        except Exception as exc:
            print(f"[eval_router] Trained policy unavailable: {exc}")

    oracle = rt.oracle_score(matrix)
    rand = rt.random_router_score(matrix, seed=config.dataset_seed)
    singles = rt.single_model_scores(matrix)

    cq_points: list[pf.CQPoint] = []
    rows: list[dict[str, Any]] = []
    for policy in policies:
        result = _replay(
            policy,
            matrix,
            registry,
            config.slm_quality_tolerance,
        )
        ece = None
        if result["confidences"]:
            try:
                ece = cal.expected_calibration_error(
                    result["confidences"],
                    result["correctness"],
                )
            except Exception:
                ece = None
        rows.append(
            {
                "policy": policy.name,
                "mean_score": result["mean_score"],
                "mean_cost": result["mean_cost"],
                "mean_latency_ms": result["mean_latency_ms"],
                "regret": rt.regret(oracle, result["mean_score"]),
                "slm_pick_rate": result["slm_pick_rate"],
                "slm_retention_rate": result["slm_retention_rate"],
                "ece": ece,
            }
        )
        cq_points.append(
            pf.CQPoint(policy.name, result["mean_cost"], result["mean_score"])
        )

    evidence = _evidence_assessment(matrix, config.minimum_holdout_items)
    report = {
        "n_items": len(matrix),
        "evaluation_split": "holdout",
        "split_seed": config.dataset_seed,
        "holdout_fraction": config.router_holdout_fraction,
        "slm_quality_tolerance": config.slm_quality_tolerance,
        "evidence": evidence,
        "oracle_score": oracle,
        "random_score": rand,
        "single_model_scores": singles,
        "policies": rows,
    }

    results_dir = config.paths.results_dir
    os.makedirs(results_dir, exist_ok=True)
    out_json = os.path.join(results_dir, "router_eval.json")
    with open(out_json, "w", encoding="utf-8") as output:
        json.dump(report, output, indent=2)

    try:
        frontier = pf.pareto_frontier(cq_points)
        pf.plot_pareto(
            cq_points,
            frontier,
            os.path.join(results_dir, "pareto.png"),
        )
    except Exception as exc:
        print(f"[eval_router] Pareto plot skipped: {exc}")

    print(f"[eval_router] holdout={len(matrix)} oracle={oracle:.3f} | wrote {out_json}")
    if evidence["status"] != "sufficient":
        print("[eval_router] INSUFFICIENT EVIDENCE:")
        for reason in evidence["reasons"]:
            print(f"  - {reason}")
    for row in rows:
        ece_text = f"{row['ece']:.3f}" if row["ece"] is not None else "n/a"
        print(
            f"  {row['policy']:<20} score={row['mean_score']:.3f} "
            f"cost=${row['mean_cost']:.5f} "
            f"latency={row['mean_latency_ms']:.0f}ms "
            f"regret={row['regret']:.3f} "
            f"retention={row['slm_retention_rate']:.3f} "
            f"ece={ece_text}"
        )
    return report
