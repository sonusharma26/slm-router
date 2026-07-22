"""Routing quality metrics: oracle, random baseline, single-model, regret."""
from __future__ import annotations

import random
from typing import Any, TYPE_CHECKING

# Defensive import of RunTrace
_RunTrace: Any = None
try:
    from slm_router.trace import RunTrace as _RunTrace  # type: ignore[import]
except ImportError:
    pass

if _RunTrace is None:
    try:
        from ml_core.routing import RunTrace as _RunTrace  # type: ignore[import]
    except ImportError:
        pass

# Provide a fallback stub so type annotations work even without the real import
if _RunTrace is None:
    from pydantic import BaseModel

    class RunTrace(BaseModel):  # type: ignore[no-redef]
        """Minimal stub for RunTrace when the real module is unavailable."""
        item_id: str = ""
        model_id: str = ""
        score: float = 0.0
        run_kind: str = "unknown"
        route_decision: str | None = None

        class model_config:
            extra = "allow"
else:
    RunTrace = _RunTrace  # type: ignore[assignment,misc]

# Type alias: item_id -> model_id -> RunTrace
OracleMatrix = dict[str, dict[str, Any]]


def _get_score(trace: Any) -> float:
    """Safely extract the numeric score from a RunTrace (or dict)."""
    if isinstance(trace, dict):
        return float(trace.get("score", 0.0))
    return float(getattr(trace, "score", 0.0))


def oracle_score(
    matrix: OracleMatrix,
    objective: str = "accuracy",
) -> float:
    """Compute the oracle score: best possible average score.

    For each item, the oracle picks the model with the highest score.

    Args:
        matrix: Nested dict {item_id -> {model_id -> RunTrace}}.
        objective: Metric to optimise (currently only 'accuracy'/score supported).

    Returns:
        Mean of per-item best scores. 0.0 if matrix is empty.
    """
    if not matrix:
        return 0.0

    per_item_best: list[float] = []
    for _item_id, model_traces in matrix.items():
        if not model_traces:
            per_item_best.append(0.0)
            continue
        best = max(_get_score(trace) for trace in model_traces.values())
        per_item_best.append(best)

    return sum(per_item_best) / len(per_item_best)


def random_router_score(
    matrix: OracleMatrix,
    seed: int = 42,
) -> float:
    """Compute the average score of a random routing policy.

    For each item, a random model is selected (seeded for reproducibility).

    Args:
        matrix: Nested dict {item_id -> {model_id -> RunTrace}}.
        seed: Random seed.

    Returns:
        Mean score of random picks. 0.0 if matrix is empty.
    """
    if not matrix:
        return 0.0

    rng = random.Random(seed)
    scores: list[float] = []

    for _item_id, model_traces in matrix.items():
        if not model_traces:
            scores.append(0.0)
            continue
        model_ids = sorted(model_traces.keys())  # sort for reproducibility
        chosen = rng.choice(model_ids)
        scores.append(_get_score(model_traces[chosen]))

    return sum(scores) / len(scores)


def single_model_scores(matrix: OracleMatrix) -> dict[str, float]:
    """Compute the average score for each model across all items.

    Items where a model is absent are scored as 0.0.

    Args:
        matrix: Nested dict {item_id -> {model_id -> RunTrace}}.

    Returns:
        Dict {model_id -> average_score}.
    """
    if not matrix:
        return {}

    # Collect all model ids
    all_models: set[str] = set()
    for model_traces in matrix.values():
        all_models.update(model_traces.keys())

    model_totals: dict[str, float] = {m: 0.0 for m in all_models}
    n_items = len(matrix)

    for _item_id, model_traces in matrix.items():
        for model_id in all_models:
            if model_id in model_traces:
                model_totals[model_id] += _get_score(model_traces[model_id])
            # Absent model contributes 0.0

    return {m: total / n_items for m, total in model_totals.items()}


def learned_router_score(router_traces: list[Any]) -> float:
    """Compute the average score of the router's picks.

    Args:
        router_traces: List of RunTrace objects produced by the router.

    Returns:
        Mean score. 0.0 if list is empty.
    """
    if not router_traces:
        return 0.0

    total = sum(_get_score(trace) for trace in router_traces)
    return total / len(router_traces)


def regret(oracle: float, router: float) -> float:
    """Compute regret as the gap between oracle and router performance.

    Args:
        oracle: Oracle score (upper bound).
        router: Router score.

    Returns:
        oracle - router (non-negative if router <= oracle).
    """
    return oracle - router
