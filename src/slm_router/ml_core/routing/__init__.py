"""routing: bandit, offline-RL, and cascade routing policies."""
from __future__ import annotations

import uuid
from typing import Any

import numpy as np

from slm_router.types import RouteDecision
from slm_router.ml_core.routing.reward import compute_reward
from slm_router.ml_core.routing.policy_base import RoutingPolicy
from slm_router.ml_core.routing.bandit import LinUCBPolicy, ThompsonSamplingPolicy
from slm_router.ml_core.routing.offline_rl import OfflineRLPolicy
from slm_router.ml_core.routing.cascade import CascadePolicy

__all__ = [
    "compute_reward",
    "RoutingPolicy",
    "LinUCBPolicy",
    "ThompsonSamplingPolicy",
    "OfflineRLPolicy",
    "CascadePolicy",
    "CheapestFirstPolicy",
    "get_policy",
]


# ---------------------------------------------------------------------------
# CheapestFirstPolicy (v0 default / safety-net policy)
# ---------------------------------------------------------------------------

class CheapestFirstPolicy(RoutingPolicy):
    """Pick the cheapest model from candidate_models or SLM tier.

    Not a learned policy — used as the sensible default when no bandit/RL
    policy has been trained yet, and as the genuine last-resort fallback if
    :func:`get_policy` fails to construct a requested policy.
    """

    policy_version: str = "cheapest-first-v0"

    def __init__(self, registry: Any, candidate_models: list[str] | None = None) -> None:
        self._registry = registry
        self._candidates = candidate_models or []

    def _pick_id(self) -> str:
        if self._candidates:
            def _price(mid: str) -> float:
                try:
                    s = self._registry.get(mid)
                    return s.price_in_per_m + s.price_out_per_m
                except KeyError:
                    return float("inf")

            return min(self._candidates, key=_price)

        slm_models = self._registry.by_tier("SLM")
        if slm_models:
            return slm_models[0].id
        all_models = self._registry.all()
        if all_models:
            return sorted(
                all_models,
                key=lambda s: s.price_in_per_m + s.price_out_per_m,
            )[0].id
        raise RuntimeError("ModelRegistry is empty — cannot pick a model.")

    def pick(self) -> str:
        """Return the model_id of the cheapest available candidate."""
        return self._pick_id()

    def select(self, ctx: np.ndarray, candidates: list) -> RouteDecision:  # type: ignore[override]
        model_id = self._pick_id()
        return RouteDecision(
            query_id=str(uuid.uuid4()),
            chosen_model=model_id,
            policy_version=self.policy_version,
            explore=False,
        )

    def update(self, trace: Any) -> None:  # type: ignore[override]
        """No-op — cheapest-first is a static, non-learned policy."""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_policy(name: str = "cheapest", **kwargs: Any) -> RoutingPolicy:
    """Construct a routing policy by name.

    Parameters
    ----------
    name:
        One of ``"cheapest"`` (default), ``"linucb"``, ``"thompson"``,
        ``"offline_rl"``, or ``"cascade"``.
    kwargs:
        Forwarded to the underlying policy constructor. See each policy
        class for its accepted keyword arguments.

    Raises
    ------
    ValueError
        If ``name`` does not match a known policy.
    KeyError
        If a required keyword argument for the requested policy is missing.
    """
    key = (name or "cheapest").strip().lower()

    if key in ("cheapest", "cheapest_first", "cheapestfirst", "default"):
        return CheapestFirstPolicy(
            registry=kwargs.get("registry"),
            candidate_models=kwargs.get("candidate_models"),
        )
    if key in ("linucb", "lin_ucb"):
        return LinUCBPolicy(
            n_arms=kwargs["n_arms"],
            dim=kwargs["dim"],
            alpha=kwargs.get("alpha", 1.0),
        )
    if key in ("thompson", "thompson_sampling"):
        return ThompsonSamplingPolicy(
            n_arms=kwargs["n_arms"],
            dim=kwargs["dim"],
            lam=kwargs.get("lam", 1.0),
            sigma2=kwargs.get("sigma2", 1.0),
        )
    if key in ("offline_rl", "offline-rl", "offlinerl"):
        return OfflineRLPolicy(
            n_arms=kwargs["n_arms"],
            dim=kwargs["dim"],
            lam=kwargs.get("lam", 0.1),
            beta=kwargs.get("beta", 0.05),
            correctness_weight=kwargs.get("correctness_weight", 1.0),
        )
    if key in ("cascade",):
        return CascadePolicy(
            confidence_predictor=kwargs["confidence_predictor"],
            threshold=kwargs.get("threshold", 0.7),
        )

    raise ValueError(f"Unknown routing policy name: {name!r}")
