"""Reward computation for the routing policy."""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Normalisation constants (can be tuned / loaded from config)
# ---------------------------------------------------------------------------
_COST_REF_USD: float = 0.01    # reference cost (e.g. GPT-4 Turbo per query)
_LATENCY_REF_MS: float = 5000.0  # reference latency (5 s)


def _normalise_cost(cost_usd: float) -> float:
    """Map cost in USD to [0, 1] via a reference value."""
    return float(np.clip(cost_usd / max(_COST_REF_USD, 1e-9), 0.0, 1.0))


def _normalise_latency(latency_ms: float) -> float:
    """Map latency in ms to [0, 1] via a reference value."""
    return float(np.clip(latency_ms / max(_LATENCY_REF_MS, 1e-9), 0.0, 1.0))


def compute_reward(
    quality: float,
    cost_usd: float,
    latency_ms: float,
    lam: float,
    beta: float,
    correctness_weight: float = 1.0,
) -> float:
    """Compute a scalar reward for the routing policy.

    The reward trades off correctness quality against cost and latency:

    .. math::

        r = w \\cdot \\text{quality} - \\lambda \\cdot \\hat{c} - \\beta \\cdot \\hat{l}

    where :math:`\\hat{c}` and :math:`\\hat{l}` are normalised cost and latency.

    Parameters
    ----------
    quality:
        Correctness / quality score in ``[0, 1]`` (e.g. from a verifier).
    cost_usd:
        Monetary cost in US dollars for this query.
    latency_ms:
        Wall-clock latency in milliseconds.
    lam:
        Cost penalty coefficient (:attr:`~slm_router.config.AppConfig.reward.lambda_cost`).
    beta:
        Latency penalty coefficient (:attr:`~slm_router.config.AppConfig.reward.beta_latency`).
    correctness_weight:
        Weight applied to the quality term (default 1.0).

    Returns
    -------
    float
        Scalar reward (can be negative).
    """
    c_norm = _normalise_cost(cost_usd)
    l_norm = _normalise_latency(latency_ms)

    reward = (
        correctness_weight * float(np.clip(quality, 0.0, 1.0))
        - lam * c_norm
        - beta * l_norm
    )
    return float(reward)
