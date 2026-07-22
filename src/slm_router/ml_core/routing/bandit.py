"""Contextual bandit routing policies: LinUCB and Thompson Sampling."""
from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

import numpy as np

from slm_router.types import ModelSpec, RouteDecision
from slm_router.ml_core.routing.policy_base import RoutingPolicy

if TYPE_CHECKING:
    from slm_router.trace import RunTrace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LinUCB Policy
# ---------------------------------------------------------------------------

class LinUCBPolicy(RoutingPolicy):
    """LinUCB (disjoint) contextual bandit.

    One ridge-regression arm per model.  The UCB score is::

        θᵢᵀ x + alpha * sqrt(xᵀ Aᵢ⁻¹ x)

    Parameters
    ----------
    n_arms:
        Number of candidate models.
    dim:
        Dimensionality of the context vector.
    alpha:
        Exploration coefficient.
    """

    policy_version: str = "linucb-v0"

    def __init__(self, n_arms: int, dim: int, alpha: float = 1.0) -> None:
        self.n_arms = n_arms
        self.dim = dim
        self.alpha = alpha

        # Per-arm parameters: A (d×d), b (d,)
        self._A: list[np.ndarray] = [np.eye(dim, dtype=np.float64) for _ in range(n_arms)]
        self._b: list[np.ndarray] = [np.zeros(dim, dtype=np.float64) for _ in range(n_arms)]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ucb_score(self, arm: int, x: np.ndarray) -> float:
        A_inv = np.linalg.inv(self._A[arm])
        theta = A_inv @ self._b[arm]
        exploit = float(theta @ x)
        explore = float(self.alpha * np.sqrt(x @ A_inv @ x))
        return exploit + explore

    # ------------------------------------------------------------------
    # RoutingPolicy interface
    # ------------------------------------------------------------------

    def select(
        self,
        ctx: np.ndarray,
        candidates: list[ModelSpec],
    ) -> RouteDecision:
        """Select arm with maximum UCB score.

        If ``len(candidates) != n_arms`` the policy clips or pads to n_arms.
        """
        x = np.asarray(ctx, dtype=np.float64).ravel()
        n = min(len(candidates), self.n_arms)
        scores = [self._ucb_score(i, x) for i in range(n)]
        best = int(np.argmax(scores))
        model = candidates[best]
        explore = bool(scores[best] < 0)  # heuristic flag

        return RouteDecision(
            query_id=str(uuid.uuid4()),
            chosen_model=model.id,
            difficulty=float(x.mean()),
            predicted_confidence={model.id: float(np.clip(scores[best], 0, 1))},
            expected_reward={model.id: float(scores[best])},
            policy_version=self.policy_version,
            explore=explore,
        )

    def update(self, trace: RunTrace) -> None:  # type: ignore[override]
        """Update the arm chosen in *trace* using the observed reward.

        Expects ``trace.arm_index`` (int) and ``trace.reward`` (float) and
        ``trace.context`` (np.ndarray) to be set.
        """
        arm: int = getattr(trace, "arm_index", 0)
        reward: float = float(getattr(trace, "reward", 0.0))
        ctx = np.asarray(getattr(trace, "context", np.zeros(self.dim)), dtype=np.float64).ravel()

        if arm < 0 or arm >= self.n_arms:
            logger.warning("LinUCBPolicy.update: invalid arm index %d.", arm)
            return

        x = ctx[: self.dim]
        self._A[arm] += np.outer(x, x)
        self._b[arm] += reward * x


# ---------------------------------------------------------------------------
# Thompson Sampling Policy
# ---------------------------------------------------------------------------

class ThompsonSamplingPolicy(RoutingPolicy):
    """Bayesian linear Thompson Sampling.

    Per-arm Bayesian linear regression with a Gaussian posterior
    ``θ ~ N(μ, Σ)``.  At each step, sample θ and pick ``argmax θᵀx``.

    Parameters
    ----------
    n_arms:
        Number of candidate models.
    dim:
        Dimensionality of the context vector.
    lam:
        Ridge / prior precision (initial precision = lam * I).
    sigma2:
        Observation noise variance.
    """

    policy_version: str = "thompson-v0"

    def __init__(
        self,
        n_arms: int,
        dim: int,
        lam: float = 1.0,
        sigma2: float = 1.0,
    ) -> None:
        self.n_arms = n_arms
        self.dim = dim
        self.lam = lam
        self.sigma2 = sigma2

        # Posterior parameters: precision matrix (Λ) and moment (m = Λ μ)
        self._Lambda: list[np.ndarray] = [
            lam * np.eye(dim, dtype=np.float64) for _ in range(n_arms)
        ]
        self._m: list[np.ndarray] = [
            np.zeros(dim, dtype=np.float64) for _ in range(n_arms)
        ]
        self._rng = np.random.default_rng(42)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _sample_theta(self, arm: int) -> np.ndarray:
        Lam = self._Lambda[arm]
        Sigma = np.linalg.inv(Lam)
        mu = Sigma @ self._m[arm]
        try:
            theta = self._rng.multivariate_normal(mu, self.sigma2 * Sigma)
        except np.linalg.LinAlgError:
            theta = mu  # fallback to mean
        return theta

    # ------------------------------------------------------------------
    # RoutingPolicy interface
    # ------------------------------------------------------------------

    def select(
        self,
        ctx: np.ndarray,
        candidates: list[ModelSpec],
    ) -> RouteDecision:
        """Sample a θ per arm and return argmax expected reward."""
        x = np.asarray(ctx, dtype=np.float64).ravel()
        n = min(len(candidates), self.n_arms)
        scores = [float(self._sample_theta(i) @ x) for i in range(n)]
        best = int(np.argmax(scores))
        model = candidates[best]

        return RouteDecision(
            query_id=str(uuid.uuid4()),
            chosen_model=model.id,
            difficulty=float(x.mean()),
            predicted_confidence={model.id: float(np.clip(scores[best], 0, 1))},
            expected_reward={model.id: float(scores[best])},
            policy_version=self.policy_version,
            explore=True,  # always stochastic
        )

    def update(self, trace: RunTrace) -> None:  # type: ignore[override]
        """Bayesian update of the posterior for the chosen arm."""
        arm: int = getattr(trace, "arm_index", 0)
        reward: float = float(getattr(trace, "reward", 0.0))
        ctx = np.asarray(getattr(trace, "context", np.zeros(self.dim)), dtype=np.float64).ravel()

        if arm < 0 or arm >= self.n_arms:
            logger.warning("ThompsonSamplingPolicy.update: invalid arm index %d.", arm)
            return

        x = ctx[: self.dim]
        self._Lambda[arm] += np.outer(x, x) / self.sigma2
        self._m[arm] += (reward / self.sigma2) * x
