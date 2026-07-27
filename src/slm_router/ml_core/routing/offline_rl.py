"""Offline RL routing policy via Fitted Q Evaluation (FQE)."""
from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Sequence

import numpy as np

from slm_router.types import ModelSpec, RouteDecision
from slm_router.ml_core.routing.policy_base import RoutingPolicy
from slm_router.ml_core.routing.reward import compute_reward

if TYPE_CHECKING:
    from slm_router.trace import RunTrace

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defensive sklearn import
# ---------------------------------------------------------------------------
try:
    from sklearn.ensemble import GradientBoostingRegressor as _GBR  # type: ignore
    from sklearn.base import clone as _sk_clone  # type: ignore
    _SKLEARN_AVAILABLE = True
except Exception:  # noqa: BLE001
    _GBR = None  # type: ignore
    _sk_clone = None  # type: ignore
    _SKLEARN_AVAILABLE = False

# Conservative penalty for under-supported (state, action) pairs
_LOW_SUPPORT_PENALTY: float = 0.2
_MIN_SUPPORT: int = 5


def _default_regressor() -> object | None:
    if _SKLEARN_AVAILABLE and _GBR is not None:
        return _GBR(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=0)
    return None


class OfflineRLPolicy(RoutingPolicy):
    """Offline fitted-Q policy with conservative low-support penalty.

    For each arm (model), a ``Q(x, a) → reward`` regressor is trained on
    logged traces.  At inference time the policy selects ``argmax_a Q(x, a)``,
    penalising arms with insufficient support in the logged data.

    Parameters
    ----------
    n_arms:
        Number of candidate models.
    dim:
        Context dimensionality.
    lam:
        :attr:`~slm_router.config.AppConfig.reward.lambda_cost`.
    beta:
        :attr:`~slm_router.config.AppConfig.reward.beta_latency`.
    correctness_weight:
        Weight on quality in reward computation.
    """

    policy_version: str = "offline-rl-v0"

    def __init__(
        self,
        n_arms: int,
        dim: int,
        lam: float = 0.1,
        beta: float = 0.05,
        correctness_weight: float = 1.0,
    ) -> None:
        self.n_arms = n_arms
        self.dim = dim
        self.lam = lam
        self.beta = beta
        self.correctness_weight = correctness_weight

        self._q_funcs: list[object | None] = [None] * n_arms
        self._support_counts: np.ndarray = np.zeros(n_arms, dtype=np.int64)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _q_value(self, arm: int, x: np.ndarray) -> float:
        q_fn = self._q_funcs[arm]
        if q_fn is None:
            base = -_LOW_SUPPORT_PENALTY
        else:
            base = float(q_fn.predict(x.reshape(1, -1))[0])  # type: ignore[union-attr]
        # Conservative penalty for under-supported arms
        if self._support_counts[arm] < _MIN_SUPPORT:
            base -= _LOW_SUPPORT_PENALTY
        return base

    # ------------------------------------------------------------------
    # RoutingPolicy interface
    # ------------------------------------------------------------------

    def select(
        self,
        ctx: np.ndarray,
        candidates: list[ModelSpec],
    ) -> RouteDecision:
        """Greedy Q-maximising arm selection.

        Raises
        ------
        ValueError
            If ``len(candidates)`` does not exactly match ``self.n_arms``.
            Silently truncating (or ignoring extra candidates) would produce
            a selection over the wrong action space, so a mismatch is
            treated as a caller error rather than tolerated.
        """
        x = np.asarray(ctx, dtype=np.float32).ravel()
        if len(candidates) != self.n_arms:
            raise ValueError(
                f"OfflineRLPolicy.select() expected {self.n_arms} candidates "
                f"(n_arms={self.n_arms}) but received {len(candidates)}."
            )
        n = self.n_arms
        q_vals = [self._q_value(i, x) for i in range(n)]
        best = int(np.argmax(q_vals))
        model = candidates[best]

        return RouteDecision(
            query_id=str(uuid.uuid4()),
            chosen_model=model.id,
            difficulty=float(x.mean()),
            predicted_confidence={model.id: float(np.clip(q_vals[best], 0, 1))},
            expected_reward={candidates[i].id: float(q_vals[i]) for i in range(n)},
            policy_version=self.policy_version,
            explore=False,
        )

    def update(self, trace: RunTrace) -> None:  # type: ignore[override]
        """No-op at inference; use :meth:`fit` for batch offline updates."""

    def training_readiness(self) -> tuple[bool, str | None]:
        """Return whether every configured arm has enough fitted support."""
        unsupported = [
            f"arm {arm}: {int(count)}/{_MIN_SUPPORT}"
            for arm, count in enumerate(self._support_counts)
            if count < _MIN_SUPPORT or self._q_funcs[arm] is None
        ]
        if unsupported:
            return False, "insufficient per-arm support (" + ", ".join(unsupported) + ")"
        return True, None

    # ------------------------------------------------------------------
    # Offline training
    # ------------------------------------------------------------------

    def fit(self, traces: Sequence[RunTrace]) -> "OfflineRLPolicy":
        """Train Q-regressors from a batch of logged traces.

        Each trace must expose:
        - ``trace.context`` (np.ndarray, shape ``(dim,)``)
        - ``trace.arm_index`` (int)
        - ``trace.quality`` (float)
        - ``trace.cost_usd`` (float)
        - ``trace.latency_ms`` (float)

        Returns
        -------
        self
        """
        # Collect (X, y) per arm
        arm_X: list[list[np.ndarray]] = [[] for _ in range(self.n_arms)]
        arm_y: list[list[float]] = [[] for _ in range(self.n_arms)]

        for tr in traces:
            arm = int(getattr(tr, "arm_index", 0))
            if arm < 0 or arm >= self.n_arms:
                continue
            ctx = np.asarray(getattr(tr, "context", np.zeros(self.dim)), dtype=np.float32)
            quality = float(getattr(tr, "quality", 0.0))
            cost = float(getattr(tr, "cost_usd", 0.0))
            latency = float(getattr(tr, "latency_ms", 0.0))
            r = compute_reward(
                quality, cost, latency,
                self.lam, self.beta, self.correctness_weight,
            )
            arm_X[arm].append(ctx)
            arm_y[arm].append(r)

        if not _SKLEARN_AVAILABLE:
            logger.warning("sklearn unavailable; OfflineRLPolicy Q-functions will be None.")
            return self

        for arm in range(self.n_arms):
            X = arm_X[arm]
            y = arm_y[arm]
            self._support_counts[arm] = len(X)
            if len(X) < _MIN_SUPPORT:
                logger.warning(
                    "Arm %d has only %d samples (< %d); Q-function not fitted.",
                    arm, len(X), _MIN_SUPPORT,
                )
                self._q_funcs[arm] = None
                continue

            X_arr = np.stack(X).astype(np.float32)
            y_arr = np.asarray(y, dtype=np.float32)
            q = _sk_clone(_default_regressor())
            q.fit(X_arr, y_arr)
            self._q_funcs[arm] = q
            logger.info("Arm %d Q-function fitted on %d traces.", arm, len(X))

        return self

    # ------------------------------------------------------------------
    # FQE value estimate (promotion gate)
    # ------------------------------------------------------------------

    def fqe_value(self, traces: Sequence[RunTrace]) -> float:
        """Doubly-robust off-policy value estimate over logged traces.

        For each trace we have a logged context ``x``, a logged action
        ``a_log`` (``trace.arm_index``), and a realised reward ``r`` computed
        the same way :meth:`fit` computes it (via
        :func:`~slm_router.ml_core.routing.reward.compute_reward`). The
        target (greedy) policy is deterministic: ``a_greedy =
        argmax_a Q(x, a)``, using :meth:`_q_value` so the low-support penalty
        is applied consistently with training and inference.

        The per-trace doubly-robust estimate is

        .. math::

            \\hat{v}(x) = Q(x, a_{greedy}) +
                \\mathbb{1}[a_{log} = a_{greedy}] \\cdot (r - Q(x, a_{log}))

        i.e. the direct-method value ``Q(x, a_greedy)`` corrected by the
        logged reward whenever the logging policy happened to take the same
        action the greedy target policy would take (the importance weight
        for a deterministic target policy is 1 in that case and 0
        otherwise). This is no longer a pure optimistic upper bound: unlike
        ``max_a Q(x, a)`` alone, it is anchored back to observed rewards
        whenever logged and greedy actions coincide.

        Parameters
        ----------
        traces:
            Evaluation traces (held-out or from a simulator). Traces whose
            ``arm_index`` falls outside ``[0, n_arms)`` are skipped, matching
            :meth:`fit`.

        Returns
        -------
        float
            Mean doubly-robust value estimate across traces.
        """
        if not traces:
            return 0.0

        values: list[float] = []
        for tr in traces:
            a_log = int(getattr(tr, "arm_index", 0))
            if a_log < 0 or a_log >= self.n_arms:
                continue

            ctx = np.asarray(getattr(tr, "context", np.zeros(self.dim)), dtype=np.float32)
            quality = float(getattr(tr, "quality", 0.0))
            cost = float(getattr(tr, "cost_usd", 0.0))
            latency = float(getattr(tr, "latency_ms", 0.0))
            r = compute_reward(
                quality, cost, latency,
                self.lam, self.beta, self.correctness_weight,
            )

            q_vals = [self._q_value(i, ctx) for i in range(self.n_arms)]
            a_greedy = int(np.argmax(q_vals))

            dm = q_vals[a_greedy]
            correction = (r - q_vals[a_log]) if a_log == a_greedy else 0.0
            values.append(dm + correction)

        if not values:
            return 0.0

        return float(np.mean(values))
