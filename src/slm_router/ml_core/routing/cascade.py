"""Cascade routing policy: try SLM first, escalate to LLM on low confidence."""
from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Sequence

import numpy as np

from slm_router.types import ModelSpec, RouteDecision
from slm_router.ml_core.routing.policy_base import RoutingPolicy
from slm_router.ml_core.confidence.predictor import ConfidencePredictor

if TYPE_CHECKING:
    from slm_router.trace import RunTrace

logger = logging.getLogger(__name__)


class CascadePolicy(RoutingPolicy):
    """Static safety-net cascade policy.

    Strategy
    --------
    1. Always attempt the *first* model in ``candidates`` (the SLM).
    2. Compute posterior confidence using ``confidence_predictor``.
    3. If ``conformal_accept`` is False **or** ``p_correct < threshold``,
       escalate to the *last* model in ``candidates`` (the LLM).
    4. Otherwise, commit to the SLM response.

    This provides a strong, interpretable baseline with guaranteed conformal
    coverage: by construction, escalation is triggered whenever the SLM
    falls outside the conformal prediction set.

    Parameters
    ----------
    confidence_predictor:
        A fitted :class:`~slm_router.ml_core.confidence.ConfidencePredictor`.
    threshold:
        Scalar p_correct threshold below which escalation is forced
        (in addition to the conformal check).
    """

    policy_version: str = "cascade-v0"

    def __init__(
        self,
        confidence_predictor: ConfidencePredictor,
        threshold: float = 0.7,
    ) -> None:
        self.confidence_predictor = confidence_predictor
        self.threshold = threshold

    # ------------------------------------------------------------------
    # RoutingPolicy interface
    # ------------------------------------------------------------------

    def select(
        self,
        ctx: np.ndarray,
        candidates: list[ModelSpec],
    ) -> RouteDecision:
        """Select a model using the cascade rule.

        Parameters
        ----------
        ctx:
            Context / difficulty feature vector.
        candidates:
            Ordered model list.  ``candidates[0]`` is the SLM,
            ``candidates[-1]`` is the most capable LLM.

        Returns
        -------
        RouteDecision
        """
        if not candidates:
            raise ValueError("candidates must be non-empty.")

        feats = np.asarray(ctx, dtype=np.float32)
        conf_out = self.confidence_predictor.predict_prior(feats)

        slm = candidates[0]
        llm = candidates[-1]

        escalate = (not conf_out.conformal_accept) or (conf_out.p_correct < self.threshold)
        chosen = llm if (escalate and len(candidates) > 1) else slm

        predicted_conf = {
            slm.id: conf_out.p_correct,
        }
        if len(candidates) > 1:
            # LLM assumed to have high confidence
            predicted_conf[llm.id] = 0.95

        expected_reward = {m.id: 0.0 for m in candidates}

        return RouteDecision(
            query_id=str(uuid.uuid4()),
            chosen_model=chosen.id,
            difficulty=float(feats.mean()),
            predicted_confidence=predicted_conf,
            expected_reward=expected_reward,
            policy_version=self.policy_version,
            explore=False,
        )

    def update(self, trace: RunTrace) -> None:  # type: ignore[override]
        """No online update — cascade is a static policy.

        To adapt the threshold, re-fit the ``confidence_predictor`` and
        adjust ``self.threshold`` offline.
        """

    # ------------------------------------------------------------------
    # Posterior escalation (call after SLM response is available)
    # ------------------------------------------------------------------

    def maybe_escalate(
        self,
        feats: np.ndarray,
        raw_signals: dict[str, float],
        candidates: list[ModelSpec],
    ) -> tuple[bool, RouteDecision]:
        """Re-evaluate escalation using posterior confidence.

        Call this after the SLM has produced a response and raw signals
        (log-probs, verifier score) are available.

        Parameters
        ----------
        feats:
            Same feature vector used in :meth:`select`.
        raw_signals:
            Posterior signals (see :meth:`~ConfidencePredictor.predict_posterior`).
        candidates:
            Same candidates list as :meth:`select`.

        Returns
        -------
        escalated: bool
            ``True`` if the LLM should be invoked.
        decision: RouteDecision
            Updated routing decision.
        """
        conf_out = self.confidence_predictor.predict_posterior(feats, raw_signals)
        slm = candidates[0]
        llm = candidates[-1]

        escalate = (not conf_out.conformal_accept) or (conf_out.p_correct < self.threshold)
        chosen = llm if (escalate and len(candidates) > 1) else slm

        decision = RouteDecision(
            query_id=str(uuid.uuid4()),
            chosen_model=chosen.id,
            difficulty=float(feats.mean()),
            predicted_confidence={slm.id: conf_out.p_correct},
            expected_reward={m.id: 0.0 for m in candidates},
            policy_version=self.policy_version + "-posterior",
            explore=False,
        )
        return escalate, decision
