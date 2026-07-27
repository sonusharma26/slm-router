"""Self-improvement loop: incremental policy updates + periodic batch retrain with a
promotion gate that only ships a new policy if it beats the incumbent under FQE.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SelfImprovementLoop:
    """Orchestrates the trace -> retrain -> promote feedback cycle.

    - ``step_online`` applies an incremental bandit update for each new labeled trace.
    - ``retrain_batch`` fits an OfflineRLPolicy on the full trace set and promotes it
      only if its FQE value beats the incumbent by ``margin`` (off-policy gate).
    """

    def __init__(self, store: Any, policy: Any | None, margin: float = 0.0):
        self.store = store
        self.policy = policy
        self.margin = margin

    def step_online(self, traces: list[Any]) -> int:
        """Incrementally update the live policy from newly labeled traces."""
        n = 0
        for t in traces:
            if getattr(t, "score", None) is None:
                continue
            try:
                self.policy.update(t)
                n += 1
            except Exception as exc:  # policy may not support online update
                logger.debug("online update skipped: %s", exc)
        return n

    def retrain_batch(
        self, candidate_policy: Any, traces: list[Any] | None = None
    ) -> dict[str, Any]:
        """Fit ``candidate_policy`` on logged traces and gate promotion via FQE.

        Returns a dict describing the decision. Does not mutate ``self.policy``
        unless the gate passes; the caller persists/promotes the returned policy.

        If ``traces`` is omitted, falls back to querying the store for
        ``live``/``router_eval`` traces (the online-serving path). Callers
        training from an oracle matrix should build samples themselves (see
        ``ml_core.routing.trace_adapter``) and pass them in directly.
        """
        if traces is None:
            traces = self.store.query(run_kind="live") + self.store.query(run_kind="router_eval")
        if not traces:
            return {"promoted": False, "reason": "no traces"}

        candidate_policy.fit(traces)
        readiness = getattr(candidate_policy, "training_readiness", None)
        if readiness is not None:
            ready, reason = readiness()
            if not ready:
                return {
                    "promoted": False,
                    "reason": reason,
                    "n_traces": len(traces),
                }
        if self.policy is None:
            incumbent_val = float("-inf")
        else:
            try:
                incumbent_val = self.policy.fqe_value(traces)
            except Exception:
                incumbent_val = float("-inf")
        candidate_val = candidate_policy.fqe_value(traces)

        promote = candidate_val > incumbent_val + self.margin
        if promote:
            self.policy = candidate_policy
        return {
            "promoted": promote,
            "incumbent_fqe": incumbent_val,
            "candidate_fqe": candidate_val,
            "margin": self.margin,
            "n_traces": len(traces),
        }
