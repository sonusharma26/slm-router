"""Delayed outcome linkage, precedence, disputes, and eligibility (V2-500..502)."""

from __future__ import annotations
from inference_control.contracts import OutcomeRecord, PolicySpec

PRECEDENCE = {"deterministic": 1, "application": 2, "human": 3, "calibrated_judge": 4, "proxy": 5}


class OutcomeStore:
    def __init__(self):
        self._records: dict[str, OutcomeRecord] = {}
        self._active: dict[str, str] = {}

    def add(self, outcome: OutcomeRecord, policy: PolicySpec) -> OutcomeRecord:
        training = (
            outcome.evaluator_type in policy.training_outcome_sources and not outcome.disputed
        )
        promotion = (
            outcome.evaluator_type in policy.promotion_outcome_sources
            and outcome.evaluator_type != "proxy"
            and not outcome.disputed
        )
        qualified = outcome.model_copy(
            update={"training_eligible": training, "promotion_eligible": promotion}
        )
        prior_id = self._active.get(outcome.decision_id)
        if prior_id:
            prior = self._records[prior_id]
            if qualified.supersedes != prior_id:
                raise ValueError("replacement outcome must explicitly supersede the active label")
            if PRECEDENCE[qualified.evaluator_type] > PRECEDENCE[prior.evaluator_type]:
                raise ValueError("lower-precedence outcome cannot supersede stronger evidence")
        self._records[qualified.outcome_id] = qualified
        self._active[qualified.decision_id] = qualified.outcome_id
        return qualified

    def active(self, decision_id: str) -> OutcomeRecord | None:
        oid = self._active.get(decision_id)
        return self._records.get(oid) if oid else None

    def dispute(self, outcome_id: str) -> OutcomeRecord:
        updated = self._records[outcome_id].model_copy(
            update={
                "disputed": True,
                "training_eligible": False,
                "promotion_eligible": False,
            }
        )
        self._records[outcome_id] = updated
        return updated
