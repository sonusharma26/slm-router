"""Typed outcome adapters with provenance-specific defaults (V2-500)."""

from __future__ import annotations
from dataclasses import dataclass
from inference_control.contracts import OutcomeRecord


@dataclass(frozen=True)
class OutcomeAdapter:
    evaluator_type: str
    evaluator_version: str
    training_default: bool
    promotion_default: bool

    def record(
        self,
        decision_id: str,
        quality: dict[str, float],
        source_artifact: str,
        *,
        confidence: float = 1,
        uncertainty: float = 0,
        scope: str = "task",
    ) -> OutcomeRecord:
        return OutcomeRecord(
            decision_id=decision_id,
            quality=quality,
            evaluator_type=self.evaluator_type,
            evaluator_version=self.evaluator_version,
            source_artifact=source_artifact,
            label_confidence=confidence,
            uncertainty=uncertainty,
            causal_scope=scope,
            training_eligible=self.training_default,
            promotion_eligible=self.promotion_default,
        )


class DeterministicOutcomeAdapter(OutcomeAdapter):
    def __init__(self, version: str):
        super().__init__("deterministic", version, True, True)


class ApplicationOutcomeAdapter(OutcomeAdapter):
    def __init__(self, version: str):
        super().__init__("application", version, True, True)


class HumanOutcomeAdapter(OutcomeAdapter):
    def __init__(self, version: str):
        super().__init__("human", version, True, True)


class CalibratedJudgeOutcomeAdapter(OutcomeAdapter):
    def __init__(self, version: str, human_agreement: float, minimum_agreement: float):
        if human_agreement < minimum_agreement:
            raise ValueError("judge lacks measured human agreement")
        super().__init__("calibrated_judge", version, True, True)


class ProxyOutcomeAdapter(OutcomeAdapter):
    def __init__(self, version: str):
        super().__init__("proxy", version, False, False)
