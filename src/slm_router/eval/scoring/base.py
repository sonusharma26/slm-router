from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from slm_router.eval.datasets.schema import EvalItem


class ScoreResult(BaseModel):
    score: float
    correct: bool | None = None
    sub_scores: dict[str, Any] = {}
    error: str | None = None


class Scorer(ABC):
    """Abstract base class for all scorers."""

    @abstractmethod
    def score(self, item: "EvalItem", model_output: str) -> ScoreResult:
        """Score a model output against a ground-truth EvalItem.

        Args:
            item: The EvalItem containing query and gold_answer.
            model_output: Raw text output from the model.

        Returns:
            A ScoreResult with at least a numeric score in [0.0, 1.0].
        """
        ...
