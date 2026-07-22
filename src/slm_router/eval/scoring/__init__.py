"""Evaluation scoring: result types, scorer base class, and dispatch utilities."""
from __future__ import annotations

from slm_router.eval.scoring.base import ScoreResult, Scorer
from slm_router.eval.scoring.dispatch import get_scorer, score_response

__all__ = [
    "ScoreResult",
    "Scorer",
    "get_scorer",
    "score_response",
]
