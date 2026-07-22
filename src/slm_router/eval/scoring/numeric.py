from __future__ import annotations

import re
from typing import TYPE_CHECKING

from slm_router.eval.scoring.base import ScoreResult, Scorer

if TYPE_CHECKING:
    from slm_router.eval.datasets.schema import EvalItem

# ---------------------------------------------------------------------------
# Number extraction helpers
# ---------------------------------------------------------------------------

# Matches \boxed{N} or \boxed{-N} or \boxed{N.M}
_BOXED_RE = re.compile(r"\\boxed\{\s*([-+]?\d[\d,\s]*\.?\d*)\s*\}")

# "answer is N" / "answer: N" / "= N"
_ANSWER_IS_RE = re.compile(
    r"(?:answer\s+is|answer:|=)\s*\$?\s*([-+]?\d[\d,\s]*\.?\d*)", re.IGNORECASE
)

# Any number in text (including negative, decimals, comma-separated)
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def _clean_number(s: str) -> str:
    """Remove commas, dollar signs, percent signs, and internal spaces."""
    return s.replace(",", "").replace("$", "").replace("%", "").replace(" ", "").strip()


def _parse_float(s: str) -> float | None:
    """Try to parse a string as float; return None on failure."""
    try:
        return float(_clean_number(s))
    except (ValueError, TypeError):
        return None


def _extract_number(text: str) -> float | None:
    """Extract the most likely numeric answer from model output.

    Priority:
    1. \\boxed{N}
    2. "answer is N"
    3. Last number in the text
    """
    # 1. \boxed{N}
    match = _BOXED_RE.search(text)
    if match:
        val = _parse_float(match.group(1))
        if val is not None:
            return val

    # 2. "answer is N"
    match = _ANSWER_IS_RE.search(text)
    if match:
        val = _parse_float(match.group(1))
        if val is not None:
            return val

    # 3. Last number anywhere in text
    all_numbers = _NUMBER_RE.findall(text)
    for candidate in reversed(all_numbers):
        val = _parse_float(candidate)
        if val is not None:
            return val

    return None


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class NumericScorer(Scorer):
    """GSM8K scorer: extract numeric answer and compare with tolerance."""

    TOLERANCE: float = 1e-6

    def score(self, item: "EvalItem", model_output: str) -> ScoreResult:
        if item.gold_answer is None:
            return ScoreResult(
                score=0.0,
                correct=None,
                error="gold_answer is None; cannot score numeric",
            )

        gold_val = _parse_float(item.gold_answer)
        if gold_val is None:
            return ScoreResult(
                score=0.0,
                correct=None,
                error=f"Could not parse gold_answer as float: {item.gold_answer!r}",
            )

        predicted_val = _extract_number(model_output)

        if predicted_val is None:
            return ScoreResult(
                score=0.0,
                correct=False,
                sub_scores={"predicted": None, "gold": gold_val},
                error="Could not extract a number from model output",
            )

        correct = abs(predicted_val - gold_val) <= self.TOLERANCE
        return ScoreResult(
            score=1.0 if correct else 0.0,
            correct=correct,
            sub_scores={"predicted": predicted_val, "gold": gold_val},
        )
