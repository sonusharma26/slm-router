from __future__ import annotations

import re
import string
from collections import Counter
from typing import TYPE_CHECKING

from slm_router.eval.scoring.base import ScoreResult, Scorer

if TYPE_CHECKING:
    from slm_router.eval.datasets.schema import EvalItem

# ---------------------------------------------------------------------------
# MCQ letter extraction patterns (tried in order, first match wins)
# ---------------------------------------------------------------------------

_LETTER_PATTERNS: list[re.Pattern[str]] = [
    # "answer is X" or "answer: X"
    re.compile(r"answer\s+is\s*[:\s]\s*\(?([A-Da-d])\)?", re.IGNORECASE),
    # \boxed{X}
    re.compile(r"\\boxed\{\s*([A-Da-d])\s*\}", re.IGNORECASE),
    # "The answer is (X)" / "answer (X)"
    re.compile(r"answer[^A-Za-z]*\(\s*([A-Da-d])\s*\)", re.IGNORECASE),
    # Parenthesised standalone: "(X)" or "(X)."
    re.compile(r"\(\s*([A-Da-d])\s*\)"),
    # Final standalone capital letter A-D at end of text
    re.compile(r"(?:^|[\s,.\n])([A-D])[\s.,\n]*$"),
    # Any capital A-D as last letter-like token
    re.compile(r"\b([A-D])\b(?![\w])"),
]


def _extract_letter(text: str) -> str | None:
    """Try to extract a choice letter (A-D) from model output."""
    stripped = text.strip()
    for pattern in _LETTER_PATTERNS:
        match = pattern.search(stripped)
        if match:
            return match.group(1).upper()
    return None


# ---------------------------------------------------------------------------
# Token F1 helpers (SQuAD-style)
# ---------------------------------------------------------------------------

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)


def _normalize_text(text: str) -> str:
    """Lowercase, remove punctuation and articles, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = _ARTICLES_RE.sub(" ", text)
    text = " ".join(text.split())
    return text


def _token_f1(prediction: str, ground_truth: str) -> tuple[float, float]:
    """Compute token-level F1 and exact match between prediction and ground_truth.

    Returns:
        (f1, em) both in [0.0, 1.0]
    """
    pred_tokens = _normalize_text(prediction).split()
    gold_tokens = _normalize_text(ground_truth).split()

    em = 1.0 if pred_tokens == gold_tokens else 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0, em

    precision = num_same / len(pred_tokens) if pred_tokens else 0.0
    recall = num_same / len(gold_tokens) if gold_tokens else 0.0

    if precision + recall == 0.0:
        return 0.0, em

    f1 = 2 * precision * recall / (precision + recall)
    return f1, em


# ---------------------------------------------------------------------------
# Scorer implementations
# ---------------------------------------------------------------------------


class ExactMatchScorer(Scorer):
    """MCQ scorer: extract the chosen letter from model output and compare to gold."""

    def score(self, item: "EvalItem", model_output: str) -> ScoreResult:
        if item.gold_answer is None:
            return ScoreResult(
                score=0.0,
                correct=None,
                error="gold_answer is None; cannot score MCQ",
            )

        gold = item.gold_answer.strip().upper()
        predicted = _extract_letter(model_output)

        if predicted is None:
            return ScoreResult(
                score=0.0,
                correct=False,
                sub_scores={"predicted": None, "gold": gold},
                error="Could not extract a letter (A-D) from model output",
            )

        correct = predicted == gold
        return ScoreResult(
            score=1.0 if correct else 0.0,
            correct=correct,
            sub_scores={"predicted": predicted, "gold": gold},
        )


class TokenF1Scorer(Scorer):
    """HotpotQA / SQuAD-style scorer: token F1 + exact match."""

    def score(self, item: "EvalItem", model_output: str) -> ScoreResult:
        gold = item.gold_answer or ""
        f1, em = _token_f1(model_output, gold)

        return ScoreResult(
            score=f1,
            correct=bool(em),
            sub_scores={"f1": f1, "em": em},
        )
