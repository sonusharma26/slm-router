from __future__ import annotations

from typing import Any

from slm_router.eval.datasets.base import BenchmarkLoader
from slm_router.eval.datasets.schema import EvalItem, TaskType

_LETTERS = ["A", "B", "C", "D"]

# Possible column names used across GPQA dataset versions
_QUESTION_COLS = ["Question", "question", "prompt", "problem"]
_CORRECT_COLS = ["Correct Answer", "correct_answer", "answer", "gold"]
_CHOICE_COL_GROUPS = [
    # (correct, wrong1, wrong2, wrong3)
    ("Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"),
    ("correct_answer", "incorrect_answer_1", "incorrect_answer_2", "incorrect_answer_3"),
    ("answer", "distractor1", "distractor2", "distractor3"),
]


def _get_col(row: dict[str, Any], candidates: list[str], default: str = "") -> str:
    for key in candidates:
        if key in row and row[key] is not None:
            return str(row[key])
    return default


def _build_choices_and_gold(
    row: dict[str, Any],
) -> tuple[list[str], str]:
    """Extract choices and gold letter from a GPQA row.

    Returns (choices_list, gold_letter). The correct answer is always placed
    at a fixed but shuffled position via the item's own ordering; for simplicity
    we place it at position A and shuffle with seed derived from question text.
    """
    import random

    # Try to find correct + incorrect columns
    for correct_col, w1_col, w2_col, w3_col in _CHOICE_COL_GROUPS:
        if correct_col in row:
            correct = str(row.get(correct_col, ""))
            wrong = [
                str(row.get(w1_col, "")),
                str(row.get(w2_col, "")),
                str(row.get(w3_col, "")),
            ]
            # Remove empty distractors
            wrong = [w for w in wrong if w.strip()]

            # Shuffle choices deterministically using question text as seed
            question = _get_col(row, _QUESTION_COLS, "")
            rng = random.Random(hash(question) & 0xFFFFFFFF)
            choices = [correct] + wrong
            rng.shuffle(choices)
            gold_index = choices.index(correct)
            gold_letter = _LETTERS[gold_index] if gold_index < len(_LETTERS) else "A"
            return choices, gold_letter

    # Fallback: look for A/B/C/D columns directly
    choices = []
    for letter in _LETTERS:
        val = row.get(letter, row.get(f"choice_{letter.lower()}", ""))
        choices.append(str(val) if val else "")

    gold_raw = _get_col(row, _CORRECT_COLS, "A")
    # Normalise: could be "A", "(A)", "a"
    gold_letter = gold_raw.strip().strip("()").upper()
    if gold_letter not in _LETTERS:
        gold_letter = "A"

    return choices, gold_letter


def _format_choices(choices: list[str]) -> str:
    parts = []
    for letter, text in zip(_LETTERS, choices):
        parts.append(f"{letter}) {text}")
    return "  ".join(parts)


class GPQALoader(BenchmarkLoader):
    """Loader for GPQA Diamond (Graduate-level Professional QA).

    HuggingFace dataset: Idavidrein/gpqa, config: gpqa_diamond
    """

    HF_DATASET = "Idavidrein/gpqa"
    HF_CONFIG = "gpqa_diamond"

    @property
    def name(self) -> str:
        return "gpqa"

    @property
    def task_type(self) -> TaskType:
        return TaskType.MCQ

    def load(
        self,
        split: str = "train",  # GPQA only has a train split
        limit: int | None = None,
        seed: int = 42,
    ) -> list[EvalItem]:
        try:
            from datasets import load_dataset  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required to load GPQA. "
                "Install it with: pip install datasets"
            ) from exc

        # GPQA Diamond only provides a single split; gracefully fall back
        try:
            ds = load_dataset(
                self.HF_DATASET,
                self.HF_CONFIG,
                split=split,
                trust_remote_code=True,
            )
        except Exception:
            ds = load_dataset(
                self.HF_DATASET,
                self.HF_CONFIG,
                split="train",
                trust_remote_code=True,
            )

        items: list[EvalItem] = []
        for row in ds:
            try:
                item = self._row_to_eval_item(row)
                items.append(item)
            except Exception:
                continue  # Skip malformed rows defensively

        return self._deterministic_sample(items, limit, seed)

    def _row_to_eval_item(self, row: dict[str, Any]) -> EvalItem:
        question = _get_col(row, _QUESTION_COLS, "")
        choices, gold_letter = _build_choices_and_gold(row)

        formatted = _format_choices(choices)
        query = (
            f"{question}\n\n"
            f"{formatted}\n\n"
            "Answer with the letter only."
        )

        # Preserve extra metadata columns
        meta: dict[str, Any] = {}
        for key in ("Subdomain", "subdomain", "difficulty", "source", "Writer's Difficulty Estimate"):
            if key in row and row[key] is not None:
                meta[str(key)] = row[key]

        return EvalItem(
            dataset=self.name,
            task_type=self.task_type,
            query=query,
            gold_answer=gold_letter,
            metadata=meta,
        )
