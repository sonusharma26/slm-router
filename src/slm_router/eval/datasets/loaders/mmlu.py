from __future__ import annotations

import math
import random
from typing import Any

from slm_router.eval.datasets.base import BenchmarkLoader
from slm_router.eval.datasets.schema import EvalItem, TaskType, stable_item_id

_LETTERS = ["A", "B", "C", "D"]
_CHOICE_KEYS = ["A", "B", "C", "D"]


def _format_choices(choices: list[str]) -> str:
    """Format a list of choice strings into 'A) ... B) ... C) ... D) ...'."""
    parts = []
    for letter, text in zip(_LETTERS, choices):
        parts.append(f"{letter}) {text}")
    return "  ".join(parts)


def _answer_index_to_letter(index: Any) -> str:
    """Convert a numeric answer index (0-3) to a letter A-D."""
    try:
        idx = int(index)
        return _LETTERS[idx]
    except (ValueError, TypeError, IndexError):
        return str(index)


class MMLULoader(BenchmarkLoader):
    """Loader for the MMLU (Massive Multitask Language Understanding) benchmark.

    HuggingFace dataset: cais/mmlu, config: all
    """

    HF_DATASET = "cais/mmlu"
    HF_CONFIG = "all"

    @property
    def name(self) -> str:
        return "mmlu"

    @property
    def task_type(self) -> TaskType:
        return TaskType.MCQ

    def load(
        self,
        split: str = "test",
        limit: int | None = None,
        seed: int = 42,
    ) -> list[EvalItem]:
        try:
            from datasets import load_dataset  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required to load MMLU. "
                "Install it with: pip install datasets"
            ) from exc

        ds = load_dataset(self.HF_DATASET, self.HF_CONFIG, split=split)

        if limit is None:
            raw_items = list(ds)
            items = [self._row_to_eval_item(row) for row in raw_items]
        else:
            # Stratified sample by subject
            items = self._stratified_sample(ds, limit, seed)

        return items

    def _row_to_eval_item(self, row: dict[str, Any]) -> EvalItem:
        choices: list[str] = list(row.get("choices", []))
        question: str = str(row.get("question", ""))
        answer_index = row.get("answer", 0)
        gold_letter = _answer_index_to_letter(answer_index)
        subject: str = str(row.get("subject", "unknown"))

        formatted_choices = _format_choices(choices)
        query = (
            f"{question}\n\n"
            f"{formatted_choices}\n\n"
            "Answer with the letter only."
        )

        return EvalItem(
            item_id=stable_item_id(self.name, query),
            dataset=self.name,
            task_type=self.task_type,
            query=query,
            gold_answer=gold_letter,
            metadata={"subject": subject},
        )

    def _stratified_sample(
        self,
        ds: Any,
        limit: int,
        seed: int,
    ) -> list[EvalItem]:
        """Sample *limit* items from the dataset, stratified by subject."""
        # Group rows by subject
        from collections import defaultdict

        groups: dict[str, list[Any]] = defaultdict(list)
        for row in ds:
            subject = str(row.get("subject", "unknown"))
            groups[subject].append(row)

        subjects = sorted(groups.keys())
        n_subjects = len(subjects)

        if n_subjects == 0:
            return []

        # Allocate slots per subject (roughly equal, handle remainder)
        base_per_subject = limit // n_subjects
        remainder = limit - base_per_subject * n_subjects

        rng = random.Random(seed)
        sampled_rows: list[Any] = []

        for i, subject in enumerate(subjects):
            subject_rows = groups[subject]
            n_take = base_per_subject + (1 if i < remainder else 0)
            n_take = min(n_take, len(subject_rows))
            if n_take > 0:
                sampled_rows.extend(rng.sample(subject_rows, n_take))

        # If we still need more (due to small subjects), top up randomly
        if len(sampled_rows) < limit:
            all_rows = [row for rows in groups.values() for row in rows]
            already = set(id(r) for r in sampled_rows)
            remaining = [r for r in all_rows if id(r) not in already]
            extra = min(limit - len(sampled_rows), len(remaining))
            if extra > 0:
                sampled_rows.extend(rng.sample(remaining, extra))

        return [self._row_to_eval_item(row) for row in sampled_rows]
