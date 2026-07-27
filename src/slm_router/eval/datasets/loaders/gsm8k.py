from __future__ import annotations

import re
from typing import Any

from slm_router.eval.datasets.base import BenchmarkLoader
from slm_router.eval.datasets.schema import EvalItem, TaskType, stable_item_id

_HASH_SEP = "####"
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")


def _extract_gold_numeric(answer_text: str) -> str | None:
    """Extract the numeric answer after '####' in a GSM8K answer string.

    Returns a normalised numeric string (commas removed), or None if not found.
    """
    if _HASH_SEP in answer_text:
        after = answer_text.split(_HASH_SEP, 1)[1].strip()
        # Remove commas used as thousands separators
        after = after.replace(",", "").strip()
        # Take just the first token (in case of trailing whitespace/text)
        parts = after.split()
        if parts:
            return parts[0]
    # Fallback: find last number in string
    matches = _NUMBER_RE.findall(answer_text)
    if matches:
        return matches[-1].replace(",", "")
    return None


class GSM8KLoader(BenchmarkLoader):
    """Loader for GSM8K (Grade School Math 8K).

    HuggingFace dataset: gsm8k, config: main
    """

    HF_DATASET = "openai/gsm8k"
    HF_CONFIG = "main"

    @property
    def name(self) -> str:
        return "gsm8k"

    @property
    def task_type(self) -> TaskType:
        return TaskType.MATH

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
                "The 'datasets' package is required to load GSM8K. "
                "Install it with: pip install datasets"
            ) from exc

        ds = load_dataset(
            self.HF_DATASET,
            self.HF_CONFIG,
            split=split,
        )

        items: list[EvalItem] = []
        for row in ds:
            try:
                item = self._row_to_eval_item(row)
                items.append(item)
            except Exception:
                continue

        return self._deterministic_sample(items, limit, seed)

    def _row_to_eval_item(self, row: dict[str, Any]) -> EvalItem:
        question: str = str(row.get("question", ""))
        raw_answer: str = str(row.get("answer", ""))

        gold = _extract_gold_numeric(raw_answer)

        query = (
            f"{question}\n\n"
            "Show your work, then end your response with the final numeric "
            "answer on its own line in the form: #### <number>"
        )

        return EvalItem(
            item_id=stable_item_id(self.name, query),
            dataset=self.name,
            task_type=self.task_type,
            query=query,
            gold_answer=gold,
            reference={"raw_answer": raw_answer},
            metadata={},
        )
