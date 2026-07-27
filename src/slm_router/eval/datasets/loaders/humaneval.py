from __future__ import annotations

from typing import Any

from slm_router.eval.datasets.base import BenchmarkLoader
from slm_router.eval.datasets.schema import EvalItem, TaskType, stable_item_id


class HumanEvalLoader(BenchmarkLoader):
    """Loader for OpenAI HumanEval (code generation benchmark).

    HuggingFace dataset: openai_humaneval
    """

    HF_DATASET = "openai/openai_humaneval"

    @property
    def name(self) -> str:
        return "humaneval"

    @property
    def task_type(self) -> TaskType:
        return TaskType.CODE

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
                "The 'datasets' package is required to load HumanEval. "
                "Install it with: pip install datasets"
            ) from exc

        ds = load_dataset(
            self.HF_DATASET,
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
        task_id: str = str(row.get("task_id", ""))
        prompt: str = str(row.get("prompt", ""))
        test: str = str(row.get("test", ""))
        entry_point: str = str(row.get("entry_point", ""))
        canonical_solution: str = str(row.get("canonical_solution", ""))

        query = (
            f"Complete the following Python function:\n\n"
            f"```python\n{prompt}\n```\n\n"
            "Return ONLY the function body implementation inside a ```python``` block."
        )

        reference: dict[str, Any] = {
            "prompt": prompt,
            "test": test,
            "entry_point": entry_point,
            "canonical_solution": canonical_solution,
        }

        return EvalItem(
            item_id=stable_item_id(self.name, task_id or query),
            dataset=self.name,
            task_type=self.task_type,
            query=query,
            gold_answer=None,  # Evaluated via execution, not string match
            reference=reference,
            metadata={"task_id": task_id},
        )
