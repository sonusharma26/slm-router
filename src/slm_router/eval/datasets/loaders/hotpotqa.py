from __future__ import annotations

from typing import Any

from slm_router.eval.datasets.base import BenchmarkLoader
from slm_router.eval.datasets.schema import EvalItem, TaskType, stable_item_id


class HotpotQALoader(BenchmarkLoader):
    """Loader for HotpotQA (multi-hop question answering).

    HuggingFace dataset: hotpot_qa, config: distractor
    """

    HF_DATASET = "hotpotqa/hotpot_qa"
    HF_CONFIG = "distractor"

    @property
    def name(self) -> str:
        return "hotpotqa"

    @property
    def task_type(self) -> TaskType:
        return TaskType.MULTIHOP_QA

    def load(
        self,
        split: str = "validation",
        limit: int | None = None,
        seed: int = 42,
    ) -> list[EvalItem]:
        try:
            from datasets import load_dataset  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required to load HotpotQA. "
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
        answer: str = str(row.get("answer", ""))
        level: str = str(row.get("level", ""))
        item_type: str = str(row.get("type", ""))
        row_id: str = str(row.get("id", ""))

        # Build supporting facts reference if available
        supporting_facts = row.get("supporting_facts")
        reference: dict[str, Any] | None = None
        if supporting_facts is not None:
            reference = {"supporting_facts": supporting_facts}

        return EvalItem(
            item_id=stable_item_id(self.name, question),
            dataset=self.name,
            task_type=self.task_type,
            query=question,
            gold_answer=answer,
            reference=reference,
            metadata={
                "id": row_id,
                "level": level,
                "type": item_type,
            },
        )
