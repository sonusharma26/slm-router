from __future__ import annotations

import random
from abc import ABC, abstractmethod

from slm_router.eval.datasets.schema import EvalItem, TaskType


class BenchmarkLoader(ABC):
    """Abstract base class for benchmark dataset loaders."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of the benchmark dataset."""
        ...

    @property
    @abstractmethod
    def task_type(self) -> TaskType:
        """The task type this loader produces."""
        ...

    @abstractmethod
    def load(
        self,
        split: str = "test",
        limit: int | None = None,
        seed: int = 42,
    ) -> list[EvalItem]:
        """Load items from the dataset.

        Args:
            split: Dataset split to load (e.g. "test", "validation", "train").
            limit: Maximum number of items to return. None = all items.
            seed: Random seed for deterministic sampling when limit is set.

        Returns:
            List of EvalItem instances.
        """
        ...

    @staticmethod
    def _deterministic_sample(
        items: list[EvalItem],
        limit: int | None,
        seed: int,
    ) -> list[EvalItem]:
        """Return a deterministic random sample of *items* of size *limit*.

        If limit is None or >= len(items) the full list is returned unchanged.
        """
        if limit is None or limit >= len(items):
            return items
        return random.Random(seed).sample(items, limit)
