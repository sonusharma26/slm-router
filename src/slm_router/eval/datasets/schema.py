from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any
import uuid

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    MCQ = "mcq"
    MULTIHOP_QA = "multihop_qa"
    MATH = "math"
    CODE = "code"
    OPEN_ENDED = "open_ended"


def stable_item_id(dataset: str, query: str) -> str:
    """Deterministic item_id derived from content, stable across process runs.

    Loaders should pass this explicitly so the same underlying dataset row
    always maps to the same item_id (needed for oracle-matrix skip-gating
    and for reconnecting stored traces back to their source query text).
    """
    return hashlib.md5(f"{dataset}:{query}".encode()).hexdigest()


class EvalItem(BaseModel, frozen=True):
    # Falls back to a random id only if a loader forgets to pass one explicitly.
    item_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dataset: str
    task_type: TaskType
    query: str  # render-ready
    gold_answer: str | None = None
    reference: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
