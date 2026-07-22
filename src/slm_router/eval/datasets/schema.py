from __future__ import annotations

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


class EvalItem(BaseModel, frozen=True):
    item_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    dataset: str
    task_type: TaskType
    query: str  # render-ready
    gold_answer: str | None = None
    reference: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
