"""Canonical, process-independent identities used by replay and evidence lineage."""
from __future__ import annotations
import hashlib
import json
import math
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any


def primitive(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return primitive(value.model_dump())
    if is_dataclass(value) and not isinstance(value, type):
        return primitive(asdict(value))
    if isinstance(value, dict):
        return {str(k): primitive(v) for k, v in sorted(value.items(), key=lambda p: str(p[0]))}
    if isinstance(value, (set, frozenset)):
        return sorted((primitive(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(value, (list, tuple)):
        return [primitive(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite evidence cannot be serialized")
    return value


def canonical(value: Any) -> str:
    return json.dumps(primitive(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class FrozenDict(dict):
    """JSON-serializable immutable mapping, including nested configuration values."""
    def _immutable(self, *args, **kwargs):
        raise TypeError("immutable contract mapping")
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable

    def __deepcopy__(self, memo):
        return self


def freeze(value):
    if isinstance(value, dict):
        return FrozenDict({key: freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(freeze(item) for item in value)
    return value
