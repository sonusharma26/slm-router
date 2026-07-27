"""Disk-backed response cache using diskcache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:
    import diskcache  # type: ignore
    _DISKCACHE_AVAILABLE = True
except ImportError:
    diskcache = None  # type: ignore
    _DISKCACHE_AVAILABLE = False

CACHE_VERSION = 2


def make_key(
    model_id: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int | None,
    logprobs: bool,
) -> str:
    """Return a stable sha256 hex key for the given call parameters."""
    payload = json.dumps(
        {
            "cache_version": CACHE_VERSION,
            "model_id": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "logprobs": logprobs,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class ResponseCache:
    """Thin wrapper around diskcache.Cache; degrades to no-op if unavailable."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if _DISKCACHE_AVAILABLE:
            self._cache: Any = diskcache.Cache(str(self._path))
        else:
            self._cache = None

    def get(self, key: str) -> dict | None:
        """Return the cached value or None."""
        if self._cache is None:
            return None
        value = self._cache.get(key)
        return value if isinstance(value, dict) else None

    def set(self, key: str, value: dict) -> None:
        """Store *value* under *key*."""
        if self._cache is None:
            return
        self._cache.set(key, value)

    def close(self) -> None:
        if self._cache is not None:
            self._cache.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
