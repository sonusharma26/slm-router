"""Thin async rate-limiter wrapper; degrades to no-op if aiolimiter is absent."""

from __future__ import annotations

try:
    from aiolimiter import AsyncLimiter as _AsyncLimiter  # type: ignore
    _AIOLIMITER_AVAILABLE = True
except ImportError:
    _AsyncLimiter = None  # type: ignore
    _AIOLIMITER_AVAILABLE = False


class _NoOpLimiter:
    """Fallback when aiolimiter is not installed."""

    async def acquire(self) -> None:  # noqa: D401
        """No-op; always succeeds immediately."""


class RateLimiter:
    """Async rate limiter wrapping aiolimiter.AsyncLimiter.

    Falls back to a no-op implementation when aiolimiter is not installed so
    that importing this module never raises ImportError.
    """

    def __init__(self, requests_per_second: float = 4.0) -> None:
        if _AIOLIMITER_AVAILABLE and _AsyncLimiter is not None:
            # aiolimiter requires max_rate >= 1 (capacity for a single acquire).
            # For sub-1 rates, stretch the time period instead of shrinking max_rate.
            if requests_per_second >= 1:
                max_rate, time_period = requests_per_second, 1.0
            else:
                max_rate, time_period = 1.0, 1.0 / requests_per_second
            self._limiter = _AsyncLimiter(max_rate=max_rate, time_period=time_period)
        else:
            self._limiter = _NoOpLimiter()

    async def acquire(self) -> None:
        """Acquire one slot; blocks until within the rate budget."""
        if isinstance(self._limiter, _NoOpLimiter):
            await self._limiter.acquire()
        else:
            async with self._limiter:
                pass
