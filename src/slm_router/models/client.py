"""OpenRouter async HTTP client with caching, rate-limiting, and retries."""

from __future__ import annotations

import math
import time
from typing import Any

from pydantic import BaseModel

from slm_router.types import (
    CostBreakdown,
    ModelResponse,
    TokenLogprob,
    Usage,
)

try:
    import httpx  # type: ignore
    _HTTPX_AVAILABLE = True
except ImportError:
    httpx = None  # type: ignore
    _HTTPX_AVAILABLE = False

try:
    import tenacity  # type: ignore
    from tenacity import (
        retry,
        retry_if_exception,
        stop_after_attempt,
        wait_exponential_jitter,
    )
    _TENACITY_AVAILABLE = True
except ImportError:
    tenacity = None  # type: ignore
    _TENACITY_AVAILABLE = False

BASE_URL = "https://openrouter.ai/api/v1"


class ChatMessage(BaseModel):
    """Simple chat message helper."""

    role: str
    content: str


def _to_dict_messages(messages: list[Any]) -> list[dict]:
    """Normalise ChatMessage objects and plain dicts into dicts."""
    result = []
    for m in messages:
        if isinstance(m, ChatMessage):
            result.append(m.model_dump())
        elif isinstance(m, dict):
            result.append(m)
        else:
            result.append({"role": str(getattr(m, "role", "user")), "content": str(m)})
    return result


def _mean_logprob(logprobs: list[TokenLogprob] | None) -> float:
    """Return mean log-probability or 0.0 if unavailable."""
    if not logprobs:
        return 0.0
    return sum(t.logprob for t in logprobs) / len(logprobs)


def _is_retryable(exc: BaseException) -> bool:
    """Return True for httpx transport errors and 429/5xx HTTP responses."""
    if httpx is None:
        return False
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {429, 500, 502, 503, 504}
    return False


def _build_retry_decorator():
    if not _TENACITY_AVAILABLE:
        return None
    return retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1, max=30),
        reraise=True,
    )


_RETRY = _build_retry_decorator()


class OpenRouterClient:
    """Async client for the OpenRouter /chat/completions endpoint."""

    def __init__(
        self,
        api_key: str,
        registry: Any,
        cache: Any | None = None,
        limiter: Any | None = None,
        client: Any | None = None,
        timeout: float = 60,
    ) -> None:
        self._api_key = api_key
        self._registry = registry
        self._cache = cache
        self._limiter = limiter
        self._timeout = timeout
        self._own_client = client is None
        if client is not None:
            self._client = client
        elif _HTTPX_AVAILABLE:
            self._client = httpx.AsyncClient(
                base_url=BASE_URL,
                timeout=httpx.Timeout(timeout),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
        else:
            self._client = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_cache_key(
        self,
        model_id: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int | None,
        logprobs: bool,
    ) -> str:
        from slm_router.models.cache import make_key
        return make_key(model_id, messages, temperature, max_tokens, logprobs)

    async def _post(self, payload: dict) -> dict:
        """POST to /chat/completions, raising on HTTP errors."""
        resp = await self._client.post("/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def _post_with_retry(self, payload: dict) -> dict:
        if _TENACITY_AVAILABLE and _RETRY is not None:
            return await _RETRY(self._post)(payload)
        return await self._post(payload)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def complete(
        self,
        model_id: str,
        messages: list[dict] | list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        logprobs: bool = False,
        use_cache: bool = True,
    ) -> ModelResponse:
        """Call the model and return a ModelResponse.

        Uses disk cache, rate-limiter, and tenacity retries when available.
        """
        spec = self._registry.get(model_id)
        dict_messages = _to_dict_messages(messages)  # type: ignore[arg-type]

        # ---- cache lookup ----
        cache_key = self._make_cache_key(
            model_id, dict_messages, temperature, max_tokens, logprobs
        )
        if use_cache and self._cache is not None:
            cached_raw = self._cache.get(cache_key)
            if cached_raw is not None:
                return self._parse_response(
                    model_id, cached_raw, spec, logprobs, cached=True
                )

        # ---- rate-limit ----
        if self._limiter is not None:
            await self._limiter.acquire()

        # ---- build payload ----
        payload: dict[str, Any] = {
            "model": model_id,
            "messages": dict_messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        want_logprobs = logprobs and spec.supports_logprobs
        if want_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 1

        # ---- HTTP call ----
        t0 = time.perf_counter()
        raw = await self._post_with_retry(payload)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # ---- store cache ----
        if use_cache and self._cache is not None:
            self._cache.set(cache_key, raw)

        return self._parse_response(
            model_id, raw, spec, logprobs, cached=False, latency_ms=latency_ms
        )

    def _parse_response(
        self,
        model_id: str,
        raw: dict,
        spec: Any,
        logprobs: bool,
        *,
        cached: bool,
        latency_ms: float = 0.0,
    ) -> ModelResponse:
        choice = raw["choices"][0]
        text: str = choice["message"]["content"] or ""

        # Usage
        raw_usage = raw.get("usage") or {}
        usage = Usage(
            prompt_tokens=raw_usage.get("prompt_tokens", 0),
            completion_tokens=raw_usage.get("completion_tokens", 0),
            total_tokens=raw_usage.get("total_tokens", 0),
        )

        # Cost — registry is authoritative
        cost = self._registry.cost(model_id, usage)

        # Logprobs
        token_logprobs: list[TokenLogprob] | None = None
        if logprobs and spec.supports_logprobs:
            lp_data = choice.get("logprobs") or {}
            content_lps = lp_data.get("content") or []
            if content_lps:
                token_logprobs = [
                    TokenLogprob(
                        token=entry.get("token", ""),
                        logprob=entry.get("logprob", 0.0),
                    )
                    for entry in content_lps
                ]

        return ModelResponse(
            model_id=model_id,
            text=text,
            usage=usage,
            cost=cost,
            latency_ms=latency_ms,
            logprobs=token_logprobs,
            raw=raw,
            cached=cached,
        )

    async def aclose(self) -> None:
        """Close the underlying httpx client if we own it."""
        if self._own_client and self._client is not None and _HTTPX_AVAILABLE:
            await self._client.aclose()
