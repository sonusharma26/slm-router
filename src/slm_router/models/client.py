"""Provider-aware async model client with caching, rate-limiting, and retries."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel

from slm_router.types import (
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
GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


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
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=8),
        reraise=True,
    )


_RETRY = _build_retry_decorator()


class OpenRouterClient:
    """Async client for OpenRouter, Google Gemini, and NVIDIA Build.

    The historical class name is retained to avoid breaking callers.
    """

    def __init__(
        self,
        api_key: str,
        registry: Any,
        google_api_key: str = "",
        cache: Any | None = None,
        limiter: Any | None = None,
        client: Any | None = None,
        timeout: float = 60,
        nvidia_api_key: str = "",
    ) -> None:
        self._api_key = api_key
        self._google_api_key = google_api_key
        self._nvidia_api_key = nvidia_api_key
        self._registry = registry
        self._cache = cache
        self._limiter = limiter
        self._timeout = timeout
        self._own_client = client is None
        if client is not None:
            self._client = client
            self._google_client = client
            self._nvidia_client = client
        elif _HTTPX_AVAILABLE:
            self._client = httpx.AsyncClient(
                base_url=BASE_URL,
                timeout=httpx.Timeout(timeout),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            self._google_client = httpx.AsyncClient(
                base_url=GOOGLE_BASE_URL,
                timeout=httpx.Timeout(timeout),
                headers={
                    "x-goog-api-key": google_api_key,
                    "Content-Type": "application/json",
                },
            )
            self._nvidia_client = httpx.AsyncClient(
                base_url=NVIDIA_BASE_URL,
                timeout=httpx.Timeout(timeout),
                headers={"Content-Type": "application/json"},
            )
        else:
            self._client = None
            self._google_client = None
            self._nvidia_client = None

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

    async def _post_nvidia(self, payload: dict) -> dict:
        resp = await self._nvidia_client.post(
            f"{NVIDIA_BASE_URL}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self._nvidia_api_key}"},
        )
        resp.raise_for_status()
        return resp.json()

    async def _post_nvidia_with_retry(self, payload: dict) -> dict:
        if _TENACITY_AVAILABLE and _RETRY is not None:
            return await _RETRY(self._post_nvidia)(payload)
        return await self._post_nvidia(payload)

    async def _post_google(self, model_id: str, payload: dict) -> dict:
        """POST to Gemini generateContent, raising on HTTP errors."""
        resp = await self._google_client.post(
            f"/models/{model_id}:generateContent", json=payload
        )
        resp.raise_for_status()
        return resp.json()

    async def _post_google_with_retry(self, model_id: str, payload: dict) -> dict:
        async def post(payload_: dict) -> dict:
            return await self._post_google(model_id, payload_)

        if _TENACITY_AVAILABLE and _RETRY is not None:
            return await _RETRY(post)(payload)
        return await post(payload)

    @staticmethod
    def _google_payload(
        messages: list[dict],
        *,
        model_id: str,
        temperature: float,
        max_tokens: int | None,
    ) -> dict:
        """Translate OpenAI-style chat messages to Gemini generateContent."""
        contents: list[dict] = []
        system_parts: list[dict] = []
        for message in messages:
            role = str(message.get("role", "user"))
            part = {"text": str(message.get("content", ""))}
            if role == "system":
                system_parts.append(part)
                continue
            contents.append(
                {
                    "role": "model" if role == "assistant" else "user",
                    "parts": [part],
                }
            )

        payload: dict[str, Any] = {"contents": contents}
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}

        generation_config: dict[str, Any] = {}
        # Gemini 3.6 deprecates sampling parameters; do not send temperature.
        if model_id != "gemini-3.6-flash":
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens
        if generation_config:
            payload["generationConfig"] = generation_config
        return payload

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
        if spec.provider == "google" and not self._google_api_key:
            raise RuntimeError(
                f"GOOGLE_API_KEY is required for Google model {model_id!r}"
            )
        if spec.provider == "openrouter" and not self._api_key:
            raise RuntimeError(
                f"OPENROUTER_API_KEY is required for OpenRouter model {model_id!r}"
            )
        if spec.provider == "nvidia-build" and not self._nvidia_api_key:
            raise RuntimeError(
                f"NVIDIA_API_KEY is required for NVIDIA Build model {model_id!r}"
            )

        # ---- cache lookup ----
        cache_key = self._make_cache_key(
            model_id, dict_messages, temperature, max_tokens, logprobs
        )
        if use_cache and self._cache is not None:
            cached_value = self._cache.get(cache_key)
            if cached_value is not None:
                cached_raw = cached_value.get("response")
                cached_latency = cached_value.get("provider_latency_ms")
                if not isinstance(cached_raw, dict) or not isinstance(
                    cached_latency, (int, float)
                ):
                    # Cache v2 keys should only contain envelopes. Treat any
                    # malformed entry as a miss instead of recording fake 0 ms.
                    cached_raw = None
                if cached_raw is not None:
                    return self._parse_response(
                        model_id,
                        cached_raw,
                        spec,
                        logprobs,
                        cached=True,
                        latency_ms=float(cached_latency),
                    )

        # ---- rate-limit ----
        if self._limiter is not None:
            await self._limiter.acquire()

        # ---- build payload ----
        want_logprobs = logprobs and spec.supports_logprobs
        if spec.provider == "google":
            payload = self._google_payload(
                dict_messages,
                model_id=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            payload = {
                "model": model_id,
                "messages": dict_messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            if want_logprobs:
                payload["logprobs"] = True
                payload["top_logprobs"] = 1

        # ---- HTTP call ----
        t0 = time.perf_counter()
        if spec.provider == "google":
            raw = await self._post_google_with_retry(model_id, payload)
        elif spec.provider == "nvidia-build":
            raw = await self._post_nvidia_with_retry(payload)
        else:
            raw = await self._post_with_retry(payload)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # ---- store cache ----
        if use_cache and self._cache is not None:
            self._cache.set(
                cache_key,
                {
                    "response": raw,
                    "provider_latency_ms": latency_ms,
                },
            )

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
        if spec.provider == "google":
            return self._parse_google_response(
                model_id, raw, cached=cached, latency_ms=latency_ms
            )

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

    def _parse_google_response(
        self,
        model_id: str,
        raw: dict,
        *,
        cached: bool,
        latency_ms: float,
    ) -> ModelResponse:
        candidates = raw.get("candidates") or []
        if not candidates:
            feedback = raw.get("promptFeedback") or {}
            raise ValueError(
                f"Gemini returned no candidates for {model_id!r}: {feedback}"
            )
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(str(part.get("text", "")) for part in parts)
        raw_usage = raw.get("usageMetadata") or {}
        usage = Usage(
            prompt_tokens=raw_usage.get("promptTokenCount", 0),
            completion_tokens=raw_usage.get("candidatesTokenCount", 0),
            total_tokens=raw_usage.get("totalTokenCount", 0),
        )
        return ModelResponse(
            model_id=model_id,
            text=text,
            usage=usage,
            cost=self._registry.cost(model_id, usage),
            latency_ms=latency_ms,
            raw=raw,
            cached=cached,
        )

    async def aclose(self) -> None:
        """Close the underlying httpx client if we own it."""
        if self._own_client and self._client is not None and _HTTPX_AVAILABLE:
            await self._client.aclose()
            await self._google_client.aclose()
            await self._nvidia_client.aclose()
