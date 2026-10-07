"""Two provider seams, no gateway reimplementation and no implicit retries."""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Protocol
import httpx


class ProviderTimeout(TimeoutError):
    pass


@dataclass(frozen=True)
class AdapterResponse:
    text: str | None
    provider_request_id: str
    revision: str
    input_tokens: int
    output_tokens: int
    spend: float | None
    latency_ms: float
    raw: dict[str, Any]
    time_to_first_token_ms: float | None = None
    usage_known: bool = True
    observed_model: str | None = None
    observed_revision: str | None = None


class ProviderAdapter(Protocol):
    def call(self, endpoint_id: str, messages: list[dict], config: dict[str, Any]) -> AdapterResponse: ...


class MockAdapter:
    def __init__(self, handler: Callable): self.handler = handler
    def call(self, endpoint_id, messages, config): return self.handler(endpoint_id, messages, config)


class OpenAICompatibleAdapter:
    def __init__(self, base_url: str, api_key: str, client: httpx.Client | None = None, *, token_limit_field: str = "max_completion_tokens"):
        url = httpx.URL(base_url)
        if url.scheme not in {"https", "http"} or not url.host or url.username or url.password:
            raise ValueError("base URL must be an operator-configured HTTP(S) endpoint without credentials")
        if token_limit_field not in {"max_tokens","max_completion_tokens"}:
            raise ValueError("unsupported token limit field")
        self.token_limit_field = token_limit_field
        self.base_url, self.api_key = base_url.rstrip("/"), api_key
        self._owns_client = client is None
        self.client = client or httpx.Client(timeout=60, follow_redirects=False)

    def call(self, endpoint_id: str, messages: list[dict], config: dict[str, Any]) -> AdapterResponse:
        config = dict(config)
        if any(k in config for k in ("model", "messages", "api_key", "api_base", "base_url", "num_retries", "max_retries")):
            raise ValueError("generation config cannot replace transport or model identity")
        if config.pop("stream", False):
            raise ValueError("streaming is not implemented; no provider call was made")
        if self.token_limit_field == "max_tokens" and "max_completion_tokens" in config:
            config["max_tokens"] = config.pop("max_completion_tokens")
        timeout = float(config.pop("timeout", 60))
        started = perf_counter()
        try:
            response = self.client.post(f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model":endpoint_id,"messages":messages,**config,"stream":False}, timeout=timeout)
        except httpx.TimeoutException as exc:
            raise ProviderTimeout("provider transport timed out") from exc
        response.raise_for_status()
        data = response.json()
        latency = (perf_counter()-started)*1000
        usage = data.get("usage") or {}
        message = data["choices"][0]["message"]
        revision = response.headers.get("x-model-revision") or data.get("model_revision")
        raw = {**data, "_response_headers": {k:v for k,v in response.headers.items()
                if k in {"x-request-id","x-model-revision"}}}
        return AdapterResponse(
            message.get("content"), response.headers.get("x-request-id",data.get("id","unknown")),
            str(revision or data.get("model",endpoint_id)),
            int(usage.get("prompt_tokens",0)), int(usage.get("completion_tokens",0)),
            None, latency, raw, None,
            "prompt_tokens" in usage and "completion_tokens" in usage,
            data.get("model"), str(revision) if revision is not None else None)

    def close(self):
        if self._owns_client: self.client.close()
