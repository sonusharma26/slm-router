"""Provider adapters; gateway behavior remains outside the control plane (V2-201)."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Protocol
import httpx


@dataclass(frozen=True)
class AdapterResponse:
    text: str
    provider_request_id: str
    revision: str
    input_tokens: int
    output_tokens: int
    spend: float
    latency_ms: float
    raw: dict[str, object]


class ProviderAdapter(Protocol):
    def call(
        self, endpoint_id: str, messages: list[dict[str, str]], config: dict[str, object]
    ) -> AdapterResponse: ...


class MockAdapter:
    def __init__(
        self, handler: Callable[[str, list[dict[str, str]], dict[str, object]], AdapterResponse]
    ):
        self.handler = handler

    def call(
        self, endpoint_id: str, messages: list[dict[str, str]], config: dict[str, object]
    ) -> AdapterResponse:
        return self.handler(endpoint_id, messages, config)


class OpenAICompatibleAdapter:
    def __init__(self, base_url: str, api_key: str, client: httpx.Client | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.client = client or httpx.Client(timeout=60)

    def call(
        self, endpoint_id: str, messages: list[dict[str, str]], config: dict[str, object]
    ) -> AdapterResponse:
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": endpoint_id, "messages": messages, **config},
        )
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage", {})
        return AdapterResponse(
            data["choices"][0]["message"]["content"],
            response.headers.get("x-request-id", data.get("id", "unknown")),
            data.get("model", endpoint_id),
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)),
            float(data.get("spend", 0)),
            float(data.get("latency_ms", 0)),
            data,
        )


class LiteLLMAdapter:
    def __init__(self, completion: Callable[..., object]):
        self.completion = completion

    def call(
        self, endpoint_id: str, messages: list[dict[str, str]], config: dict[str, object]
    ) -> AdapterResponse:
        result = self.completion(model=endpoint_id, messages=messages, **config)
        usage = result.usage
        return AdapterResponse(
            result.choices[0].message.content,
            str(getattr(result, "id", "unknown")),
            str(getattr(result, "model", endpoint_id)),
            int(usage.prompt_tokens),
            int(usage.completion_tokens),
            float(getattr(result, "_hidden_params", {}).get("response_cost", 0)),
            0,
            (
                result.model_dump()
                if hasattr(result, "model_dump")
                else vars(result)
                if hasattr(result, "__dict__")
                else dict(result)
            ),
        )
