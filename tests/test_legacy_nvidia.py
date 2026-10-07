from __future__ import annotations

import json

import httpx
import pytest

from slm_router.models import ModelRegistry, OpenRouterClient
from slm_router.types import ModelSpec


@pytest.mark.asyncio
async def test_nvidia_delivery_and_existing_provider_isolation(monkeypatch):
    registry = ModelRegistry.from_yaml("configs/models.yaml")
    extra = {"tier": "SLM", "price_in_per_m": 1.0, "price_out_per_m": 2.0, "context_len": 4096}
    registry = ModelRegistry(registry.all() + [
        ModelSpec(id="openrouter-test", name="or", provider="openrouter", **extra),
        ModelSpec(id="google-test", name="google", provider="google", **extra),
    ])
    requests = []

    def handle(request):
        requests.append(request)
        if request.url.host == "generativelanguage.googleapis.com":
            return httpx.Response(200, json={
                "candidates": [{"content": {"parts": [{"text": "google answer"}]}}],
                "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2, "totalTokenCount": 6},
            })
        return httpx.Response(200, json={
            "model": "served-upstream-model",
            "choices": [{"message": {"content": "def add(a, b): return a + b"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18},
        })

    async_client = httpx.AsyncClient

    def create_client(**kwargs):
        return async_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", create_client)
    client = OpenRouterClient(
        "dummy-openrouter-key", registry, google_api_key="dummy-google-key",
        nvidia_api_key="dummy-nvidia-key",
    )
    try:
        answer = await client.complete(
            "openai/gpt-oss-20b", [{"role": "user", "content": "Write add"}],
            max_tokens=128, temperature=0.2, use_cache=False,
        )
        assert answer.text == "def add(a, b): return a + b"
        assert answer.model_id == "openai/gpt-oss-20b"
        assert answer.raw["model"] == "served-upstream-model"
        assert answer.usage.model_dump() == {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18}
        assert answer.cost.total_usd == 0.0
        request = requests[0]
        assert str(request.url) == "https://integrate.api.nvidia.com/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer dummy-nvidia-key"
        assert "x-goog-api-key" not in request.headers
        assert json.loads(request.content) == {
            "model": "openai/gpt-oss-20b", "messages": [{"role": "user", "content": "Write add"}],
            "temperature": 0.2, "max_tokens": 128,
        }
        openrouter = await client.complete("openrouter-test", [], use_cache=False)
        assert openrouter.text == "def add(a, b): return a + b"
        assert str(requests[1].url) == "https://openrouter.ai/api/v1/chat/completions"
        assert requests[1].headers["authorization"] == "Bearer dummy-openrouter-key"
        assert "dummy-nvidia-key" not in str(requests[1].headers)
        google = await client.complete("google-test", [], use_cache=False)
        assert google.text == "google answer"
        assert google.usage.total_tokens == 6
        assert str(requests[2].url) == "https://generativelanguage.googleapis.com/v1beta/models/google-test:generateContent"
        assert requests[2].headers["x-goog-api-key"] == "dummy-google-key"
        assert "dummy-nvidia-key" not in str(requests[2].headers)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_nvidia_requires_own_key_and_does_not_expose_error_body(caplog):
    registry = ModelRegistry.from_yaml("configs/models.yaml")
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(401, text="sensitive-provider-body dummy-nvidia-key")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http_client:
        client = OpenRouterClient("dummy-openrouter-key", registry, client=http_client)
        with pytest.raises(RuntimeError, match="NVIDIA_API_KEY is required"):
            await client.complete("openai/gpt-oss-20b", [], use_cache=False)
        assert requests == []
        client = OpenRouterClient(
            "", registry, client=http_client, nvidia_api_key="dummy-nvidia-key",
        )
        with pytest.raises(httpx.HTTPStatusError) as caught:
            await client.complete("openai/gpt-oss-20b", [], use_cache=False)
        assert len(requests) == 1
        assert "sensitive-provider-body" not in str(caught.value)
        assert "dummy-nvidia-key" not in str(caught.value)
        assert "dummy-nvidia-key" not in caplog.text
