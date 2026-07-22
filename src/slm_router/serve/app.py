"""FastAPI application: SLM Meta-Router."""

from __future__ import annotations

import math
from contextlib import asynccontextmanager
from typing import Any

try:
    from fastapi import FastAPI, HTTPException  # type: ignore
    _FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore
    HTTPException = None  # type: ignore
    _FASTAPI_AVAILABLE = False

from slm_router.models.cache import ResponseCache
from slm_router.models.client import OpenRouterClient
from slm_router.models.registry import ModelRegistry
from slm_router.models.ratelimit import RateLimiter
from slm_router.serve.schemas import RouteRequest, RouteResponse
from slm_router.settings import load_settings
from slm_router.types import ModelResponse

# Try to import routing policy from Track C; degrade gracefully.
try:
    from slm_router.ml_core.routing import get_policy
    _ROUTING_AVAILABLE = True
except ImportError:
    _ROUTING_AVAILABLE = False


# ---------------------------------------------------------------------------
# Confidence heuristic
# ---------------------------------------------------------------------------

def _confidence(response: ModelResponse) -> float:
    """Return a simple confidence estimate for a model response.

    Uses mean log-probability if logprobs are present; otherwise returns 0.5.
    """
    if response.logprobs:
        mean_lp = sum(t.logprob for t in response.logprobs) / len(response.logprobs)
        # Convert log-prob to [0, 1] via exp; clamp to [0, 1].
        return float(min(1.0, max(0.0, math.exp(mean_lp))))
    return 0.5


# ---------------------------------------------------------------------------
# CheapestFirstPolicy (v0 fallback)
# ---------------------------------------------------------------------------

class CheapestFirstPolicy:
    """Pick the cheapest model from candidate_models or SLM tier."""

    def __init__(self, registry: ModelRegistry, candidate_models: list[str]) -> None:
        self._registry = registry
        self._candidates = candidate_models

    def pick(self) -> str:
        """Return the model_id of the cheapest available candidate."""
        if self._candidates:
            # Sort by total price (in + out) ascending.
            def _price(mid: str) -> float:
                try:
                    s = self._registry.get(mid)
                    return s.price_in_per_m + s.price_out_per_m
                except KeyError:
                    return float("inf")

            return min(self._candidates, key=_price)

        slm_models = self._registry.by_tier("SLM")
        if slm_models:
            return slm_models[0].id
        all_models = self._registry.all()
        if all_models:
            return sorted(
                all_models,
                key=lambda s: s.price_in_per_m + s.price_out_per_m,
            )[0].id
        raise RuntimeError("ModelRegistry is empty — cannot pick a model.")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: Any):  # type: ignore[valid-type]
    settings = load_settings()

    registry = ModelRegistry.from_yaml(settings.models_path)
    cache = ResponseCache(settings.cache_dir)
    limiter = RateLimiter(settings.requests_per_second)
    client = OpenRouterClient(
        api_key=settings.openrouter_api_key,
        registry=registry,
        cache=cache,
        limiter=limiter,
    )

    # Load AppConfig for candidate_models if config_path is accessible.
    candidate_models: list[str] = []
    try:
        from slm_router.config import AppConfig
        cfg = AppConfig.from_yaml(settings.config_path)
        candidate_models = cfg.candidate_models
    except Exception:
        pass

    if _ROUTING_AVAILABLE:
        try:
            policy = get_policy(registry=registry, candidate_models=candidate_models)
        except Exception:
            policy = CheapestFirstPolicy(registry, candidate_models)
    else:
        policy = CheapestFirstPolicy(registry, candidate_models)

    app.state.registry = registry
    app.state.client = client
    app.state.policy = policy

    yield

    await client.aclose()
    cache.close()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

if _FASTAPI_AVAILABLE and FastAPI is not None:
    app = FastAPI(title="SLM Meta-Router", lifespan=_lifespan)

    @app.get("/healthz")
    async def healthz() -> dict:
        """Liveness probe."""
        return {"status": "ok"}

    @app.post("/route", response_model=RouteResponse)
    async def route(request: RouteRequest, _app: Any = None) -> RouteResponse:
        from fastapi import Request  # noqa: PLC0415 (local import is fine here)
        # Access app.state via the global app object.
        registry: ModelRegistry = app.state.registry
        client: OpenRouterClient = app.state.client
        policy: Any = app.state.policy

        # Pick model
        model_id: str = policy.pick()
        spec = registry.get(model_id)

        # Build messages
        messages: list[dict] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.query})

        try:
            response: ModelResponse = await client.complete(
                model_id,
                messages,
                max_tokens=request.max_tokens,
                logprobs=spec.supports_logprobs,
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        confidence = _confidence(response)

        return RouteResponse(
            answer=response.text,
            model_used=model_id,
            cost_usd=response.cost.total_usd,
            confidence=confidence,
            latency_ms=response.latency_ms,
            tier=spec.tier,
        )

else:
    # Stub so the module is importable even without FastAPI.
    app = None  # type: ignore
