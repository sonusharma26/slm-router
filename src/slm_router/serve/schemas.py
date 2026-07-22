"""Request / response schemas for the serving layer."""

from __future__ import annotations

from pydantic import BaseModel


class RouteRequest(BaseModel):
    """Incoming routing request."""

    query: str
    system: str | None = None
    max_tokens: int | None = None


class RouteResponse(BaseModel):
    """Response returned by POST /route."""

    answer: str
    model_used: str
    cost_usd: float
    confidence: float
    latency_ms: float
    tier: str
