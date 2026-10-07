"""Shared data contracts used across every subsystem.

Kept dependency-light (pydantic + stdlib) so models/, ml_core/, eval/, trace/
can all import from here without circular deps.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Tier = Literal["SLM", "small", "frontier"]
Provider = Literal["openrouter", "google", "nvidia-build"]


class Query(BaseModel):
    """A single user query flowing into the meta-router."""

    id: str
    text: str
    context: list[str] | None = None  # retrieved docs, for ambiguity features
    domain: str | None = None


class ModelSpec(BaseModel):
    """Static description of a candidate model (from models.yaml)."""

    id: str
    name: str
    tier: Tier
    provider: Provider = "openrouter"
    price_in_per_m: float
    price_out_per_m: float
    context_len: int
    supports_logprobs: bool = False
    p50_latency_ms: float = 0.0  # rolling estimate, updated from traces


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CostBreakdown(BaseModel):
    input_usd: float = 0.0
    output_usd: float = 0.0
    total_usd: float = 0.0


class TokenLogprob(BaseModel):
    token: str
    logprob: float


class ModelResponse(BaseModel):
    """Result of one model call, with everything reward math needs."""

    model_id: str
    text: str
    usage: Usage = Field(default_factory=Usage)
    cost: CostBreakdown = Field(default_factory=CostBreakdown)
    latency_ms: float = 0.0
    logprobs: list[TokenLogprob] | None = None
    raw: dict = Field(default_factory=dict)
    cached: bool = False


class ConfidenceOut(BaseModel):
    """Output of the confidence engine for one (query, model)."""

    p_correct: float
    epistemic_var: float = 0.0
    conformal_accept: bool = True
    lower_bound: float = 0.0


class RouteDecision(BaseModel):
    """A routing decision, logged for attribution and off-policy eval."""

    query_id: str
    chosen_model: str
    difficulty: float = 0.5
    predicted_confidence: dict[str, float] = Field(default_factory=dict)
    expected_reward: dict[str, float] = Field(default_factory=dict)
    policy_version: str = "v0"
    explore: bool = False
