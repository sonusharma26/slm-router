"""FastAPI application: SLM Meta-Router."""

from __future__ import annotations

import logging
import math
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

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
from slm_router.types import ModelResponse, Query

logger = logging.getLogger("uvicorn.error")

# Try to import routing policy from Track C; degrade gracefully.
try:
    from slm_router.ml_core.routing import get_policy
    _ROUTING_AVAILABLE = True
except ImportError:
    _ROUTING_AVAILABLE = False

# Try to import the calibrated confidence engine; degrade gracefully.
try:
    from slm_router.ml_core.confidence import ConfidencePredictor
    _CONFIDENCE_AVAILABLE = True
except ImportError:
    ConfidencePredictor = None  # type: ignore
    _CONFIDENCE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Confidence estimation
# ---------------------------------------------------------------------------

# Neutral prior used only when neither a calibrated predictor nor logprobs
# are available for a response (e.g. free models with no logprob support and
# no fitted ConfidencePredictor artifact on disk). Documented constant so it
# is never confused with a real calibrated estimate.
_NEUTRAL_CONFIDENCE_PRIOR = 0.5


def _confidence(
    response: ModelResponse,
    predictor: Any | None,
    feats: Any | None,
) -> float:
    """Return a calibrated confidence estimate for a model response.

    Preferred path: run the fitted :class:`ConfidencePredictor` (temperature
    scaling + conformal calibration over difficulty features and posterior
    signals such as mean log-probability) to get a calibrated p(correct).

    Fallback path (predictor/features unavailable): mean log-probability
    converted to [0, 1] via ``exp`` when logprobs are present, else a named
    neutral prior — never a bare inline magic number.
    """
    raw_signals: dict[str, float] = {}
    if response.logprobs:
        mean_lp = sum(t.logprob for t in response.logprobs) / len(response.logprobs)
        raw_signals["mean_log_prob"] = mean_lp

    if predictor is not None and feats is not None:
        try:
            out = predictor.predict_posterior(feats, raw_signals)
            return float(min(1.0, max(0.0, out.p_correct)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("ConfidencePredictor failed; falling back: %s", exc)

    if "mean_log_prob" in raw_signals:
        return float(min(1.0, max(0.0, math.exp(raw_signals["mean_log_prob"]))))
    return _NEUTRAL_CONFIDENCE_PRIOR


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
        google_api_key=settings.google_api_key,
        cache=cache,
        limiter=limiter,
    )

    # Load AppConfig for candidate models and the promoted-policy location.
    candidate_models: list[str] = []
    policy_path: Path | None = None
    cfg: Any = None
    try:
        from slm_router.config import AppConfig
        cfg = AppConfig.from_yaml(settings.config_path)
        candidate_models = cfg.candidate_models
        policy_path = Path(cfg.paths.models_dir) / "policy.joblib"
    except Exception as exc:
        logger.warning("Could not load router config; using cheapest-first: %s", exc)

    featurizer: Any = None
    candidate_specs: list[Any] = []
    policy_source = "cheapest-first"

    # Build the calibrated confidence predictor (Track ml_core/confidence).
    # Loaded once at startup and reused per-request; falls back to an
    # unfitted (heuristic) predictor or None if unavailable.
    confidence_predictor: Any = None
    confidence_featurizer: Any = None
    if _CONFIDENCE_AVAILABLE:
        try:
            from slm_router.ml_core.features import DifficultyFeaturizer, Embedder

            confidence_featurizer = DifficultyFeaturizer(Embedder())
            confidence_predictor = ConfidencePredictor()
            confidence_path = None
            if cfg is not None:
                confidence_path = Path(cfg.paths.models_dir) / "confidence.joblib"
            if confidence_path is not None and confidence_path.exists():
                confidence_predictor.load(confidence_path)
                logger.info("Loaded calibrated confidence predictor from %s", confidence_path)
            else:
                logger.info(
                    "No fitted confidence predictor artifact found; using "
                    "unfitted ConfidencePredictor (heuristic prior + signal blend)."
                )
        except Exception as exc:
            logger.warning(
                "Could not initialise ConfidencePredictor; confidence will use "
                "the logprob/neutral-prior fallback: %s", exc
            )
            confidence_predictor = None
            confidence_featurizer = None

    if _ROUTING_AVAILABLE:
        try:
            if policy_path is None or not policy_path.exists():
                raise FileNotFoundError(policy_path or "policy.joblib")
            if not candidate_models:
                raise ValueError("candidate_models is empty")

            from slm_router.ml_core.features import DifficultyFeaturizer, Embedder
            from slm_router.ml_core.routing.trace_adapter import SPLIT_VERSION

            candidate_specs = [registry.get(model_id) for model_id in candidate_models]
            policy = get_policy(
                "offline_rl",
                n_arms=len(candidate_models),
                dim=9,
                lam=cfg.reward.lambda_cost,
                beta=cfg.reward.beta_latency,
                correctness_weight=cfg.reward.correctness_weight,
            )
            policy.load(policy_path)
            if getattr(policy, "candidate_models", None) != candidate_models:
                raise ValueError(
                    "saved candidate model order does not match the current config"
                )
            if (
                getattr(policy, "split_seed", None) != cfg.dataset_seed
                or getattr(policy, "holdout_fraction", None)
                != cfg.router_holdout_fraction
                or getattr(policy, "split_version", None) != SPLIT_VERSION
            ):
                raise ValueError(
                    "saved train/holdout split does not match the current config"
                )
            ready, reason = policy.training_readiness()
            if not ready:
                raise ValueError(f"saved policy is not deployable: {reason}")
            featurizer = DifficultyFeaturizer(Embedder())
            feature_dim = len(
                featurizer.vectorize(Query(id="startup-validation", text=""))
            )
            if policy.dim != feature_dim:
                raise ValueError(
                    "saved context dimension does not match the serving "
                    f"featurizer ({policy.dim} != {feature_dim})"
                )
            policy_source = str(policy_path)
            logger.info("Loaded promoted routing policy from %s", policy_path)
        except Exception as exc:
            logger.warning(
                "Could not load promoted routing policy; using cheapest-first: %s", exc
            )
            policy = CheapestFirstPolicy(registry, candidate_models)
    else:
        policy = CheapestFirstPolicy(registry, candidate_models)

    app.state.registry = registry
    app.state.client = client
    app.state.policy = policy
    app.state.featurizer = featurizer
    app.state.candidate_specs = candidate_specs
    app.state.policy_source = policy_source
    app.state.confidence_predictor = confidence_predictor
    app.state.confidence_featurizer = confidence_featurizer

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
        policy = app.state.policy
        return {
            "status": "ok",
            "routing_policy": type(policy).__name__,
            "policy_version": getattr(policy, "policy_version", "unknown"),
            "policy_source": app.state.policy_source,
        }

    @app.post("/route", response_model=RouteResponse)
    async def route(request: RouteRequest, _app: Any = None) -> RouteResponse:
        from fastapi import Request  # noqa: PLC0415 (local import is fine here)
        # Access app.state via the global app object.
        registry: ModelRegistry = app.state.registry
        client: OpenRouterClient = app.state.client
        policy: Any = app.state.policy

        # Learned policies use the same query features and candidate ordering as
        # offline training. The fallback policy retains its no-argument picker.
        confidence_feats: Any = None
        if app.state.featurizer is not None:
            context = app.state.featurizer.vectorize(
                Query(id=str(uuid4()), text=request.query)
            )
            confidence_feats = context
            if len(context) != policy.dim:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Promoted routing policy is incompatible with the "
                        f"serving feature vector ({policy.dim} != {len(context)})."
                    ),
                )
            decision = policy.select(context, app.state.candidate_specs)
            model_id = decision.chosen_model
        else:
            model_id = policy.pick()
            if app.state.confidence_featurizer is not None:
                try:
                    confidence_feats = app.state.confidence_featurizer.vectorize(
                        Query(id=str(uuid4()), text=request.query)
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not compute confidence features: %s", exc)
                    confidence_feats = None
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

        confidence = _confidence(
            response, app.state.confidence_predictor, confidence_feats
        )

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
