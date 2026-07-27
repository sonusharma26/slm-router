"""CLI entry-point for the SLM meta-router.

Registered in pyproject.toml as:
    [project.scripts]
    slm = "slm_router.cli:app"
"""

from __future__ import annotations

from pathlib import Path

try:
    import typer  # type: ignore
    _TYPER_AVAILABLE = True
except ImportError:
    typer = None  # type: ignore
    _TYPER_AVAILABLE = False

if _TYPER_AVAILABLE and typer is not None:
    app = typer.Typer(name="slm", help="SLM Meta-Router CLI")

    @app.command()
    def serve(
        host: str = typer.Option("127.0.0.1", help="Bind host"),
        port: int = typer.Option(8000, help="Bind port"),
        reload: bool = typer.Option(False, help="Enable auto-reload (dev only)"),
    ) -> None:
        """Start the FastAPI serving process via uvicorn."""
        try:
            import uvicorn  # type: ignore
        except ImportError:
            typer.echo("uvicorn is not installed. Run: pip install uvicorn", err=True)
            raise typer.Exit(1)

        uvicorn.run(
            "slm_router.serve.app:app",
            host=host,
            port=port,
            reload=reload,
        )

    def _bootstrap(config_path: str):
        """Build Settings, AppConfig, ModelRegistry, TraceStore, and a client."""
        from slm_router.config import AppConfig
        from slm_router.models import ModelRegistry, OpenRouterClient, ResponseCache
        from slm_router.models.ratelimit import RateLimiter
        from slm_router.settings import load_settings
        from slm_router.trace import TraceStore

        settings = load_settings()
        cfg = AppConfig.from_yaml(config_path)
        registry = ModelRegistry.from_yaml(settings.models_path)
        store = TraceStore(f"{cfg.paths.traces_dir}/router.db")
        cache = ResponseCache(settings.cache_dir)
        # OpenRouter free-tier models (":free" suffix) enforce a strict ~20
        # req/min cap; pace well under that so retries don't just re-429.
        limiter = RateLimiter(requests_per_second=0.15)
        client = OpenRouterClient(
            settings.openrouter_api_key,
            registry,
            google_api_key=settings.google_api_key,
            cache=cache,
            limiter=limiter,
        )
        return settings, cfg, registry, store, client

    @app.command()
    def collect_traces(
        config: str = typer.Option("configs/config.yaml", help="Path to config YAML"),
        run_kind: str = typer.Option(
            "live", help="Trace run_kind to collect: live, router_eval, or oracle"
        ),
    ) -> None:
        """Collect logged inference traces from the trace store (phase 2)."""
        _settings, _cfg, _registry, store, _client = _bootstrap(config)
        traces = store.query(run_kind=run_kind)
        typer.echo(
            f"collect_traces: found {len(traces)} '{run_kind}' trace(s) in {store.path}"
        )

    @app.command()
    def build_oracle(
        config: str = typer.Option("configs/config.yaml", help="Path to config YAML"),
        limit: int = typer.Option(None, help="Max items per dataset (overrides config)"),
        confirm: bool = typer.Option(True, help="Prompt before incurring API cost"),
        dataset: str = typer.Option(
            None, help="Restrict to a single dataset name (overrides config datasets list)"
        ),
        refresh: bool = typer.Option(
            False,
            help=(
                "Re-call and replace existing oracle cells, bypassing response cache "
                "(incurs API usage)"
            ),
        ),
    ) -> None:
        """Run every candidate model on every eval item to build the oracle matrix."""
        import asyncio

        _settings, cfg, registry, store, client = _bootstrap(config)
        asyncio.run(
            __import__(
                "slm_router.eval.workflows.build_oracle", fromlist=["build_oracle"]
            ).build_oracle(
                client, registry, store, cfg,
                datasets=[dataset] if dataset else cfg.datasets,
                limit=limit if limit is not None else cfg.dataset_limit,
                confirm=confirm,
                concurrency=1,
                refresh=refresh,
            )
        )

    @app.command()
    def train_router(
        config: str = typer.Option("configs/config.yaml", help="Path to config YAML"),
        context_dim: int = typer.Option(9, help="Context feature vector dimensionality"),
        margin: float = typer.Option(
            0.0, help="Minimum FQE improvement over the incumbent required to promote"
        ),
    ) -> None:
        """Train the learned routing policy from the oracle matrix (phase 4)."""
        from slm_router.eval.datasets import load_dataset_items
        from slm_router.eval.workflows.build_oracle import config_hash
        from slm_router.feedback.loop import SelfImprovementLoop
        from slm_router.ml_core.features import DifficultyFeaturizer, Embedder
        from slm_router.ml_core.routing import get_policy
        from slm_router.ml_core.routing.trace_adapter import (
            SPLIT_VERSION,
            build_training_samples,
            select_holdout_items,
        )

        _settings, cfg, registry, store, _client = _bootstrap(config)
        candidate_models = list(cfg.candidate_models)
        n_arms = len(candidate_models) or len(registry.all())

        cfg_hash = config_hash(cfg)
        oracle_traces = [
            t for t in store.query(run_kind="oracle") if t.config_hash == cfg_hash
        ]
        if not oracle_traces:
            typer.echo("train_router: no oracle traces found for current config.")
            raise typer.Exit(1)

        seed = getattr(cfg, "dataset_seed", 13)
        holdout_fraction = cfg.router_holdout_fraction
        usable_models: dict[str, set[str]] = {}
        for trace in oracle_traces:
            if not trace.error and trace.model in candidate_models:
                usable_models.setdefault(trace.item_id, set()).add(trace.model)
        complete_items = {
            item_id
            for item_id, models in usable_models.items()
            if models == set(candidate_models)
        }
        holdout_items = select_holdout_items(
            complete_items, seed, holdout_fraction
        )
        training_items = complete_items - holdout_items
        training_traces = [
            trace for trace in oracle_traces if trace.item_id in training_items
        ]
        if not holdout_items:
            successful_by_model = {
                model_id: sum(
                    1
                    for trace in oracle_traces
                    if trace.model == model_id and not trace.error
                )
                for model_id in candidate_models
            }
            typer.echo(
                "train_router: need at least 2 complete oracle items "
                f"(found {len(complete_items)}). Successful rows by model: "
                f"{successful_by_model}. Re-run build-oracle without --refresh "
                "to retry only failed/missing cells."
            )
            raise typer.Exit(1)
        item_texts: dict[str, str] = {}
        for ds_name in cfg.datasets:
            try:
                for it in load_dataset_items(
                    ds_name, split="test", limit=cfg.dataset_limit, seed=seed
                ):
                    item_texts[it.item_id] = it.query
            except Exception as exc:
                typer.echo(f"train_router: WARNING could not reload '{ds_name}': {exc}")

        featurizer = DifficultyFeaturizer(Embedder())
        samples = build_training_samples(
            training_traces, candidate_models, featurizer, item_texts
        )
        if not samples:
            typer.echo("train_router: no usable samples after feature extraction.")
            raise typer.Exit(1)
        feature_dim = len(samples[0].context)
        if context_dim != feature_dim:
            typer.echo(
                f"train_router: context_dim={context_dim} does not match "
                f"the featurizer output dimension ({feature_dim})."
            )
            raise typer.Exit(1)

        policy_kwargs = dict(
            n_arms=n_arms,
            dim=context_dim,
            lam=cfg.reward.lambda_cost,
            beta=cfg.reward.beta_latency,
            correctness_weight=cfg.reward.correctness_weight,
        )
        policy_path = Path(cfg.paths.models_dir) / "policy.joblib"
        incumbent = None
        if policy_path.exists():
            try:
                incumbent = get_policy("offline_rl", **policy_kwargs)
                incumbent.load(policy_path)
                saved_candidates = getattr(incumbent, "candidate_models", None)
                if saved_candidates != candidate_models:
                    raise ValueError(
                        "saved candidate model order does not match the current config"
                    )
                if (
                    getattr(incumbent, "split_seed", None) != seed
                    or getattr(incumbent, "holdout_fraction", None)
                    != holdout_fraction
                    or getattr(incumbent, "split_version", None) != SPLIT_VERSION
                ):
                    raise ValueError(
                        "saved train/holdout split does not match the current config"
                    )
            except Exception as exc:
                typer.echo(
                    f"train_router: WARNING could not use incumbent "
                    f"'{policy_path}': {exc}"
                )
                incumbent = None
        candidate = get_policy("offline_rl", **policy_kwargs)

        loop = SelfImprovementLoop(store, incumbent, margin=margin)
        result = loop.retrain_batch(candidate, traces=samples)
        if result["promoted"]:
            candidate.candidate_models = candidate_models
            candidate.split_seed = seed
            candidate.holdout_fraction = holdout_fraction
            candidate.split_version = SPLIT_VERSION
            candidate.training_item_count = len(
                {trace.item_id for trace in training_traces}
            )
            candidate.save(policy_path)
            result["policy_path"] = str(policy_path)
        typer.echo(
            f"train_router: n_samples={len(samples)} "
            f"holdout_items={len(holdout_items)} {result}"
        )

    @app.command()
    def train_confidence(
        config: str = typer.Option("configs/config.yaml", help="Path to config YAML"),
        quality_threshold: float = typer.Option(
            0.5, help="quality >= threshold counts as correct (score is in [0, 1])"
        ),
    ) -> None:
        """Fit the ConfidencePredictor on oracle traces and save confidence.joblib."""
        import numpy as np

        from slm_router.eval.datasets import load_dataset_items
        from slm_router.eval.workflows.build_oracle import config_hash
        from slm_router.ml_core.confidence.metrics import expected_calibration_error
        from slm_router.ml_core.features import DifficultyFeaturizer, Embedder

        try:
            from slm_router.ml_core.confidence import ConfidencePredictor
        except ImportError as exc:
            typer.echo(f"train_confidence: ConfidencePredictor unavailable: {exc}")
            raise typer.Exit(1)

        _settings, cfg, _registry, store, _client = _bootstrap(config)
        candidate_models = list(cfg.candidate_models)

        cfg_hash = config_hash(cfg)
        oracle_traces = [
            t for t in store.query(run_kind="oracle") if t.config_hash == cfg_hash
        ]
        if not oracle_traces:
            typer.echo("train_confidence: no oracle traces found for current config.")
            raise typer.Exit(1)

        seed = getattr(cfg, "dataset_seed", 13)
        item_texts: dict[str, str] = {}
        for ds_name in cfg.datasets:
            try:
                for it in load_dataset_items(
                    ds_name, split="test", limit=cfg.dataset_limit, seed=seed
                ):
                    item_texts[it.item_id] = it.query
            except Exception as exc:
                typer.echo(
                    f"train_confidence: WARNING could not reload '{ds_name}': {exc}"
                )

        featurizer = DifficultyFeaturizer(Embedder())
        from slm_router.ml_core.routing.trace_adapter import build_training_samples

        samples = build_training_samples(
            oracle_traces, candidate_models, featurizer, item_texts
        )
        if not samples:
            typer.echo("train_confidence: no usable samples after feature extraction.")
            raise typer.Exit(1)

        X = np.stack([s.context for s in samples])
        y = np.array(
            [1.0 if s.quality >= quality_threshold else 0.0 for s in samples],
            dtype=np.float32,
        )

        predictor = ConfidencePredictor()
        predictor.fit(X, y)

        models_dir = Path(cfg.paths.models_dir)
        confidence_path = models_dir / "confidence.joblib"
        predictor.save(confidence_path)

        summary = (
            f"train_confidence: n_examples={len(y)} n_positive={int(y.sum())} "
            f"saved to {confidence_path}"
        )
        if predictor._fitted:
            p_train = predictor._raw_proba(X)
            ece = expected_calibration_error(p_train, y)
            summary += f" train_ece={ece:.4f}"
        typer.echo(summary)

    @app.command()
    def evaluate(
        config: str = typer.Option("configs/config.yaml", help="Path to config YAML"),
        dataset: str = typer.Option(None, help="Restrict to one dataset"),
    ) -> None:
        """Evaluate routing policies vs the oracle matrix (regret, ECE, Pareto)."""
        from slm_router.eval.workflows.eval_router import eval_router

        _settings, cfg, registry, store, _client = _bootstrap(config)
        eval_router(store, registry, cfg, dataset=dataset)

else:
    # Minimal stub when typer is not installed.
    class _StubApp:  # type: ignore
        def __call__(self, *args, **kwargs):
            print("typer is not installed; CLI unavailable.")

    app = _StubApp()  # type: ignore
