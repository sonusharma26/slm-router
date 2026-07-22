"""CLI entry-point for the SLM meta-router.

Registered in pyproject.toml as:
    [project.scripts]
    slm = "slm_router.cli:app"
"""

from __future__ import annotations

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
        from slm_router.settings import load_settings
        from slm_router.trace import TraceStore

        settings = load_settings()
        cfg = AppConfig.from_yaml(config_path)
        registry = ModelRegistry.from_yaml(settings.models_path)
        store = TraceStore(f"{cfg.paths.traces_dir}/router.db")
        cache = ResponseCache(settings.cache_dir)
        client = OpenRouterClient(settings.openrouter_api_key, registry, cache=cache)
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
    ) -> None:
        """Run every candidate model on every eval item to build the oracle matrix."""
        import asyncio

        _settings, cfg, registry, store, client = _bootstrap(config)
        asyncio.run(
            __import__(
                "slm_router.eval.workflows.build_oracle", fromlist=["build_oracle"]
            ).build_oracle(
                client, registry, store, cfg,
                datasets=cfg.datasets,
                limit=limit if limit is not None else cfg.dataset_limit,
                confirm=confirm,
            )
        )

    @app.command()
    def train_router(
        config: str = typer.Option("configs/config.yaml", help="Path to config YAML"),
        context_dim: int = typer.Option(8, help="Context feature vector dimensionality"),
        margin: float = typer.Option(
            0.0, help="Minimum FQE improvement over the incumbent required to promote"
        ),
    ) -> None:
        """Train the learned routing policy from traces (phase 4)."""
        from slm_router.feedback.loop import SelfImprovementLoop
        from slm_router.ml_core.routing import get_policy

        _settings, cfg, registry, store, _client = _bootstrap(config)
        n_arms = len(cfg.candidate_models) or len(registry.all())

        policy_kwargs = dict(
            n_arms=n_arms,
            dim=context_dim,
            lam=cfg.reward.lambda_cost,
            beta=cfg.reward.beta_latency,
            correctness_weight=cfg.reward.correctness_weight,
        )
        incumbent = get_policy("offline_rl", **policy_kwargs)
        candidate = get_policy("offline_rl", **policy_kwargs)

        loop = SelfImprovementLoop(store, incumbent, margin=margin)
        result = loop.retrain_batch(candidate)
        typer.echo(f"train_router: {result}")

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
