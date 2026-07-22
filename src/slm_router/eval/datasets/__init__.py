"""Registry mapping name->loader class for evaluation benchmark datasets."""
from __future__ import annotations

from typing import TYPE_CHECKING

from slm_router.eval.datasets.schema import EvalItem

if TYPE_CHECKING:
    from slm_router.eval.datasets.base import BenchmarkLoader

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type] = {}


def _register_all() -> None:
    """Lazily import and register all known loaders."""
    global _REGISTRY

    loader_map: dict[str, str] = {
        "mmlu": "slm_router.eval.datasets.loaders.mmlu.MMLULoader",
        "gpqa": "slm_router.eval.datasets.loaders.gpqa.GPQALoader",
        "hotpotqa": "slm_router.eval.datasets.loaders.hotpotqa.HotpotQALoader",
        "gsm8k": "slm_router.eval.datasets.loaders.gsm8k.GSM8KLoader",
        "humaneval": "slm_router.eval.datasets.loaders.humaneval.HumanEvalLoader",
    }

    for name, dotted_path in loader_map.items():
        if name in _REGISTRY:
            continue
        try:
            module_path, class_name = dotted_path.rsplit(".", 1)
            import importlib

            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            _REGISTRY[name] = cls
        except Exception:
            pass  # Loader unavailable; skip silently


_register_all()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_loader(name: str) -> "BenchmarkLoader":
    """Return an instantiated BenchmarkLoader for the given dataset name.

    Args:
        name: One of "mmlu", "gpqa", "hotpotqa", "gsm8k", "humaneval".

    Raises:
        KeyError: If the name is not registered.
        ImportError: If the required dependencies are not installed.
    """
    _register_all()  # Ensure registry is populated

    if name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise KeyError(
            f"Unknown dataset '{name}'. Available datasets: {available}"
        )

    cls = _REGISTRY[name]
    return cls()


def load_dataset_items(
    name: str,
    split: str = "test",
    limit: int | None = None,
    seed: int = 42,
) -> list[EvalItem]:
    """Convenience function: get loader and call load() in one step.

    Args:
        name: Dataset name (e.g. "mmlu", "gsm8k").
        split: Dataset split ("train", "validation", "test").
        limit: Maximum number of items. None = all.
        seed: Random seed for deterministic sampling.

    Returns:
        List of EvalItem instances.
    """
    loader = get_loader(name)
    return loader.load(split=split, limit=limit, seed=seed)


def list_datasets() -> list[str]:
    """Return sorted list of registered dataset names."""
    _register_all()
    return sorted(_REGISTRY.keys())


__all__ = [
    "get_loader",
    "load_dataset_items",
    "list_datasets",
    "EvalItem",
]
