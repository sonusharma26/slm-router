from __future__ import annotations

try:
    from slm_router.eval.datasets.loaders.mmlu import MMLULoader
except ImportError:
    MMLULoader = None  # type: ignore[assignment,misc]

try:
    from slm_router.eval.datasets.loaders.gpqa import GPQALoader
except ImportError:
    GPQALoader = None  # type: ignore[assignment,misc]

try:
    from slm_router.eval.datasets.loaders.hotpotqa import HotpotQALoader
except ImportError:
    HotpotQALoader = None  # type: ignore[assignment,misc]

try:
    from slm_router.eval.datasets.loaders.gsm8k import GSM8KLoader
except ImportError:
    GSM8KLoader = None  # type: ignore[assignment,misc]

try:
    from slm_router.eval.datasets.loaders.humaneval import HumanEvalLoader
except ImportError:
    HumanEvalLoader = None  # type: ignore[assignment,misc]

__all__ = [
    "MMLULoader",
    "GPQALoader",
    "HotpotQALoader",
    "GSM8KLoader",
    "HumanEvalLoader",
]
