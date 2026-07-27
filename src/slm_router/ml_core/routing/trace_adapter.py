"""Adapts stored oracle RunTrace rows into training samples for offline-RL fitting.

RunTrace carries score/cost/latency but not the context vector or arm index that
OfflineRLPolicy.fit()/fqe_value() expect; this module bridges the two.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


SPLIT_VERSION = 2


def is_holdout_item(item_id: str, seed: int, holdout_fraction: float) -> bool:
    """Legacy per-item split predicate retained for artifact compatibility."""
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")
    digest = hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return value < holdout_fraction


def select_holdout_items(
    item_ids: set[str] | list[str],
    seed: int,
    holdout_fraction: float,
) -> set[str]:
    """Select an exact, deterministic item-level holdout.

    Hash-threshold sampling can produce an empty holdout on a small oracle
    matrix. Ranking by the same stable hash preserves order independence while
    guaranteeing at least one train and one holdout item whenever possible.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")
    unique_ids = set(item_ids)
    if len(unique_ids) < 2:
        return set()

    ranked = sorted(
        unique_ids,
        key=lambda item_id: hashlib.sha256(
            f"{seed}:{item_id}".encode("utf-8")
        ).digest(),
    )
    count = round(len(ranked) * holdout_fraction)
    count = max(1, min(len(ranked) - 1, count))
    return set(ranked[:count])


class TrainingSample:
    """Duck-typed stand-in for RunTrace exposing what OfflineRLPolicy needs."""

    __slots__ = ("context", "arm_index", "quality", "cost_usd", "latency_ms")

    def __init__(
        self,
        context: np.ndarray,
        arm_index: int,
        quality: float,
        cost_usd: float,
        latency_ms: float,
    ) -> None:
        self.context = context
        self.arm_index = arm_index
        self.quality = quality
        self.cost_usd = cost_usd
        self.latency_ms = latency_ms


def build_training_samples(
    traces: list[Any],
    candidate_models: list[str],
    featurizer: Any,
    item_texts: dict[str, str],
) -> list[TrainingSample]:
    """Convert error-free oracle traces into (context, arm, reward-inputs) samples.

    Traces whose model isn't in ``candidate_models`` or whose item text is
    unknown are skipped. Feature vectors are cached per item so each item's
    text is only embedded/featurized once even though it appears once per arm.
    """
    from slm_router.types import Query

    samples: list[TrainingSample] = []
    ctx_cache: dict[str, np.ndarray] = {}

    for tr in traces:
        if tr.error or tr.model not in candidate_models:
            continue
        text = item_texts.get(tr.item_id)
        if text is None:
            continue

        ctx = ctx_cache.get(tr.item_id)
        if ctx is None:
            ctx = featurizer.vectorize(Query(id=tr.item_id, text=text))
            ctx_cache[tr.item_id] = ctx

        samples.append(
            TrainingSample(
                context=ctx,
                arm_index=candidate_models.index(tr.model),
                quality=tr.score,
                cost_usd=tr.cost_usd,
                latency_ms=tr.latency_ms,
            )
        )
    return samples
