"""Batch inference helpers."""

from __future__ import annotations

import asyncio
from typing import Any

from slm_router.types import ModelResponse

try:
    from tqdm.asyncio import tqdm_asyncio  # type: ignore
    _TQDM_AVAILABLE = True
except ImportError:
    tqdm_asyncio = None  # type: ignore
    _TQDM_AVAILABLE = False


async def run_batch(
    client: Any,
    model_ids: list[str],
    prompts: list[str],
    *,
    concurrency: int = 8,
    **complete_kwargs: Any,
) -> list[ModelResponse]:
    """Run multiple (model_id, prompt) pairs concurrently.

    *model_ids* and *prompts* are zipped; they must have equal length.
    Returns ModelResponse objects in the same order.
    """
    if len(model_ids) != len(prompts):
        raise ValueError(
            f"model_ids ({len(model_ids)}) and prompts ({len(prompts)}) must have equal length"
        )

    sem = asyncio.Semaphore(concurrency)

    async def _one(model_id: str, prompt: str) -> ModelResponse:
        async with sem:
            messages = [{"role": "user", "content": prompt}]
            return await client.complete(model_id, messages, **complete_kwargs)

    tasks = [_one(mid, p) for mid, p in zip(model_ids, prompts)]

    if _TQDM_AVAILABLE and tqdm_asyncio is not None:
        results: list[ModelResponse] = await tqdm_asyncio.gather(
            *tasks, desc="batch inference"
        )
    else:
        results = list(await asyncio.gather(*tasks))

    return results
