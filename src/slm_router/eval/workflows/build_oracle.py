"""Workflow: build an oracle matrix by running every candidate model on every eval item.

The oracle matrix (every model's score+cost+latency on every item) is the frozen
offline simulator that router evaluation replays against for free.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from slm_router.eval.datasets import load_dataset_items
from slm_router.eval.scoring.dispatch import score_response
from slm_router.trace.models import RunTrace


def _md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def config_hash(config: Any) -> str:
    """Stable hash of the config fields that affect generation/scoring."""
    candidate_models = sorted(getattr(config, "candidate_models", []))
    judge_model = str(getattr(config, "judge_model", ""))
    payload = json.dumps({"models": candidate_models, "judge": judge_model}, sort_keys=True)
    return _md5(payload)


def _run_id(item_id: str, model_id: str, cfg_hash: str, kind: str) -> str:
    return _md5(f"{kind}:{item_id}:{model_id}:{cfg_hash}")


async def build_oracle(
    client: Any,
    registry: Any,
    store: Any,
    config: Any,
    datasets: list[str],
    limit: int | None = None,
    confirm: bool = True,
    concurrency: int = 8,
) -> None:
    """Call each candidate model on each eval item, score it, and store an oracle RunTrace.

    Idempotent: already-stored (item, model, config_hash, 'oracle') cells are skipped,
    so re-running resumes. Cost is pre-estimated and gated behind ``confirm``.
    """
    seed = getattr(config, "dataset_seed", 13)

    items = []
    for ds_name in datasets:
        try:
            items.extend(load_dataset_items(ds_name, split="test", limit=limit, seed=seed))
        except Exception as exc:  # dataset libs not installed / network off
            print(f"[build_oracle] WARNING: could not load '{ds_name}': {exc}")

    candidate_models: list[str] = list(getattr(config, "candidate_models", []))
    if not candidate_models or not items:
        print("[build_oracle] Nothing to do (no items or no candidate_models).")
        return

    cfg_hash = config_hash(config)

    # Rough pre-estimate: 500 prompt + 200 completion tokens per call.
    from slm_router.types import Usage

    est = 0.0
    for mid in candidate_models:
        try:
            est += registry.cost(mid, Usage(prompt_tokens=500, completion_tokens=200)).total_usd
        except Exception:
            pass
    est *= len(items)
    n_calls = len(items) * len(candidate_models)
    print(
        f"[build_oracle] {len(items)} items x {len(candidate_models)} models = "
        f"{n_calls} calls. Estimated cost: ${est:.4f}"
    )

    if confirm:
        try:
            ans = input("Proceed? [y/N] ").strip().lower()
        except EOFError:
            ans = "n"
        if ans not in ("y", "yes"):
            print("[build_oracle] Aborted.")
            return

    sem = asyncio.Semaphore(concurrency)
    done = skipped = errors = 0

    async def process(item: Any, model_id: str) -> RunTrace | None:
        nonlocal done, skipped, errors
        if store.has_run(item.item_id, model_id, cfg_hash, "oracle"):
            skipped += 1
            return None
        async with sem:
            try:
                resp = await client.complete(
                    model_id,
                    [{"role": "user", "content": item.query}],
                    temperature=0.0,
                    max_tokens=1024,
                )
            except Exception as exc:
                errors += 1
                return RunTrace(
                    run_id=_run_id(item.item_id, model_id, cfg_hash, "oracle"),
                    item_id=item.item_id,
                    dataset=item.dataset,
                    task_type=str(item.task_type),
                    model=model_id,
                    model_tier=_tier(registry, model_id),
                    run_kind="oracle",
                    config_hash=cfg_hash,
                    error=str(exc),
                )
        try:
            sr = score_response(item, resp.text)
        except Exception as exc:
            errors += 1
            sr = None
            score, correct, err = 0.0, None, str(exc)
        else:
            score, correct, err = sr.score, sr.correct, sr.error
        done += 1
        return RunTrace(
            run_id=_run_id(item.item_id, model_id, cfg_hash, "oracle"),
            item_id=item.item_id,
            dataset=item.dataset,
            task_type=str(item.task_type),
            model=model_id,
            model_tier=_tier(registry, model_id),
            raw_output=resp.text,
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            cost_usd=resp.cost.total_usd,
            latency_ms=resp.latency_ms,
            score=score,
            correct=correct,
            run_kind="oracle",
            config_hash=cfg_hash,
            error=err,
        )

    results = await asyncio.gather(
        *(process(it, mid) for it in items for mid in candidate_models)
    )
    traces = [r for r in results if r is not None]
    if traces:
        store.insert_many(traces)
    print(
        f"[build_oracle] Done. stored={done} skipped={skipped} errors={errors}"
    )


def _tier(registry: Any, model_id: str) -> str:
    try:
        return registry.get(model_id).tier
    except Exception:
        return "unknown"
