"""Workflow: build an oracle matrix by running every candidate model on every eval item.

The oracle matrix (every model's score+cost+latency on every item) is the frozen
offline simulator that router evaluation replays against for free.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
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
    refresh: bool = False,
) -> None:
    """Call each candidate model on each eval item, score it, and store an oracle RunTrace.

    By default, already-stored cells are skipped so re-running resumes.
    ``refresh=True`` deliberately re-calls and replaces them without using the
    response cache. Cost is always pre-estimated and gated behind ``confirm``.
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

    pending = [
        (
            item,
            model_id,
            store.has_run(item.item_id, model_id, cfg_hash, "oracle"),
        )
        for item in items
        for model_id in candidate_models
        if refresh
        or not store.has_run(
            item.item_id, model_id, cfg_hash, "oracle", successful_only=True
        )
    ]
    if not pending:
        print("[build_oracle] Oracle matrix is already complete; 0 calls needed.")
        return

    # Rough pre-estimate: 500 prompt + 200 completion tokens per pending call.
    from slm_router.types import Usage

    per_model_est: dict[str, float] = {}
    for mid in candidate_models:
        try:
            per_model_est[mid] = registry.cost(
                mid, Usage(prompt_tokens=500, completion_tokens=200)
            ).total_usd
        except Exception:
            per_model_est[mid] = 0.0
    est = sum(per_model_est[mid] for _, mid, _ in pending)
    n_calls = len(pending)
    print(
        f"[build_oracle] {n_calls} pending of "
        f"{len(items) * len(candidate_models)} matrix cells. "
        f"Estimated cost: ${est:.4f}"
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
    done = errors = completed = 0
    skipped = len(items) * len(candidate_models) - len(pending)
    stored_by_model: Counter[str] = Counter()
    errors_by_model: Counter[str] = Counter()
    print(
        f"[build_oracle] Starting {len(pending)} calls; progress is reported "
        "after the first completion and then every 5.",
        flush=True,
    )

    async def process(
        item: Any,
        model_id: str,
        retrying_failed_cell: bool,
    ) -> RunTrace | None:
        nonlocal completed, done, errors
        async with sem:
            trace = await _call_and_score(
                item,
                model_id,
                bypass_cache=refresh or retrying_failed_cell,
            )
        # Persist immediately so a killed/interrupted run keeps whatever
        # completed so far, instead of losing everything to a final batch insert.
        if trace is not None:
            store.insert_many([trace])
            if trace.error:
                errors_by_model[model_id] += 1
            else:
                stored_by_model[model_id] += 1
        completed += 1
        if completed == 1 or completed == len(pending) or completed % 5 == 0:
            print(
                f"[build_oracle] Progress: {completed}/{len(pending)} "
                f"(stored={done}, errors={errors})",
                flush=True,
            )
        return trace

    async def _call_and_score(
        item: Any,
        model_id: str,
        bypass_cache: bool,
    ) -> RunTrace | None:
        nonlocal done, errors
        try:
            resp = await client.complete(
                model_id,
                [{"role": "user", "content": item.query}],
                temperature=0.0,
                max_tokens=1024,
                use_cache=not bypass_cache,
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

    # process() persists each trace as soon as it completes, so results here
    # are only used for the summary counts, not a final batch insert.
    await asyncio.gather(
        *(process(it, mid, retrying) for it, mid, retrying in pending)
    )
    print(
        f"[build_oracle] Done. stored={done} skipped={skipped} errors={errors}"
    )
    for model_id in candidate_models:
        if stored_by_model[model_id] or errors_by_model[model_id]:
            print(
                f"  {model_id}: stored={stored_by_model[model_id]} "
                f"errors={errors_by_model[model_id]}"
            )


def _tier(registry: Any, model_id: str) -> str:
    try:
        return registry.get(model_id).tier
    except Exception:
        return "unknown"
