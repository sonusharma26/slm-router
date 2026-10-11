"""Evaluate slm-router on LLMRouterBench using the benchmark's own loader and split.

Setup (on a machine that can reach GitHub and Hugging Face):

    git clone https://github.com/ynulihao/LLMRouterBench
    # Put the result data where LLMRouterBench's config expects it (results/bench/...),
    # following its README (Hugging Face dataset NPULH/LLMRouterBench).
    pip install -e ".[dev]"                       # this repository
    pip install -r ../LLMRouterBench/requirements.txt   # its loader needs loguru, pandas, ...

Run (from this repository's root):

    python scripts/run_llmrouterbench.py --lrb-root ../LLMRouterBench \\
        --config config/baseline_config_performance_cost.yaml --seeds 42,43,44 \\
        --out results/llmrouterbench

Per seed, the script
1. loads records with LLMRouterBench's ``BaselineDataLoader`` and splits them with its
   ``split_by_dataset_then_prompt`` (train ratio and seed as given), i.e. the same test
   prompts its baselines are scored on;
2. builds one full-information matrix per prompt. Missing model scores are filled with 0,
   as LLMRouterBench's adaptors do, and missing costs with that model's mean cost; the
   counts are reported;
3. divides LLMRouterBench's training prompts into train / calibration / validation by a
   prompt hash, leaving its test prompts untouched;
4. runs the comparison harness (slm-router at several quality targets, the
   answer-agreement cascade and majority-vote family built from each model's stored extracted
   answer, kNN, threshold, best single model, every single model, and the oracle as a
   reference) and writes ``report.json`` and ``report.md``.

Votes call the top-k models (k = 3, 5, ... up to ``--max-voters``) and serve the answer that
reaches a quorum, comparing the stored ``prediction`` fields after the same normalization the
agreement cascade uses; missing, blank or NaN predictions are not comparable. The calibration
picks plain routing, an agreement cascade or a vote, whichever is cheapest among those that
pass; ``--max-voters 0`` removes the votes (agreement cascades only).

By default the router never sees which dataset a prompt came from (``--task-hints``
exposes it; that is a different, easier setting). Features are hashed word n-grams, or
sentence-transformer embeddings with ``--embedder st:<model>``.
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference_control.benchmarks.compare import (
    LOCAL_TARGETS,
    run_compare,
    write_compare_report,
)
from inference_control.benchmarks.data import BenchmarkDataset, Query, Truth
from inference_control.contracts import EndpointSnapshot, RequestContext
from inference_control.routing.features import (
    NGRAM_FEATURE_VERSION,
    SentenceEmbedder,
    ngram_features,
)
from inference_control.util import digest

FIT_SPLITS = ((.6, "train"), (.85, "calibration"), (1., "validation"))  # shares of LLMRouterBench's train prompts
VOTE_SIZES = (3, 5, 7)  # majority-vote sizes tried, capped by --max-voters


def featurizer(spec: str | None):
    if not spec:
        return NGRAM_FEATURE_VERSION, lambda texts: [ngram_features(t) for t in texts]
    if spec.startswith("st:"):
        embedder = SentenceEmbedder(spec[3:])
        return embedder.feature_version, embedder
    raise ValueError("--embedder must look like st:<sentence-transformers model>")


def stored_answer(record):
    """The model's stored extracted answer; None (not comparable) when missing or NaN.

    Normalization (case, whitespace, trailing period) is applied later by the cascade module.
    """
    value = getattr(record, "prediction", None) if record is not None else None
    return None if isinstance(value, float) and math.isnan(value) else value


def fit_split(dataset_id: str, prompt: str, seed: int) -> str:
    bucket = int(digest(("lrb-fit", seed, dataset_id, " ".join(prompt.split())))[:8], 16) / 2**32
    return next(name for share, name in FIT_SPLITS if bucket < share)


def build_dataset(train_records, test_records, *, seed, featurize, feature_version, task_hints, name):
    groups, duplicates = {}, 0
    for side, records in (("train", train_records), ("test", test_records)):
        for record in records:
            key = (record.dataset_id, record.prompt)
            group = groups.setdefault(key, {"dataset": record.dataset_id, "side": side, "rows": {},
                                            "text": record.origin_query or record.prompt})
            if group["side"] != side:
                raise ValueError("a prompt appears in both LLMRouterBench train and test; check the split")
            if record.model_name in group["rows"]:
                duplicates += 1
                continue
            group["rows"][record.model_name] = record
    models = sorted({m for g in groups.values() for m in g["rows"]})
    costs = defaultdict(list)
    for g in groups.values():
        for model, record in g["rows"].items():
            if record.cost is not None:
                costs[model].append(float(record.cost))
    mean_cost = {m: mean(costs[m]) if costs[m] else 0. for m in models}
    endpoints = [EndpointSnapshot(endpoint_id=m, provider="llmrouterbench", upstream_model=m, revision="archived",
        region="archived", context_window=10**9, config_hash="llmrouterbench", price_version="recorded-cost",
        input_price_per_million=0., output_price_per_million=0., governance=frozenset({"archived"}))
        for m in models]
    keys = sorted(groups)
    features = featurize([groups[k]["text"] for k in keys])
    queries, outcomes, filled, slice_of, answers = [], [], 0, {}, {}
    for key, vector in zip(keys, features):
        g = groups[key]
        dataset_id, prompt = key
        request_id = digest(("llmrouterbench", dataset_id, " ".join(prompt.split())))[:24]
        split = "test" if g["side"] == "test" else fit_split(dataset_id, prompt, seed)
        tokens = max((int(r.prompt_tokens or 0) for r in g["rows"].values()), default=0)
        request = RequestContext(request_id=request_id, application_id="llmrouterbench", tenant_policy_id="benchmark",
            input_tokens=tokens, max_output_tokens=1_000_000, feature_version=feature_version,
            task_hint=dataset_id if task_hints else None,
            traffic_slices=frozenset({dataset_id}) if task_hints else frozenset(), query_features=tuple(vector))
        queries.append(Query(request=request, text=g["text"], split=split,
                             group_id=digest((dataset_id, " ".join(prompt.split())))))
        slice_of[request_id] = dataset_id
        for model in models:
            record = g["rows"].get(model)
            # Extracted answers drive agreement checks and votes; missing ones never agree.
            answers[(request_id, model)] = stored_answer(record)
            if record is None:
                filled += 1
                score, cost, output = 0., mean_cost[model], 0
            else:
                score = float(record.score)
                if not 0 <= score <= 1:
                    raise ValueError(f"score outside [0,1] for {model} on {dataset_id}; normalize first")
                cost = float(record.cost) if record.cost is not None else mean_cost[model]
                output = int(record.completion_tokens or 0)
            outcomes.append(Truth(request_id=request_id, endpoint_id=model, quality=score, cost=max(0., cost),
                latency_ms=None, input_tokens=tokens, output_tokens=output, evaluator_type="benchmark",
                evaluator_version="LLMRouterBench"))
    from inference_control.benchmarks.runner import default_policy
    from inference_control.policies.validator import eligibility_reasons
    probe = default_policy(missing_latency=True)
    blocked = {e.endpoint_id: eligibility_reasons(e, queries[0].request, probe) for e in endpoints}
    if any(blocked.values()):
        raise ValueError(f"benchmark endpoints are ineligible for imported requests: {blocked}")
    dataset = BenchmarkDataset(name=name, evidence_kind="full_information", queries=queries, outcomes=outcomes,
        endpoints=endpoints, provenance={"source": "https://github.com/ynulihao/LLMRouterBench", "seed": seed,
            "filled_missing_scores": filled, "duplicate_records_ignored": duplicates,
            "fit_splits_of_lrb_train": {n: s for s, n in FIT_SPLITS}, "task_hints": task_hints})
    return dataset, slice_of, answers


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lrb-root", type=Path, required=True)
    parser.add_argument("--config", default="config/baseline_config_performance_cost.yaml")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--train-ratio", type=float, default=.8)
    parser.add_argument("--ood-datasets", default="")
    parser.add_argument("--embedder", default=None, help="st:<sentence-transformers model>; default hashed n-grams")
    parser.add_argument("--task-hints", action="store_true", help="let the router see the dataset id (easier setting)")
    parser.add_argument("--quality-risk", type=float, default=.05)
    parser.add_argument("--include-previous", action="store_true", help="also run the previous local router (slow)")
    parser.add_argument("--no-cascade", action="store_true", help="skip the answer-agreement cascade and votes")
    parser.add_argument("--max-voters", type=int, default=5,
                        help="largest majority vote tried (sizes 3, 5, 7 up to this); 0 keeps agreement cascades only")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "llmrouterbench")
    args = parser.parse_args()
    lrb = args.lrb_root.resolve()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    sys.path.insert(0, str(lrb))
    # The split helper lives in baselines.adaptors, whose __init__ imports every adaptor
    # (torch, pandas, ...). Register the package without running that __init__ so only
    # adaptors/common.py loads; LLMRouterBench's own split code still runs unchanged.
    import types
    adaptors = types.ModuleType("baselines.adaptors")
    adaptors.__path__ = [str(lrb / "baselines" / "adaptors")]
    sys.modules.setdefault("baselines.adaptors", adaptors)
    previous = Path.cwd()
    os.chdir(lrb)  # LLMRouterBench resolves its results directory relative to its root
    try:
        from baselines.data_loader import BaselineDataLoader
        loader = BaselineDataLoader(config_path=str(lrb / args.config))
        records = loader.load_all_records()
        splits = {seed: loader.split_by_dataset_then_prompt(records, train_ratio=args.train_ratio, random_seed=seed,
                      ood_datasets=[d for d in args.ood_datasets.split(",") if d]) for seed in seeds}
    finally:
        os.chdir(previous)
    try:
        commit = subprocess.run(["git", "-C", str(lrb), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        commit = None
    feature_version, featurize = featurizer(args.embedder)
    vote_sizes = tuple(v for v in VOTE_SIZES if v <= args.max_voters)
    datasets, slice_of, answers = {}, {}, {}
    for seed, (train, test) in splits.items():
        name = f"LLMRouterBench:{Path(args.config).stem}:seed{seed}"
        datasets[seed], groups, seed_answers = build_dataset(train, test, seed=seed, featurize=featurize,
            feature_version=feature_version, task_hints=args.task_hints, name=name)
        slice_of.update(groups)
        answers.update(seed_answers)
        print(f"seed {seed}: {sum(q.split=='test' for q in datasets[seed].queries)} test prompts, "
              f"{len(datasets[seed].endpoints)} models, filled {datasets[seed].provenance['filled_missing_scores']} "
              "missing scores", flush=True)
    report = run_compare(datasets=datasets, slice_of=slice_of, quality_risk=args.quality_risk,
                         answers=None if args.no_cascade else answers,
                         cascade_voters=vote_sizes,
                         local_targets=LOCAL_TARGETS if args.include_previous else (),
                         policy_overrides={"max_expected_spend": 1e9, "max_absolute_spend": 1e9},
                         progress=lambda message: print(message, flush=True))
    report["llmrouterbench"] = {"root_commit": commit, "config": args.config, "train_ratio": args.train_ratio,
        "records": len(records), "seeds": seeds, "features": feature_version, "task_hints": args.task_hints,
        "vote_sizes": [] if args.no_cascade else list(vote_sizes),
        "comparability": ("Same loader, split function, seeds and test prompts as LLMRouterBench baselines. "
                          "Compare 'Best per-dataset average' with the paper's AvgAcc and the cost columns with its "
                          "performance-cost results; run LLMRouterBench's own baselines for a direct table.")}
    path = write_compare_report(report, args.out)
    print(f"wrote {path} and {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
