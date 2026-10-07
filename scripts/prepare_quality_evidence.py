"""Import four preselected, pinned archive runs without multiplying duplicate prompts."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import tarfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference_control.benchmarks.data import BenchmarkDataset, import_llmrouterbench
from inference_control.contracts import EndpointSnapshot
from inference_control.util import digest

REVISION = "0e5af1b84bf73437a01a1849c0f1d2468baa93fc"
RUNS = {
    "bench-release/winogrande/valid/DeepHermes-3-Llama-3-8B-Preview/winogrande-valid-DeepHermes-3-Llama-3-8B-Preview-20251019_064322.json": "fac9c5c814cd482862fbbc71640268adafcac9d4a7a36086b8bae70b5d66e354",
    "bench-release/winogrande/valid/DeepSeek-R1-Distill-Qwen-7B/winogrande-valid-DeepSeek-R1-Distill-Qwen-7B-20251019_064315.json": "3bbb0e90b617e7ebd49f1569fac2c192013166c3d7851e740c6c616c085045bb",
    "bench-release/arcc/test/DeepHermes-3-Llama-3-8B-Preview/arcc-test-DeepHermes-3-Llama-3-8B-Preview-20251019_064036.json": "cf9034603469d5fe3744a565558b6200d74963595705c11461677008d56c256c",
    "bench-release/arcc/test/DeepSeek-R1-Distill-Qwen-7B/arcc-test-DeepSeek-R1-Distill-Qwen-7B-20251019_064029.json": "c51ebe4e10893f8f4809ca5e39d0e8a1fbce55510ccb5d9a4ee1a9c2962a5f19",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-prefix", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exclude-dataset", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    provenance = []
    with gzip.open(args.archive_prefix, "rb") as raw, tarfile.open(fileobj=raw, mode="r|") as archive:
        for member in archive:
            if member.offset + member.size > 768*1024*1024:
                raise ValueError("archive exceeds the decompressed byte budget")
            if member.name not in RUNS:
                continue
            if not member.isfile() or member.size > 32*1024*1024:
                raise ValueError("unexpected selected member shape")
            payload = archive.extractfile(member).read(member.size+1)
            if len(payload) != member.size or hashlib.sha256(payload).hexdigest() != RUNS[member.name]:
                raise ValueError("selected archive member integrity mismatch")
            data = json.loads(payload)
            original_count = len(data["records"])
            seen, retained = set(), []
            for row in data["records"]:
                text = row.get("origin_query") or row.get("prompt")
                if not isinstance(text, str):
                    raise TypeError("missing source prompt")
                group = digest(" ".join(text.split()))
                if group not in seen:
                    seen.add(group)
                    retained.append(row)
            data["records"] = retained
            target = args.out / "curated" / Path(*member.name.split("/")[1:])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(data), encoding="utf-8")
            provenance.append({"member": member.name, "source_sha256": RUNS[member.name],
                "curated_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "source_records": original_count, "unique_prompt_groups": len(retained),
                "duplicates_removed": original_count-len(retained)})
            if len(provenance) == len(RUNS):
                break
    if len(provenance) != len(RUNS):
        raise ValueError("archive prefix is missing a preselected complete run")
    models = sorted({name.split("/")[-2] for name in RUNS})
    endpoints = [EndpointSnapshot(endpoint_id=model, provider="archived-benchmark", upstream_model=model,
        revision="archived-unverified-provider-revision", region="archived", context_window=131072,
        config_hash=digest((REVISION, model)), price_version="training-cost-fit-v1",
        input_price_per_million=1., output_price_per_million=1.) for model in models]
    dataset = import_llmrouterbench(args.out / "curated", endpoints=endpoints, seed=42, max_output_tokens=1024)
    old = BenchmarkDataset.load(args.exclude_dataset)
    if {q.group_id for q in old.queries} & {q.group_id for q in dataset.queries}:
        raise ValueError("new evidence overlaps excluded prompt groups")
    train_ids = {q.request.request_id for q in dataset.queries if q.split == "train"}
    prices, priced_endpoints = {}, []
    for endpoint in endpoints:
        rows = [r for r in dataset.outcomes if r.request_id in train_ids and r.endpoint_id == endpoint.endpoint_id]
        xx = sum(r.input_tokens**2 for r in rows)
        yy = sum(r.output_tokens**2 for r in rows)
        xy = sum(r.input_tokens*r.output_tokens for r in rows)
        xc = sum(r.input_tokens*r.cost for r in rows)
        yc = sum(r.output_tokens*r.cost for r in rows)
        determinant = xx*yy-xy*xy
        if determinant <= 0:
            raise ValueError("underdetermined historical price fit")
        input_rate = (xc*yy-yc*xy)/determinant
        output_rate = (yc*xx-xc*xy)/determinant
        residual = max(abs(r.cost-input_rate*r.input_tokens-output_rate*r.output_tokens) for r in rows)
        if min(input_rate, output_rate) < 0 or residual > 1e-8:
            raise ValueError("historical training cost does not support fixed linear token pricing")
        prices[endpoint.endpoint_id] = {"input_per_million": input_rate*1e6,
            "output_per_million": output_rate*1e6, "training_rows": len(rows),
            "maximum_training_cost_residual": residual}
        priced_endpoints.append(endpoint.model_copy(update={"input_price_per_million": input_rate*1e6,
                                                          "output_price_per_million": output_rate*1e6}))
    dataset.endpoints = priced_endpoints
    dataset.provenance.update({"archive_revision": REVISION, "source_members": provenance,
        "selection_rule": "first two model IDs in archive order; first two additional complete tasks; fixed before scores",
        "duplicate_rule": "retain first normalized-prompt occurrence in source order, independent of labels",
        "excluded_prompt_overlap": 0, "historical_prices": prices,
        "price_method": "linear token prices fitted only to training costs; not current provider prices",
        "scope": "historical ARC/WinoGrande scores; distinct from old coding workload; no live-provider or latency claim"})
    dataset.save(args.out / "imported.json")
    (args.out / "provenance.json").write_text(json.dumps(dataset.provenance, indent=2), encoding="utf-8")
    print(json.dumps({"requests": len(dataset.queries),
        "split_counts": dict(Counter(q.split for q in dataset.queries)),
        "duplicates_removed_per_model_run": [p["duplicates_removed"] for p in provenance],
        "dataset_hash": dataset.fingerprint, "split_hash": dataset.split_hash,
        "output": str(args.out / "imported.json")}, indent=2))


if __name__ == "__main__":
    main()
