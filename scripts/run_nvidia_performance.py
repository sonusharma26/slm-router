"""Bounded live endpoint latency benchmark; no credentials or response text retained."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import os

from run_nvidia_live_pilot import load_env_value, normalize_secret

from inference_control.adapters.providers import OpenAICompatibleAdapter
from inference_control.configuration import load_endpoints


def percentile(values, quantile):
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lower = int(position)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)


def measure(endpoint, key, count, timeout):
    adapter = OpenAICompatibleAdapter("https://integrate.api.nvidia.com/v1", key, token_limit_field="max_tokens")
    samples = []
    start = perf_counter()
    try:
        for index in range(count + 1):
            left, right = 13 + index * 7, 5 + index * 3
            config = dict(endpoint.inference_config)
            config.update(max_completion_tokens=512, timeout=timeout)
            began = perf_counter()
            row = {"sample":index, "warmup":index == 0}
            try:
                response = adapter.call(endpoint.upstream_model,
                    [{"role":"user", "content":f"Calculate {left} + {right}. Return only the integer answer, without explanation."}],
                    config)
                row.update(success=True, latency_ms=response.latency_ms,
                    exact_answer=(response.text or "").strip() == str(left + right),
                    input_tokens=response.input_tokens, output_tokens=response.output_tokens,
                    usage_known=response.usage_known,
                    model_identity_matches=response.observed_model == endpoint.upstream_model,
                    immutable_revision_observed=response.observed_revision is not None,
                    response_characters=len(response.text or ""))
            except (httpx.HTTPError, TimeoutError, ValueError, KeyError, TypeError, IndexError) as exc:
                response = getattr(exc, "response", None)
                row.update(success=False, latency_ms=(perf_counter() - began) * 1000,
                           error_type=type(exc).__name__, http_status=getattr(response, "status_code", None))
            samples.append(row)
            if index == 0 and not row["success"]:
                break
    finally:
        adapter.close()
    elapsed = perf_counter() - start
    measured = [row for row in samples if not row["warmup"]]
    successful = [row for row in measured if row["success"]]
    latencies = [row["latency_ms"] for row in successful]
    measured_seconds = sum(row["latency_ms"] for row in measured) / 1000
    return {"endpoint_id":endpoint.endpoint_id, "model":endpoint.upstream_model,
        "requested_measured_calls":count, "measured_calls":len(measured),
        "successful_calls":len(successful), "wall_seconds_including_warmup":elapsed,
        "warmup":samples[0], "samples":measured,
        "metrics":{"mean_latency_ms":mean(latencies) if latencies else None,
            "p50_latency_ms":percentile(latencies, .5) if latencies else None,
            "p95_latency_ms":percentile(latencies, .95) if latencies else None,
            "error_rate":1 - len(successful) / len(measured) if measured else None,
            "exact_answer_accuracy_all_measured":sum(row.get("exact_answer", False) for row in measured) / len(measured) if measured else None,
            "serial_successful_requests_per_second":len(successful) / measured_seconds if measured_seconds else None,
            "input_tokens":sum(row["input_tokens"] for row in successful),
            "output_tokens":sum(row["output_tokens"] for row in successful),
            "usage_coverage":sum(row["usage_known"] for row in successful) / len(successful) if successful else None,
            "actual_billed_cost":None}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--requests-per-model", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=45)
    args = parser.parse_args()
    if not 1 <= args.requests_per_model <= 20 or not 0 < args.timeout <= 180:
        parser.error("requests must be 1..20 and timeout 0..180 seconds")
    args.out.mkdir(parents=True, exist_ok=False)
    key = normalize_secret(os.environ.get("NVIDIA_API_KEY") or load_env_value(ROOT / ".env.local", "NVIDIA_API_KEY"))
    endpoints = [endpoint for endpoint in load_endpoints(ROOT / "examples/nvidia-live/endpoints.json")
                 if endpoint.endpoint_id in {"nvidia-fast", "nvidia-strong"}]
    started = perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(measure, endpoint, key, args.requests_per_model, args.timeout) for endpoint in endpoints]
        results = []
        for future in futures:
            row = future.result()
            results.append(row)
            print(json.dumps({"endpoint_id":row["endpoint_id"], "metrics":row["metrics"], "warmup_success":row["warmup"]["success"]}), flush=True)
    report = {"kind":"live_endpoint_performance_microbenchmark", "tested_at_utc":datetime.now(UTC).isoformat(),
        "parallel_endpoint_workers":2, "wall_seconds":perf_counter() - started,
        "max_output_tokens_per_call":512, "timeout_seconds":args.timeout, "results":results,
        "limitations":["Generated integer-addition workload; not coding or production quality evidence.",
            "Small descriptive latency sample; p95 is not a statistical tail guarantee.",
            "Direct endpoint measurements do not establish router certification or routing coverage.",
            "Warm-up excluded from measured latency/accuracy; cold-start latency reported separately.",
            "Two endpoints measured concurrently; throughput is serial per endpoint, not a load-capacity estimate.",
            "NVIDIA did not supply billed dollar cost; token usage is reported without inventing free pricing."]}
    path = args.out / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report":str(path), "wall_seconds":report["wall_seconds"]}), flush=True)
    return int(any(row["measured_calls"] != args.requests_per_model or
                   row["successful_calls"] != args.requests_per_model for row in results))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, httpx.HTTPError, TimeoutError, KeyError, TypeError, IndexError) as exc:
        print(f"Benchmark failed: {type(exc).__name__}", file=sys.stderr)
        raise SystemExit(1) from None
