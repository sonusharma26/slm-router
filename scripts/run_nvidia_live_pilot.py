"""Run a three-call, real-provider SLM Router pilot without retaining raw prompts or answers.

This is a functional test only: two deterministic calibration calls establish a tiny
uncertified map, and the router then executes one new request through its selected
endpoint. It is not a quality benchmark or a production-certification workflow.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PILOT_MAX_OUTPUT_TOKENS = 512
PILOT_PROVIDER_TIMEOUT_SECONDS = 180.0
PILOT_ENDPOINT_IDS = ("nvidia-fast", "nvidia-strong")

from inference_control.adapters.providers import OpenAICompatibleAdapter
from inference_control.adapters.runtime import RuntimeProviderAdapter
from inference_control.api.app import ControlPlane, DecideRequest
from inference_control.capability.conditional import ConditionalCapabilityMap, EvidenceObservation
from inference_control.configuration import load_endpoints
from inference_control.contracts import RequestContext
from inference_control.execution import Executor
from inference_control.features.query import query_features
from inference_control.ledger import SQLiteLedger
from inference_control.planning import Planner
from inference_control.policies.compiler import compile_policy


def load_env_value(path: Path, name: str) -> str:
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            value = value.strip().strip('"').strip("'")
            if value:
                return value
    raise ValueError(f"{name} is not set in {path}")


def normalize_secret(value: str) -> str:
    normalized = value.strip().strip('"').strip("'")
    if not normalized:
        raise ValueError("NVIDIA_API_KEY is empty")
    return normalized


def request_for(policy_id: str, request_id: str, messages: list[dict], text: str) -> RequestContext:
    # Match the byte-based admission shape used by the OpenAI-compatible API.
    serialized = json.dumps({"messages": messages}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return RequestContext(
        request_id=request_id,
        application_id="nvidia-live-pilot",
        tenant_policy_id=policy_id,
        input_tokens=len(serialized),
        max_output_tokens=PILOT_MAX_OUTPUT_TOKENS,
        feature_version="hash-text-v1",
        task_hint="exact-number",
        traffic_slices=frozenset({"pilot"}),
        privacy_classification="public",
        query_features=query_features(text),
    )


def exact_number(text: str | None, expected: str) -> float:
    return float((text or "").strip() == expected)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(out: Path, env_file: Path) -> dict:
    if out.exists():
        raise ValueError(f"output directory already exists: {out}")
    out.mkdir(parents=True)
    report = {
        "kind": "real_provider_functional_pilot",
        "limitations": [
            "Three live calls only; this is not a quality or cost benchmark.",
            "The generated map has one calibration observation per endpoint and cannot issue a certificate.",
            "Developer-program free-endpoint prices are represented as zero only for this prototype profile.",
            "No raw prompt, response text, API key, or provider request ID is written to this report.",
        ],
        "tested_at_utc": datetime.now(timezone.utc).isoformat(),
        "calibration": [],
    }
    providers = {}
    ledger: SQLiteLedger | None = None
    active_endpoint_id: str | None = None
    try:
        templates = {
            endpoint.endpoint_id: endpoint
            for endpoint in load_endpoints(ROOT / "examples" / "nvidia-live" / "endpoints.json")
        }
        missing = set(PILOT_ENDPOINT_IDS) - set(templates)
        if missing:
            raise ValueError(f"pilot endpoint templates missing: {sorted(missing)}")
        endpoints = [templates[endpoint_id] for endpoint_id in PILOT_ENDPOINT_IDS]
        policy = compile_policy(ROOT / "examples" / "nvidia-live" / "policy-live-pilot.yaml")
        api_key = normalize_secret(os.environ.get("NVIDIA_API_KEY") or load_env_value(env_file, "NVIDIA_API_KEY"))
        providers = {
            endpoint.endpoint_id: OpenAICompatibleAdapter(
                "https://integrate.api.nvidia.com/v1", api_key, token_limit_field="max_tokens"
            )
            for endpoint in endpoints
        }
        calibration_text, expected = "Return only the number 4.", "4"
        calibration_messages = [{"role": "user", "content": calibration_text}]
        cmap = ConditionalCapabilityMap(k=16, min_samples=1, max_age_seconds=policy.evidence_max_age_seconds)
        observed_endpoints = []
        for template in endpoints:
            endpoint = template
            active_endpoint_id = endpoint.endpoint_id
            generation_config = dict(endpoint.inference_config)
            generation_config.update({
                "max_tokens": PILOT_MAX_OUTPUT_TOKENS,
                "timeout": PILOT_PROVIDER_TIMEOUT_SECONDS,
            })
            response = providers[endpoint.endpoint_id].call(
                endpoint.upstream_model, calibration_messages, generation_config
            )
            if not response.usage_known:
                raise RuntimeError(f"{endpoint.endpoint_id} did not provide prompt and completion token usage")
            if response.observed_model != endpoint.upstream_model:
                raise RuntimeError(
                    f"{endpoint.endpoint_id} returned model identity {response.observed_model!r}, "
                    f"expected {endpoint.upstream_model!r}"
                )
            if response.observed_revision is not None:
                endpoint = endpoint.model_copy(update={"revision": response.observed_revision})
            observed_endpoints.append(endpoint)
            request = request_for(policy.policy_id, f"pilot-calibration-{endpoint.endpoint_id}", calibration_messages, calibration_text)
            cmap.add(EvidenceObservation(
                sample_id=f"pilot:{endpoint.endpoint_id}:exact-number-4",
                request=request,
                target_id=endpoint.endpoint_id,
                endpoint_ids=(endpoint.endpoint_id,),
                endpoint_revisions=(endpoint.capability_revision,),
                quality=exact_number(response.text, expected),
                cost=(response.input_tokens * endpoint.input_price_per_million
                      + response.output_tokens * endpoint.output_price_per_million) / 1_000_000,
                latency_ms=response.latency_ms,
                output_tokens=response.output_tokens,
                failed=False,
                split="calibration",
                evaluator_type="deterministic",
                evaluator_version="exact-number-pilot-v1",
            ))
            report["calibration"].append({
                "endpoint_id": endpoint.endpoint_id,
                "model_requested": endpoint.upstream_model,
                "model_returned": response.observed_model,
                "response_id_present": response.provider_request_id != "unknown",
                "usage_complete": response.usage_known,
                "prompt_tokens": response.input_tokens,
                "completion_tokens": response.output_tokens,
                "latency_ms": round(response.latency_ms, 1),
                "deterministic_exact_match": bool(exact_number(response.text, expected)),
                "revision_observed": response.observed_revision is not None,
            })

        endpoints = observed_endpoints
        write_json(out / "endpoints.json", [endpoint.model_dump(mode="json") for endpoint in endpoints])
        write_json(out / "capability-map.json", cmap.export())
        bridge = RuntimeProviderAdapter({endpoint.endpoint_id: endpoint for endpoint in endpoints}, providers)
        ledger = SQLiteLedger(out / "live-pilot.sqlite3")
        control = ControlPlane(Planner(endpoints, capability_map=cmap), ledger, Executor(bridge), default_policy=policy)
        route_text = "Return only the number 7."
        route_messages = [{"role": "user", "content": route_text}]
        decision = control.decide(DecideRequest(
            request=request_for(policy.policy_id, "pilot-router-request", route_messages, route_text), policy=policy
        ))
        if decision.selected_plan.plan_type == "abstain":
            raise RuntimeError("pilot router abstained; inspect rejected_alternatives in the ledger")
        active_endpoint_id = decision.selected_plan.steps[0].endpoint_id
        selected = []
        execution = control.execute(decision.decision_id, messages=route_messages, result_sink=selected.append)
        report["routing"] = {
            "decision_id": decision.decision_id,
            "plan_type": decision.selected_plan.plan_type,
            "selected_endpoint_id": decision.selected_plan.steps[0].endpoint_id,
            "certificate_status": decision.certificate_status,
            "execution_state": execution.state,
            "execution_selected_endpoint_id": execution.selected_endpoint_id,
            "provider_request_id_present": bool(execution.provider_request_ids),
            "usage": {"prompt_tokens": execution.input_tokens, "completion_tokens": execution.output_tokens},
            "latency_ms": round(execution.total_latency_ms, 1),
            "constraint_violations": list(execution.constraint_violations),
            "response_content_length": len((selected[0].message or {}).get("content") or "") if selected else 0,
        }
        report["ledger_verified"] = ledger.verify()
        report["endpoint_snapshot"] = str(out / "endpoints.json")
        write_json(out / "pilot-report.json", report)
        return report
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        if active_endpoint_id is not None:
            report["failed_endpoint_id"] = active_endpoint_id
        response = getattr(exc, "response", None)
        if response is not None:
            report["http_status_code"] = response.status_code
            try:
                provider_error = (response.json() or {}).get("error") or {}
                if isinstance(provider_error, dict):
                    if provider_error.get("code") is not None:
                        report["provider_error_code"] = str(provider_error["code"])
                    if provider_error.get("type") is not None:
                        report["provider_error_type"] = str(provider_error["type"])
                    if provider_error.get("message") is not None:
                        report["provider_error_message"] = str(provider_error["message"])[:300]
            except (TypeError, ValueError):
                pass
        write_json(out / "pilot-report.json", report)
        raise
    finally:
        if ledger is not None:
            ledger.close()
        for provider in providers.values():
            provider.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a three-call NVIDIA Build SLM Router functional pilot.")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "nvidia-live-pilot")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env.local")
    args = parser.parse_args()
    try:
        report = run(args.out.resolve(), args.env_file.resolve())
    except Exception as exc:
        print(f"Pilot failed: {type(exc).__name__}. See {args.out / 'pilot-report.json'}", file=sys.stderr)
        return 1
    print(json.dumps({
        "report": str(args.out / "pilot-report.json"),
        "capability_map": str(args.out / "capability-map.json"),
        "endpoint_snapshot": str(args.out / "endpoints.json"),
        "selected_endpoint_id": report["routing"]["selected_endpoint_id"],
        "execution_state": report["routing"]["execution_state"],
        "certificate_status": report["routing"]["certificate_status"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
