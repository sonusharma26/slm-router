#!/usr/bin/env python3
"""Exercise the configured NVIDIA models through the adaptive router.

This is a small live functional matrix, not a benchmark or production calibration.
Raw prompts, responses, provider request IDs, and credentials are never persisted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference_control.adapters.providers import OpenAICompatibleAdapter
from inference_control.adapters.runtime import RuntimeProviderAdapter
from inference_control.api.app import ControlPlane, DecideRequest
from inference_control.capability.conditional import ConditionalCapabilityMap, EvidenceObservation
from inference_control.configuration import load_endpoints
from inference_control.contracts import EndpointSnapshot, RequestContext
from inference_control.execution import Executor
from inference_control.features.query import query_features
from inference_control.ledger import SQLiteLedger
from inference_control.planning import Planner
from inference_control.policies.compiler import compile_policy

MAX_OUTPUT_TOKENS = 64
PROVIDER_TIMEOUT_SECONDS = 180.0
APPLICATION_ID = "nvidia-matrix-live"
TASK_HINT = "exact-number"
TRAFFIC_SLICE = "pilot"
POLICY_ID = "nvidia-matrix-live"


def load_env_value(path: Path, name: str) -> str:
    if not path.exists():
        raise ValueError(f"{name} is not set and {path} does not exist")
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


def secret(env_file: Path, name: str) -> str:
    value = (os.environ.get(name) or load_env_value(env_file, name)).strip().strip('"').strip("'")
    if not value:
        raise ValueError(f"{name} is empty")
    return value


def request_for(policy_id: str, request_id: str, messages: list[dict], text: str) -> RequestContext:
    serialized = json.dumps({"messages": messages}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return RequestContext(
        request_id=request_id,
        application_id=APPLICATION_ID,
        tenant_policy_id=policy_id,
        input_tokens=len(serialized),
        max_output_tokens=MAX_OUTPUT_TOKENS,
        feature_version="hash-text-v1",
        task_hint=TASK_HINT,
        traffic_slices=frozenset({TRAFFIC_SLICE}),
        privacy_classification="public",
        query_features=query_features(text),
    )


def exact_number(text: str | None, expected: str) -> bool:
    return (text or "").strip() == expected


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def provider_error(exc: Exception) -> dict[str, object]:
    item: dict[str, object] = {"error_type": type(exc).__name__}
    response = getattr(exc, "response", None)
    if isinstance(response, httpx.Response):
        item["http_status_code"] = response.status_code
        try:
            body = response.json()
            error = body.get("error") if isinstance(body, dict) else None
            if isinstance(error, dict):
                for source, target in (("code", "provider_error_code"), ("type", "provider_error_type")):
                    if error.get(source) is not None:
                        item[target] = str(error[source])[:100]
                if error.get("message") is not None:
                    item["provider_error_message"] = str(error["message"])[:300]
        except (TypeError, ValueError):
            pass
    return item


def policy_document(endpoints: list[EndpointSnapshot], *, policy_id: str = POLICY_ID,
                    endpoint_id: str | None = None) -> dict:
    return {
        "policy_id": policy_id,
        "version": "1",
        "minimum_quality": 0.0,
        "cost": {"expected_max": 0.0, "absolute_max": 0.0},
        "latency": {"p95_max_ms": 300000},
        "privacy": {
            "providers": sorted({endpoint.provider for endpoint in endpoints}),
            "endpoints": [endpoint_id] if endpoint_id else [],
            "boundary": "any",
        },
        "plans": {"allowed": ["direct", "abstain"], "max_calls": 1, "max_candidates": 32},
        "evidence": {
            "require_certificate": False,
            "min_samples": 1,
            "max_age_seconds": 3600,
            "certificate_ttl_seconds": 300,
            "quality_risk": 0.05,
            "latency_risk": 0.05,
        },
        "objective": "latency",
        "on_infeasible": "abstain",
    }


def generated_service(endpoints: list[EndpointSnapshot]) -> dict:
    providers = []
    for endpoint in endpoints:
        if endpoint.provider != "nvidia-build":
            raise ValueError(f"unsupported live provider: {endpoint.provider}")
        providers.append({
            "endpoint_id": endpoint.endpoint_id,
            "kind": "openai_compatible",
            "base_url": "https://integrate.api.nvidia.com/v1",
            "api_key_env": "NVIDIA_API_KEY",
            "token_limit_field": "max_tokens",
        })
    return {
        "endpoints": "endpoints.json",
        "policy": "policy.yaml",
        "capability_map": "capability-map.json",
        "ledger": "service.sqlite3",
        "api_key_env": "SLM_ROUTER_API_KEY",
        "providers": providers,
    }


def run(out: Path, env_file: Path) -> dict:
    if out.exists():
        raise ValueError(f"output directory already exists: {out}")
    out.mkdir(parents=True)
    report: dict[str, object] = {
        "kind": "real_nvidia_functional_matrix",
        "tested_at_utc": datetime.now(UTC).isoformat(),
        "limitations": [
            "This is a small functional test, not a quality, latency, or cost benchmark.",
            "One calibration observation per endpoint cannot issue a meaningful certificate.",
            "Zero prices are taken from the checked-in prototype/free-tier declarations.",
            "No raw prompt, response, provider request ID, or credential is persisted.",
        ],
        "calibration": [],
        "endpoint_routes": [],
    }
    write_json(out / "pilot-report.json", report)

    templates = load_endpoints(ROOT / "examples" / "nvidia-live" / "endpoints.json")
    api_key = secret(env_file, "NVIDIA_API_KEY")
    providers = {
        endpoint.endpoint_id: OpenAICompatibleAdapter(
            "https://integrate.api.nvidia.com/v1", api_key, token_limit_field="max_tokens"
        )
        for endpoint in templates
    }
    ledger: SQLiteLedger | None = None
    try:
        calibration_text, expected = "Return only the number 4.", "4"
        calibration_messages = [{"role": "user", "content": calibration_text}]
        cmap = ConditionalCapabilityMap(k=16, min_samples=1, max_age_seconds=3600)
        endpoints: list[EndpointSnapshot] = []
        for template in templates:
            item: dict[str, object] = {
                "endpoint_id": template.endpoint_id,
                "provider": template.provider,
                "model_requested": template.upstream_model,
            }
            try:
                config = dict(template.inference_config)
                config.update({"max_tokens": MAX_OUTPUT_TOKENS, "timeout": PROVIDER_TIMEOUT_SECONDS})
                response = providers[template.endpoint_id].call(
                    template.upstream_model, calibration_messages, config
                )
                if not response.usage_known:
                    raise RuntimeError("provider returned incomplete token usage")
                if response.observed_model != template.upstream_model:
                    raise RuntimeError("provider returned a different model identity")
                endpoint = template
                if response.observed_revision is not None:
                    endpoint = template.model_copy(update={"revision": response.observed_revision})
                endpoints.append(endpoint)
                request = request_for(
                    POLICY_ID,
                    f"matrix-calibration-{endpoint.endpoint_id}",
                    calibration_messages,
                    calibration_text,
                )
                match = exact_number(response.text, expected)
                cmap.add(EvidenceObservation(
                    sample_id=f"matrix:{endpoint.endpoint_id}:exact-number-4",
                    request=request,
                    target_id=endpoint.endpoint_id,
                    endpoint_ids=(endpoint.endpoint_id,),
                    endpoint_revisions=(endpoint.capability_revision,),
                    quality=float(match),
                    cost=(response.input_tokens * endpoint.input_price_per_million
                          + response.output_tokens * endpoint.output_price_per_million) / 1_000_000,
                    latency_ms=response.latency_ms,
                    output_tokens=response.output_tokens,
                    failed=False,
                    split="calibration",
                    evaluator_type="deterministic",
                    evaluator_version="exact-number-provider-matrix-v1",
                ))
                item.update({
                    "status": "completed",
                    "model_returned_matches": True,
                    "usage_complete": True,
                    "prompt_tokens": response.input_tokens,
                    "completion_tokens": response.output_tokens,
                    "latency_ms": round(response.latency_ms, 1),
                    "deterministic_exact_match": match,
                    "revision_observed": response.observed_revision is not None,
                })
            except Exception as exc:  # noqa: BLE001 - isolate each external provider failure
                item.update({"status": "failed", **provider_error(exc)})
            report["calibration"].append(item)
            write_json(out / "pilot-report.json", report)

        if not endpoints:
            raise RuntimeError("all provider calibration calls failed")

        write_json(out / "endpoints.json", [endpoint.model_dump(mode="json") for endpoint in endpoints])
        write_json(out / "capability-map.json", cmap.export())
        policy_data = policy_document(endpoints)
        (out / "policy.yaml").write_text(yaml.safe_dump(policy_data, sort_keys=False), encoding="utf-8")
        (out / "service.yaml").write_text(
            yaml.safe_dump(generated_service(endpoints), sort_keys=False), encoding="utf-8"
        )

        active_providers = {endpoint.endpoint_id: providers[endpoint.endpoint_id] for endpoint in endpoints}
        bridge = RuntimeProviderAdapter({endpoint.endpoint_id: endpoint for endpoint in endpoints}, active_providers)
        ledger = SQLiteLedger(out / "pilot.sqlite3")
        default_policy = compile_policy(policy_data)
        control = ControlPlane(
            Planner(endpoints, capability_map=cmap),
            ledger,
            Executor(bridge),
            default_policy=default_policy,
        )

        route_text, route_expected = "Return only the number 7.", "7"
        route_messages = [{"role": "user", "content": route_text}]
        for endpoint in endpoints:
            forced_id = f"matrix-{endpoint.endpoint_id}"
            forced_policy = compile_policy(
                policy_document(endpoints, policy_id=forced_id, endpoint_id=endpoint.endpoint_id)
            )
            decision = control.decide(DecideRequest(
                request=request_for(
                    forced_id,
                    f"matrix-route-{endpoint.endpoint_id}",
                    route_messages,
                    route_text,
                ),
                policy=forced_policy,
            ))
            selected = []
            execution = control.execute(
                decision.decision_id,
                messages=route_messages,
                result_sink=selected.append,
            )
            report["endpoint_routes"].append({
                "endpoint_id": endpoint.endpoint_id,
                "provider": endpoint.provider,
                "plan_type": decision.selected_plan.plan_type,
                "execution_state": execution.state,
                "selected_endpoint_id": execution.selected_endpoint_id,
                "usage": {
                    "prompt_tokens": execution.input_tokens,
                    "completion_tokens": execution.output_tokens,
                },
                "latency_ms": round(execution.total_latency_ms, 1),
                "accounting_complete": execution.accounting_complete,
                "constraint_violations": list(execution.constraint_violations),
                "provider_errors": list(execution.provider_errors),
                "deterministic_exact_match": bool(
                    selected and exact_number((selected[0].message or {}).get("content"), route_expected)
                ),
                "replay_matched": control.replay(decision.decision_id)["matched"],
            })
            write_json(out / "pilot-report.json", report)

        combined = control.decide(DecideRequest(
            request=request_for(
                POLICY_ID,
                "matrix-combined-decision",
                route_messages,
                route_text,
            ),
            policy=default_policy,
        ))
        report["combined_routing"] = {
            "decision_id": combined.decision_id,
            "plan_type": combined.selected_plan.plan_type,
            "selected_endpoint_id": (
                combined.selected_plan.steps[0].endpoint_id
                if combined.selected_plan.plan_type != "abstain" else None
            ),
            "eligible_endpoints": list(combined.eligible_endpoints),
            "rejected_alternatives": [item.model_dump(mode="json") for item in combined.rejected_alternatives],
            "certificate_status": combined.certificate_status,
            "replay_matched": control.replay(combined.decision_id)["matched"],
        }
        report["ledger_verified"] = ledger.verify()
        report["successful_endpoints"] = [endpoint.endpoint_id for endpoint in endpoints]
        report["failed_endpoints"] = [
            item["endpoint_id"] for item in report["calibration"] if item["status"] == "failed"
        ]
        report["generated_service_config"] = "service.yaml"
        write_json(out / "pilot-report.json", report)
        return report
    finally:
        if ledger is not None:
            ledger.close()
        for provider in providers.values():
            provider.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a live NVIDIA provider matrix.")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "nvidia-matrix-live")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env.local")
    args = parser.parse_args()
    try:
        report = run(args.out.resolve(), args.env_file.resolve())
    except Exception as exc:  # noqa: BLE001 - CLI must leave a sanitized failure report
        print(f"Provider matrix failed: {type(exc).__name__}. See {args.out / 'pilot-report.json'}", file=sys.stderr)
        return 1
    print(json.dumps({
        "report": str(args.out / "pilot-report.json"),
        "successful_endpoints": report["successful_endpoints"],
        "failed_endpoints": report["failed_endpoints"],
        "combined_selected_endpoint_id": report["combined_routing"]["selected_endpoint_id"],
        "ledger_verified": report["ledger_verified"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
