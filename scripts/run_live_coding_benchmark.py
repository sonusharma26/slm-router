"""Frozen live coding evidence and actual routed holdout execution."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from time import perf_counter

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_nvidia_live_pilot import load_env_value, normalize_secret

from inference_control.adapters.providers import OpenAICompatibleAdapter
from inference_control.adapters.runtime import RuntimeProviderAdapter
from inference_control.api.app import ControlPlane, DecideRequest
from inference_control.benchmarks.coding import CodingTask, prepare_tasks, score_response, task_messages
from inference_control.capability.conditional import (
    ConditionalCapabilityMap,
    EvidenceObservation,
    quantile_tolerance_rank,
    stratum,
)
from inference_control.configuration import load_endpoints
from inference_control.contracts import Call, EndpointSnapshot, PolicySpec, RequestContext
from inference_control.execution import Executor
from inference_control.features.query import query_features
from inference_control.ledger import SQLiteLedger
from inference_control.planning import Planner
from inference_control.util import digest

COUNTS = {"train": 16, "calibration": 128, "validation": 16, "test": 32}
OUTPUT_CAP = 1024
TIMEOUT = 8.0


@dataclass(frozen=True)
class Receipt:
    call_id: str
    task_id: str
    endpoint_id: str
    arm: str
    split: str
    family: str
    observed_at: str
    success: bool
    quality: float
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    usage_known: bool = False
    model_matches: bool = False
    revision_observed: bool = False
    output_hash: str | None = None
    rejection: str | None = None
    error_type: str | None = None
    http_status: int | None = None
    actual_billed_cost: float | None = None
    provider_latency_ms: float | None = None
    execution_latency_ms: float | None = None
    decision_ms: float = 0
    constraint_violations: tuple[str, ...] = ()
    preflight_expression: str | None = None


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def request_for(task):
    messages = task_messages(task)
    return RequestContext(
        request_id=task.task_id, application_id="live-bounded-coding-v1", tenant_policy_id="live-coding",
        input_tokens=len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")).encode("utf-8")),
        max_output_tokens=OUTPUT_CAP, task_hint="bounded-python-expression-v1",
        traffic_slices=frozenset({"bounded-coding"}), feature_version="hash-text-v1",
        query_features=query_features(task.prompt),
    )


def policy_for(deadline):
    return PolicySpec(
        policy_id="live-coding", version=f"quality065-risk005-deadline{deadline}-v1", minimum_quality=.65,
        quality_risk=.05, latency_risk=.05, max_expected_spend=0, max_absolute_spend=0,
        deadline_ms=deadline, minimum_evidence_samples=8, require_certificate=False,
        objective="latency", permitted_plan_types=frozenset({"direct", "abstain"}),
    )


def candidates():
    endpoints = load_endpoints(ROOT / "examples/nvidia-live/endpoints.json")
    template = endpoints[0].model_dump()
    extras = [
        ("nvidia-fast-low", "openai/gpt-oss-20b", {"temperature": 1.0, "top_p": 1.0, "reasoning_effort": "low"}),
        ("nvidia-lightning", "nvidia/nemotron-3.5-lightning-30b-a3b", {
            "temperature": 1.0, "top_p": .95, "chat_template_kwargs": {"enable_thinking": False}}),
        ("nvidia-lightning-coding", "nvidia/nemotron-3.5-lightning-30b-a3b", {
            "temperature": 0.0, "top_p": .95, "chat_template_kwargs": {"enable_thinking": False}}),
        ("nvidia-gemma-small", "google/gemma-3-4b-it", {"temperature": 0.0, "top_p": 1.0}),
        ("nvidia-codegemma", "google/codegemma-1.1-7b", {"temperature": 0.0, "top_p": 1.0}),
    ]
    for endpoint_id, model, config in extras:
        endpoints.append(EndpointSnapshot.model_validate({
            **template, "endpoint_id": endpoint_id, "upstream_model": model, "inference_config": config,
            "revision": "nvidia-build-catalog-unversioned-2026-10-07", "config_hash": digest(config),
        }))
    return endpoints


def prepare(out, candidate_ids=None, *, seed=43, prompt_style="compact", calibration_requests=128):
    counts = {**COUNTS, "calibration": calibration_requests}
    if calibration_requests < 128:
        raise ValueError("at least 128 calibration requests are required")
    tasks = prepare_tasks(seed=seed, counts=counts, prompt_style=prompt_style)
    requests = [request_for(task) for task in tasks]
    if Counter(task.split for task in tasks) != Counter(counts):
        raise ValueError("task split counts differ from the frozen design")
    if len({stratum(request) for request in requests}) != 1:
        raise ValueError("workload crosses exact evidence strata")
    if len({task.group_id for task in tasks}) != len(tasks):
        raise ValueError("duplicate coding specifications")
    for task in tasks:
        if score_response(task, json.dumps({"expression": task.reference_expression})).quality != 1:
            raise ValueError("trusted solution fails the hidden grader")
    endpoints = candidates()
    if candidate_ids:
        available = {endpoint.endpoint_id for endpoint in endpoints}
        if not set(candidate_ids) <= available or len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate IDs must be unique known endpoints")
        endpoints = [endpoint for endpoint in endpoints if endpoint.endpoint_id in candidate_ids]
    if not endpoints:
        raise ValueError("at least one candidate endpoint is required")
    selected_max = min(2, len(endpoints))
    manifest = {
        "kind": "bounded_python_expression_live_benchmark", "created_at_utc": datetime.now(UTC).isoformat(),
        "counts": counts, "seed": seed, "prompt_style": prompt_style, "quality_target": .65, "risk": .05, "primary_deadline_ms": 2500,
        "secondary_deadline_ms": 5000, "max_output_tokens": OUTPUT_CAP, "measurement_timeout_seconds": TIMEOUT,
        "pricing_status": "unknown; nominal zero snapshot prices are planner placeholders only",
        "production_certification_eligible": False, "preflight_call_cap": 2 * len(endpoints),
        "total_call_cap": 2 * len(endpoints) + selected_max * sum(counts[split] for split in ("train", "calibration", "validation")) + (2 + selected_max) * counts["test"],
        "futility_rule": "Stop only when every selected endpoint has more calibration latencies above 5000ms than the planned binomial tolerance rank can tolerate; both deadline profiles are then impossible irrespective of future samples.",
        "candidate_selection": "at least one accounted model-matching correct preflight completion; rank deadline-compatible successes then correctness then total latency then id; calibration decides quality and failures",
        "input_admission_bytes": {"min": min(r.input_tokens for r in requests), "max": max(r.input_tokens for r in requests)},
        "exact_stratum": stratum(requests[0]), "tasks": [asdict(task) for task in tasks],
        "endpoints": [endpoint.model_dump(mode="json") for endpoint in endpoints],
        "policies": [policy_for(value).model_dump(mode="json") for value in (2500, 5000)],
        "source_hashes": {name: digest((ROOT / name).read_text(encoding="utf-8")) for name in (
            "src/inference_control/benchmarks/coding.py", "scripts/run_live_coding_benchmark.py",
            "src/inference_control/contracts/models.py", "src/inference_control/capability/conditional.py",
            "src/inference_control/planning/planner.py", "src/inference_control/execution/runtime.py",
            "src/inference_control/api/app.py", "src/inference_control/adapters/providers.py",
            "src/inference_control/adapters/runtime.py")},
    }
    manifest["fingerprint"] = digest(manifest)
    out.mkdir(parents=True, exist_ok=False)
    for name in manifest["source_hashes"]:
        target = out / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((ROOT / name).read_text(encoding="utf-8"), encoding="utf-8")
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"prepared": str(out), "counts": counts, "input_bytes": manifest["input_admission_bytes"]}), flush=True)


def load_run(out):
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    fingerprint = manifest.pop("fingerprint")
    if digest(manifest) != fingerprint:
        raise ValueError("manifest changed after preparation")
    for name, expected in manifest["source_hashes"].items():
        if digest((ROOT / name).read_text(encoding="utf-8")) != expected:
            raise ValueError("benchmark source changed after preparation")
    tasks = [CodingTask(**{**row, "cases": tuple((case[0], case[1]) for case in row["cases"])}) for row in manifest["tasks"]]
    endpoints = [EndpointSnapshot.model_validate(row) for row in manifest["endpoints"]]
    return manifest, tasks, endpoints


class Attempts:
    def __init__(self, out, cap):
        self.path = out / "attempts.jsonl"
        self.cap = cap
        self.started = set()
        self.results = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["kind"] == "started":
                    self.started.add(row["call_id"])
                else:
                    self.results[row["receipt"]["call_id"]] = row["receipt"]
        if self.started - self.results.keys():
            raise ValueError("an attempt has an uncertain outcome; automatic retry is forbidden")

    def append(self, row):
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def claim(self, call_id):
        if call_id in self.results:
            return self.results[call_id]
        if len(self.started) >= self.cap:
            raise ValueError("frozen call cap exhausted")
        self.append({"kind": "started", "call_id": call_id})
        self.started.add(call_id)
        return None

    def complete(self, receipt):
        row = asdict(receipt)
        self.append({"kind": "completed", "receipt": row})
        self.results[receipt.call_id] = row
        return row


def measure(task, endpoint, adapter, attempts, arm):
    call_id = f"{arm}:{task.task_id}:{endpoint.endpoint_id}"
    existing = attempts.claim(call_id)
    if existing is not None:
        return existing
    started = perf_counter()
    observed = datetime.now(UTC).isoformat()
    try:
        response = adapter.call(endpoint.upstream_model, task_messages(task), {
            **endpoint.inference_config, "max_completion_tokens": OUTPUT_CAP, "timeout": TIMEOUT})
        caller_ms = (perf_counter() - started) * 1000
        score = score_response(task, response.text or "")
        expression = None
        if arm == "preflight":
            try:
                expression = json.loads(response.text or "").get("expression")
                if not isinstance(expression, str) or len(expression) > 4096:
                    expression = None
            except (ValueError, AttributeError):
                pass
        model_matches = response.observed_model == endpoint.upstream_model
        admitted = response.input_tokens <= request_for(task).input_tokens + endpoint.input_token_overhead and response.output_tokens <= OUTPUT_CAP
        receipt = Receipt(call_id, task.task_id, endpoint.endpoint_id, arm, task.split, task.family, observed,
            True, score.quality if model_matches and response.usage_known and admitted else 0, caller_ms,
            response.input_tokens, response.output_tokens, response.usage_known, model_matches,
            response.observed_revision is not None, digest(response.text or ""), score.rejection,
            provider_latency_ms=response.latency_ms, preflight_expression=expression)
    except (httpx.HTTPError, TimeoutError, ValueError, KeyError, TypeError, IndexError) as exc:
        receipt = Receipt(call_id, task.task_id, endpoint.endpoint_id, arm, task.split, task.family, observed,
            False, 0, (perf_counter() - started) * 1000, error_type=type(exc).__name__,
            http_status=getattr(getattr(exc, "response", None), "status_code", None))
    return attempts.complete(receipt)


class IdentityCheckedAdapter(OpenAICompatibleAdapter):
    def call(self, endpoint_id, messages, config):
        response = super().call(endpoint_id, messages, config)
        if response.observed_model != endpoint_id:
            raise ValueError("provider model identity was not verified")
        return response


def providers_for(endpoints):
    key = normalize_secret(os.environ.get("NVIDIA_API_KEY") or load_env_value(ROOT / ".env.local", "NVIDIA_API_KEY"))
    return {endpoint.endpoint_id: IdentityCheckedAdapter(
        "https://integrate.api.nvidia.com/v1", key, token_limit_field="max_tokens") for endpoint in endpoints}


def preflight(out):
    manifest, tasks, endpoints = load_run(out)
    probes = [task for task in tasks if task.split == "train"][:2]
    attempts = Attempts(out, manifest["total_call_cap"])
    providers = providers_for(endpoints)
    rows = []
    try:
        for endpoint in endpoints:
            samples = [measure(task, endpoint, providers[endpoint.endpoint_id], attempts, "preflight") for task in probes]
            row = {"endpoint_id": endpoint.endpoint_id, "samples": samples,
                "correct": sum(sample["quality"] == 1 for sample in samples),
                "within_primary": sum(sample["quality"] == 1 and sample["latency_ms"] <= 2500 for sample in samples)}
            rows.append(row)
            print(json.dumps({"stage": "preflight", "endpoint": endpoint.endpoint_id, "correct": row["correct"],
                "within_primary": row["within_primary"], "latencies_ms": [round(sample["latency_ms"]) for sample in samples]}), flush=True)
    finally:
        for provider in providers.values():
            provider.close()
    viable = sorted((row for row in rows if row["correct"] >= 1),
        key=lambda row: (-row["within_primary"], -row["correct"], sum(sample["latency_ms"] for sample in row["samples"]), row["endpoint_id"]))
    report = {"results": rows, "selected_endpoint_ids": [row["endpoint_id"] for row in viable[:2]],
              "small_sample_selection_only": True,
              "selection_mode": "constrained_model_selection" if len(viable[:2]) == 1 else "two_endpoint_routing",
              "adaptive_gain_claimed": False}
    write_json(out / "preflight.json", report)
    print(json.dumps({"selected": report["selected_endpoint_ids"]}), flush=True)


def percentile(values, quantile):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * quantile
    lower = int(position)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)


def summarize(rows, deadline):
    served = [row for row in rows if row.get("endpoint_id")]
    return {"requests": len(rows), "served": len(served), "correct": sum(row["quality"] == 1 for row in rows),
        "on_time_correct": sum(row["quality"] == 1 and row["latency_ms"] <= deadline for row in rows),
        "all_request_quality": mean(row["quality"] for row in rows) if rows else None,
        "served_quality": mean(row["quality"] for row in served) if served else None,
        "failed": sum(not row["success"] for row in served),
        "deadline_misses": sum(row["latency_ms"] > deadline for row in served),
        "p50_latency_ms": percentile([row["latency_ms"] for row in served], .5),
        "p95_latency_ms_descriptive": percentile([row["latency_ms"] for row in served], .95),
        "p50_provider_latency_ms": percentile([row["provider_latency_ms"] for row in served if row.get("provider_latency_ms") is not None], .5),
        "p50_decision_ms": percentile([row.get("decision_ms", 0) for row in served], .5),
        "input_tokens": sum(row["input_tokens"] for row in served), "output_tokens": sum(row["output_tokens"] for row in served),
        "timeout_failures": sum("Timeout" in (row.get("error_type") or "") for row in served),
        "usage_known_requests": sum(row["usage_known"] for row in served),
        "identity_verified_requests": sum(row["model_matches"] for row in served),
        "rejection_counts": dict(Counter(row["rejection"] for row in served if row.get("rejection"))),
        "per_family": {family: {"requests": len(group), "correct": sum(row["quality"] == 1 for row in group),
            "on_time_correct": sum(row["quality"] == 1 and row["latency_ms"] <= deadline for row in group)}
            for family in sorted({row["family"] for row in rows})
            for group in [[row for row in rows if row["family"] == family]]},
        "actual_billed_cost": None, "selected_endpoints": dict(Counter(row["endpoint_id"] for row in served))}


def collect(out):
    manifest, tasks, endpoints = load_run(out)
    selected = json.loads((out / "preflight.json").read_text(encoding="utf-8"))["selected_endpoint_ids"]
    if not 1 <= len(selected) <= 2:
        raise ValueError("one or two models must pass bounded coding preflight")
    endpoints = [endpoint for endpoint in endpoints if endpoint.endpoint_id in selected]
    attempts = Attempts(out, manifest["total_call_cap"])
    providers = providers_for(endpoints)
    rng = random.Random(manifest["seed"])
    rows = []
    try:
        for index, task in enumerate(task for task in tasks if task.split != "test"):
            order = list(endpoints)
            rng.shuffle(order)
            rows.extend(measure(task, endpoint, providers[endpoint.endpoint_id], attempts, "evidence") for endpoint in order)
            calibration_count = manifest["counts"]["calibration"]
            rank = quantile_tolerance_rank(calibration_count, .95, .05 / len(endpoints))
            allowed_exceedances = calibration_count - rank if rank is not None else -1
            misses = {endpoint.endpoint_id: sum(row["split"] == "calibration" and row["latency_ms"] > 5000
                for row in rows if row["endpoint_id"] == endpoint.endpoint_id) for endpoint in endpoints}
            if all(value > allowed_exceedances for value in misses.values()):
                write_json(out / "evidence.json", rows)
                write_json(out / "report.json", {
                    "kind": manifest["kind"], "complete_benchmark": False, "stage": "calibration_futility",
                    "both_profiles_mathematically_futile": True, "planned_calibration_requests": calibration_count,
                    "planned_binomial_tolerance_rank": rank, "calibration_exceedances_above_5000_ms": misses,
                    "completed_provider_calls": len(attempts.results), "ended_calls": list(attempts.results),
                    "actual_provider_attempts": len(attempts.results), "abstention_claims": 0,
                    "evidence_rows": len(rows), "production_certification_eligible": False,
                    "evidence_summary": {endpoint.endpoint_id: summarize(
                        [row for row in rows if row["endpoint_id"] == endpoint.endpoint_id], 5000) for endpoint in endpoints},
                    "actual_billed_cost": None,
                    "limitations": ["Partial evidence cannot support a full calibration map or certificate."],
                })
                print(json.dumps({"stage": "calibration_futility", "exceedances": misses}), flush=True)
                return
            if (index + 1) % 8 == 0:
                print(json.dumps({"stage": "evidence", "completed_tasks": index + 1,
                    "summary": {endpoint.endpoint_id: summarize([row for row in rows if row["endpoint_id"] == endpoint.endpoint_id], 2500) for endpoint in endpoints}}), flush=True)
    finally:
        for provider in providers.values():
            provider.close()
    write_json(out / "evidence.json", rows)
    cmap = ConditionalCapabilityMap(k=manifest["counts"]["calibration"], min_samples=8, quality_method="local-mean-kl")
    by_id = {task.task_id: task for task in tasks}
    by_endpoint = {endpoint.endpoint_id: endpoint for endpoint in endpoints}
    observations = [EvidenceObservation(
        sample_id=row["call_id"], request=request_for(by_id[row["task_id"]]), target_id=row["endpoint_id"],
        endpoint_ids=(row["endpoint_id"],), endpoint_revisions=(by_endpoint[row["endpoint_id"]].capability_revision,),
        quality=row["quality"], cost=0, latency_ms=row["latency_ms"], output_tokens=row["output_tokens"],
        failed=not row["success"], split=row["split"], evaluator_type="benchmark", certification_eligible=False,
        evaluator_version="bounded-python-expression-v1", observed_at=datetime.fromisoformat(row["observed_at"]),
        metrics=frozenset({"quality", "failure", "latency"}),
    ) for row in rows if row["split"] in {"train", "calibration"}]
    cmap.add_many(observations)
    expected = manifest["counts"]["train"] + manifest["counts"]["calibration"]
    if Counter(row.target_id for row in cmap.rows.values()) != Counter({endpoint.endpoint_id: expected for endpoint in endpoints}):
        raise ValueError("persisted evidence count differs from the frozen design")
    write_json(out / "capability-map.json", cmap.export())
    planner = Planner(endpoints, capability_map=cmap)
    representative = request_for(next(task for task in tasks if task.split == "test"))
    assessment = {}
    for deadline in (2500, 5000):
        policy = policy_for(deadline)
        value = planner.assess(representative, policy)
        assessment[str(deadline)] = {
            "selected": value.selection[0].evidence_key if value.selection else None,
            "candidates": [{"endpoint_id": plan.evidence_key, "quality_lower": estimate.quality.lower,
                "latency_p95_upper_ms": estimate.latency.upper, "calibration_count": estimate.lineage.calibration_size,
                "reasons": planner.reasons(plan, estimate, policy), "certification_reasons": estimate.lineage.reasons}
                for plan, estimate in value.candidates if estimate],
        }
    write_json(out / "assessment.json", assessment)
    print(json.dumps({"assessment": assessment}), flush=True)


def holdout(out):
    manifest, tasks, endpoints = load_run(out)
    selected = json.loads((out / "preflight.json").read_text(encoding="utf-8"))["selected_endpoint_ids"]
    endpoints = [endpoint for endpoint in endpoints if endpoint.endpoint_id in selected]
    attempts = Attempts(out, manifest["total_call_cap"])
    providers = providers_for(endpoints)
    if not 1 <= len(endpoints) <= 2:
        raise ValueError("one or two selected endpoints are required")
    frozen_map = json.loads((out / "capability-map.json").read_text(encoding="utf-8"))
    profiles = {}
    for deadline in (2500, 5000):
        policy = policy_for(deadline)
        ledger = SQLiteLedger(out / f"routed-{deadline}.sqlite3")
        control = ControlPlane(Planner(endpoints, capability_map=ConditionalCapabilityMap.restore(frozen_map)), ledger,
            Executor(RuntimeProviderAdapter({endpoint.endpoint_id: endpoint for endpoint in endpoints}, providers)), default_policy=policy)
        profiles[f"router-{deadline}"] = (deadline, policy, ledger, control)
    rows, choices = [], []
    rng = random.Random(73)
    try:
        for task in (task for task in tasks if task.split == "test"):
            decisions = {}
            for arm, (_, policy, _, control) in profiles.items():
                began = perf_counter()
                decision = control.decide(DecideRequest(request=request_for(task), policy=policy))
                decision_ms = (perf_counter() - began) * 1000
                calls = [step for step in decision.selected_plan.steps if isinstance(step, Call)]
                decisions[arm] = (decision, decision_ms, calls)
                choices.append({"task_id": task.task_id, "arm": arm,
                    "selected": calls[0].endpoint_id if calls else None,
                    "certificate_status": decision.certificate_status,
                    "rejected": [row.model_dump(mode="json") for row in decision.rejected_alternatives]})
            write_json(out / "choices.json", choices)
            order = [*profiles, *[endpoint.endpoint_id for endpoint in endpoints]]
            rng.shuffle(order)
            for arm in order:
                if arm not in profiles:
                    endpoint = next(endpoint for endpoint in endpoints if endpoint.endpoint_id == arm)
                    rows.append(measure(task, endpoint, providers[arm], attempts, "fixed"))
                    continue
                _, _, _, control = profiles[arm]
                decision, decision_ms, calls = decisions[arm]
                call_id = f"{arm}:{task.task_id}"
                existing = attempts.claim(call_id)
                if existing is not None:
                    raise ValueError("held-out router execution already exists; do not repeat it")
                routed_start = perf_counter()
                results = []
                execution = control.execute(decision.decision_id, messages=task_messages(task), result_sink=results.append)
                control_ms = (perf_counter() - routed_start) * 1000
                text = (results[0].message or {}).get("content") if results else None
                score = score_response(task, text or "") if text else None
                row = attempts.complete(Receipt(call_id, task.task_id, calls[0].endpoint_id if calls else "",
                    arm, task.split, task.family, datetime.now(UTC).isoformat(), execution.state == "completed",
                    score.quality if score else 0, decision_ms + control_ms,
                    execution.input_tokens, execution.output_tokens, execution.accounting_complete,
                    bool(results), results[0].revision_observed if results else False,
                    digest(text) if text else None, score.rejection if score else None,
                    execution.provider_errors[0] if execution.provider_errors else None,
                    provider_latency_ms=results[0].provider_latency_ms if results else None,
                    execution_latency_ms=execution.total_latency_ms, decision_ms=decision_ms,
                    constraint_violations=tuple(execution.constraint_violations)))
                rows.append(row)
            write_json(out / "holdout.json", rows)
            print(json.dumps({"stage": "holdout", "task": task.task_id, "route": choices[-1]["selected"]}), flush=True)
        report = {
            "kind": manifest["kind"], "tested_at_utc": datetime.now(UTC).isoformat(),
            "complete_benchmark": True,
            "primary_policy": policy_for(2500).model_dump(mode="json"),
            "profile_policies": {arm: policy.model_dump(mode="json") for arm, (_, policy, _, _) in profiles.items()},
            "ledger_verified": all(ledger.verify() for _, _, ledger, _ in profiles.values()),
            "profile_ledgers_verified": {arm: ledger.verify() for arm, (_, _, ledger, _) in profiles.items()},
            "production_certification_eligible": False, "recorded_attempts_including_abstentions": len(attempts.started),
            "actual_provider_attempts": sum(row["arm"] not in profiles for row in attempts.results.values())
                + sum(event.payload["attempted_calls"] for _, _, ledger, _ in profiles.values() for event in ledger.events("execution")),
            "abstention_claims": sum(not choice["selected"] for choice in choices),
            "selection_mode": "constrained_model_selection" if len(endpoints) == 1 else "two_endpoint_routing",
            "adaptive_gain_claimed": False,
            "arms": {arm: summarize([row for row in rows if row["arm"] == arm], deadline)
                for arm, (deadline, _, _, _) in profiles.items()},
            "fixed_endpoints": {endpoint.endpoint_id: summarize([row for row in rows if row["arm"] == "fixed" and row["endpoint_id"] == endpoint.endpoint_id], 2500) for endpoint in endpoints},
            "fixed_endpoints_by_profile": {str(deadline): {endpoint.endpoint_id: summarize(
                [row for row in rows if row["arm"] == "fixed" and row["endpoint_id"] == endpoint.endpoint_id], deadline)
                for endpoint in endpoints} for deadline in (2500, 5000)},
            "assessment": json.loads((out / "assessment.json").read_text(encoding="utf-8")),
            "limitations": ["Narrow generated Python-expression tasks; not general coding or production traffic.",
                "Unknown billed cost and unverified revisions prevent dollar savings and production certification claims.",
                "Two preflight tasks select candidates; preflight observations are excluded from fitted evidence.",
                "Held-out hidden scores never update the capability map. Actual runtime drift protection remains enabled.",
                "Latency percentiles include failed attempts and measure request termination; timeouts are censored completions.",
                "Small single-run heldout comparison is exploratory; it cannot establish a general speedup or quality noninferiority."],
        }
        paired = {}
        for arm in profiles:
            routed = {row["task_id"]: row for row in rows if row["arm"] == arm}
            paired[arm] = {}
            for endpoint in endpoints:
                fixed = {row["task_id"]: row for row in rows if row["arm"] == "fixed" and row["endpoint_id"] == endpoint.endpoint_id}
                differences = [routed[task_id]["quality"] - fixed[task_id]["quality"] for task_id in sorted(routed)]
                bootstrap_rng = random.Random(manifest["seed"])
                samples = [mean(bootstrap_rng.choices(differences, k=len(differences))) for _ in range(2000)]
                paired[arm][endpoint.endpoint_id] = {"quality_difference": mean(differences),
                    "paired_bootstrap_95_interval": [percentile(samples, .025), percentile(samples, .975)],
                    "router_wins": sum(value > 0 for value in differences), "router_losses": sum(value < 0 for value in differences),
                    "exploratory_only": True, "noninferiority_proven": False}
        report["paired_quality_comparisons"] = paired
        write_json(out / "report.json", report)
        print(json.dumps({"report": str(out / "report.json"), "routers": report["arms"], "fixed": report["fixed_endpoints"]}), flush=True)
    finally:
        for _, _, ledger, _ in profiles.values():
            ledger.close()
        for provider in providers.values():
            provider.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "preflight", "collect", "holdout"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidate-ids", nargs="+")
    parser.add_argument("--prompt-style", choices=("compact", "verbose"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--calibration-requests", type=int)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.out, args.candidate_ids, seed=args.seed if args.seed is not None else 43,
            prompt_style=args.prompt_style or "compact", calibration_requests=args.calibration_requests or 128)
    elif args.candidate_ids or args.prompt_style is not None or args.seed is not None or args.calibration_requests is not None:
        parser.error("candidate IDs, prompt style, seed and calibration count are frozen by prepare")
    else:
        {"preflight": preflight, "collect": collect, "holdout": holdout}[args.stage](args.out)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, httpx.HTTPError, TimeoutError, KeyError, TypeError, IndexError) as exc:
        print(json.dumps({"benchmark_error_type": type(exc).__name__}), file=sys.stderr)
        raise SystemExit(1) from None
