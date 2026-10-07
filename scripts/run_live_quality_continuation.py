"""Finish fixed-count quality diagnosis after the stricter latency experiment stops."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import httpx
from run_live_coding_benchmark import (
    TIMEOUT,
    Attempts,
    IdentityCheckedAdapter,
    Receipt,
    load_run,
    measure,
    policy_for,
    providers_for,
    request_for,
    summarize,
    write_json,
)

from inference_control.adapters.runtime import RuntimeProviderAdapter
from inference_control.api.app import ControlPlane, DecideRequest
from inference_control.benchmarks.coding import score_response, task_messages
from inference_control.capability.conditional import ConditionalCapabilityMap, EvidenceObservation
from inference_control.contracts import Call
from inference_control.execution import Executor
from inference_control.ledger import SQLiteLedger
from inference_control.planning import Planner
from inference_control.util import digest


class FixedWaitAdapter(IdentityCheckedAdapter):
    def call(self, endpoint_id, messages, config):
        config = {**config, "timeout": min(float(config.get("timeout", TIMEOUT)), TIMEOUT)}
        return super().call(endpoint_id, messages, config)


def run(out):
    manifest, tasks, endpoints = load_run(out)
    selected = json.loads((out / "preflight.json").read_text(encoding="utf-8"))["selected_endpoint_ids"]
    endpoints = [endpoint for endpoint in endpoints if endpoint.endpoint_id in selected]
    if len(endpoints) != 1:
        raise ValueError("this quality continuation studies the single selected configuration")
    if not (out / "strict-profile-report.json").exists():
        write_json(out / "strict-profile-report.json", json.loads((out / "report.json").read_text(encoding="utf-8")))
    decision = {
        "purpose": "complete the original128 quality-calibration observations; stricter latency failure remains recorded",
        "quality_target": .65, "risk": .05, "latency_diagnostic_only_deadline_ms": 10000,
        "provider_wait_cap_seconds": TIMEOUT, "posthoc_diagnostic": True,
        "reuse": "all completed original evidence, including failures; no uncertain attempt repeats",
        "done": "fixed counts, quality lower bound, unchanged strict policy assessments,32 actual diagnostic holdout requests",
        "maximum_total_provider_attempts": manifest["total_call_cap"],
        "source_hash": digest(Path(__file__).read_text(encoding="utf-8")),
    }
    path = out / "quality-continuation.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != decision:
        raise ValueError("quality continuation design changed")
    write_json(path, decision)
    attempts = Attempts(out, manifest["total_call_cap"])
    providers = providers_for(endpoints)
    endpoint = endpoints[0]
    rows = []
    try:
        for index, task in enumerate(task for task in tasks if task.split != "test"):
            rows.append(measure(task, endpoint, providers[endpoint.endpoint_id], attempts, "evidence"))
            if (index + 1) % 16 == 0:
                print(json.dumps({"stage": "quality_diagnosis", "completed": index + 1,
                    "metrics": summarize(rows, 2500)}), flush=True)
    finally:
        for provider in providers.values():
            provider.close()
    write_json(out / "quality-evidence.json", rows)
    by_task = {task.task_id: task for task in tasks}
    cmap = ConditionalCapabilityMap(k=manifest["counts"]["calibration"], min_samples=8, quality_method="local-mean-kl")
    observations = [EvidenceObservation(sample_id=row["call_id"], request=request_for(by_task[row["task_id"]]),
        target_id=endpoint.endpoint_id, endpoint_ids=(endpoint.endpoint_id,), endpoint_revisions=(endpoint.capability_revision,),
        quality=row["quality"], cost=0, latency_ms=row["latency_ms"], output_tokens=row["output_tokens"],
        failed=not row["success"], split=row["split"], evaluator_type="benchmark", certification_eligible=False,
        evaluator_version="bounded-python-expression-v1", observed_at=datetime.fromisoformat(row["observed_at"]),
        metrics=frozenset({"quality", "failure", "latency"}))
        for row in rows if row["split"] in {"train", "calibration"}]
    cmap.add_many(observations)
    if len(cmap.rows) != manifest["counts"]["train"] + manifest["counts"]["calibration"]:
        raise ValueError("quality evidence incomplete")
    write_json(out / "quality-capability-map.json", cmap.export())
    planner = Planner(endpoints, capability_map=cmap)
    representative = request_for(next(task for task in tasks if task.split == "test"))
    assessments = {}
    for deadline in (2500, 5000, 10000):
        policy = policy_for(deadline)
        assessment = planner.assess(representative, policy)
        assessments[str(deadline)] = {"selected": assessment.selection[0].evidence_key if assessment.selection else None,
            "candidates": [{"quality_lower": estimate.quality.lower, "latency_p95_upper_ms": estimate.latency.upper,
                "calibration_count": estimate.lineage.calibration_size,
                "reasons": planner.reasons(plan, estimate, policy)} for plan, estimate in assessment.candidates if estimate]}
    write_json(out / "quality-assessment.json", assessments)
    print(json.dumps({"assessment": assessments}), flush=True)
    policy = policy_for(10000)
    providers = providers_for(endpoints)
    original = providers[endpoint.endpoint_id]
    providers[endpoint.endpoint_id] = FixedWaitAdapter(original.base_url, original.api_key, original.client,
        token_limit_field="max_tokens")
    ledger = SQLiteLedger(out / "diagnostic-10000.sqlite3")
    control = ControlPlane(Planner(endpoints, capability_map=cmap), ledger,
        Executor(RuntimeProviderAdapter({endpoint.endpoint_id: endpoint}, providers)), default_policy=policy)
    heldout = []
    try:
        for task in (task for task in tasks if task.split == "test"):
            call_id = f"diagnostic-router:{task.task_id}"
            if attempts.claim(call_id) is not None:
                raise ValueError("do not repeat a completed diagnostic routed execution")
            began = perf_counter()
            route = control.decide(DecideRequest(request=request_for(task), policy=policy))
            decision_ms = (perf_counter() - began) * 1000
            calls = [step for step in route.selected_plan.steps if isinstance(step, Call)]
            results = []
            execution = control.execute(route.decision_id, messages=task_messages(task), result_sink=results.append)
            elapsed = (perf_counter() - began) * 1000
            text = (results[0].message or {}).get("content") if results else None
            score = score_response(task, text or "") if text else None
            heldout.append(attempts.complete(Receipt(call_id, task.task_id, endpoint.endpoint_id if calls else "",
                "diagnostic-router", "test", task.family, datetime.now(UTC).isoformat(), execution.state == "completed",
                score.quality if score else 0, elapsed, execution.input_tokens, execution.output_tokens,
                execution.accounting_complete, bool(results), results[0].revision_observed if results else False,
                digest(text) if text else None, score.rejection if score else None,
                execution.provider_errors[0] if execution.provider_errors else None,
                provider_latency_ms=results[0].provider_latency_ms if results else None,
                execution_latency_ms=execution.total_latency_ms, decision_ms=decision_ms,
                constraint_violations=tuple(execution.constraint_violations))))
            print(json.dumps({"stage": "diagnostic_holdout", "completed": len(heldout), "quality": heldout[-1]["quality"]}), flush=True)
        report = {"kind": "posthoc_live_quality_diagnosis_and_10000ms_routed_pilot",
            "complete_quality_calibration": True, "complete_primary_benchmark": False,
            "assessments": assessments, "evidence": summarize(rows, 2500),
            "actual_diagnostic_holdout": summarize(heldout, 10000), "ledger_verified": ledger.verify(),
            "actual_provider_attempts": sum(row["arm"] != "diagnostic-router" for row in attempts.results.values())
                + sum(event.payload["attempted_calls"] for event in ledger.events("execution")),
            "production_certification_eligible": False, "actual_billed_cost": None,
            "limitations": ["10000ms is an exploratory diagnostic profile chosen after stricter latency failure; it does not pass the original2500ms policy.",
                "The8-second provider cap measures request termination, including timeout failures, rather than successful-response latency.",
                "Operational drift treats provider timeouts as deadline adverse events even under the10-second policy; consecutive timeouts can cause later abstentions.",
                "One selected model; no request-adaptive or fixed-baseline speedup claim.",
                "Generated bounded coding tasks; original29 recorded coding requests remain unproven.",
                "128 fixed calibration rows include all timeout failures; no quality-target or risk-budget relaxation.",
                "Missing immutable provider revisions and unknown prices prevent production certification and dollar-savings claims."]}
        write_json(out / "quality-diagnostic-report.json", report)
    finally:
        ledger.close()
        original.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.out)
    except (OSError, ValueError, RuntimeError, httpx.HTTPError, TimeoutError, KeyError, TypeError, IndexError) as exc:
        print(json.dumps({"diagnosis_error_type": type(exc).__name__}), flush=True)
        raise SystemExit(1) from None
