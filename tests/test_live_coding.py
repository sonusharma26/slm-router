import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

from inference_control.adapters.providers import AdapterResponse
from inference_control.capability.conditional import ConditionalCapabilityMap
from inference_control.ledger import SQLiteLedger


def test_live_evidence_and_both_routed_profiles_without_network(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    spec = importlib.util.spec_from_file_location("live_coding_runner", root / "scripts/run_live_coding_benchmark.py")
    runner = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, runner)
    spec.loader.exec_module(runner)
    out = tmp_path / "live"
    runner.prepare(out, ["nvidia-lightning-coding"])
    manifest, tasks, endpoints = runner.load_run(out)
    answers = {task.prompt: task.reference_expression for task in tasks}
    provider_calls = []

    class TrustedProvider:
        def call(self, model, messages, config):
            provider_calls.append((model, messages[-1]["content"]))
            text = json.dumps({"expression": answers[messages[-1]["content"]]})
            return AdapterResponse(text, f"offline-{len(provider_calls)}", endpoints[0].revision,
                100, 30, None, 1, {}, observed_model=model)

        def close(self):
            pass

    monkeypatch.setattr(runner, "providers_for", lambda selected: {
        endpoint.endpoint_id: TrustedProvider() for endpoint in selected})
    runner.preflight(out)
    runner.collect(out)
    evidence = json.loads((out / "evidence.json").read_text())
    assert Counter(row["split"] for row in evidence) == {"train": 16, "calibration": 128, "validation": 16}
    frozen = json.loads((out / "capability-map.json").read_text())
    restored = ConditionalCapabilityMap.restore(frozen)
    assert len(restored.rows) == 144
    assert Counter(row.split for row in restored.rows.values()) == {"train": 16, "calibration": 128}
    assert all(not row.certification_eligible for row in restored.rows.values())
    assessment = json.loads((out / "assessment.json").read_text())
    assert all(profile["selected"] == "nvidia-lightning-coding" for profile in assessment.values())
    assert all(profile["candidates"][0]["quality_lower"] > .65 for profile in assessment.values())
    assert all(profile["candidates"][0]["calibration_count"] == 128 for profile in assessment.values())
    runner.holdout(out)
    report = json.loads((out / "report.json").read_text())
    assert report["complete_benchmark"] is True
    assert report["selection_mode"] == "constrained_model_selection"
    assert report["actual_provider_attempts"] == len(provider_calls) == 258
    assert report["abstention_claims"] == 0
    assert report["recorded_attempts_including_abstentions"] == 258
    assert manifest["total_call_cap"] == 258
    assert report["profile_ledgers_verified"] == {"router-2500": True, "router-5000": True}
    assert all(arm["correct"] == 32 and arm["all_request_quality"] == 1 for arm in report["arms"].values())
    assert all(arm["on_time_correct"] == 32 for arm in report["arms"].values())
    choices = json.loads((out / "choices.json").read_text())
    assert len(choices) == 64
    assert all(choice["certificate_status"] != "current" for choice in choices)
    for deadline in (2500, 5000):
        ledger = SQLiteLedger(out / f"routed-{deadline}.sqlite3")
        try:
            assert ledger.verify()
            assert len(ledger.events("decision")) == 32
            assert sum(event.payload["attempted_calls"] for event in ledger.events("execution")) == 32
            assert ledger.events("outcome") == []
        finally:
            ledger.close()
    assert json.loads((out / "capability-map.json").read_text()) == frozen
    slower_out = tmp_path / "primary-abstains"
    runner.prepare(slower_out, ["nvidia-lightning-coding"])
    runner.preflight(slower_out)
    clock = iter(range(0, 100000, 3))
    with monkeypatch.context() as timing:
        timing.setattr(runner, "perf_counter", lambda: next(clock))
        runner.collect(slower_out)
    before_holdout = len(provider_calls)
    runner.holdout(slower_out)
    slower_report = json.loads((slower_out / "report.json").read_text())
    assert slower_report["arms"]["router-2500"]["served"] == 0
    assert slower_report["arms"]["router-5000"]["correct"] == 32
    assert len(provider_calls) - before_holdout == 64
    assert slower_report["actual_provider_attempts"] == 226
    assert slower_report["abstention_claims"] == 32
    assert slower_report["recorded_attempts_including_abstentions"] == 258
