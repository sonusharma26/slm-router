import time

from fastapi.testclient import TestClient

from inference_control.api.perf import PerfRunner, create_perf_app

KEY = "k" * 24
AUTH = {"Authorization": f"Bearer {KEY}"}

FAKE = """
import argparse, json, pathlib, sys
p = argparse.ArgumentParser(); p.add_argument("--out"); p.add_argument("--requests-per-model", type=int)
a = p.parse_args()
if a.requests_per_model == 13:
    print("Benchmark failed: ValueError", file=sys.stderr); sys.exit(1)
out = pathlib.Path(a.out); out.mkdir(parents=True)
(out / "report.json").write_text(json.dumps({"results": [], "limitations": [], "tested_at_utc": "t", "wall_seconds": 1}))
"""


def client(tmp_path):
    script = tmp_path / "scripts" / "fake.py"
    script.parent.mkdir()
    script.write_text(FAKE)
    runner = PerfRunner(script, tmp_path / "results")
    return TestClient(create_perf_app(KEY, runner)), runner


def wait(c):
    for _ in range(100):
        s = c.get("/api/status", headers=AUTH).json()
        if s["state"] != "running":
            return s
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_auth_required_but_page_is_public(tmp_path):
    c, _ = client(tmp_path)
    assert c.get("/").status_code == 200
    assert c.get("/api/status").status_code == 401
    assert c.post("/api/run", json={}).status_code == 401
    assert c.post("/api/run", json={}, headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_run_completes_and_returns_report(tmp_path):
    c, _ = client(tmp_path)
    assert c.get("/api/status", headers=AUTH).json()["state"] == "idle"
    assert c.post("/api/run", json={"requests_per_model": 3}, headers=AUTH).status_code == 202
    s = wait(c)
    assert s["state"] == "completed" and s["report"]["wall_seconds"] == 1


def test_failed_run_reports_error_without_report(tmp_path):
    c, _ = client(tmp_path)
    c.post("/api/run", json={"requests_per_model": 13}, headers=AUTH)
    s = wait(c)
    assert s["state"] == "failed" and s["report"] is None and "ValueError" in s["error"]


def test_bounds_and_single_flight(tmp_path):
    c, runner = client(tmp_path)
    assert c.post("/api/run", json={"requests_per_model": 21}, headers=AUTH).status_code == 422
    assert c.post("/api/run", json={"requests_per_model": 0}, headers=AUTH).status_code == 422
    with runner._lock:
        runner._status["state"] = "running"
    assert c.post("/api/run", json={}, headers=AUTH).status_code == 409
