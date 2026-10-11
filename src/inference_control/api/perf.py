"""On-demand live performance run with a one-button page; one run at a time, report kept in memory."""

import hmac
import json
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


PERF_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SLM Router Performance Run</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; padding: 24px; background: #080a0b; color: #f1f5f2; font: 14px/1.5 system-ui, sans-serif; }
  main { max-width: 960px; margin: 0 auto; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  p.sub { color: #929c99; margin: 0 0 20px; }
  .card { background: #101314; border: 1px solid #252b2c; border-radius: 12px; padding: 16px; margin-bottom: 16px; }
  label { display: block; color: #b5bfbb; margin-bottom: 4px; }
  input { background: #0c0f10; color: inherit; border: 1px solid #343c3d; border-radius: 8px; padding: 8px 10px; width: 100%; max-width: 360px; }
  .row { display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-end; }
  button { background: #b7f34a; color: #101600; border: 0; border-radius: 8px; padding: 9px 18px; font-weight: 600; cursor: pointer; }
  button:disabled { opacity: .5; cursor: default; }
  #state { font-weight: 600; }
  .err { color: #ff7e8c; }
  table { border-collapse: collapse; width: 100%; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #252b2c; }
  th { color: #929c99; font-weight: 500; }
  .scroll { overflow-x: auto; }
  pre { background: #0c0f10; padding: 12px; border-radius: 8px; overflow: auto; max-height: 420px; }
</style>
</head>
<body>
<main>
  <h1>Live performance run</h1>
  <p class="sub">Calls the configured NVIDIA endpoints with a small arithmetic workload. Each run makes billable provider calls.</p>
  <div class="card row">
    <div><label for="key">Router API key (kept in memory only)</label><input id="key" type="password" autocomplete="off"></div>
    <div><label for="n">Measured calls per model (1-20)</label><input id="n" type="number" min="1" max="20" value="6" style="max-width:120px"></div>
    <button id="run">Run performance test</button>
  </div>
  <div class="card">Status: <span id="state">unknown</span> <span id="msg" class="err"></span></div>
  <div id="results"></div>
</main>
<script>
const $ = id => document.getElementById(id);
let timer = null;
const hdr = () => ({ "Authorization": "Bearer " + $("key").value });
const fmt = v => v === null || v === undefined ? "n/a" : (typeof v === "number" ? (Math.round(v * 1000) / 1000).toString() : v);

function render(s) {
  $("state").textContent = s.state + (s.state === "running" ? " (" + s.requests_per_model + " calls per model)" : "");
  $("msg").textContent = s.error || "";
  $("run").disabled = s.state === "running";
  if (!s.report) { $("results").textContent = ""; return; }
  const cols = [["mean_latency_ms", "Mean ms"], ["p50_latency_ms", "p50 ms"], ["p95_latency_ms", "p95 ms"],
    ["error_rate", "Error rate"], ["exact_answer_accuracy_all_measured", "Exact-answer acc."],
    ["serial_successful_requests_per_second", "Req/s (serial)"], ["input_tokens", "Input tok"], ["output_tokens", "Output tok"]];
  let h = '<div class="card scroll"><table><tr><th>Endpoint</th><th>Model</th><th>OK / measured</th>' +
    cols.map(c => "<th>" + c[1] + "</th>").join("") + "</tr>";
  for (const r of s.report.results) {
    h += "<tr><td>" + esc(r.endpoint_id) + "</td><td>" + esc(r.model) + "</td><td>" + r.successful_calls + " / " + r.measured_calls + "</td>" +
      cols.map(c => "<td>" + fmt(r.metrics[c[0]]) + "</td>").join("") + "</tr>";
  }
  h += "</table><p class='sub'>Tested " + esc(s.report.tested_at_utc) + ", wall " + fmt(s.report.wall_seconds) + " s</p></div>";
  h += '<div class="card"><strong>Limitations</strong><ul>' + s.report.limitations.map(l => "<li>" + esc(l) + "</li>").join("") + "</ul></div>";
  h += '<div class="card"><strong>Full report</strong><pre>' + esc(JSON.stringify(s.report, null, 2)) + "</pre></div>";
  $("results").innerHTML = h;
}
function esc(v) { return String(v).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }

async function poll() {
  try {
    const res = await fetch("api/status", { headers: hdr() });
    if (res.status === 401) { $("msg").textContent = "Invalid API key"; clearInterval(timer); timer = null; return; }
    const s = await res.json();
    render(s);
    if (s.state !== "running" && timer) { clearInterval(timer); timer = null; }
  } catch (e) { $("msg").textContent = "Status request failed"; }
}
function ensurePolling() { if (!timer) timer = setInterval(poll, 3000); }

$("run").addEventListener("click", async () => {
  $("msg").textContent = "";
  const res = await fetch("api/run", { method: "POST", headers: { ...hdr(), "Content-Type": "application/json" },
    body: JSON.stringify({ requests_per_model: Number($("n").value) }) });
  if (res.status === 401) { $("msg").textContent = "Invalid API key"; return; }
  if (!res.ok) { $("msg").textContent = (await res.json()).detail || "Could not start run"; return; }
  await poll(); ensurePolling();
});
$("key").addEventListener("change", () => { poll().then(() => { if ($("state").textContent.startsWith("running")) ensurePolling(); }); });
</script>
</body>
</html>
"""


class RunRequest(BaseModel):
    requests_per_model: int = Field(default=6, ge=1, le=20)


class PerfRunner:
    """Runs scripts/run_nvidia_performance.py in a subprocess; at most one run at a time."""

    def __init__(self, script: Path, out_root: Path, *, python: str = sys.executable, timeout_seconds: float = 1800):
        self.script, self.out_root, self.python, self.timeout_seconds = script, out_root, python, timeout_seconds
        self._lock = threading.Lock()
        self._status: dict = {"state": "idle", "requests_per_model": None, "started_at": None,
                              "finished_at": None, "exit_code": None, "error": None, "report": None}

    def status(self) -> dict:
        with self._lock:
            return dict(self._status)

    def start(self, requests_per_model: int) -> bool:
        with self._lock:
            if self._status["state"] == "running":
                return False
            self._status = {"state": "running", "requests_per_model": requests_per_model,
                            "started_at": datetime.now(UTC).isoformat(), "finished_at": None,
                            "exit_code": None, "error": None, "report": None}
        threading.Thread(target=self._run, args=(requests_per_model,), daemon=True).start()
        return True

    def _run(self, requests_per_model: int) -> None:
        out = self.out_root / f"nvidia-performance-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
        update: dict = {}
        try:
            self.out_root.mkdir(parents=True, exist_ok=True)
            done = subprocess.run([self.python, str(self.script), "--out", str(out),
                                   "--requests-per-model", str(requests_per_model)],
                                  capture_output=True, text=True, timeout=self.timeout_seconds,
                                  cwd=str(self.script.parent.parent))
            update["exit_code"] = done.returncode
            report_path = out / "report.json"
            if report_path.is_file():
                update["report"] = json.loads(report_path.read_text(encoding="utf-8"))
            if done.returncode != 0:
                tail = (done.stderr or "").strip().splitlines()[-1:] or ["run reported failures"]
                update["error"] = ("Not every call succeeded; " if update.get("report") else "") + tail[0][:200]
            update["state"] = "failed" if update.get("report") is None else "completed"
        except subprocess.TimeoutExpired:
            update.update(state="failed", error="run timed out")
        except (OSError, ValueError) as exc:
            update.update(state="failed", error=f"{type(exc).__name__} while running")
        update["finished_at"] = datetime.now(UTC).isoformat()
        with self._lock:
            self._status.update(update)


def create_perf_app(api_key: str | None, runner: PerfRunner) -> FastAPI:
    perf = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def authorize(authorization: str | None = Header(default=None)):
        if not api_key or not hmac.compare_digest(authorization or "", f"Bearer {api_key}"):
            raise HTTPException(401, "invalid API credential")

    @perf.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index():
        return HTMLResponse(PERF_HTML, headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": ("default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
                                        "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"),
            "Referrer-Policy": "no-referrer", "X-Content-Type-Options": "nosniff"})

    @perf.get("/api/status", dependencies=[Depends(authorize)])
    def status():
        return runner.status()

    @perf.post("/api/run", status_code=202, dependencies=[Depends(authorize)])
    def run(body: RunRequest):
        if not runner.start(body.requests_per_model):
            raise HTTPException(409, "a performance run is already in progress")
        return runner.status()

    return perf
