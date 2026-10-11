#!/usr/bin/env bash
# Render web-service entrypoint for the free tier (ephemeral filesystem, no disk,
# no cron). State under results/ is rebuilt on every start, which costs three live
# NVIDIA calls; the router ledger therefore resets whenever the service restarts.
# The performance test does not run here: open /perf/ and press the button.
set -euo pipefail

PILOT_DIR="results/nvidia-live-pilot"
rm -rf "$PILOT_DIR"
python scripts/run_nvidia_live_pilot.py --out "$PILOT_DIR" || echo "Pilot failed; starting server anyway (see $PILOT_DIR/pilot-report.json)" >&2

exec python scripts/render_server.py --config examples/nvidia-live/service.yaml --host 0.0.0.0 --port "${PORT:-10000}"
