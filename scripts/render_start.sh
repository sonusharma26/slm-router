#!/usr/bin/env bash
# Render web-service entrypoint. results/ is a persistent disk, so the pilot
# endpoint map, capability map and SQLite ledger survive redeploys.
set -euo pipefail

PILOT_DIR="results/nvidia-live-pilot"

# The pilot writes its maps before its live calls finish, so only a marker written
# after success proves the state is usable; a failed attempt is discarded and retried.
if [ ! -f "$PILOT_DIR/.pilot-ok" ]; then
  echo "Running the three-call NVIDIA pilot to create $PILOT_DIR"
  rm -rf "$PILOT_DIR"
  python scripts/run_nvidia_live_pilot.py --out "$PILOT_DIR"
  touch "$PILOT_DIR/.pilot-ok"
fi

exec slm serve --config examples/nvidia-live/service.yaml --host 0.0.0.0 --port "${PORT:-10000}"
