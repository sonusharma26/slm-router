#!/usr/bin/env bash
# Render web-service entrypoint for the free tier (ephemeral filesystem, no disk,
# no cron). State under results/ is rebuilt on every start, which costs three live
# NVIDIA calls; the router ledger therefore resets whenever the service restarts.
set -euo pipefail

PILOT_DIR="results/nvidia-live-pilot"
rm -rf "$PILOT_DIR"
python scripts/run_nvidia_live_pilot.py --out "$PILOT_DIR"

# Optional: set RUN_PERF_ON_START=1 to run the live NVIDIA performance
# microbenchmark on boot and print its sanitized report to the service logs.
# A failure here must not keep the router from starting.
if [ "${RUN_PERF_ON_START:-0}" = "1" ]; then
  PERF_DIR="results/nvidia-performance-$(date -u +%Y%m%dT%H%M%SZ)"
  if python scripts/run_nvidia_performance.py --out "$PERF_DIR" --requests-per-model "${PERF_REQUESTS_PER_MODEL:-6}"; then
    echo "===== PERFORMANCE REPORT ====="
    find "$PERF_DIR" -name '*.json' -print -exec cat {} \;
    echo "===== END REPORT ====="
  else
    echo "Performance run failed; starting the router anyway" >&2
  fi
fi

exec slm serve --config examples/nvidia-live/service.yaml --host 0.0.0.0 --port "${PORT:-10000}"
