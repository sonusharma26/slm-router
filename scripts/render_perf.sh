#!/usr/bin/env bash
# Render cron-job entrypoint: run the live NVIDIA latency/accuracy microbenchmark
# and print the sanitized report to the job logs. No disk needed; the output
# directory is fresh per run because the script refuses to overwrite.
set -euo pipefail

OUT="/tmp/nvidia-performance-$(date -u +%Y%m%dT%H%M%SZ)"
python scripts/run_nvidia_performance.py --out "$OUT" --requests-per-model "${PERF_REQUESTS_PER_MODEL:-6}"

echo "===== REPORT ====="
find "$OUT" -name '*.json' -print -exec cat {} \;
