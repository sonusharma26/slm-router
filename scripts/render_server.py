"""Render entrypoint: the router service plus a /perf/ page that runs the live performance test on demand."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import uvicorn

from inference_control.api.perf import PerfRunner, create_perf_app
from inference_control.configuration import load_service


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "10000")))
    args = parser.parse_args()
    api, control = load_service(args.config)
    runner = PerfRunner(ROOT / "scripts" / "run_nvidia_performance.py", ROOT / "results")
    api.mount("/perf", create_perf_app(os.environ.get("SLM_ROUTER_API_KEY"), runner))
    try:
        uvicorn.run(api, host=args.host, port=args.port, workers=1)
    finally:
        control.ledger.close()
        for adapter in getattr(control.executor.adapter, "providers", {}).values():
            if hasattr(adapter, "close"):
                adapter.close()


if __name__ == "__main__":
    main()
