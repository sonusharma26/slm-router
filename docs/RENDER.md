# Deploying on Render (free tier)

The free tier has no Blueprints, persistent disks or cron jobs, so deploy one web service by hand.

1. Render dashboard → **New → Web Service** → connect `sonusharma26/slm-router`, branch `main`.
2. Settings:
   - Runtime: **Python 3**
   - Build command: `pip install .`
   - Start command: `bash scripts/render_start.sh`
   - Instance type: **Free**
3. Environment variables:

   | Key | Value |
   |---|---|
   | `PYTHON_VERSION` | `3.12.8` |
   | `NVIDIA_API_KEY` | your NVIDIA Build key |
   | `OPENROUTER_API_KEY` | your OpenRouter key (not used by the service config yet) |
   | `SLM_ROUTER_API_KEY` | a random value of 24+ characters; use it as the bearer key |
   | `RUN_PERF_ON_START` | `1` to run the performance microbenchmark on boot (optional) |
   | `PERF_REQUESTS_PER_MODEL` | calls per model, 1–20, default `6` (optional) |

4. Deploy. The performance report appears in the service **Logs** between `===== PERFORMANCE REPORT =====` markers. To rerun it, use **Manual Deploy → Restart**.

## Free-tier limits

- The filesystem is ephemeral: the pilot, router ledger and any reports are rebuilt on every start and lost on restart or redeploy. Copy reports out of the logs.
- Every start makes three live NVIDIA calls (the pilot), plus the performance calls if `RUN_PERF_ON_START=1`. Free services spin down after about 15 minutes idle, so a cold start repeats them.
- There is no health check because `/health` requires the API key.
- Only NVIDIA is wired into the service config. Other providers need new endpoint config.
