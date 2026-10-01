# NVIDIA Build live functional pilot

The base service profile connects the control plane to two NVIDIA Build hosted endpoints:

- `openai/gpt-oss-20b` (`nvidia-fast`)
- `meta/muse-glimmer-30b` (`nvidia-strong`)

The optional provider-matrix script can also exercise these four additional endpoints:

- `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` (`nvidia-nemotron-omni`)
- `z-ai/glm-5.3-flash` (`nvidia-glm-flash`)
- `moonshotai/kimi-k3` (`nvidia-kimi-k3`)
- `google/gemma-4-31b-it` (`nvidia-gemma-4`)

The adaptive chat surface is text-only, so the functional matrix exercises the
multimodal-capable models with text requests only. Image routing is not claimed.

It is a three-call functional pilot, not a benchmark or production calibration:

1. Send one deterministic exact-number prompt to each endpoint.
2. Reject missing usage or a returned model ID that differs from the configured model.
3. Seed one explicitly **uncertified** observation per endpoint, then have the real router select and execute one new request.

Both endpoints expose reasoning separately from final answer content. The pilot therefore allows up to 512 completion tokens per call; smaller limits can be consumed entirely by reasoning and leave `message.content` empty.
The endpoint snapshots also reserve 64 input tokens for NVIDIA's provider-side chat template. This is a conservative pilot value based on the observed usage delta, not a production-certified tokenizer bound.

The report contains only sanitized metadata. Prompts, answers, provider request IDs and the API key are not written to disk.

## Prepare the environment

Install the project with a Python version in the declared range, then set a separate local router credential. The pilot script reads `NVIDIA_API_KEY` from the process environment first and then `.env.local`.

```powershell
python -m pip install -e ".[dev]"
$env:SLM_ROUTER_API_KEY = "replace-with-a-separate-random-value-of-at-least-24-characters"
```

Do not put `SLM_ROUTER_API_KEY` in source control. `.env.local` is ignored by this repository.

## Run the three-call pilot

```powershell
python scripts/run_nvidia_live_pilot.py
Get-Content results/nvidia-live-pilot/pilot-report.json
```

The command refuses to overwrite `results/nvidia-live-pilot`. Use a different `--out` path to rerun it.

A successful functional result has `execution_state: "completed"`, no constraint violations, and `certificate_status: "uncertified"`. The latter is expected: one labelled observation per model cannot establish a meaningful quality or latency certificate.

## Start the local OpenAI-compatible router

After a successful pilot has created `capability-map.json` and `endpoints.json`, start the profile on loopback:

```powershell
slm serve --config examples/nvidia-live/service.yaml --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/dashboard/` for the local test dashboard. Enter the independent `SLM_ROUTER_API_KEY` when prompted; the page keeps it in memory only and does not expose the NVIDIA provider credential.

In another PowerShell window, submit a non-streaming request using the independent router credential:

```powershell
$headers = @{ Authorization = "Bearer $env:SLM_ROUTER_API_KEY"; "Content-Type" = "application/json" }
$body = @{ model = "nvidia-live-pilot"; messages = @(@{ role = "user"; content = "Return 9." }); max_tokens = 512; metadata = @{ application_id = "nvidia-live-pilot"; task = "exact-number"; traffic_slice = "pilot"; privacy = "public" } } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/chat/completions -Headers $headers -Body $body
```

The response's `slm_router` object contains the router decision and execution IDs. Inspect the service ledger or use `slm replay` to verify a decision; neither reproduces stored raw response text.

## Pilot-only configuration boundary

NVIDIA presently describes Developer Program access to hosted NIM endpoints as free for prototyping. The zero prices in the template `endpoints.json` are therefore restricted to this functional developer-program profile. The pilot copies a returned immutable revision into its generated snapshot when NVIDIA supplies one. Before any paid or production use, replace the pricing with an operator-verified pricing contract, verify the model context limits and token overhead, and create application-specific train/calibration evidence. Do not promote this pilot policy or use its single observations as quality evidence.
