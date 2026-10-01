"""Local, dependency-free operator dashboard for exercising the control API."""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>SLM Router Test Bench</title>
  <style>
    :root {
      color-scheme: dark;
      --canvas: #080a0b;
      --surface: #101314;
      --surface-raised: #15191a;
      --surface-soft: #0c0f10;
      --border: #252b2c;
      --border-strong: #343c3d;
      --text: #f1f5f2;
      --muted: #929c99;
      --muted-strong: #b5bfbb;
      --accent: #b7f34a;
      --accent-ink: #101600;
      --cyan: #69e4d4;
      --amber: #f6c96b;
      --danger: #ff7e8c;
      --radius-lg: 18px;
      --radius-md: 12px;
      --shadow: 0 22px 70px rgba(0, 0, 0, .32);
    }
    * { box-sizing: border-box; }
    html { min-width: 320px; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      background:
        radial-gradient(circle at 82% -8%, rgba(105, 228, 212, .08), transparent 27rem),
        radial-gradient(circle at -10% 12%, rgba(183, 243, 74, .07), transparent 31rem),
        var(--canvas);
      font: 14px/1.5 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    button, input, textarea, select { font: inherit; }
    button { color: inherit; }
    button:focus-visible, input:focus-visible, textarea:focus-visible, select:focus-visible {
      outline: 2px solid rgba(183, 243, 74, .72);
      outline-offset: 2px;
    }
    [hidden] { display: none !important; }
    .app-shell { width: min(1500px, calc(100% - 32px)); margin: 0 auto; padding: 22px 0 44px; }
    .topbar { display: flex; align-items: center; justify-content: space-between; gap: 24px; margin-bottom: 18px; }
    .brand { display: flex; align-items: center; gap: 12px; min-width: 245px; }
    .brand-mark {
      display: grid;
      width: 42px;
      height: 42px;
      place-items: center;
      border: 1px solid rgba(183, 243, 74, .24);
      border-radius: 13px;
      background: #11180d;
      box-shadow: inset 0 0 22px rgba(183, 243, 74, .06);
    }
    .brand h1 { margin: 0; font-size: 16px; letter-spacing: -.02em; }
    .brand p { margin: 2px 0 0; color: var(--muted); font-size: 11px; letter-spacing: .02em; }
    .connection-bar { display: flex; align-items: center; justify-content: flex-end; gap: 9px; flex: 1; }
    .token-wrap { position: relative; width: min(360px, 42vw); }
    .token-wrap input { padding-right: 42px; }
    .icon-button {
      position: absolute;
      top: 50%;
      right: 6px;
      width: 32px;
      height: 32px;
      padding: 0;
      transform: translateY(-50%);
      border: 0;
      border-radius: 8px;
      color: var(--muted);
      background: transparent;
      cursor: pointer;
    }
    .connection-state {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      min-width: max-content;
      padding: 9px 11px;
      border: 1px solid var(--border);
      border-radius: 999px;
      color: var(--muted);
      background: rgba(16, 19, 20, .82);
      font-size: 12px;
    }
    .connection-state .dot { width: 7px; height: 7px; border-radius: 50%; background: #5a6261; }
    .connection-state.ok { color: var(--accent); border-color: rgba(183, 243, 74, .2); }
    .connection-state.ok .dot { background: var(--accent); box-shadow: 0 0 0 4px rgba(183, 243, 74, .09); }
    .workspace { display: grid; grid-template-columns: minmax(420px, .88fr) minmax(520px, 1.12fr); gap: 16px; }
    .panel {
      min-width: 0;
      border: 1px solid var(--border);
      border-radius: var(--radius-lg);
      background: rgba(16, 19, 20, .94);
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .panel-heading { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; padding: 18px 20px; border-bottom: 1px solid var(--border); }
    .panel-heading h2 { margin: 0; font-size: 15px; letter-spacing: -.01em; }
    .panel-heading p { margin: 4px 0 0; color: var(--muted); font-size: 12px; }
    .panel-body { padding: 18px 20px 20px; }
    .section-label { margin: 0 0 9px; color: var(--muted); font-size: 10px; font-weight: 750; letter-spacing: .12em; text-transform: uppercase; }
    .preset-row { display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 19px; }
    .preset { padding: 7px 10px; border: 1px solid var(--border); border-radius: 999px; color: var(--muted-strong); background: var(--surface-soft); cursor: pointer; font-size: 12px; }
    .preset:hover { border-color: var(--border-strong); color: var(--text); }
    .preset.active { color: var(--accent); border-color: rgba(183, 243, 74, .32); background: rgba(183, 243, 74, .07); }
    .preset-note { margin: -10px 0 17px; color: var(--muted); font-size: 11px; }
    .preset-note strong { color: var(--accent); font-weight: 700; }
    label { display: block; margin: 0 0 6px; color: var(--muted-strong); font-size: 12px; }
    input, textarea, select {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 10px;
      color: var(--text);
      background: #0b0e0f;
      padding: 10px 11px;
      outline: none;
      transition: border-color .15s, box-shadow .15s;
    }
    input::placeholder, textarea::placeholder { color: #626b68; }
    input:focus, textarea:focus, select:focus { border-color: rgba(183, 243, 74, .4); box-shadow: 0 0 0 3px rgba(183, 243, 74, .06); }
    textarea { resize: vertical; }
    #systemPrompt { min-height: 67px; }
    #userPrompt { min-height: 138px; font-size: 14px; line-height: 1.6; }
    .field { min-width: 0; margin-bottom: 13px; }
    .field-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0 11px; }
    .full { grid-column: 1 / -1; }
    .helper { margin-top: 5px; color: var(--muted); font-size: 11px; }
    .advanced { margin: 4px 0 15px; border: 1px solid var(--border); border-radius: var(--radius-md); background: var(--surface-soft); overflow: hidden; }
    .advanced summary { display: flex; align-items: center; justify-content: space-between; padding: 12px 13px; color: var(--muted-strong); cursor: pointer; list-style: none; font-size: 12px; font-weight: 650; }
    .advanced summary::-webkit-details-marker { display: none; }
    .advanced summary::after { content: "+"; color: var(--muted); font-size: 18px; font-weight: 400; line-height: 1; }
    .advanced[open] summary::after { content: "-"; }
    .advanced-content { padding: 1px 13px 13px; border-top: 1px solid var(--border); }
    .switch-row { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 12px 0; border-bottom: 1px solid var(--border); }
    .switch-copy strong { display: block; font-size: 12px; }
    .switch-copy span { display: block; margin-top: 2px; color: var(--muted); font-size: 11px; }
    .switch { position: relative; width: 38px; height: 22px; flex: 0 0 auto; }
    .switch input { position: absolute; opacity: 0; pointer-events: none; }
    .slider { position: absolute; inset: 0; border: 1px solid var(--border-strong); border-radius: 999px; background: #1a1f20; cursor: pointer; transition: .18s; }
    .slider::before { content: ""; position: absolute; width: 16px; height: 16px; left: 2px; top: 2px; border-radius: 50%; background: #77807d; transition: .18s; }
    .switch input:checked + .slider { border-color: rgba(183, 243, 74, .45); background: rgba(183, 243, 74, .13); }
    .switch input:checked + .slider::before { transform: translateX(16px); background: var(--accent); }
    .config-block { padding-top: 12px; }
    .config-block textarea { min-height: 110px; font: 11px/1.5 ui-monospace, SFMono-Regular, Consolas, monospace; }
    .composer-actions { display: flex; align-items: center; gap: 10px; }
    .button { display: inline-flex; align-items: center; justify-content: center; gap: 7px; min-height: 39px; padding: 9px 14px; border: 1px solid var(--border); border-radius: 10px; background: var(--surface-raised); cursor: pointer; font-weight: 700; font-size: 12px; }
    .button:hover { border-color: var(--border-strong); }
    .button.primary { min-width: 145px; color: var(--accent-ink); border-color: var(--accent); background: var(--accent); }
    .button.primary:hover { filter: brightness(1.05); }
    .button.ghost { background: transparent; color: var(--muted-strong); }
    .button:disabled { opacity: .48; cursor: not-allowed; filter: none; }
    .keyboard-hint { color: var(--muted); font-size: 11px; }
    kbd { padding: 2px 5px; border: 1px solid var(--border-strong); border-bottom-width: 2px; border-radius: 5px; color: var(--muted-strong); background: #0a0c0d; font: 10px ui-monospace, monospace; }
    .health-strip { display: flex; align-items: stretch; gap: 1px; border-bottom: 1px solid var(--border); background: var(--border); }
    .health-item { flex: 1; min-width: 0; padding: 11px 14px; background: var(--surface-soft); }
    .health-item span { display: block; color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .08em; }
    .health-item strong { display: block; margin-top: 4px; overflow: hidden; color: var(--muted-strong); font: 600 11px ui-monospace, SFMono-Regular, Consolas, monospace; text-overflow: ellipsis; white-space: nowrap; }
    .result-topline { display: flex; align-items: center; gap: 8px; }
    .run-state { display: inline-flex; align-items: center; min-height: 25px; padding: 4px 9px; border: 1px solid var(--border); border-radius: 999px; color: var(--muted); background: var(--surface-soft); font-size: 10px; font-weight: 800; letter-spacing: .07em; text-transform: uppercase; }
    .run-state.success { color: var(--accent); border-color: rgba(183, 243, 74, .28); background: rgba(183, 243, 74, .06); }
    .run-state.failed { color: var(--danger); border-color: rgba(255, 126, 140, .28); background: rgba(255, 126, 140, .06); }
    .run-state.running { color: var(--cyan); border-color: rgba(105, 228, 212, .25); }
    .run-state.running::before { content: ""; width: 7px; height: 7px; margin-right: 6px; border: 1px solid currentColor; border-top-color: transparent; border-radius: 50%; animation: spin .8s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .tabs { display: flex; gap: 4px; padding: 11px 14px 0; border-bottom: 1px solid var(--border); }
    .tab { position: relative; padding: 8px 10px 10px; border: 0; color: var(--muted); background: transparent; cursor: pointer; font-size: 12px; font-weight: 700; }
    .tab.active { color: var(--text); }
    .tab.active::after { content: ""; position: absolute; right: 8px; bottom: -1px; left: 8px; height: 2px; border-radius: 2px; background: var(--accent); }
    .result-content { min-height: 545px; }
    .tab-panel { padding: 18px 20px 20px; }
    .answer-card { min-height: 178px; padding: 18px; border: 1px solid var(--border); border-radius: 14px; background: #0b0e0f; }
    .answer-label { color: var(--cyan); font-size: 10px; font-weight: 800; letter-spacing: .11em; text-transform: uppercase; }
    .answer { margin-top: 12px; color: var(--text); font: 500 16px/1.65 ui-monospace, SFMono-Regular, Consolas, monospace; overflow-wrap: anywhere; white-space: pre-wrap; }
    .answer.placeholder { color: #69726f; font: 400 13px/1.55 Inter, ui-sans-serif, system-ui, sans-serif; }
    .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 10px; }
    .metric { min-width: 0; padding: 12px; border: 1px solid var(--border); border-radius: 11px; background: var(--surface-soft); }
    .metric span { display: block; color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .07em; }
    .metric strong { display: block; margin-top: 6px; overflow: hidden; font: 650 12px ui-monospace, SFMono-Regular, Consolas, monospace; text-overflow: ellipsis; white-space: nowrap; }
    .trace-ids { margin-top: 10px; padding: 4px 13px; border: 1px solid var(--border); border-radius: 11px; background: var(--surface-soft); }
    .trace-row { display: grid; grid-template-columns: 82px minmax(0, 1fr) auto; align-items: center; gap: 8px; padding: 8px 0; border-bottom: 1px solid var(--border); }
    .trace-row:last-child { border-bottom: 0; }
    .trace-row span { color: var(--muted); font-size: 11px; }
    .trace-row code { overflow: hidden; color: var(--muted-strong); font-size: 11px; text-overflow: ellipsis; white-space: nowrap; }
    .copy-button { padding: 3px 6px; border: 0; border-radius: 5px; color: var(--muted); background: transparent; cursor: pointer; font-size: 10px; }
    .copy-button:hover { color: var(--text); background: var(--surface-raised); }
    .error-box { margin-bottom: 12px; padding: 11px 13px; border: 1px solid rgba(255, 126, 140, .28); border-radius: 10px; color: #ffd5da; background: rgba(255, 126, 140, .07); font-size: 12px; white-space: pre-wrap; }
    .decision-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
    .decision-stat { padding: 11px 12px; border: 1px solid var(--border); border-radius: 10px; background: var(--surface-soft); }
    .decision-stat span { display: block; color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: .07em; }
    .decision-stat strong { display: block; margin-top: 6px; font: 650 11px/1.35 ui-monospace, SFMono-Regular, Consolas, monospace; overflow-wrap: anywhere; }
    .route-section { margin-top: 17px; }
    .route-section h3 { margin: 0 0 8px; color: var(--muted-strong); font-size: 11px; letter-spacing: .05em; text-transform: uppercase; }
    .chips { display: flex; flex-wrap: wrap; gap: 6px; }
    .chip { padding: 5px 8px; border: 1px solid var(--border); border-radius: 7px; color: var(--muted-strong); background: var(--surface-soft); font: 10px ui-monospace, SFMono-Regular, Consolas, monospace; }
    .chip.accent { color: var(--cyan); border-color: rgba(105, 228, 212, .22); }
    .empty-line { color: var(--muted); font-size: 12px; }
    .rejection-list { display: grid; gap: 6px; }
    .rejection { padding: 9px 10px; border-left: 2px solid var(--amber); border-radius: 4px 8px 8px 4px; background: rgba(246, 201, 107, .055); }
    .rejection strong { display: block; font: 650 11px ui-monospace, SFMono-Regular, Consolas, monospace; }
    .rejection span { display: block; margin-top: 4px; color: var(--muted); font-size: 10px; overflow-wrap: anywhere; }
    .replay-row { display: flex; align-items: center; gap: 9px; margin-top: 17px; }
    .replay-result { color: var(--muted); font-size: 11px; }
    .replay-result.pass { color: var(--accent); }
    .replay-result.fail { color: var(--danger); }
    .raw-section { margin-bottom: 10px; border: 1px solid var(--border); border-radius: 10px; background: #090b0c; overflow: hidden; }
    .raw-section summary { padding: 10px 12px; color: var(--muted-strong); cursor: pointer; font-size: 11px; font-weight: 700; }
    pre { margin: 0; padding: 0 12px 13px; max-height: 290px; overflow: auto; color: #b8c5c1; font: 11px/1.55 ui-monospace, SFMono-Regular, Consolas, monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
    .history { grid-column: 1 / -1; }
    .history .panel-heading { align-items: center; }
    .history-count { color: var(--muted); font: 11px ui-monospace, monospace; }
    .history-table { width: 100%; border-collapse: collapse; }
    .history-table th { padding: 9px 16px; border-bottom: 1px solid var(--border); color: var(--muted); font-size: 10px; font-weight: 700; letter-spacing: .08em; text-align: left; text-transform: uppercase; }
    .history-table td { padding: 11px 16px; border-bottom: 1px solid var(--border); color: var(--muted-strong); font-size: 11px; }
    .history-table tr:last-child td { border-bottom: 0; }
    .history-table code { font-size: 10px; }
    .history-table .outcome { font-weight: 800; text-transform: uppercase; }
    .history-table .outcome.success { color: var(--accent); }
    .history-table .outcome.failed { color: var(--danger); }
    .history-empty { padding: 22px; color: var(--muted); font-size: 12px; text-align: center; }
    .footer-note { margin: 14px 2px 0; color: #69726f; font-size: 11px; text-align: center; }
    .toast { position: fixed; right: 22px; bottom: 22px; z-index: 20; padding: 9px 12px; border: 1px solid var(--border-strong); border-radius: 9px; color: var(--text); background: #171b1c; box-shadow: var(--shadow); font-size: 11px; }
    @media (max-width: 1040px) {
      .workspace { grid-template-columns: 1fr; }
      .history { grid-column: 1; }
      .result-content { min-height: 0; }
    }
    @media (max-width: 720px) {
      .app-shell { width: min(100% - 20px, 1500px); padding-top: 12px; }
      .topbar { align-items: stretch; flex-direction: column; gap: 12px; }
      .connection-bar { justify-content: flex-start; flex-wrap: wrap; }
      .token-wrap { width: 100%; }
      .connection-state { order: -1; }
      .metrics, .decision-grid { grid-template-columns: 1fr 1fr; }
      .history-scroll { overflow-x: auto; }
      .history-table { min-width: 650px; }
    }
    @media (max-width: 470px) {
      .field-grid, .metrics, .decision-grid { grid-template-columns: 1fr; }
      .composer-actions { align-items: stretch; flex-direction: column; }
      .button.primary { width: 100%; }
      .keyboard-hint { text-align: center; }
      .health-strip { display: grid; grid-template-columns: 1fr 1fr; }
    }
  </style>
</head>
<body>
  <main class="app-shell">
    <header class="topbar">
      <div class="brand">
        <div class="brand-mark" aria-hidden="true">
          <svg width="23" height="23" viewBox="0 0 24 24" fill="none">
            <path d="M4 6.8 12 2.8l8 4v10.4l-8 4-8-4V6.8Z" stroke="#b7f34a" stroke-width="1.5"/>
            <path d="m4.5 7 7.5 4 7.5-4M12 11v10" stroke="#69e4d4" stroke-width="1.5"/>
          </svg>
        </div>
        <div><h1>SLM Router</h1><p>Adaptive inference test bench</p></div>
      </div>
      <div class="connection-bar">
        <div id="connectionState" class="connection-state"><span class="dot"></span><span id="connectionText">Disconnected</span></div>
        <div class="token-wrap">
          <label for="apiKey" hidden>Router API key</label>
          <input id="apiKey" type="password" autocomplete="off" spellcheck="false" placeholder="Enter SLM_ROUTER_API_KEY">
          <button id="revealKey" class="icon-button" type="button" title="Show API key" aria-label="Show API key">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true"><path d="M2.5 12s3.5-6 9.5-6 9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6Z" stroke="currentColor" stroke-width="1.7"/><circle cx="12" cy="12" r="2.5" stroke="currentColor" stroke-width="1.7"/></svg>
          </button>
        </div>
        <button id="connectButton" class="button" type="button">Connect</button>
      </div>
    </header>

    <section class="workspace">
      <section class="panel" aria-labelledby="requestTitle">
        <div class="panel-heading">
          <div><h2 id="requestTitle">Test request</h2><p>Compose one OpenAI-compatible, non-streaming router call.</p></div>
          <span class="run-state">Local only</span>
        </div>
        <div class="panel-body">
          <p class="section-label">Quick tests</p>
          <div class="preset-row" aria-label="Prompt presets">
            <button class="preset active" type="button" data-preset="exact">Exact answer - route</button>
            <button class="preset" type="button" data-preset="reasoning">Reasoning gate</button>
            <button class="preset" type="button" data-preset="json">JSON gate</button>
            <button class="preset" type="button" data-preset="tool">Tool gate</button>
            <button class="preset" type="button" data-preset="privacy">Privacy gate</button>
          </div>
          <p class="preset-note"><strong>Exact answer</strong> matches the live pilot evidence. Gate presets intentionally test safe abstention when evidence or capabilities are unavailable.</p>
          <div class="field-grid">
            <div class="field full"><label for="routerModel">Router policy model</label><select id="routerModel" class="payload-source"><option value="">Connect to load models</option></select></div>
            <div class="field full"><label for="systemPrompt">System message <span class="helper">(optional)</span></label><textarea id="systemPrompt" class="payload-source" placeholder="Define the assistant behavior..."></textarea></div>
            <div class="field full"><label for="userPrompt">User message</label><textarea id="userPrompt" class="payload-source">Return 9.</textarea></div>
            <div class="field"><label for="applicationId">Application</label><input id="applicationId" class="payload-source" value="nvidia-live-pilot"></div>
            <div class="field"><label for="sessionId">Session ID <span class="helper">(optional)</span></label><input id="sessionId" class="payload-source" placeholder="evaluation-session-1"></div>
            <div class="field"><label for="taskHint">Task hint</label><input id="taskHint" class="payload-source" value="exact-number"></div>
            <div class="field"><label for="trafficSlice">Traffic slices</label><input id="trafficSlice" class="payload-source" value="pilot" placeholder="pilot,code"></div>
            <div class="field"><label for="privacy">Privacy classification</label><select id="privacy" class="payload-source"><option value="public">Public</option><option value="internal">Internal</option><option value="confidential">Confidential</option><option value="restricted">Restricted</option></select></div>
            <div class="field"><label for="maxTokens">Max output tokens</label><input id="maxTokens" class="payload-source" type="number" min="1" max="131072" value="512"></div>
          </div>
          <details id="capabilityOptions" class="advanced">
            <summary>Capability checks and request payload</summary>
            <div class="advanced-content">
              <div class="switch-row">
                <div class="switch-copy"><strong>Structured output</strong><span>Require an endpoint with JSON schema support.</span></div>
                <label class="switch" aria-label="Enable structured output"><input id="structuredOutput" class="payload-source" type="checkbox"><span class="slider"></span></label>
              </div>
              <div id="schemaBlock" class="config-block" hidden><label for="jsonSchema">JSON schema</label><textarea id="jsonSchema" class="payload-source" spellcheck="false">{"type":"object","properties":{"answer":{"type":"number"}},"required":["answer"],"additionalProperties":false}</textarea></div>
              <div class="switch-row">
                <div class="switch-copy"><strong>Tool calling</strong><span>Require an endpoint with tool-use support.</span></div>
                <label class="switch" aria-label="Enable tool calling"><input id="toolCalling" class="payload-source" type="checkbox"><span class="slider"></span></label>
              </div>
              <div id="toolBlock" class="config-block" hidden><label for="toolDefinition">OpenAI tool definition</label><textarea id="toolDefinition" class="payload-source" spellcheck="false">{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}}}</textarea></div>
              <div class="config-block"><label for="payloadPreview">Request preview</label><textarea id="payloadPreview" readonly spellcheck="false"></textarea><div class="helper">The authorization header and provider credentials are never shown.</div></div>
            </div>
          </details>
          <div class="composer-actions">
            <button id="runButton" class="button primary" type="button" disabled><svg width="14" height="14" viewBox="0 0 20 20" fill="none" aria-hidden="true"><path d="m4 3 13 7-13 7V3Z" fill="currentColor"/></svg>Run router test</button>
            <span class="keyboard-hint"><kbd>Ctrl</kbd> + <kbd>Enter</kbd> to run</span>
          </div>
        </div>
      </section>

      <section class="panel" aria-labelledby="resultTitle">
        <div class="panel-heading">
          <div><h2 id="resultTitle">Router result</h2><p>Provider output and the evidence behind the route.</p></div>
          <div class="result-topline"><span id="certificateBadge" class="run-state">No run</span><span id="runState" class="run-state">Idle</span></div>
        </div>
        <div class="health-strip">
          <div class="health-item"><span>Service</span><strong id="serviceHealth">Not connected</strong></div>
          <div class="health-item"><span>Ledger</span><strong id="ledgerHealth">--</strong></div>
          <div class="health-item"><span>Capability map</span><strong id="mapVersion">--</strong></div>
          <div class="health-item"><span>Policy</span><strong id="policyName">--</strong></div>
        </div>
        <div class="tabs" role="tablist" aria-label="Result views">
          <button class="tab active" type="button" role="tab" aria-selected="true" data-panel="responsePanel">Response</button>
          <button class="tab" type="button" role="tab" aria-selected="false" data-panel="decisionPanel">Decision</button>
          <button class="tab" type="button" role="tab" aria-selected="false" data-panel="rawPanel">Raw JSON</button>
        </div>
        <div class="result-content">
          <div id="responsePanel" class="tab-panel" role="tabpanel">
            <div id="responseError" class="error-box" hidden></div>
            <div class="answer-card"><div class="answer-label">Assistant response</div><div id="answer" class="answer placeholder">Connect to the local router, choose a test, and run it. The selected endpoint and trace will appear here.</div></div>
            <div class="metrics">
              <div class="metric"><span>Endpoint</span><strong id="endpoint">--</strong></div>
              <div class="metric"><span>Round trip</span><strong id="roundTrip">--</strong></div>
              <div class="metric"><span>Tokens</span><strong id="tokenUsage">--</strong></div>
              <div class="metric"><span>Realized spend</span><strong id="realizedSpend">--</strong></div>
            </div>
            <div class="trace-ids">
              <div class="trace-row"><span>Decision</span><code id="decisionId">--</code><button class="copy-button" type="button" data-copy="decisionId">Copy</button></div>
              <div class="trace-row"><span>Execution</span><code id="executionId">--</code><button class="copy-button" type="button" data-copy="executionId">Copy</button></div>
              <div class="trace-row"><span>Request</span><code id="requestId">--</code><button class="copy-button" type="button" data-copy="requestId">Copy</button></div>
            </div>
          </div>
          <div id="decisionPanel" class="tab-panel" role="tabpanel" hidden>
            <div class="decision-grid">
              <div class="decision-stat"><span>Plan</span><strong id="planType">--</strong></div>
              <div class="decision-stat"><span>Router latency</span><strong id="decisionLatency">--</strong></div>
              <div class="decision-stat"><span>Fallback</span><strong id="fallback">--</strong></div>
              <div class="decision-stat"><span>Quality estimate</span><strong id="qualityEstimate">--</strong></div>
              <div class="decision-stat"><span>Cost estimate</span><strong id="costEstimate">--</strong></div>
              <div class="decision-stat"><span>Latency estimate</span><strong id="latencyEstimate">--</strong></div>
            </div>
            <div class="route-section"><h3>Selected steps</h3><div id="selectedSteps" class="chips"><span class="empty-line">No decision loaded.</span></div></div>
            <div class="route-section"><h3>Eligible endpoints</h3><div id="eligibleEndpoints" class="chips"><span class="empty-line">No decision loaded.</span></div></div>
            <div class="route-section"><h3>Rejected alternatives</h3><div id="rejectedAlternatives" class="rejection-list"><span class="empty-line">No decision loaded.</span></div></div>
            <div class="replay-row"><button id="replayButton" class="button ghost" type="button" disabled>Verify replay</button><span id="replayResult" class="replay-result">Load a decision to verify deterministic routing.</span></div>
          </div>
          <div id="rawPanel" class="tab-panel" role="tabpanel" hidden>
            <details class="raw-section" open><summary>Request payload</summary><pre id="rawRequest">No request yet.</pre></details>
            <details class="raw-section" open><summary>API response</summary><pre id="rawResponse">No response yet.</pre></details>
            <details class="raw-section"><summary>Decision record</summary><pre id="rawDecision">No decision yet.</pre></details>
          </div>
        </div>
      </section>

      <section class="panel history" aria-labelledby="historyTitle">
        <div class="panel-heading"><div><h2 id="historyTitle">Session runs</h2><p>Recent tests are kept in this page only and disappear on refresh.</p></div><span id="historyCount" class="history-count">0 runs</span></div>
        <div id="historyEmpty" class="history-empty">Your completed and rejected tests will appear here.</div>
        <div id="historyScroll" class="history-scroll" hidden><table class="history-table"><thead><tr><th>Time</th><th>Test</th><th>Outcome</th><th>Endpoint / plan</th><th>Latency</th><th>Decision</th></tr></thead><tbody id="historyBody"></tbody></table></div>
      </section>
    </section>
    <p class="footer-note">The router credential remains in page memory only. No prompt, response, or credential is persisted by this dashboard.</p>
  </main>
  <div id="toast" class="toast" role="status" hidden></div>

  <script>
    const byId = (id) => document.getElementById(id);
    const state = { bearer: "", connected: false, busy: false, history: [], decision: null, response: null };
    const presets = {
      exact: { label: "Exact answer", application: "nvidia-live-pilot", slice: "pilot", system: "", prompt: "Return 9.", task: "exact-number", privacy: "public", structured: false, tools: false },
      reasoning: { label: "Reasoning gate", application: "nvidia-live-pilot", slice: "pilot", system: "Give a concise final answer and a short explanation.", prompt: "A train travels 180 km in 2.5 hours. What is its average speed in km/h?", task: "reasoning", privacy: "public", structured: false, tools: false },
      json: { label: "JSON gate", application: "nvidia-live-pilot", slice: "pilot", system: "Return data that matches the supplied JSON schema.", prompt: "Return the numeric answer to 6 multiplied by 7.", task: "structured-output", privacy: "public", structured: true, tools: false },
      tool: { label: "Tool gate", application: "nvidia-live-pilot", slice: "pilot", system: "Use an available tool when it is needed.", prompt: "What is the weather in Bengaluru?", task: "tool-use", privacy: "public", structured: false, tools: true },
      privacy: { label: "Privacy gate", application: "nvidia-live-pilot", slice: "pilot", system: "Handle this request according to its privacy classification.", prompt: "Summarize this restricted internal incident: service latency exceeded the private SLO.", task: "privacy-check", privacy: "restricted", structured: false, tools: false }
    };

    function authHeaders(json = false) {
      const headers = { Authorization: `Bearer ${state.bearer}` };
      if (json) headers["Content-Type"] = "application/json";
      return headers;
    }
    async function api(path, options = {}) {
      const response = await fetch(path, options);
      const text = await response.text();
      let data;
      try { data = text ? JSON.parse(text) : {}; } catch { data = { detail: text }; }
      if (!response.ok) {
        const error = new Error(`HTTP ${response.status}`);
        error.status = response.status;
        error.data = data;
        throw error;
      }
      return data;
    }
    function makeId() {
      if (globalThis.crypto && typeof globalThis.crypto.randomUUID === "function") return globalThis.crypto.randomUUID();
      return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
    }
    function setConnection(connected, text) {
      state.connected = connected;
      byId("connectionState").classList.toggle("ok", connected);
      byId("connectionText").textContent = text;
      byId("connectButton").textContent = connected ? "Disconnect" : "Connect";
      byId("runButton").disabled = !connected || state.busy;
      if (!connected) {
        byId("serviceHealth").textContent = "Not connected";
        byId("ledgerHealth").textContent = "--";
        byId("mapVersion").textContent = "--";
        byId("policyName").textContent = "--";
      }
    }
    function notify(message) {
      const toast = byId("toast");
      toast.textContent = message;
      toast.hidden = false;
      window.clearTimeout(notify.timer);
      notify.timer = window.setTimeout(() => { toast.hidden = true; }, 1800);
    }
    function formatError(error) {
      if (error && error.status === 401) return "The router API key was rejected.";
      const detail = error && error.data ? error.data.detail : null;
      if (typeof detail === "string") return detail;
      if (detail && typeof detail === "object") return detail.error || JSON.stringify(detail, null, 2);
      if (error && error.data) return JSON.stringify(error.data, null, 2);
      return error && error.message ? error.message : "The request failed.";
    }
    function setBusy(busy) {
      state.busy = busy;
      byId("runButton").disabled = busy || !state.connected;
      byId("connectButton").disabled = busy;
      byId("runState").className = `run-state${busy ? " running" : ""}`;
      byId("runState").textContent = busy ? "Routing" : "Idle";
      if (busy) {
        byId("answer").className = "answer placeholder";
        byId("answer").textContent = "Evaluating constraints, selecting a plan, and executing the chosen endpoint...";
        byId("responseError").hidden = true;
      }
    }
    function parseJsonField(id, label) {
      try { return JSON.parse(byId(id).value); }
      catch (error) { throw new Error(`${label} is not valid JSON: ${error.message}`); }
    }
    function buildPayload(requestId = "dashboard-<generated-on-send>") {
      const prompt = byId("userPrompt").value.trim();
      const model = byId("routerModel").value;
      const maxTokens = Number(byId("maxTokens").value);
      if (!prompt) throw new Error("Enter a user message.");
      if (!model) throw new Error("Connect to a router policy model first.");
      if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 131072) throw new Error("Max output tokens must be between 1 and 131072.");
      const messages = [];
      const systemPrompt = byId("systemPrompt").value.trim();
      if (systemPrompt) messages.push({ role: "system", content: systemPrompt });
      messages.push({ role: "user", content: prompt });
      const metadata = {
        application_id: byId("applicationId").value.trim() || "router-dashboard",
        request_id: requestId,
        task: byId("taskHint").value.trim(),
        traffic_slice: byId("trafficSlice").value.trim(),
        privacy: byId("privacy").value
      };
      const sessionId = byId("sessionId").value.trim();
      if (sessionId) metadata.session_id = sessionId;
      const payload = { model, messages, max_tokens: maxTokens, stream: false, metadata };
      if (byId("structuredOutput").checked) {
        payload.response_format = { type: "json_schema", json_schema: { name: "dashboard_response", strict: true, schema: parseJsonField("jsonSchema", "JSON schema") } };
      }
      if (byId("toolCalling").checked) {
        const tool = parseJsonField("toolDefinition", "Tool definition");
        payload.tools = Array.isArray(tool) ? tool : [tool];
        payload.tool_choice = "auto";
      }
      return payload;
    }
    function updatePayloadPreview() {
      try { byId("payloadPreview").value = JSON.stringify(buildPayload(), null, 2); }
      catch (error) { byId("payloadPreview").value = `Preview unavailable: ${error.message}`; }
    }
    function applyPreset(name) {
      const preset = presets[name];
      if (!preset) return;
      document.querySelectorAll(".preset").forEach((button) => button.classList.toggle("active", button.dataset.preset === name));
      byId("systemPrompt").value = preset.system;
      byId("userPrompt").value = preset.prompt;
      byId("applicationId").value = preset.application;
      byId("taskHint").value = preset.task;
      byId("trafficSlice").value = preset.slice;
      byId("privacy").value = preset.privacy;
      byId("structuredOutput").checked = preset.structured;
      byId("toolCalling").checked = preset.tools;
      byId("schemaBlock").hidden = !preset.structured;
      byId("toolBlock").hidden = !preset.tools;
      if (preset.structured || preset.tools) byId("capabilityOptions").open = true;
      updatePayloadPreview();
    }
    async function connect() {
      if (state.connected) {
        state.bearer = "";
        byId("apiKey").value = "";
        setConnection(false, "Disconnected");
        notify("Disconnected; the in-memory key was cleared.");
        return;
      }
      const key = byId("apiKey").value.trim();
      if (!key) { notify("Enter the router API key first."); return; }
      state.bearer = key;
      byId("connectButton").disabled = true;
      byId("connectionText").textContent = "Connecting...";
      try {
        const [health, models] = await Promise.all([api("/health", { headers: authHeaders() }), api("/v1/models", { headers: authHeaders() })]);
        if (!models.data || models.data.length === 0) throw new Error("No default router policy model is configured.");
        byId("routerModel").replaceChildren(...models.data.map((item) => new Option(item.id, item.id)));
        byId("serviceHealth").textContent = health.status || "ok";
        byId("ledgerHealth").textContent = health.ledger_verified ? "verified" : "verification failed";
        byId("mapVersion").textContent = health.map_version || "static estimates";
        byId("policyName").textContent = models.data[0].id;
        setConnection(true, "Connected");
        updatePayloadPreview();
      } catch (error) {
        state.bearer = "";
        setConnection(false, "Connection failed");
        notify(formatError(error));
      } finally { byId("connectButton").disabled = false; }
    }
    function setText(id, value) { byId(id).textContent = value == null || value === "" ? "--" : String(value); }
    function money(value) { const number = Number(value); return Number.isFinite(number) ? `$${number.toFixed(6)}` : "--"; }
    function prediction(value, kind) {
      if (!value) return "--";
      const suffix = kind === "latency" ? " ms" : "";
      const transform = kind === "cost" ? (item) => `$${Number(item).toFixed(6)}` : (item) => Number(item).toFixed(3);
      return `${transform(value.mean)} [${transform(value.lower)} - ${transform(value.upper)}]${suffix}`;
    }
    function renderChips(containerId, values, accent = false) {
      const container = byId(containerId);
      container.replaceChildren();
      if (!values || values.length === 0) {
        const empty = document.createElement("span"); empty.className = "empty-line"; empty.textContent = "None"; container.append(empty); return;
      }
      values.forEach((value) => { const chip = document.createElement("span"); chip.className = `chip${accent ? " accent" : ""}`; chip.textContent = value; container.append(chip); });
    }
    function renderDecision(decision) {
      state.decision = decision;
      byId("rawDecision").textContent = decision ? JSON.stringify(decision, null, 2) : "Decision record was not available.";
      if (!decision) {
        ["planType", "decisionLatency", "fallback", "qualityEstimate", "costEstimate", "latencyEstimate"].forEach((id) => setText(id, "--"));
        renderChips("selectedSteps", []); renderChips("eligibleEndpoints", []);
        const rejected = byId("rejectedAlternatives");
        const empty = document.createElement("span");
        empty.className = "empty-line";
        empty.textContent = "Decision record was not available.";
        rejected.replaceChildren(empty);
        byId("replayButton").disabled = true;
        return;
      }
      setText("planType", decision.selected_plan && decision.selected_plan.plan_type);
      setText("decisionLatency", Number.isFinite(decision.decision_latency_ms) ? `${decision.decision_latency_ms.toFixed(2)} ms` : "--");
      setText("fallback", decision.fallback);
      setText("qualityEstimate", prediction(decision.estimated_quality, "quality"));
      setText("costEstimate", prediction(decision.estimated_cost, "cost"));
      setText("latencyEstimate", prediction(decision.estimated_latency, "latency"));
      const steps = (decision.selected_plan && decision.selected_plan.steps || []).map((step) => {
        if (step.endpoint_id) return `${step.operator}: ${step.endpoint_id}`;
        if (step.verifier) return `${step.operator}: ${step.verifier}`;
        if (step.selector) return `${step.operator}: ${step.selector}`;
        return `${step.operator}: ${step.reason || ""}`;
      });
      renderChips("selectedSteps", steps, true);
      renderChips("eligibleEndpoints", decision.eligible_endpoints || []);
      const rejected = byId("rejectedAlternatives");
      rejected.replaceChildren();
      if (!decision.rejected_alternatives || decision.rejected_alternatives.length === 0) {
        const empty = document.createElement("span"); empty.className = "empty-line"; empty.textContent = "No rejected alternatives."; rejected.append(empty);
      } else {
        decision.rejected_alternatives.forEach((item) => {
          const row = document.createElement("div"); row.className = "rejection";
          const title = document.createElement("strong"); title.textContent = `${item.endpoint_id} / ${item.plan_type || "direct"}`;
          const reasons = document.createElement("span"); reasons.textContent = (item.reason_codes || []).join(" | ") || "No reason supplied";
          row.append(title, reasons); rejected.append(row);
        });
      }
      byId("replayButton").disabled = !decision.decision_id;
      byId("replayResult").className = "replay-result";
      byId("replayResult").textContent = "Replay is ready for deterministic verification.";
    }
    async function fetchDecision(id) {
      if (!id) return null;
      try { return await api(`/v2/decisions/${encodeURIComponent(id)}`, { headers: authHeaders() }); } catch { return null; }
    }
    function findDecisionId(error) {
      const detail = error && error.data ? error.data.detail : null;
      if (detail && typeof detail === "object") return detail.decision_id || (detail.detail && detail.detail.decision_id);
      return error && error.data ? error.data.decision_id : null;
    }
    function renderSuccess(data, decision, elapsed, requestId) {
      const route = data.slm_router || {};
      const message = data.choices && data.choices[0] ? data.choices[0].message || {} : {};
      let content = message.content;
      if (!content && message.tool_calls) content = JSON.stringify(message.tool_calls, null, 2);
      byId("answer").className = "answer";
      byId("answer").textContent = content == null || content === "" ? "(The endpoint returned empty content.)" : String(content);
      setText("endpoint", data.model);
      setText("roundTrip", `${elapsed.toFixed(0)} ms`);
      const usage = data.usage || {};
      setText("tokenUsage", Number.isFinite(usage.total_tokens) ? `${usage.total_tokens} total` : "--");
      setText("realizedSpend", money(route.realized_spend));
      setText("decisionId", route.decision_id); setText("executionId", route.execution_id); setText("requestId", requestId);
      byId("certificateBadge").className = "run-state"; byId("certificateBadge").textContent = route.certificate_status || "uncertified";
      byId("runState").className = "run-state success"; byId("runState").textContent = "Completed";
      renderDecision(decision);
    }
    function renderFailure(error, decision, elapsed, requestId, decisionId) {
      const rejected = decision && decision.rejected_alternatives ? decision.rejected_alternatives : [];
      const reasons = rejected.map((item) => `${item.endpoint_id}: ${(item.reason_codes || []).join(", ")}`).join("\n");
      const reasonCodes = rejected.flatMap((item) => item.reason_codes || []);
      const estimateHint = reasonCodes.includes("ESTIMATE_MISSING")
        ? "\n\nNo evidence matches this request context. Use the Exact answer preset for the current NVIDIA pilot, or refresh the pilot evidence if that preset also abstains."
        : "";
      const message = reasons ? `Router rejected the available alternatives:\n${reasons}${estimateHint}` : formatError(error);
      byId("responseError").hidden = false; byId("responseError").textContent = message;
      byId("answer").className = "answer placeholder"; byId("answer").textContent = "No provider response was returned. Inspect the Decision tab for the router evidence.";
      setText("endpoint", decision && decision.selected_plan ? decision.selected_plan.plan_type : "not executed");
      setText("roundTrip", `${elapsed.toFixed(0)} ms`); setText("tokenUsage", "--"); setText("realizedSpend", "--");
      setText("decisionId", decisionId); setText("executionId", error && error.data && error.data.detail && error.data.detail.execution_id); setText("requestId", requestId);
      byId("certificateBadge").className = "run-state"; byId("certificateBadge").textContent = decision && decision.certificate_status || "no certificate";
      byId("runState").className = "run-state failed"; byId("runState").textContent = decision && decision.selected_plan && decision.selected_plan.plan_type === "abstain" ? "Abstained" : "Failed";
      renderDecision(decision);
    }
    function addHistory(entry) {
      state.history.unshift(entry); state.history = state.history.slice(0, 10);
      const body = byId("historyBody"); body.replaceChildren();
      state.history.forEach((item) => {
        const row = document.createElement("tr");
        const values = [item.time, item.test, item.outcome, item.endpoint, `${item.latency.toFixed(0)} ms`, item.decision || "--"];
        values.forEach((value, index) => {
          const cell = document.createElement("td");
          if (index === 2) cell.className = `outcome ${item.ok ? "success" : "failed"}`;
          if (index === 5) { const code = document.createElement("code"); code.textContent = value; cell.append(code); } else cell.textContent = value;
          row.append(cell);
        });
        body.append(row);
      });
      byId("historyCount").textContent = `${state.history.length} ${state.history.length === 1 ? "run" : "runs"}`;
      byId("historyEmpty").hidden = true; byId("historyScroll").hidden = false;
    }
    function currentTestLabel() {
      const active = document.querySelector(".preset.active");
      return active && presets[active.dataset.preset] ? presets[active.dataset.preset].label : (byId("taskHint").value.trim() || "Custom");
    }
    async function run() {
      if (!state.connected || state.busy) return;
      let payload;
      const requestId = `dashboard-${makeId()}`;
      try { payload = buildPayload(requestId); } catch (error) { notify(error.message); return; }
      setBusy(true); state.decision = null; state.response = null;
      byId("rawRequest").textContent = JSON.stringify(payload, null, 2);
      byId("rawResponse").textContent = "Waiting for API response...";
      byId("rawDecision").textContent = "Waiting for decision record...";
      byId("replayButton").disabled = true;
      const started = performance.now();
      try {
        const data = await api("/v1/chat/completions", { method: "POST", headers: authHeaders(true), body: JSON.stringify(payload) });
        const elapsed = performance.now() - started;
        state.response = data; byId("rawResponse").textContent = JSON.stringify(data, null, 2);
        const route = data.slm_router || {};
        const decision = await fetchDecision(route.decision_id);
        renderSuccess(data, decision, elapsed, requestId);
        addHistory({ time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }), test: currentTestLabel(), outcome: "completed", ok: true, endpoint: data.model || "--", latency: elapsed, decision: route.decision_id });
      } catch (error) {
        const elapsed = performance.now() - started;
        state.response = error.data || { error: error.message }; byId("rawResponse").textContent = JSON.stringify(state.response, null, 2);
        const decisionId = findDecisionId(error);
        const decision = await fetchDecision(decisionId);
        renderFailure(error, decision, elapsed, requestId, decisionId);
        const plan = decision && decision.selected_plan ? decision.selected_plan.plan_type : "not executed";
        addHistory({ time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }), test: currentTestLabel(), outcome: plan === "abstain" ? "abstained" : "failed", ok: false, endpoint: plan, latency: elapsed, decision: decisionId });
      } finally {
        state.busy = false; byId("runButton").disabled = !state.connected; byId("connectButton").disabled = false;
      }
    }
    async function replayDecision() {
      if (!state.decision || !state.decision.decision_id) return;
      const button = byId("replayButton"); button.disabled = true;
      byId("replayResult").className = "replay-result"; byId("replayResult").textContent = "Replaying against historical evidence...";
      try {
        const result = await api(`/v2/decisions/${encodeURIComponent(state.decision.decision_id)}/replay`, { headers: authHeaders() });
        byId("replayResult").className = `replay-result ${result.matched ? "pass" : "fail"}`;
        byId("replayResult").textContent = result.matched ? "Verified: decision fingerprint matched." : "Mismatch: inspect the replay fingerprints in Raw JSON.";
        byId("rawDecision").textContent = JSON.stringify({ decision: state.decision, replay: result }, null, 2);
      } catch (error) {
        byId("replayResult").className = "replay-result fail"; byId("replayResult").textContent = `Replay failed: ${formatError(error)}`;
      } finally { button.disabled = false; }
    }
    function selectTab(panelId) {
      document.querySelectorAll(".tab").forEach((tab) => {
        const active = tab.dataset.panel === panelId; tab.classList.toggle("active", active); tab.setAttribute("aria-selected", String(active));
      });
      ["responsePanel", "decisionPanel", "rawPanel"].forEach((id) => { byId(id).hidden = id !== panelId; });
    }
    async function copyValue(id) {
      const value = byId(id).textContent;
      if (!value || value === "--") return;
      try { await navigator.clipboard.writeText(value); notify("Copied to clipboard."); } catch { notify("Clipboard access is unavailable."); }
    }
    byId("connectButton").addEventListener("click", connect);
    byId("runButton").addEventListener("click", run);
    byId("replayButton").addEventListener("click", replayDecision);
    byId("apiKey").addEventListener("keydown", (event) => { if (event.key === "Enter") connect(); });
    byId("apiKey").addEventListener("input", () => {
      if (state.connected && byId("apiKey").value.trim() !== state.bearer) { state.bearer = ""; setConnection(false, "Key changed; reconnect"); }
    });
    byId("revealKey").addEventListener("click", () => {
      const input = byId("apiKey"); const reveal = input.type === "password"; input.type = reveal ? "text" : "password";
      byId("revealKey").title = reveal ? "Hide API key" : "Show API key"; byId("revealKey").setAttribute("aria-label", byId("revealKey").title);
    });
    byId("structuredOutput").addEventListener("change", () => { byId("schemaBlock").hidden = !byId("structuredOutput").checked; updatePayloadPreview(); });
    byId("toolCalling").addEventListener("change", () => { byId("toolBlock").hidden = !byId("toolCalling").checked; updatePayloadPreview(); });
    document.querySelectorAll(".preset").forEach((button) => button.addEventListener("click", () => applyPreset(button.dataset.preset)));
    document.querySelectorAll(".payload-source").forEach((input) => {
      const changed = () => {
        document.querySelectorAll(".preset").forEach((button) => button.classList.remove("active"));
        updatePayloadPreview();
      };
      input.addEventListener("input", changed);
      input.addEventListener("change", changed);
    });
    document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => selectTab(tab.dataset.panel)));
    document.querySelectorAll("[data-copy]").forEach((button) => button.addEventListener("click", () => copyValue(button.dataset.copy)));
    document.addEventListener("keydown", (event) => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { event.preventDefault(); run(); } });
    applyPreset("exact");
    setConnection(false, "Disconnected");
  </script>
</body>
</html>"""


def create_dashboard_app() -> FastAPI:
    dashboard = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @dashboard.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index():
        return HTMLResponse(
            DASHBOARD_HTML,
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": (
                    "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
                    "connect-src 'self'; img-src 'self' data:; base-uri 'none'; "
                    "form-action 'none'; frame-ancestors 'none'"
                ),
                "Referrer-Policy": "no-referrer",
                "X-Content-Type-Options": "nosniff",
            },
        )

    return dashboard
