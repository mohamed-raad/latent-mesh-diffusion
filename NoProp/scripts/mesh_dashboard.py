"""
Mesh Training Dashboard — FastAPI webapp with live VRAM/speed/loss monitoring,
full recap panel (benchmarks, accuracy, confidence), and workflow controls.
Run: uvicorn mesh_dashboard:app --reload --port 8765
"""
import os
import json
import sys
import subprocess
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# Set HF token for HuggingFace downloads
os.environ["HF_TOKEN"] = "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"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

MONITOR_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "training_status.json")
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_PROJ, "scripts")

app = FastAPI(title="Mesh Training Dashboard")


def read_status() -> dict:
    if os.path.exists(MONITOR_FILE):
        try:
            with open(MONITOR_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "step": 0, "loss_history": [], "steps_per_sec": 0, "node_count": 0,
        "gpu": {}, "vram_pct": 0,
        "benchmarks": {}, "workflow": {"current": "idle"},
        "generation_stats": {},
    }


@app.get("/status")
def get_status():
    return read_status()


@app.post("/status_update")
def status_update(body: dict):
    """Webhook endpoint for generators to push progress updates."""
    current = read_status()
    if body.get("total"):
        current["generation_stats"] = current.get("generation_stats", {})
        current["generation_stats"]["total_generated"] = body["total"]
    if body.get("message"):
        current["workflow"] = {"current": "running", "progress": body["message"]}
    with open(MONITOR_FILE, "w") as f:
        json.dump(current, f, indent=2)
    return {"ok": True}


@app.get("/trigger/{action}")
def trigger_action(action: str):
    s = read_status()
    w = s.get("workflow", {})
    if w.get("current") not in ("idle", "", None):
        return {"status": "error", "message": f"Workflow busy: {w['current']}"}

    def _run_monitored(cmd: str, title: str):
        """Run a command as a monitored subprocess, writing status updates."""
        s = read_status()
        s["workflow"] = {"current": "running", "progress": f"Starting {title}...", "started_at": time.time()}
        with open(MONITOR_FILE, "w") as f:
            json.dump(s, f, indent=2)

        def _monitor():
            try:
                proc = subprocess.Popen(
                    cmd, shell=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, cwd=_PROJ,
                    env={**os.environ, "HF_TOKEN": "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",
                         "PYTHONUNBUFFERED": "1"},
                )
                for line in iter(proc.stdout.readline, ""):
                    if not line:
                        break
                    line = line.rstrip()
                    if line:
                        s = read_status()
                        s["workflow"]["current"] = "running"
                        s["workflow"]["progress"] = line[-120:] if len(line) > 120 else line
                        with open(MONITOR_FILE, "w") as f:
                            json.dump(s, f, indent=2)
                proc.wait()
                s = read_status()
                s["workflow"] = {"current": "idle", "progress": f"{title} completed (exit={proc.returncode})"}
                with open(MONITOR_FILE, "w") as f:
                    json.dump(s, f, indent=2)
            except Exception as e:
                s = read_status()
                s["workflow"] = {"current": "idle", "progress": f"{title} failed: {e}"}
                with open(MONITOR_FILE, "w") as f:
                    json.dump(s, f, indent=2)

        threading.Thread(target=_monitor, daemon=True).start()
        return {"status": "ok", "message": f"Started {title}"}

    TRIGGERS = {
        "generate": f'uv run --no-sync --package noprop-mesh python scripts/generate_with_llm.py --phases Language_Grammar Conversation_Patterns Writing_Style',
        "ingest": f'uv run --no-sync --package noprop-mesh python scripts/ingest_docs.py --max-pages 3 --sources airesearch',
        "agk_gen": f'uv run --no-sync --package noprop-mesh python scripts/agk_data_generator.py --output agk_data --phases 1 3 5 10',
        "train": f'uv run --no-sync --package noprop-mesh python scripts/train_on_text.py --data agk_llm --epochs 5 --batch 8 --embed-dim 768 --num-heads 8 --canvas-len 512 --canvas-steps 5 --lr 3e-4',
        "benchmark": f'uv run --no-sync --package noprop-mesh python scripts/benchmark.py --quick',
        "full_pipeline": (
            f'uv run --no-sync --package noprop-mesh python scripts/agk_data_generator.py --output agk_data --phases 1 3 5 10 && '
            f'uv run --no-sync --package noprop-mesh python scripts/train_on_text.py --data agk_llm --epochs 5 --batch 8 --embed-dim 768 --num-heads 8 --canvas-len 512 --canvas-steps 5 --lr 3e-4 && '
            f'uv run --no-sync --package noprop-mesh python scripts/benchmark.py --quick'
        ),
    }

    if action in TRIGGERS:
        return _run_monitored(TRIGGERS[action], action)


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return HTMLResponse(HTML_PAGE)


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mesh Training Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }
h1 { color: #58a6ff; margin-bottom: 16px; font-size: 22px; }
h2 { color: #8b949e; font-size: 14px; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 20px; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
.card .value { font-size: 28px; font-weight: 700; color: #f0f6fc; }
.card .label { font-size: 11px; color: #8b949e; text-transform: uppercase; margin-top: 4px; }
.card .sub { font-size: 12px; color: #8b949e; margin-top: 2px; }
.green { color: #3fb950 !important; }
.orange { color: #d29922 !important; }
.red { color: #f85149 !important; }
.blue { color: #58a6ff !important; }
.chart-container { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 20px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; color: #8b949e; font-weight: 600; padding: 8px 12px; border-bottom: 1px solid #30363d; }
td { padding: 8px 12px; border-bottom: 1px solid #21262d; }
tr:hover td { background: #1c2128; }
.bar-bg { background: #21262d; border-radius: 4px; height: 20px; width: 100%; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 4px; transition: width 0.5s; }

/* Tabs */
.tabs { display: flex; gap: 4px; margin-bottom: 20px; border-bottom: 1px solid #30363d; padding-bottom: 0; }
.tab { padding: 10px 20px; cursor: pointer; border: 1px solid #30363d; border-bottom: none; border-radius: 8px 8px 0 0; background: #161b22; color: #8b949e; font-size: 13px; font-weight: 600; user-select: none; }
.tab:hover { background: #1c2128; }
.tab.active { background: #0d1117; color: #58a6ff; border-color: #58a6ff; }
.tab-content { display: none; }
.tab-content.active { display: block; }

/* Workflow buttons */
.workflow-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin-bottom: 20px; }
.btn { display: inline-flex; align-items: center; justify-content: center; gap: 8px; padding: 12px 20px; border: 1px solid #30363d; border-radius: 8px; background: #21262d; color: #c9d1d9; font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.2s; user-select: none; }
.btn:hover { background: #30363d; border-color: #58a6ff; }
.btn:active { transform: scale(0.97); }
.btn .icon { font-size: 16px; }
.btn-primary { background: #1f6feb; border-color: #1f6feb; color: #fff; }
.btn-primary:hover { background: #388bfd; }
.btn-success { background: #238636; border-color: #238636; color: #fff; }
.btn-success:hover { background: #2ea043; }
.btn-warning { background: #9e6a03; border-color: #9e6a03; color: #fff; }
.btn-warning:hover { background: #bb8009; }
.btn-danger { background: #da3633; border-color: #da3633; color: #fff; }
.btn-danger:hover { background: #f85149; }

/* Recap panel */
.recap-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 8px; margin-bottom: 16px; }
.recap-item { text-align: center; padding: 12px; background: #161b22; border: 1px solid #30363d; border-radius: 8px; }
.recap-item .val { font-size: 24px; font-weight: 700; }
.recap-item .lbl { font-size: 10px; color: #8b949e; text-transform: uppercase; margin-top: 4px; }

/* Badge for workflow status */
.workflow-badge { display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; }
.workflow-badge.idle { background: #21262d; color: #8b949e; }
.workflow-badge.running { background: #1f6feb33; color: #58a6ff; animation: pulse 1.5s infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.6; } }
</style>
</head>
<body>

<h1> Mesh Training Dashboard</h1>

<!-- Workflow Status Badge -->
<div style="margin-bottom:16px;">
  <span class="workflow-badge idle" id="workflow-badge">idle</span>
  <span style="font-size:12px;color:#8b949e;margin-left:8px;" id="workflow-progress"></span>
</div>

<!-- Tabs -->
<div class="tabs">
  <div class="tab active" data-tab="recap"> Recap</div>
  <div class="tab" data-tab="training"> Training</div>
  <div class="tab" data-tab="workflows"> Workflows</div>
  <div class="tab" data-tab="benchmarks"> Benchmarks</div>
</div>

<!-- Tab: Recap -->
<div class="tab-content active" id="tab-recap">
  <h2>Full Recap</h2>
  <div class="recap-grid">
    <div class="recap-item"><div class="val green" id="r-speed">0</div><div class="lbl">Steps/sec</div></div>
    <div class="recap-item"><div class="val blue" id="r-loss">--</div><div class="lbl">Loss</div></div>
    <div class="recap-item"><div class="val orange" id="r-accuracy">--</div><div class="lbl">Accuracy</div></div>
    <div class="recap-item"><div class="val" id="r-confidence">--</div><div class="lbl">Confidence</div></div>
    <div class="recap-item"><div class="val green" id="r-perplexity">--</div><div class="lbl">Perplexity</div></div>
    <div class="recap-item"><div class="val blue" id="r-spec-speedup">1.0x</div><div class="lbl">Spec Speedup</div></div>
    <div class="recap-item"><div class="val orange" id="r-nodes">0</div><div class="lbl">Mesh Nodes</div></div>
    <div class="recap-item"><div class="val" id="r-generated">0</div><div class="lbl">Generated</div></div>
  </div>
  <div class="chart-container">
    <h2>Benchmark History</h2>
    <canvas id="benchChart" height="150"></canvas>
  </div>
</div>

<!-- Tab: Training -->
<div class="tab-content" id="tab-training">
  <div class="grid">
    <div class="card"><div class="value" id="t-step">0</div><div class="label">Steps</div></div>
    <div class="card"><div class="value" id="t-speed">0</div><div class="label">Steps/sec</div></div>
    <div class="card"><div class="value" id="t-loss">--</div><div class="label">Current Loss</div></div>
    <div class="card"><div class="value" id="t-nodes">0</div><div class="label">Mesh Nodes</div></div>
    <div class="card"><div class="value" id="t-vram">0 MB</div><div class="label">VRAM</div><div class="sub" id="t-vram-sub"></div></div>
    <div class="card"><div class="value" id="t-gpu-util">0%</div><div class="label">GPU Util</div><div class="sub" id="t-gpu-temp"></div></div>
  </div>
  <div class="chart-container">
    <h2>Training Loss</h2>
    <canvas id="lossChart" height="200"></canvas>
  </div>
  <div class="chart-container">
    <h2>VRAM Usage</h2>
    <div class="bar-bg"><div class="bar-fill" id="vram-bar" style="width:0%;"></div></div>
    <div style="display:flex;justify-content:space-between;font-size:11px;color:#8b949e;margin-top:4px;">
      <span id="vram-used-label">0 MB used</span>
      <span id="vram-total-label">0 MB total</span>
    </div>
  </div>
  <div class="card">
    <h2>Details</h2>
    <table><thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>
        <tr><td>GPU</td><td id="t-gpu-name">--</td></tr>
        <tr><td>Uptime</td><td id="t-uptime">--</td></tr>
        <tr><td>VRAM %</td><td id="t-vram-pct">--</td></tr>
        <tr><td>Power</td><td id="t-power">--</td></tr>
        <tr><td>Dataset Size</td><td id="t-dataset-size">--</td></tr>
      </tbody>
    </table>
  </div>
</div>

<!-- Tab: Workflows -->
<div class="tab-content" id="tab-workflows">
  <h2>Workflow Controls</h2>
  <p style="font-size:12px;color:#8b949e;margin-bottom:16px;">
    Each button opens a new terminal window showing live output. Close it when done.
  </p>

  <h2>Data Generation</h2>
  <div class="workflow-grid">
    <div class="btn btn-primary" onclick="trigger('generate')"><span class="icon">+</span> LLM Generate</div>
    <div class="btn btn-primary" onclick="trigger('agk_gen')"><span class="icon">+</span> AGK Template</div>
    <div class="btn btn-success" onclick="trigger('ingest')"><span class="icon">+</span> Ingest Papers</div>
  </div>

  <h2>Training &amp; Benchmark</h2>
  <div class="workflow-grid">
    <div class="btn btn-warning" onclick="trigger('train')"><span class="icon">+</span> Train Mesh</div>
    <div class="btn btn-warning" onclick="trigger('benchmark')"><span class="icon">+</span> Benchmark</div>
    <div class="btn btn-danger" onclick="trigger('full_pipeline')"><span class="icon">+</span> Full Pipeline</div>
  </div>

  <h2>Server Control</h2>
  <div class="workflow-grid">
    <div class="btn btn-success" onclick="trigger('server_start')"><span class="icon">+</span> Start llama-server</div>
    <div class="btn btn-danger" onclick="trigger('server_stop')"><span class="icon">+</span> Stop llama-server</div>
  </div>

  <div class="card" style="margin-top:20px;">
    <h2>Generation Stats</h2>
    <table><thead><tr><th>Metric</th><th>Count</th></tr></thead>
      <tbody>
        <tr><td>LLM Generated Samples</td><td id="gs-generated">0</td></tr>
        <tr><td>Ingested Docs Chunks</td><td id="gs-ingested">0</td></tr>
      </tbody>
    </table>
  </div>
</div>

<!-- Tab: Benchmarks -->
<div class="tab-content" id="tab-benchmarks">
  <h2>Benchmark Suite</h2>
  <div class="chart-container">
    <h2>Accuracy Over Time</h2>
    <canvas id="accuracyChart" height="150"></canvas>
  </div>
  <div class="chart-container">
    <h2>Confidence Over Time</h2>
    <canvas id="confidenceChart" height="150"></canvas>
  </div>
  <div class="chart-container">
    <h2>Speculative Decoding Speedup</h2>
    <canvas id="specChart" height="150"></canvas>
  </div>
</div>

<script>
let lossChart, benchChart, accuracyChart, confidenceChart, specChart;
let _lastData = '';

function initCharts() {
  lossChart = new Chart(document.getElementById('lossChart'), {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'Loss', data: [], borderColor: '#58a6ff', backgroundColor: 'rgba(88,166,255,0.1)', fill: true, tension: 0.3, pointRadius: 1 }] },
    options: { responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { display: false } }, scales: { x: { grid: { color: '#21262d' }, ticks: { color: '#8b949e', maxTicksLimit: 10 } }, y: { grid: { color: '#21262d' }, ticks: { color: '#8b949e' }, beginAtZero: true } } }
  });
  benchChart = new Chart(document.getElementById('benchChart'), {
    type: 'line',
    data: { labels: [], datasets: [
      { label: 'Accuracy', data: [], borderColor: '#3fb950', backgroundColor: 'rgba(63,185,80,0.1)', fill: true, tension: 0.3, pointRadius: 2, yAxisID: 'y' },
      { label: 'Confidence', data: [], borderColor: '#d29922', backgroundColor: 'rgba(210,153,34,0.1)', fill: true, tension: 0.3, pointRadius: 2, yAxisID: 'y' },
    ]},
    options: { responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { labels: { color: '#8b949e', font: { size: 10 } } } }, scales: { x: { grid: { color: '#21262d' }, ticks: { color: '#8b949e', maxTicksLimit: 10 } }, y: { grid: { color: '#21262d' }, ticks: { color: '#8b949e' }, beginAtZero: true, max: 1 } } }
  });
  accuracyChart = new Chart(document.getElementById('accuracyChart'), {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'Accuracy', data: [], borderColor: '#3fb950', backgroundColor: 'rgba(63,185,80,0.1)', fill: true, tension: 0.3, pointRadius: 2 }] },
    options: { responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { labels: { color: '#8b949e', font: { size: 10 } } } }, scales: { x: { grid: { color: '#21262d' }, ticks: { color: '#8b949e', maxTicksLimit: 10 } }, y: { grid: { color: '#21262d' }, ticks: { color: '#8b949e' }, beginAtZero: true, max: 1 } } }
  });
  confidenceChart = new Chart(document.getElementById('confidenceChart'), {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'Confidence', data: [], borderColor: '#d29922', backgroundColor: 'rgba(210,153,34,0.1)', fill: true, tension: 0.3, pointRadius: 2 }] },
    options: { responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { labels: { color: '#8b949e', font: { size: 10 } } } }, scales: { x: { grid: { color: '#21262d' }, ticks: { color: '#8b949e', maxTicksLimit: 10 } }, y: { grid: { color: '#21262d' }, ticks: { color: '#8b949e' }, beginAtZero: true, max: 1 } } }
  });
  specChart = new Chart(document.getElementById('specChart'), {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'Spec Speedup', data: [], borderColor: '#58a6ff', backgroundColor: 'rgba(88,166,255,0.1)', fill: true, tension: 0.3, pointRadius: 2 }] },
    options: { responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { labels: { color: '#8b949e', font: { size: 10 } } } }, scales: { x: { grid: { color: '#21262d' }, ticks: { color: '#8b949e', maxTicksLimit: 10 } }, y: { grid: { color: '#21262d' }, ticks: { color: '#8b949e' }, beginAtZero: true } } }
  });
}

function fmt(t) {
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = Math.floor(t % 60);
  return h + 'h ' + m + 'm ' + s + 's';
}

function lastOr(arr, fallback) {
  return arr && arr.length > 0 ? arr[arr.length - 1].value : fallback;
}

async function trigger(action) {
  try {
    const r = await fetch('/trigger/' + action);
    const d = await r.json();
    console.log('Trigger:', d);
  } catch(e) { console.log('trigger error:', e); }
}

async function refresh() {
  try {
    const r = await fetch('/status');
    const d = await r.json();
    const dataKey = JSON.stringify({lh:d.loss_history?.length,step:d.step,loss:d.loss,node:d.node_count,ac:d.benchmarks?.accuracy?.length,co:d.benchmarks?.confidence?.length,sp:d.benchmarks?.spec_speedup?.length});
    const changed = dataKey !== _lastData;
    _lastData = dataKey;

    // Workflow
    const wf = d.workflow || {};
    const wb = document.getElementById('workflow-badge');
    const wc = (wf.current || 'idle').toLowerCase();
    wb.textContent = wc;
    wb.className = 'workflow-badge ' + (wc === 'idle' ? 'idle' : 'running');
    document.getElementById('workflow-progress').textContent = wf.progress || '';

    // GPU
    const g = d.gpu || {};
    const vram = g.vram_used_mb || 0;
    const vramTotal = g.vram_total_mb || 1;
    const pct = d.vram_pct || 0;
    const speed = d.steps_per_sec || 0;
    const loss = d.loss;
    const nodes = d.node_count || 0;

    // Recap
    const bm = d.benchmarks || {};
    document.getElementById('r-speed').textContent = speed.toFixed(2);
    document.getElementById('r-loss').textContent = loss != null ? loss.toFixed(6) : '--';
    document.getElementById('r-accuracy').textContent = lastOr(bm.accuracy, '--');
    document.getElementById('r-confidence').textContent = lastOr(bm.confidence, '--');
    document.getElementById('r-perplexity').textContent = lastOr(bm.perplexity, '--');
    const specV = lastOr(bm.spec_speedup, null);
    document.getElementById('r-spec-speedup').textContent = specV ? specV.toFixed(2) + 'x' : '1.0x';
    document.getElementById('r-nodes').textContent = nodes;

    const gs = d.generation_stats || {};
    document.getElementById('r-generated').textContent = (gs.total_generated || 0) + (gs.total_ingested || 0);

    // Training tab
    document.getElementById('t-step').textContent = d.step || 0;
    document.getElementById('t-speed').textContent = speed.toFixed(2);
    document.getElementById('t-loss').textContent = loss != null ? loss.toFixed(6) : '--';
    document.getElementById('t-nodes').textContent = nodes;
    document.getElementById('t-vram').textContent = vram + ' MB';
    document.getElementById('t-vram-sub').textContent = 'of ' + vramTotal + ' MB total';
    document.getElementById('t-gpu-util').textContent = (g.gpu_util_pct || 0) + '%';
    document.getElementById('t-gpu-temp').textContent = (g.gpu_temp_c || '?') + ' C';

    document.getElementById('vram-bar').style.width = pct + '%';
    const barColor = pct > 85 ? '#f85149' : pct > 70 ? '#d29922' : '#3fb950';
    document.getElementById('vram-bar').style.background = barColor;
    document.getElementById('vram-used-label').textContent = vram + ' MB used';
    document.getElementById('vram-total-label').textContent = vramTotal + ' MB total';

    document.getElementById('t-gpu-name').textContent = g.gpu_name || '--';
    document.getElementById('t-uptime').textContent = fmt(d.uptime_seconds || 0);
    document.getElementById('t-vram-pct').textContent = pct + '%';
    document.getElementById('t-power').textContent = (g.power_w || '--') + ' W';

    // Generation stats
    document.getElementById('gs-generated').textContent = gs.total_generated || 0;
    document.getElementById('gs-ingested').textContent = gs.total_ingested || 0;

    // Charts — only update when data actually changed
    if (changed) {
      if (d.loss_history && lossChart) {
        lossChart.data.labels = d.loss_history.map(x => x.step);
        lossChart.data.datasets[0].data = d.loss_history.map(x => x.loss);
        lossChart.update('none');
      }

      function updateBenchChart(chart, key) {
        const data = bm[key] || [];
        if (chart && data.length > 0) {
          chart.data.labels = data.map((_, i) => i + 1);
          chart.data.datasets[0].data = data.map(x => x.value);
          chart.update('none');
        }
      }
      updateBenchChart(accuracyChart, 'accuracy');
      updateBenchChart(confidenceChart, 'confidence');
      updateBenchChart(specChart, 'spec_speedup');

      if (benchChart) {
        const acc = bm.accuracy || [];
        const conf = bm.confidence || [];
        const maxLen = Math.max(acc.length, conf.length);
        benchChart.data.labels = Array.from({length: maxLen}, (_, i) => i + 1);
        benchChart.data.datasets[0].data = acc.length > 0 ? acc.map(x => x.value) : [null];
        benchChart.data.datasets[1].data = conf.length > 0 ? conf.map(x => x.value) : [null];
        benchChart.update('none');
      }
    }
  } catch(e) { console.log('refresh error:', e); }
}

// Tab switching
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
  });
});

initCharts();
refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    print("Dashboard: http://127.0.0.1:8765")
    uvicorn.run(app, host="127.0.0.1", port=8765)
