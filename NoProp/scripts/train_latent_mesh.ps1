# Train Latent Mesh Diffusion Computer — d_model=1024, canvas_len=1024, ~64 experts
# With 3-tier storage, Latent Observatory, and CSV instrumentation.

$REPO_ROOT = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$LAUNCHER = Join-Path $REPO_ROOT "scripts\run-noprop.ps1"

$SCRIPT = "train_mesh.py"

Write-Host "=== Latent Mesh Training ===" -ForegroundColor Cyan
Write-Host "Model: small (d_model=1024) | Batch: 4 | Canvas: 1024x25" -ForegroundColor Yellow
Write-Host "Experts: 64 | GPU:8 RAM:32 Disk:rest | Log: logs/latent-mesh-1" -ForegroundColor Yellow
Write-Host ""

& $LAUNCHER -Script $SCRIPT `
    "--model-size" "small" `
    "--batch-size" "4" `
    "--canvas-len" "1024" `
    "--canvas-steps" "25" `
    "--num-epochs" "3" `
    "--num-samples" "500" `
    "--checkpoint-dir" "checkpoints/latent-mesh-small" `
    "--max-experts" "64" `
    "--max-gpu-experts" "8" `
    "--max-ram-experts" "32" `
    "--log-dir" "logs/latent-mesh-1" `
    "--lr" "5e-4"
