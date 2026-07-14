# Train core engine only (canvas + latent + speculator, NO experts)
# Use this for rapid iteration on the reasoning backbone.
# Then fine-tune with full experts using train_latent_mesh.ps1

$REPO_ROOT = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$LAUNCHER = Join-Path $REPO_ROOT "scripts\run-noprop.ps1"

$SCRIPT = "train_mesh.py"

Write-Host "=== Core-Only Training ===" -ForegroundColor Cyan
Write-Host "Model: small (d_model=1024) | Batch: 4 | Canvas: 1024x25" -ForegroundColor Yellow
Write-Host "Experts: NONE | Max steps: 5000 | Log: logs/core-only-1" -ForegroundColor Yellow
Write-Host ""

& $LAUNCHER -Script $SCRIPT `
    "--model-size" "small" `
    "--batch-size" "4" `
    "--canvas-len" "1024" `
    "--canvas-steps" "25" `
    "--max-steps" "5000" `
    "--checkpoint-dir" "checkpoints/core-only-small" `
    "--log-dir" "logs/core-only-1" `
    "--lr" "3e-4" `
    "--core-only"
