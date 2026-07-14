# Train with mixed-domain streaming (fineweb-edu 70% + github-code 30%)
# Domain-aware packing groups sequences by domain for max router cache hits.

$REPO_ROOT = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$LAUNCHER = Join-Path $REPO_ROOT "scripts\run-noprop.ps1"

$SCRIPT = "train_mesh.py"

Write-Host "=== Mixed-Domain Training ===" -ForegroundColor Cyan
Write-Host "Model: small (d_model=1024) | Batch: 4 | Canvas: 1024x25" -ForegroundColor Yellow
Write-Host "Mix: fineweb-edu 70% + github-code 30% | Experts: 64 | Domain-aware packing" -ForegroundColor Yellow
Write-Host ""

& $LAUNCHER -Script $SCRIPT `
    "--model-size" "small" `
    "--batch-size" "4" `
    "--canvas-len" "1024" `
    "--canvas-steps" "25" `
    "--max-steps" "20000" `
    "--checkpoint-dir" "checkpoints/latent-mesh-mix" `
    "--max-experts" "64" `
    "--max-gpu-experts" "8" `
    "--max-ram-experts" "32" `
    "--log-dir" "logs/latent-mesh-mix-1" `
    "--lr" "5e-4" `
    "--packing" `
    "--mix" "HuggingFaceFW/fineweb-edu:0.7,codeparrot/github-code:0.3"
