# Train with Nemotron-quality settings

$REPO_ROOT = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$LAUNCHER = Join-Path $REPO_ROOT "scripts\run-noprop.ps1"

$SCRIPT = "train_mesh.py"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Latent Mesh Training — Full Config" -ForegroundColor Cyan
Write-Host " d_model=1024  canvas=1024x50" -ForegroundColor Cyan
Write-Host " 3 parallel canvases  64 experts" -ForegroundColor Cyan
Write-Host " dynamic quant  registry+memory" -ForegroundColor Cyan
Write-Host " fineweb-edu + HelpSteer2" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

& $LAUNCHER -Script $SCRIPT `
    "--model-size" "small" `
    "--canvas-len" "1024" `
    "--canvas-steps" "50" `
    "--batch-size" "4" `
    "--max-experts" "64" `
    "--max-gpu-experts" "8" `
    "--max-ram-experts" "32" `
    "--experts-count" "96" `
    "--lr" "3e-4" `
    "--mtp-weight" "0.1" `
    "--checkpoint-dir" "checkpoints/latent-mesh-1024" `
    "--log-dir" "logs/latent-mesh-1024" `
    "--expert-registry" "registry.json" `
    "--mesh-memory" "mesh_mem" `
    "--parallel-canvases" "3" `
    "--dynamic-quant" `
    "--packing" `
    "--resume" `
    "--max-steps" "100000" `
    "--phase" "reasoning" `
    "--domain" "general" `
    "--mix" "HuggingFaceFW/fineweb-edu:0.7,nvidia/HelpSteer2:0.3"
