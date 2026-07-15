# Launch Phase 1 Core Training in a separate window
param(
    [switch]$NoPacking = $true
)

$packArg = if ($NoPacking) { "--no-packing" } else { "" }

Write-Host "=== Launching Phase 1 — Core 250M ==="
Write-Host "Canvas: 512x20 | Latents: 64@192 | LR: 5e-4 | BS: 8"
Write-Host "Dataset: 70% EN / 20% AR / 10% code"
Write-Host "Checkpoints: ./checkpoints/phase_phase_1_%e2%80%94_core_250m"
Write-Host "Dashboard: tensorboard --logdir=./training_logs"
Write-Host ""

$cmd = "cd '$PWD'; .\.venv\Scripts\python.exe -u NoProp/src/train_pipeline.py --phase phase1 $packArg 2>&1"
Start-Process -WindowStyle Normal -Wait -NoNewWindow powershell -ArgumentList "-NoExit", "-Command", $cmd
