param(
    [Parameter(Mandatory = $true)]
    [string]$Script,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$ErrorActionPreference = "Stop"
$REPO_ROOT = Split-Path -Parent $PSScriptRoot
$VENV_DIR = Join-Path $REPO_ROOT ".venv"
$NOPROP_DIR = Join-Path $REPO_ROOT "NoProp"

Write-Host "=== NN Mesh: run-noprop ===" -ForegroundColor Cyan

# 1. Activate or create venv
if (-not (Test-Path (Join-Path $VENV_DIR "Scripts\Activate.ps1"))) {
    Write-Host "Creating uv venv..." -ForegroundColor Yellow
    Push-Location $REPO_ROOT
    uv venv
    uv sync --package noprop-mesh
    Pop-Location
}
. (Join-Path $VENV_DIR "Scripts\Activate.ps1")

# 2. Verify bitsandbytes
Write-Host "Verifying bitsandbytes..." -ForegroundColor Yellow
python -c "import bitsandbytes; print(f'bitsandbytes {bitsandbytes.__version__}')" 2>$null
if (-not $?) {
    Write-Host "ERROR: bitsandbytes not available. Install it: uv sync --package noprop-mesh" -ForegroundColor Red
    exit 1
}

# 3. Verify 4-bit loading works
Write-Host "Verifying 4-bit base-model loading..." -ForegroundColor Yellow
python -c @'
import torch
if torch.cuda.is_available():
    import bitsandbytes as bnb
    print("CUDA available, bitsandbytes ready for NF4 loading")
else:
    print("CUDA not available — running in CPU-only fallback mode")
    print("4-bit quantization requires CUDA")
'@

# 4. Set matmul precision
$env:TORCH_FLOAT32_MATMUL_PRECISION = "high"

# 5. Build target path
$targetScript = Join-Path $NOPROP_DIR "src" $Script
if (-not (Test-Path $targetScript)) {
    Write-Host "ERROR: Script not found: $targetScript" -ForegroundColor Red
    Write-Host "Available scripts in NoProp/src/:" -ForegroundColor Yellow
    Get-ChildItem (Join-Path $NOPROP_DIR "src") -Filter "*.py" | ForEach-Object { "  $($_.Name)" }
    exit 1
}

Write-Host "Running: python $targetScript $Args" -ForegroundColor Green
python $targetScript @Args
