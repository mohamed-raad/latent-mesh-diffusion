param(
    [string]$TestPath = "NoProp/src/tests/",
    [switch]$Verbose,
    [switch]$NoHeader
)

$ErrorActionPreference = "Stop"
$REPO_ROOT = Split-Path -Parent $PSScriptRoot
$VENV_DIR = Join-Path $REPO_ROOT ".venv"

if (-not $NoHeader) {
    Write-Host "=== NN Mesh: run-tests ===" -ForegroundColor Cyan
}

# Activate venv
if (-not (Test-Path (Join-Path $VENV_DIR "Scripts\Activate.ps1"))) {
    Write-Host "Creating uv venv..." -ForegroundColor Yellow
    Push-Location $REPO_ROOT
    uv venv
    uv sync --package noprop-mesh --group dev
    Pop-Location
}
. (Join-Path $VENV_DIR "Scripts\Activate.ps1")

$vFlag = if ($Verbose) { "-v" } else { "" }
$testDir = Join-Path $REPO_ROOT $TestPath

Write-Host "Running pytest on $testDir" -ForegroundColor Green
python -m pytest $testDir $vFlag --tb=short
