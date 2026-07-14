param(
    [string]$Target = "DeepSpec",
    [string]$WslDistro = "Ubuntu"
)

$REPO_ROOT = Split-Path -Parent $PSScriptRoot
$WSL_REPO = $REPO_ROOT -replace '\\', '/' -replace '^([A-Z]):', '/mnt/$1'

$validTargets = @{ "DeepSpec" = $true; "gemma" = $true }
if (-not $validTargets.ContainsKey($Target)) {
    Write-Host "ERROR: Target must be DeepSpec or gemma" -ForegroundColor Red
    exit 1
}

Write-Host "=== NN Mesh: WSL Reference Shell ===" -ForegroundColor Cyan
Write-Host "Target: $Target/" -ForegroundColor Yellow
Write-Host "Opening WSL shell at $WSL_REPO ..." -ForegroundColor Yellow
Write-Host "NOTE: $Target/ is READ-ONLY. Do not edit files." -ForegroundColor Magenta

wsl -d $WslDistro --cd "$WSL_REPO/$Target"
