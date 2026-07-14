param(
    [string]$WslDistro = "Ubuntu"
)

Write-Host "=== NN Mesh: WSL2 Bootstrap ===" -ForegroundColor Cyan

$wslCheck = wsl --status 2>$null
if (-not $?) {
    Write-Host "WSL is not installed. Install it first: wsl --install -d Ubuntu" -ForegroundColor Red
    exit 1
}

Write-Host "Bootstrapping $WslDistro with Python 3.12 + uv + CUDA toolkit..." -ForegroundColor Yellow

wsl -d $WslDistro -- bash -c @'
set -euo pipefail
echo "=== Updating apt packages ==="
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential curl wget python3 python3-pip python3-venv

echo "=== Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"

echo "=== Setting Python 3.12 ==="
sudo apt-get install -y -qq python3.12 python3.12-venv || true

echo "=== uv version ==="
uv --version

echo "=== Setup complete ==="
echo "From Windows, run: wsl -d Ubuntu --cd /mnt/e/path/to/repo"
'@

Write-Host "WSL2 bootstrap complete." -ForegroundColor Green
Write-Host "Run .\scripts\run-wsl-ref.ps1 to enter a WSL shell in the repo." -ForegroundColor Cyan
