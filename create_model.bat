@echo off
setlocal enabledelayedexpansion

echo ============================================
echo  Create a new mesh model from Obsidian vault
echo ============================================
echo.

set VAULT=%1
set CKPT=%2
set NODES=%3

if "%VAULT%"=="" set VAULT=vault
if "%CKPT%"=="" set CKPT=checkpoints\mesh
if "%NODES%"=="" set NODES=nodes

echo Vault: %VAULT%
echo Checkpoints: %CKPT%
echo Nodes: %NODES%
echo.

uv run --no-sync --package noprop-mesh python scripts\create_mesh_model.py --vault "%VAULT%" --ckpt "%CKPT%" --nodes "%NODES%"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Model creation failed.
    pause
    exit /b 1
)
echo.
echo Done. Model created at %CKPT%
echo Run train_from_text.bat to train on data.
echo.
pause
