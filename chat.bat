@echo off
setlocal enabledelayedexpansion

echo ============================================
echo  Interactive chat with mesh model
echo ============================================
echo.

set CKPT=%1
if "%CKPT%"=="" set CKPT=checkpoints\mesh

echo Loading mesh from %CKPT%
echo.
echo Commands: /quit  /reset  /summary
echo.

uv run --no-sync --package noprop-mesh python scripts\chat.py --checkpoint-dir "%CKPT%"
if %ERRORLEVEL% NEQ 0 (
    pause
    exit /b 1
)
