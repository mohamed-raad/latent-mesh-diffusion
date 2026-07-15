@echo off
setlocal enabledelayedexpansion

echo ============================================
echo  Generate text from a prompt
echo ============================================
echo.

set PROMPT=%1
set CKPT=%2

if "%PROMPT%"=="" (
    set /p PROMPT="Enter prompt: "
)
if "%CKPT%"=="" set CKPT=checkpoints\mesh

echo Prompt: %PROMPT%
echo Checkpoints: %CKPT%
echo.

uv run --no-sync --package noprop-mesh python scripts\generate_from_prompt.py --prompt "%PROMPT%" --ckpt "%CKPT%"
if %ERRORLEVEL% NEQ 0 (
    pause
    exit /b 1
)
echo.
pause
