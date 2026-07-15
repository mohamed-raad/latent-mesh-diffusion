@echo off
setlocal enabledelayedexpansion

echo ============================================
echo  LLM Dataset Generator (uses llama-cli.exe)
echo  Generates high-quality datasets via local GGUF model
echo  Output goes to agk_llm/ (separate from template agk_data/)
echo ============================================
echo.

set MODEL=%1
set OUTPUT=%2
set SAMPLES=%3

if "%MODEL%"=="" set MODEL=E:\my apps\LLAMA\gemma-4-E2B-it-Q4_K_S.gguf
if "%OUTPUT%"=="" set OUTPUT=agk_llm
if "%SAMPLES%"=="" set SAMPLES=0

if not exist "%MODEL%" (
    echo Model not found: %MODEL%
    echo.
    echo Usage: run_llm_generate.bat [model_path] [output_dir] [samples_per_topic]
    echo   model_path   - Path to GGUF file (default: gemma-4 GGUF in LLAMA/)
    echo   output_dir   - Output directory (default: agk_llm)
    echo   samples      - Samples per topic (0 = use per-topic defaults)
    echo.
    pause
    exit /b 1
)

echo Model:      %MODEL%
echo Output:     %OUTPUT%
echo Samples:    %SAMPLES% (0 = per-topic defaults)
echo.

set CMD=--model "%MODEL%" --output "%OUTPUT%"
if not "%SAMPLES%"=="0" set CMD=%CMD% --samples %SAMPLES%

echo Starting LLM-powered dataset generation...
echo.
uv run --no-sync --package noprop-mesh python scripts\generate_with_llm.py %CMD%
if !ERRORLEVEL! NEQ 0 (
    echo.
    echo LLM generation failed.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  LLM dataset generated in %OUTPUT%
echo  Train: train_from_text.bat %OUTPUT%
echo  Dashboard: start_dashboard.bat
echo ============================================
echo.
pause
