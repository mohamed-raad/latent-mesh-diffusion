@echo off
title Thinker Curriculum Generator
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ============================================================
:: generate_curriculum.bat
:: Starts speed-optimized llama-server + generates thinker-weighted JSONL datasets
:: Core phases (linguistics, reasoning, math) get 3-4x more samples
:: Every sample saved instantly ? Ctrl+C safe, re-run to resume
:: ============================================================

set "LLAMA_DIR=E:\my apps\LLAMA"
set "LLAMA_API=http://127.0.0.1:8081"
set "OUTPUT_DIR=%~dp0NoProp\curriculum_data"
set "PHASES=0,1,2,3,4,5,6,7,8,9,10"

:: Default: 50 base per phase, thinker-weighted (core gets 3-4x, support gets 1x)
:: Estim. total: 50*(3+2+2+4+2+3+1+1+1+1+2) = 50*22 = 1100 samples
set "SAMPLES=50"
set "RESUME="
set "FRESH="

:: --- Parse args ---
if not "%1"=="" set "PHASES=%1"
if not "%2"=="" set "SAMPLES=%2"
if "%3"=="--fresh" set "FRESH=--fresh"
if "%3"=="--equal" set "EQUAL=--equal"

:: --- Server config (speed mode) ---
set "SERVER_EXE=%LLAMA_DIR%\llama-server.exe"
set "MODEL=%LLAMA_DIR%\gemma-4-E2B-it-Q4_K_S.gguf"
set "CTX=8192"
set "BATCH=2048"
set "UBATCH=2048"
set "THREADS=12"
set "PORT=8080"

echo ============================================================
echo   THINKER CURRICULUM GENERATOR
echo ============================================================
echo  Server:    gemma-4-E2B-it-Q4_K_S.gguf (ctx=%CTX%, batch=%BATCH%)
echo  Phases:    %PHASES%  (thinker-weighted: core 3-4x, support 1x)
echo  Base:      %SAMPLES% per phase
if "%EQUAL%"=="--equal" echo  Mode:      Balanced (all phases equal)
if "%EQUAL%"=="" echo  Mode:      Thinker (linguistics/reasoning/math prioritized)
if "%FRESH%"=="--fresh" echo  Mode:      Fresh start
if not "%FRESH%"=="--fresh" echo  Resume:    Auto ^(re-run to continue^)
echo ============================================================
echo.

:: ======= STEP 1: Start speed-optimized server =======
echo [1/4] Starting llama-server (speed mode)...
echo         (minimized window ? check taskbar)

powershell -Command "try { $r = Invoke-WebRequest -Uri '%LLAMA_API%/v1/models' -UseBasicParsing -TimeoutSec 3; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1

if !ERRORLEVEL! equ 0 (
    echo  Server already running ? reusing.
) else (
    if not exist "%SERVER_EXE%" ( echo ERROR: server.exe missing! & pause & exit /b )
    if not exist "%MODEL%" ( echo ERROR: model missing! & pause & exit /b )

    start "llama-server" /min cmd /c ""%SERVER_EXE%" ^
        --model "%MODEL%" --ctx-size %CTX% --n-gpu-layers 99 ^
        --host 127.0.0.1 --port %PORT% ^
        --batch-size %BATCH% --ubatch-size %UBATCH% --parallel 1 ^
        --cache-type-k q8_0 --cache-type-v q4_0 --flash-attn on ^
        --threads %THREADS% --threads-batch %THREADS% ^
        --mlock --no-mmap --cont-batching --no-warmup"

    echo  Waiting for server...
    set "WAIT=0"
    :wait
    powershell -Command "try { $r = Invoke-WebRequest -Uri '%LLAMA_API%/v1/models' -UseBasicParsing -TimeoutSec 3; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
    if !ERRORLEVEL! equ 0 ( echo  Ready! ) else (
        set /a WAIT+=1
        if !WAIT! geq 20 ( echo  [WARN] Still waiting... check server window )
        timeout /t 3 /nobreak >nul & goto wait
    )
)

:: ======= STEP 2: Output dir =======
echo [2/4] Preparing output...
if "%FRESH%"=="--fresh" ( if exist "%OUTPUT_DIR%" rmdir /s /q "%OUTPUT_DIR%" 2>nul )
mkdir "%OUTPUT_DIR%" 2>nul
echo  Output: %OUTPUT_DIR%

:: ======= STEP 3: Activate Python =======
echo [3/4] Activating Python...
set "VENV_DIR=%~dp0.venv"
set "PYTHON_CMD=python"
if exist "%VENV_DIR%\Scripts\activate.bat" (
    call "%VENV_DIR%\Scripts\activate.bat"
    set "PYTHON_CMD=%VENV_DIR%\Scripts\python.exe"
)

:: ======= STEP 4: Generate =======
echo [4/4] Generating thinker curriculum...
echo.
echo  ? CORE phases (3-4x): Foundation(3), Reasoning(4), Mathematics(3)
echo  ? HIGH phases (2x):   Knowledge(2), Relationships(2), Programming(2), Self_Improvement(2)
echo  Support (1x):          Tool_Use, Long_Context, Memory, Multi_Agent
echo.
echo  Every sample saved instantly. Ctrl+C = resume anytime.
echo.

"%PYTHON_CMD%" "%~dp0NoProp\scripts\curriculum_generator.py" ^
    --api "%LLAMA_API%" ^
    --output "%OUTPUT_DIR%" ^
    --phases "%PHASES%" ^
    --samples %SAMPLES% ^
    %FRESH% %EQUAL%

if !ERRORLEVEL! neq 0 (
    echo.
    echo  [ERROR] Code !ERRORLEVEL! ? re-run to resume.
    pause
    exit /b !ERRORLEVEL!
)

:: ======= DONE =======
echo.
echo ============================================================
echo   DONE!
echo ============================================================
echo.
echo  Location: %OUTPUT_DIR%
echo.
echo  Re-run:
echo    generate_curriculum.bat                  -- thinker mode, 50 base
echo    generate_curriculum.bat 3,5 2000         -- reasoning + math, 2000 base
echo    generate_curriculum.bat "" "" --equal     -- balanced weights
echo    generate_curriculum.bat 0 500            -- foundation only, 500 base
echo.

pause
