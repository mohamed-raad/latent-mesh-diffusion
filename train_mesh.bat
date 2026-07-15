@echo off
REM ============================================================
REM  Mesh Training Pipeline — hierarchical expert growth
REM  Usage:  train_mesh.bat [phase]
REM
REM  Phases:
REM    1   Core brain   (general reasoning, 128ctx, 35K steps)
REM    2   Domains      (coding, math, science, 256ctx, 10K)
REM    3   Topics       (python, rust, react, 512ctx, 5K/topic)
REM    4   Specialists  (python-api, js-react, 512ctx, 2K/spec)
REM    all (default)    runs 1 -> 2 -> 3 -> 4 in sequence
REM
REM  To use real data instead of synthetic, add:
REM     --curriculum-dir "path\to\jsonl"   or
REM     --agk-dir "path\to\md"
REM ============================================================
setlocal enabledelayedexpansion

set PHASE=%~1
if "%PHASE%"=="" set PHASE=all

set VENV=%~dp0.venv\Scripts\python.exe
set SCRIPT=%~dp0NoProp\src\train_mesh.py
set CKPT_DIR=%USERPROFILE%\checkpoints\mesh
set NODES_DIR=%~dp0NoProp\nodes

set BASE_ARGS=--nodes-dir "%NODES_DIR%" --vocab-size 151643 --num-heads 8 --top-k 4 --seed 42
set BASE_ARGS=%BASE_ARGS% --no-external-nodes --num-epochs 999 --packing
if "%PHASE%"=="1" set BASE_ARGS=%BASE_ARGS% --token-budget 512
if "%PHASE%"=="2" set BASE_ARGS=%BASE_ARGS% --token-budget 1024
if "%PHASE%"=="3" set BASE_ARGS=%BASE_ARGS% --token-budget 2048
if "%PHASE%"=="4" set BASE_ARGS=%BASE_ARGS% --token-budget 2048

if /i "%PHASE%"=="1" goto :phase1
if /i "%PHASE%"=="2" goto :phase2
if /i "%PHASE%"=="3" goto :phase3
if /i "%PHASE%"=="4" goto :phase4
if /i "%PHASE%"=="all" (
    call :phase1 || exit /b 1
    call :phase2 || exit /b 1
    call :phase3 || exit /b 1
    call :phase4 || exit /b 1
    goto :done
)
echo Unknown phase: %PHASE%
echo Valid phases: 1 2 3 4 all
exit /b 1

:phase1
echo ===== PHASE 1 — Core Brain =====
echo Model: small (1024-dim)  Canvas: 128ctx  Batch: 2  Steps: 35000
echo -----------------------------------------------------------
"%VENV%" "%SCRIPT%" ^
    --model-size small ^
    --canvas-len 128 ^
    --batch-size 2 ^
    --lr 1e-3 ^
    --max-steps 35000 ^
    --mitosis-threshold 0.5 ^
    --phase core ^
    --checkpoint-dir "%CKPT_DIR%\core" ^
    %BASE_ARGS% ^
    --resume
if errorlevel 1 (
    echo Phase 1 FAILED
    exit /b 1
)
goto :eof

:phase2
echo ===== PHASE 2 — Domain Experts =====
echo Model: medium (768-dim)  Canvas: 256ctx  Batch: 1  Steps: 10000
echo -----------------------------------------------------------
"%VENV%" "%SCRIPT%" ^
    --model-size medium ^
    --canvas-len 256 ^
    --batch-size 1 ^
    --lr 5e-4 ^
    --max-steps 10000 ^
    --mitosis-threshold 0.4 ^
    --phase domain ^
    --checkpoint-dir "%CKPT_DIR%\domain" ^
    %BASE_ARGS% ^
    --resume
if errorlevel 1 (
    echo Phase 2 FAILED
    exit /b 1
)
goto :eof

:phase3
echo ===== PHASE 3 — Topic Specialization =====
echo Model: standard (512-dim)  Canvas: 512ctx  Batch: 1  Steps: 5000
echo -----------------------------------------------------------
"%VENV%" "%SCRIPT%" ^
    --model-size standard ^
    --canvas-len 512 ^
    --batch-size 1 ^
    --lr 2e-4 ^
    --max-steps 5000 ^
    --mitosis-threshold 0.3 ^
    --phase topic ^
    --checkpoint-dir "%CKPT_DIR%\topic" ^
    %BASE_ARGS% ^
    --resume
if errorlevel 1 (
    echo Phase 3 FAILED
    exit /b 1
)
goto :eof

:phase4
echo ===== PHASE 4 — Specialist Experts =====
echo Model: standard (512-dim)  Canvas: 512ctx  Batch: 1  Steps: 2000
echo -----------------------------------------------------------
"%VENV%" "%SCRIPT%" ^
    --model-size standard ^
    --canvas-len 512 ^
    --batch-size 1 ^
    --lr 1e-4 ^
    --max-steps 2000 ^
    --mitosis-threshold 0.25 ^
    --phase specialist ^
    --checkpoint-dir "%CKPT_DIR%\specialist" ^
    %BASE_ARGS% ^
    --resume
if errorlevel 1 (
    echo Phase 4 FAILED
    exit /b 1
)
goto :eof

:done
echo ===== ALL PHASES COMPLETE =====
echo Final checkpoint: %CKPT_DIR%\specialist\step_latest.pt
echo.
echo To continue training with real data, run:
echo   train_mesh.bat 1 --curriculum-dir "path\to\jsonl"
echo   train_mesh.bat 2 --agk-dir "path\to\md"
