@echo off
setlocal enabledelayedexpansion

echo ============================================
echo  Train mesh model on text data
echo ============================================
echo.

set DATA_DIR=%1
set CKPT=%2
set NODES=%3

if "%DATA_DIR%"=="" set DATA_DIR=data
if "%CKPT%"=="" set CKPT=checkpoints\mesh
if "%NODES%"=="" set NODES=nodes

if not exist "%DATA_DIR%" (
    echo Data directory '%DATA_DIR%' not found.
    echo Put your .txt .md .py .json files in a folder named 'data'
    echo in the project root, or pass the path as first argument.
    pause
    exit /b 1
)

set EPOCHS=%4
if "%EPOCHS%"=="" set EPOCHS=10

set BATCH=%5
if "%BATCH%"=="" set BATCH=4

echo Data:     %DATA_DIR%
echo Epochs:   %EPOCHS%
echo Batch:    %BATCH%
echo Checkpts: %CKPT%
echo Nodes:    %NODES%
echo.

uv run --no-sync --package noprop-mesh python scripts\train_on_text.py --data "%DATA_DIR%" --ckpt "%CKPT%" --nodes "%NODES%" --epochs %EPOCHS% --batch %BATCH%
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Training failed.
    pause
    exit /b 1
)
echo.
echo Training complete.
pause
