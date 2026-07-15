@echo off
setlocal enabledelayedexpansion

echo ============================================
echo  AGK Data Generator — Automated General Knowledge
echo  Generates multi-phase AR/EN datasets for mesh training
echo ============================================
echo.

set PHASES=%1
if "%PHASES%"=="" set PHASES=1 2 3 4 5 6 7 8 9 10 11

set OUTPUT=%2
if "%OUTPUT%"=="" set OUTPUT=agk_data

set SAMPLES=%3
if "%SAMPLES%"=="" set SAMPLES=0

echo Output:  %OUTPUT%
echo Phases:  %PHASES%
echo Samples: %SAMPLES% (0 = phase defaults)
echo.

rem Build the command
set CMD=--output "%OUTPUT%" --phases %PHASES%
if not "%SAMPLES%"=="0" set CMD=%CMD% --samples %SAMPLES%

echo Starting generation...
echo.
uv run --no-sync --package noprop-mesh python scripts\agk_data_generator.py %CMD%
if !ERRORLEVEL! NEQ 0 (
    echo.
    echo Generation failed.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  AGK data generated in %OUTPUT%
echo  Run train_from_text.bat to train the mesh:
echo     train_from_text.bat %OUTPUT%
echo ============================================
echo.
pause
