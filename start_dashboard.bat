@echo off
setlocal enabledelayedexpansion

set PORT=8765

REM Check if already running
netstat -an 2>nul | find ":%PORT%" >nul
if !ERRORLEVEL! EQU 0 (
    echo Dashboard already running at http://127.0.0.1:%PORT%
    echo Opening in browser...
    start http://127.0.0.1:%PORT%
    exit /b 0
)

echo ============================================
echo  Mesh Training Dashboard
echo  Starting on http://127.0.0.1:%PORT%
echo ============================================
echo.
echo Commands can be triggered from the dashboard UI.
echo.

REM Launch in background, open browser
start /B "" uv run --no-sync --package noprop-mesh python NoProp\scripts\mesh_dashboard.py
timeout /t 2 /nobreak >nul
start http://127.0.0.1:%PORT%

echo Dashboard launched in background.
echo http://127.0.0.1:%PORT%
echo.
