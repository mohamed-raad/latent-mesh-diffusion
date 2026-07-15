@echo off
setlocal enabledelayedexpansion

echo ============================================
echo  "Up-to-Date" Ingestion Workflow
echo  Fetches docs from AI agent frameworks,
echo  Python/NodeJS libraries, and Q&A sites.
echo  Creates training datasets in ingested_data/
echo ============================================
echo.

set SOURCES=%1
set MAX_PAGES=%2

if "%SOURCES%"=="" set SOURCES=all
if "%MAX_PAGES%"=="" set MAX_PAGES=5

echo Sources:    %SOURCES%
echo Max Pages:  %MAX_PAGES%
echo.
echo Available: agent (AI frameworks), python (Python lib docs)
echo            nodejs (NodeJS lib docs), qa (Stack Overflow)
echo            airesearch (AI/ML research papers)
echo            all (everything)
echo.

uv run --no-sync --package noprop-mesh python scripts\ingest_docs.py --sources %SOURCES% --max-pages %MAX_PAGES%
if !ERRORLEVEL! NEQ 0 (
    echo.
    echo Ingestion failed.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Ingestion complete — check ingested_data/
echo  Train: train_from_text.bat ingested_data
echo ============================================
echo.
pause
