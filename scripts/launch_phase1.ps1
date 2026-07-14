$env:HF_TOKEN = "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"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

& "E:\my apps\NN\.venv\Scripts\python.exe" -u "E:\my apps\NN\NoProp\src\run_phase1.py"
pause
