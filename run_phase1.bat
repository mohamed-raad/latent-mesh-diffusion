@echo off
cd /d "E:\my apps\NN"
echo ============================================================
echo  Phase 1 - Core 250M Training
echo  Started at: %date% %time%
echo ============================================================
.\.venv\Scripts\python.exe -u NoProp/src/train_pipeline.py --phase phase1 --no-packing 2>&1
echo.
echo ============================================================
echo  Training exited at: %date% %time%
echo ============================================================
pause
