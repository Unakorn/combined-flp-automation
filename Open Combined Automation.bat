@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" "Combined Automation.py" %*
) else (
  py -3.12 "Combined Automation.py" %*
)
if errorlevel 1 pause
