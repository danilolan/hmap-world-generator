@echo off
rem World generator tuning tool (WORLDGEN.md section 4): double-click to start.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creating the Python environment...
  py -3 -m venv .venv 2>nul || python -m venv .venv
)
".venv\Scripts\python.exe" -m pip install -q -r requirements.txt
".venv\Scripts\python.exe" server.py --open
pause
