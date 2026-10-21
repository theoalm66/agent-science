@echo off
rem Start the Pandora web UI from this folder (so config.yaml and reports/ are found) and open the browser.
rem Runs the module through python.exe instead of the pip-generated pandora.exe launcher: that launcher
rem is unsigned, and application-control policies (Device Guard / WDAC) often block it.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Pandora is not installed. Run:  py -m venv .venv  and  .venv\Scripts\python -m pip install -e ".[dev]"
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m science_agent.web %*
if errorlevel 1 pause
