@echo off
rem Start the Pandora web UI from this folder (so config.yaml and reports/ are found) and open the browser.
cd /d "%~dp0"
if not exist ".venv\Scripts\pandora.exe" (
  echo Pandora is not installed. Run:  py -m venv .venv  and  .venv\Scripts\python -m pip install -e ".[dev]"
  pause
  exit /b 1
)
".venv\Scripts\pandora.exe" %*
