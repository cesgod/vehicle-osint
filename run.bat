@echo off
rem Double-click to start. First run creates .venv and installs dependencies (a few minutes).
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto :deps

set "PY="
for %%V in (3.13 3.12 3.11 3.10) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
  )
)
if not defined PY (
  echo Python 3.10 or newer was not found. Install it from https://www.python.org/downloads/ and run this again.
  pause
  exit /b 1
)
echo Creating virtual environment with %PY% ...
%PY% -m venv .venv || (pause & exit /b 1)

:deps
if not exist ".venv\.deps-installed" (
  echo Installing dependencies, please wait ...
  ".venv\Scripts\python.exe" -m pip install -q --disable-pip-version-check -r requirements.txt || (pause & exit /b 1)
  echo ok> ".venv\.deps-installed"
)

".venv\Scripts\python.exe" run.py
pause
