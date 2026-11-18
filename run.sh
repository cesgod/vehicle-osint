#!/usr/bin/env sh
# macOS / Linux launcher. First run creates .venv and installs dependencies.
set -e
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || python3 -m venv .venv
[ -f .venv/.deps-installed ] || { .venv/bin/python -m pip install -q -r requirements.txt && touch .venv/.deps-installed; }
exec .venv/bin/python run.py
