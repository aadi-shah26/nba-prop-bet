#!/bin/bash
# Double-click this file (macOS) to open the NBA model in your browser.
# Close the Terminal window that opens to stop the app.
# Follow a Desktop/Dock symlink back to the project folder
cd "$(dirname "$(readlink -f "$0")")" || exit 1
# The project's .venv if there is one (cron uses it too), else the system Python
PY=python3
[ -x .venv/bin/python ] && PY=.venv/bin/python
# Install or upgrade when a package is missing or Streamlit is older than requirements.txt needs
if ! "$PY" -c "import nba_api, scipy, streamlit as st, sys; sys.exit(tuple(map(int, st.__version__.split('.')[:2])) < (1, 66))" 2>/dev/null; then
  echo "Installing requirements..."
  "$PY" -m pip install -r requirements.txt || { echo "Install failed"; read -r; exit 1; }
fi
exec "$PY" -m streamlit run app.py
