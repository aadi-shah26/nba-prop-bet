#!/bin/bash
# Double-click this file (macOS) to open the NBA model in your browser.
# Close the Terminal window that opens to stop the app.
cd "$(dirname "$0")" || exit 1
if ! python3 -c "import streamlit, nba_api, scipy" 2>/dev/null; then
  echo "First run: installing requirements..."
  python3 -m pip install -r requirements.txt || { echo "Install failed"; read -r; exit 1; }
fi
exec python3 -m streamlit run app.py
