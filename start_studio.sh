#!/usr/bin/env bash
# Walk Studio for macOS and Linux: run ./start_studio.sh. The first start sets everything up.
set -e
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "Setting up Walk Studio for the first time (downloads about 1 GB, takes a few minutes)."
  PYTHON=""
  for candidate in python3.12 python3.13 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then PYTHON="$candidate"; break; fi
  done
  if [ -z "$PYTHON" ]; then
    echo "Python 3 was not found. Install Python 3.12 from https://www.python.org/downloads/ and try again."
    exit 1
  fi
  "$PYTHON" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  if [ "$(uname)" = "Linux" ]; then
    .venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  fi
  .venv/bin/python -m pip install -r requirements.txt
fi

echo "Starting Walk Studio. Your browser opens in a moment. Press Ctrl+C here to stop."
exec .venv/bin/python -m studio "$@"
