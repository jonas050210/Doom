#!/usr/bin/env bash
set -euo pipefail

echo "=== NeuralDoom-Lab Setup ==="

# Check Python 3.11+
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: python3 not found. Please install Python 3.11 or newer."
  exit 1
fi

PY_VERSION=$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
  echo "Error: Python 3.11+ is required (found $PY_VERSION)."
  exit 1
fi

echo "Using Python $PY_VERSION"

# Create virtual environment
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment in .venv..."
  "$PYTHON_BIN" -m venv .venv
else
  echo "Using existing .venv..."
fi

# Activate virtual environment
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Creating runs/ and checkpoints/ directories..."
mkdir -p runs checkpoints

echo ""
echo "=== Setup Complete ==="
echo "Activate the environment with:"
echo "  source .venv/bin/activate"
