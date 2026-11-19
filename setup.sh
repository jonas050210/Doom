#!/usr/bin/env bash
# Create .venv, install dependencies, and verify the install actually works.
set -euo pipefail

cd "$(dirname "$0")"

echo "=== NeuralDoom-Lab Setup ==="

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: $PYTHON_BIN not found. Install Python 3.11-3.12 and retry." >&2
  exit 1
fi

PY_VERSION=$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=${PY_VERSION%%.*}
PY_MINOR=${PY_VERSION##*.}

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
  echo "Error: Python 3.11+ is required (found $PY_VERSION)." >&2
  exit 1
fi
if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 13 ]; then
  echo "Warning: Python $PY_VERSION is newer than the tested 3.11/3.12 range;"
  echo "         prebuilt wheels for vizdoom/torch may not exist yet." >&2
fi

echo "Using Python $PY_VERSION ($("$PYTHON_BIN" -c 'import sys; print(sys.executable)'))"

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment in .venv ..."
  "$PYTHON_BIN" -m venv .venv
else
  echo "Reusing existing .venv ..."
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "Upgrading pip ..."
python -m pip install --quiet --upgrade pip

echo "Installing dependencies from requirements.txt (this can take a few minutes) ..."
pip install -r requirements.txt

mkdir -p runs checkpoints

echo "Verifying imports ..."
python - <<'PY'
import vizdoom, gymnasium, numpy, cv2, torch, stable_baselines3
from src.config import SCENARIOS

print(f"  vizdoom            {vizdoom.__version__}")
print(f"  gymnasium          {gymnasium.__version__}")
print(f"  stable-baselines3  {stable_baselines3.__version__}")
print(f"  torch              {torch.__version__} (cuda: {torch.cuda.is_available()})")
print(f"  numpy              {numpy.__version__}")
print(f"  scenarios          {', '.join(sorted(SCENARIOS))}")
PY

echo ""
echo "=== Setup Complete ==="
echo "Activate the environment with:"
echo "  source .venv/bin/activate"
echo ""
echo "Then try:"
echo "  python -m pytest -v"
echo "  python -m src.play --random --episodes 1"
