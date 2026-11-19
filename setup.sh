#!/usr/bin/env bash
# Create .venv, install dependencies (CUDA torch on Ubuntu GPU machines),
# and verify the install actually works.
#
# Usage:
#   ./setup.sh              # runtime + dev deps, CUDA torch when available
#   ./setup.sh --no-dev     # runtime deps only
#   ./setup.sh --cpu-only   # force CPU-only torch (no CUDA download)
#
# Primary target: Ubuntu workstation with an NVIDIA GPU (RTX 4060 Ti).
# Also works on the Windows laptop via WSL2/Ubuntu and on CPU-only machines.
set -euo pipefail

cd "$(dirname "$0")"

INSTALL_DEV=1
CPU_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --no-dev) INSTALL_DEV=0 ;;
    --dev) INSTALL_DEV=1 ;;
    --cpu-only) CPU_ONLY=1 ;;
    --help|-h)
      echo "Usage: ./setup.sh [--dev|--no-dev] [--cpu-only]"
      exit 0
      ;;
    *) echo "Unknown argument: $arg (try --help)" >&2; exit 1 ;;
  esac
done

echo "=== NeuralDoom-Lab Setup (Ubuntu/Linux) ==="

# --- OS / GPU pre-checks (informational; never fatal) -----------------------
if [ -r /proc/version ] && grep -qi microsoft /proc/version 2>/dev/null; then
  echo "Detected WSL. For GPU passthrough, make sure the Windows NVIDIA driver"
  echo "is installed and 'nvidia-smi' works inside WSL."
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA driver detected:"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null \
    | sed 's/^/  /' || echo "  (nvidia-smi present but query failed — check driver)"
else
  echo "No NVIDIA GPU detected (nvidia-smi not found). Training will use CPU."
  echo "On Ubuntu, install the recommended NVIDIA driver with ubuntu-drivers, then reboot."
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: $PYTHON_BIN not found. Install Python 3.11-3.12 and retry." >&2
  echo "  sudo apt update && sudo apt install python3.11 python3.11-venv" >&2
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

if [ "$CPU_ONLY" -eq 1 ]; then
  echo "Installing CPU-only torch (CUDA disabled) ..."
  pip install --quiet "torch>=2.2,<3.0" --index-url https://download.pytorch.org/whl/cpu
  pip install -r requirements.txt
else
  echo "Installing dependencies from requirements.txt (platform-default torch;"
  echo "this can take a few minutes) ..."
  pip install -r requirements.txt
fi

if [ "$INSTALL_DEV" -eq 1 ]; then
  echo "Installing dev dependencies (pytest, ruff) ..."
  pip install -r requirements-dev.txt
fi

mkdir -p runs checkpoints

echo "Verifying imports ..."
python - <<'PY'
import vizdoom, gymnasium, numpy, cv2, torch, stable_baselines3
from src.config import SCENARIOS

print(f"  vizdoom            {vizdoom.__version__}")
print(f"  gymnasium          {gymnasium.__version__}")
print(f"  stable-baselines3  {stable_baselines3.__version__}")
print(f"  torch              {torch.__version__} (cuda: {torch.cuda.is_available()})")
if torch.cuda.is_available():
    print(f"  GPU                {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU memory         {props.total_memory // (1024*1024)} MB")
else:
    print("  GPU                none (CPU training)")
print(f"  numpy              {numpy.__version__}")
print(f"  scenarios          {', '.join(sorted(SCENARIOS))}")
PY

echo ""
echo "=== Setup Complete ==="
echo "Activate the environment with:"
echo "  source .venv/bin/activate"
echo ""
echo "Then try:"
echo "  python -m pytest -q"
echo "  python -m src.play --random --episodes 1"
echo "  python -c \"import torch; print(torch.cuda.is_available())\"  # GPU check"
echo ""
echo "Train on the GPU (Ubuntu workstation):"
echo "  python -m src.train --scenario my_way_home --n-envs 4 --timesteps 100000"
