# Create .venv, install dependencies, and verify the install on Windows.
#
# Usage (PowerShell):
#   .\setup.ps1              # runtime + dev deps, CUDA torch when available
#   .\setup.ps1 -NoDev       # runtime deps only
#   .\setup.ps1 -CpuOnly     # force CPU-only torch (no CUDA download)
#
# Primary target is Ubuntu (see setup.sh); this script keeps the Windows
# laptop (PowerShell) at parity: same venv layout, same verification, same
# GPU check. The laptop trains on CPU (--low-ram); a Windows machine with an
# NVIDIA GPU gets CUDA torch automatically.
param(
    [switch]$NoDev,
    [switch]$CpuOnly
)

$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

Write-Host "=== NeuralDoom-Lab Setup (Windows PowerShell) ==="

# --- GPU pre-check (informational; never fatal) ------------------------------
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "NVIDIA driver detected:"
    try {
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | ForEach-Object { Write-Host "  $_" }
    } catch {
        Write-Host "  (nvidia-smi present but query failed — check driver)"
    }
} else {
    Write-Host "No NVIDIA GPU detected (nvidia-smi not found). Training will use CPU."
    Write-Host "On a Windows GPU machine, install the NVIDIA driver from nvidia.com first."
}

# NOTE: keep the executable and its arguments separate. Building a single
# string such as "py -3" and calling `& $pyCmd` treats it as one command name
# and fails when only the py launcher is installed.
$pyExe = $null
$pyArgs = @()
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pyExe = "python"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $pyExe = "py"
    $pyArgs = @("-3")
} else {
    Write-Error "Python 3.11+ not found. Install it from python.org and re-run."
    exit 1
}

$pyVersion = & $pyExe @pyArgs -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$parts = $pyVersion.Split(".")
if ([int]$parts[0] -lt 3 -or ([int]$parts[0] -eq 3 -and [int]$parts[1] -lt 11)) {
    Write-Error "Python 3.11+ is required (found $pyVersion)."
    exit 1
}
if ([int]$parts[0] -eq 3 -and [int]$parts[1] -ge 13) {
    Write-Host "Warning: Python $pyVersion is newer than the tested 3.11/3.12 range;"
    Write-Host "         prebuilt wheels for vizdoom/torch may not exist yet."
}
Write-Host "Using Python $pyVersion"

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment in .venv ..."
    & $pyExe @pyArgs -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "Python command failed (exit $LASTEXITCODE)." }
} else {
    Write-Host "Reusing existing .venv ..."
}

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) { throw "Virtual environment creation failed." }

Write-Host "Upgrading pip ..."
& $venvPython -m pip install --quiet --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Python command failed (exit $LASTEXITCODE)." }

if ($CpuOnly) {
    Write-Host "Installing CPU-only torch (CUDA disabled) ..."
    & $venvPython -m pip install --quiet "torch>=2.2,<3.0" --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw "Python command failed (exit $LASTEXITCODE)." }
    & $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Python command failed (exit $LASTEXITCODE)." }
} else {
    Write-Host "Installing dependencies (platform-default torch; this can take a few minutes) ..."
    & $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Python command failed (exit $LASTEXITCODE)." }
}

if (-not $NoDev) {
    Write-Host "Installing dev dependencies (pytest, ruff) ..."
    & $venvPython -m pip install -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) { throw "Python command failed (exit $LASTEXITCODE)." }
}

New-Item -ItemType Directory -Force -Path runs, checkpoints | Out-Null

Write-Host "Verifying imports ..."
& $venvPython -c @"
import vizdoom, gymnasium, numpy, cv2, torch, stable_baselines3
from src.config import SCENARIOS
print(f'  vizdoom            {vizdoom.__version__}')
print(f'  gymnasium          {gymnasium.__version__}')
print(f'  stable-baselines3  {stable_baselines3.__version__}')
print(f'  torch              {torch.__version__} (cuda: {torch.cuda.is_available()})')
if torch.cuda.is_available():
    print(f'  GPU                {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'  GPU memory         {props.total_memory // (1024*1024)} MB')
else:
    print('  GPU                none (CPU training)')
print(f'  numpy              {numpy.__version__}')
print('  scenarios          ' + ', '.join(sorted(SCENARIOS)))
"@
if ($LASTEXITCODE -ne 0) { throw "Python command failed (exit $LASTEXITCODE)." }

Write-Host ""
Write-Host "=== Setup Complete ==="
Write-Host "Activate the environment with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Then try:"
Write-Host "  python -m pytest -q"
Write-Host "  python -m src.play --random --episodes 1"
Write-Host "  python -c `"import torch; print(torch.cuda.is_available())`"  # GPU check"
Write-Host ""
Write-Host "Train on the laptop (CPU, 2 GB RAM):"
Write-Host "  python -m src.train --scenario my_way_home --low-ram --timesteps 2048"
