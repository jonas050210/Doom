# Create .venv, install dependencies, and verify the install on Windows.
$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

Write-Host "=== NeuralDoom-Lab Windows Setup ==="

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
Write-Host "Using Python $pyVersion"

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment in .venv ..."
    & $pyExe @pyArgs -m venv .venv
} else {
    Write-Host "Reusing existing .venv ..."
}

& .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip ..."
python -m pip install --quiet --upgrade pip

Write-Host "Installing dependencies (this can take a few minutes) ..."
pip install -r requirements.txt

New-Item -ItemType Directory -Force -Path runs, checkpoints | Out-Null

Write-Host "Verifying imports ..."
python -c @"
import vizdoom, gymnasium, numpy, cv2, torch, stable_baselines3
from src.config import SCENARIOS
print(f'  vizdoom            {vizdoom.__version__}')
print(f'  gymnasium          {gymnasium.__version__}')
print(f'  stable-baselines3  {stable_baselines3.__version__}')
print(f'  torch              {torch.__version__}')
print(f'  numpy              {numpy.__version__}')
print('  scenarios          ' + ', '.join(sorted(SCENARIOS)))
"@

Write-Host ""
Write-Host "=== Setup Complete ==="
Write-Host "Activate the environment with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Then try:"
Write-Host "  python -m pytest -v"
Write-Host "  python -m src.play --random --episodes 1"
