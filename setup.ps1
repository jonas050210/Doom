$ErrorActionPreference = "Stop"

Write-Host "=== NeuralDoom-Lab Windows Setup ==="

# Check Python 3.11+
$pyCmd = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } elseif (Get-Command py -ErrorAction SilentlyContinue) { "py -3" } else {
    Write-Error "Python 3.11+ not found. Please install Python 3.11 or newer from python.org."
    exit 1
}

$pyVersion = & $pyCmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$parts = $pyVersion.Split(".")
if ([int]$parts[0] -lt 3 -or ([int]$parts[0] -eq 3 -and [int]$parts[1] -lt 11)) {
    Write-Error "Python 3.11+ is required (found $pyVersion)."
    exit 1
}

Write-Host "Using Python $pyVersion"

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment in .venv..."
    & $pyCmd -m venv .venv
} else {
    Write-Host "Using existing .venv..."
}

Write-Host "Activating .venv..."
& .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

New-Item -ItemType Directory -Force -Path runs, checkpoints | Out-Null

Write-Host ""
Write-Host "=== Setup Complete ==="
Write-Host "Activate the environment with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
