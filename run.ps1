# Fast startup: single script. Creates venv on first run, installs deps once, then starts the server.
# Model path: C:\daytrader-swingtrader-main\models\GRiP.i1-Q4_K_S.gguf

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$venvPy = Join-Path $root "venv\Scripts\python.exe"
$req = Join-Path $root "requirements.txt"
$configPath = Join-Path $root "config\ai-coder.yaml"
$modelPath = "C:\daytrader-swingtrader-main\models\GRiP.i1-Q4_K_S.gguf"
$setupMarker = Join-Path $root "venv\.setup_done"

Write-Host "== Launching AI Runtime =="

# Ensure model exists
if (-not (Test-Path $modelPath)) {
    Write-Host "Model file missing:"
    Write-Host "  $modelPath"
    Write-Host "Place a valid GGUF file there and rerun."
    exit 1
}

# Create venv if missing
if (-not (Test-Path $venvPy)) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
    $venvPy = Join-Path $root "venv\Scripts\python.exe"
}

# One-time dependency install
if (-not (Test-Path $setupMarker)) {
    Write-Host "Installing dependencies (one-time)..."
    & $venvPy -m pip install --upgrade pip setuptools wheel
    & $venvPy -m pip install -r $req
    New-Item -ItemType File -Path $setupMarker -Force | Out-Null
    Write-Host "Dependencies installed."
}

# Sanity check: config references the model path
$cfg = Get-Content $configPath -Raw
$expected = $modelPath.Replace('\','/')
if ($cfg -notmatch [Regex]::Escape($expected)) {
    Write-Host "WARNING: config/ai-coder.yaml gguf_path does not appear to reference:"
    Write-Host "  $expected"
    Write-Host "Server will still start, but the model may not load."
}

Write-Host "Starting server at http://127.0.0.1:8000 ..."
& $venvPy -m server.main