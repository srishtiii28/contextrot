param([int]$Port=8000)
$ErrorActionPreference='Stop'
$backendPath=Join-Path $PSScriptRoot 'context-rot-backend'
$pythonPath=Join-Path $backendPath '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    $parentPython=Join-Path (Split-Path $PSScriptRoot -Parent) '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $parentPython) {
        $pythonPath=$parentPython
    }
}
if (-not (Test-Path -LiteralPath $pythonPath)) {
    $bootstrap=Get-Command python -ErrorAction SilentlyContinue
    if ($bootstrap) { & $bootstrap.Source -m venv (Join-Path $backendPath '.venv') }
    else {
        $bundledPython=Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
        if (-not (Test-Path -LiteralPath $bundledPython)) { throw 'Install Python 3.11 or newer, then run this script again.' }
        & $bundledPython -m venv (Join-Path $backendPath '.venv')
    }
    if ($LASTEXITCODE -ne 0) { throw 'Could not create virtual environment.' }
    & $pythonPath -m pip install -r (Join-Path $backendPath 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
}
Write-Host "Demo: http://127.0.0.1:$Port/demo/lab.html"
Write-Host "API docs: http://127.0.0.1:$Port/docs"
Write-Host 'Press Ctrl+C to stop. Default mode is offline; existing .env settings take precedence.'
Push-Location $backendPath
try { & $pythonPath -m uvicorn app.main:app --host 127.0.0.1 --port $Port }
finally { Pop-Location }
