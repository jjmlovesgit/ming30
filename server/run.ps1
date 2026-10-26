# Start the Ming 3.0 speech stack without Docker.
#
#   .\run.ps1                  # engine (warm model) + adapter, Ctrl+C stops both
#   .\run.ps1 -AdapterOnly     # adapter only, engine already running
#   .\run.ps1 -Python C:\path\to\python.exe
#
# The engine is the native audio.cpp server; the adapter is this FastAPI app.
# Both are plain processes, so nothing here needs a container runtime.

param(
    [string]$CudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin",
    [string]$Runtime = "..\audio.cpp-0.9.3-cuda12.4",
    [string]$Config = "config\dev.server.json",
    [int]$BackendPort = 8090,
    [int]$Port = 8070,
    [string]$PythonPath = "",
    [switch]$AdapterOnly
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# Pick a Python that can actually import the adapter's dependencies. Candidates
# are tried in order; the first one whose imports succeed wins.
$candidates = @()
if ($PythonPath) { $candidates += , @($PythonPath) }
if ($env:MING30_PYTHON) { $candidates += , @($env:MING30_PYTHON) }
$candidates += , @('python')
$candidates += , @('py', '-3.12')
$candidates += , @('py', '-3.11')
$candidates += , @('py', '-3.10')

$pythonExe = $null
$pythonArgs = @()
$pythonLabel = $null
foreach ($candidate in $candidates) {
    $exe = $candidate[0]
    $pre = @()
    if ($candidate.Count -gt 1) { $pre = $candidate[1..($candidate.Count - 1)] }
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }

    # PowerShell 5.1 turns a native command's stderr into an error record and
    # $ErrorActionPreference = 'Stop' would abort on the first candidate that
    # lacks the packages, so only the exit code is used. The snippet stays
    # quote-free: 5.1 mangles embedded double quotes for native commands.
    $code = 1
    $previous = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'SilentlyContinue'
        & $exe @($pre + @('-c', 'import fastapi,uvicorn,httpx,multipart')) 2>$null | Out-Null
        $code = $LASTEXITCODE
    }
    catch {
        $code = 1
    }
    finally {
        $ErrorActionPreference = $previous
    }

    if ($code -eq 0) {
        $pythonExe = $exe
        $pythonArgs = $pre
        $pythonLabel = ("{0} {1}" -f $exe, ($pre -join ' ')).Trim()
        break
    }
}

if (-not $pythonExe) {
    throw "No Python with fastapi/uvicorn/httpx/python-multipart found. Run 'pip install -r requirements.txt' or pass -PythonPath <path>."
}

$env:PATH = "$CudaBin;$env:PATH"
$env:MING30_BACKEND_URL = "http://127.0.0.1:$BackendPort"
$env:MING30_PORT = "$Port"

New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot "voices") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot ".ming30-tmp") | Out-Null

$backend = $null

if (-not $AdapterOnly) {
    $exe = Join-Path (Resolve-Path $Runtime) "audiocpp_server.exe"
    if (-not (Test-Path $exe)) { throw "audiocpp_server.exe not found at $exe (set -Runtime)" }
    Write-Host "engine : $exe"
    Write-Host "config : $(Resolve-Path $Config)"
    $backend = Start-Process -FilePath $exe -ArgumentList @("--config", (Resolve-Path $Config), "--no-ui") -PassThru -NoNewWindow

    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$BackendPort/health" -TimeoutSec 2
            if ($health.status -eq "ok") {
                Write-Host "engine : ready, audio.cpp $($health.version), $($health.models) models"
                $ready = $true
                break
            }
        }
        catch { }
        if ($backend.HasExited) { throw "engine exited with code $($backend.ExitCode); pass a CUDA bin that its DLLs need" }
    }
    if (-not $ready) { throw "engine did not become ready on port $BackendPort" }

    try {
        $ids = (Invoke-RestMethod -Uri "http://127.0.0.1:$BackendPort/v1/models").data | ForEach-Object { $_.id }
        Write-Host "models : $($ids -join ', ')"
    }
    catch { }
}

try {
    Write-Host "adapter: http://127.0.0.1:$Port  ($pythonLabel)"
    Write-Host "e.g.   : python scripts\smoke_test.py"
    & $pythonExe @($pythonArgs + @('-m', 'ming30'))
}
finally {
    if ($backend -and -not $backend.HasExited) {
        Write-Host "engine : stopping"
        Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
    }
}
