# Start the audio.cpp backend (model stays warm) and the ming30 adapter.
#
#   .\run.ps1                 # backend + adapter
#   .\run.ps1 -AdapterOnly    # adapter only, backend already running
#
# Ctrl+C stops both processes.

param(
    [string]$CudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin",
    [string]$Runtime = "..\audio.cpp-0.9.3-cuda12.4",
    [string]$Config = "config\dev.server.json",
    [int]$BackendPort = 8090,
    [int]$Port = 8070,
    [switch]$AdapterOnly
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$env:PATH = "$CudaBin;$env:PATH"
$env:MING30_BACKEND_URL = "http://127.0.0.1:$BackendPort"
$env:MING30_PORT = "$Port"

$backend = $null
if (-not $AdapterOnly) {
    $exe = Join-Path (Resolve-Path $Runtime) "audiocpp_server.exe"
    if (-not (Test-Path $exe)) { throw "audiocpp_server.exe not found at $exe" }
    Write-Host "starting backend: $exe --config $Config"
    $backend = Start-Process -FilePath $exe -ArgumentList @("--config", (Resolve-Path $Config), "--no-ui") -PassThru -NoNewWindow

    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$BackendPort/health" -TimeoutSec 2
            if ($health.status -eq "ok") { Write-Host "backend ready: audio.cpp $($health.version) ($($health.models) models)"; break }
        } catch { }
        if ($backend.HasExited) { throw "backend exited with code $($backend.ExitCode)" }
    }
}

try {
    Write-Host "starting adapter on http://127.0.0.1:$Port"
    python -m ming30
}
finally {
    if ($backend -and -not $backend.HasExited) {
        Write-Host "stopping backend"
        Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
    }
}
