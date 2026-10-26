@echo off
rem Launcher for run.ps1 that works under the default PowerShell execution policy.
rem Usage: run.cmd [-AdapterOnly] [-Port 8070] [-BackendPort 8090] ...
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
exit /b %ERRORLEVEL%
