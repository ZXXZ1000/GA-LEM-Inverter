@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PS_SCRIPT=%SCRIPT_DIR%setup_environment.ps1"

where powershell >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Windows PowerShell was not found.
    echo Install PowerShell or run setup_environment.sh from Git Bash.
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" %*
exit /b %ERRORLEVEL%
