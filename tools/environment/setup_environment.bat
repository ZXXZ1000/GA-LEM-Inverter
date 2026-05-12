@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PS_SCRIPT=%SCRIPT_DIR%setup_environment.ps1"
for %%I in ("%SCRIPT_DIR%..\..") do set "PROJECT_ROOT=%%~fI"

where powershell >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Windows PowerShell was not found.
    echo Install PowerShell or run setup_environment.sh from Git Bash.
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" -ProjectDir "%PROJECT_ROOT%" %*
set "SETUP_EXIT=%ERRORLEVEL%"
if not "%SETUP_EXIT%"=="0" (
    echo.
    echo [ERROR] Environment setup failed. See "%SCRIPT_DIR%setup_environment.log".
    echo Press any key to close this window.
    pause >nul
)
exit /b %SETUP_EXIT%
