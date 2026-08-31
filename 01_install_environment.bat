@echo off
chcp 65001 >nul
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File ".\setup_environment.ps1"
echo.
echo Environment setup finished. Check the messages above for errors.
pause
