@echo off
chcp 65001 >nul
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File ".\self_check.ps1"
echo.
echo Self-check finished. A successful run shows status: passed.
pause
