@echo off
chcp 65001 >nul
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File ".\run_open_world_gui.ps1"
if errorlevel 1 pause
