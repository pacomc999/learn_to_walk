@echo off
rem Double-click to drive the newest duck brain live, with a speed slider and a Reset button
cd /d "%~dp0"
.venv\Scripts\python -m duck.drive
if errorlevel 1 pause
