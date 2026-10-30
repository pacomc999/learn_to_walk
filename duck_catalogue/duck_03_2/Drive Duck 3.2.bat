@echo off
rem Double-click to drive Duck 3.2 (the runner) live, with a speed slider from 0.1 to 2.0 m/s and a Reset button
rem The project folder is two levels up from this duck's folder
cd /d "%~dp0..\.."
.venv\Scripts\python -m duck.drive duck_catalogue\duck_03_2\brain
if errorlevel 1 pause
