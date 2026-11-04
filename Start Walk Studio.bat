@echo off
rem Walk Studio for Windows: double-click to start. The first start sets everything up.
setlocal
cd /d "%~dp0"
title Walk Studio

if exist ".venv\Scripts\python.exe" goto run

echo.
echo  Setting up Walk Studio for the first time.
echo  This downloads the physics and learning libraries (about 1 GB) and takes a few minutes.
echo.
set "PYTHON="
where py >nul 2>nul && (py -3.12 -c "" >nul 2>nul && set "PYTHON=py -3.12")
if not defined PYTHON (where py >nul 2>nul && (py -3 -c "" >nul 2>nul && set "PYTHON=py -3"))
if not defined PYTHON (where python >nul 2>nul && set "PYTHON=python")
if not defined PYTHON (
  echo  Python was not found. Install Python 3.12 from https://www.python.org/downloads/
  echo  and tick "Add python.exe to PATH" during the installation. Then start this file again.
  pause
  exit /b 1
)
%PYTHON% -m venv .venv || goto failed
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto failed
".venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cpu || goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto failed
echo.
echo  All set up.

:run
echo.
echo  Starting Walk Studio. Your browser opens in a moment.
echo  Keep this window open while you use the studio; close it to stop.
echo.
".venv\Scripts\python.exe" -m studio
pause
exit /b 0

:failed
echo.
echo  The setup did not finish. Check your internet connection and start this file again.
echo  (If it keeps failing, delete the .venv folder next to this file first.)
rmdir /s /q .venv 2>nul
pause
exit /b 1
