@echo off
cd /d "%~dp0"
echo === Visual Check setup (only needed once) ===
set PY=
where py >nul 2>nul && set PY=py -3
if "%PY%"=="" ( where python >nul 2>nul && set PY=python )
if "%PY%"=="" (
  echo Python not found. Install Python from https://www.python.org/downloads/
  echo and tick "Add python.exe to PATH" during install. Then run setup.bat again.
  pause
  exit /b 1
)
%PY% -m venv .venv
if errorlevel 1 ( echo Could not make the Python environment. & pause & exit /b 1 )
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 ( echo Install failed. Check the internet connection and try again. & pause & exit /b 1 )
mkdir masters 2>nul
mkdir photos 2>nul
echo.
echo Setup finished. Now double-click run.bat
pause
