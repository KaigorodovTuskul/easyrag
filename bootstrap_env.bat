@echo off
setlocal

set ROOT=%~dp0
set PYTHON=%ROOT%tools\python-portable\python.exe
set GETPIP=%TEMP%\easyrag-get-pip.py

powershell -ExecutionPolicy Bypass -File "%ROOT%scripts\bootstrap_python.ps1"
if errorlevel 1 exit /b %errorlevel%

powershell -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%GETPIP%'"
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" "%GETPIP%"
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m pip install --upgrade pip
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m pip install -r "%ROOT%requirements.txt"
if errorlevel 1 exit /b %errorlevel%

if exist "%GETPIP%" del "%GETPIP%"

echo Bootstrap completed.
