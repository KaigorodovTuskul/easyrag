@echo off
setlocal

set ROOT=%~dp0
set PYTHON=%ROOT%tools\python-portable\python.exe

powershell -ExecutionPolicy Bypass -File "%ROOT%scripts\bootstrap_python.ps1"
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m ensurepip
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m pip install --upgrade pip
if errorlevel 1 exit /b %errorlevel%

"%PYTHON%" -m pip install -r "%ROOT%requirements.txt"
if errorlevel 1 exit /b %errorlevel%

echo Bootstrap completed.
