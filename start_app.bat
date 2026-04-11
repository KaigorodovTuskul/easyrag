@echo off
setlocal

set ROOT=%~dp0
set PYTHON=%ROOT%tools\python-portable\python.exe

if not exist "%PYTHON%" (
    echo Portable Python not found. Run bootstrap_env.bat first.
    exit /b 1
)

"%PYTHON%" -m streamlit run "%ROOT%app\ui\streamlit_app.py" 
