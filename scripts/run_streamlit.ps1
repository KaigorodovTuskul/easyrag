$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root "tools\python-portable\python.exe"

if (-not (Test-Path $python)) {
    throw "Portable Python not found at $python"
}

& $python -m streamlit run "$root\app\ui\streamlit_app.py"
