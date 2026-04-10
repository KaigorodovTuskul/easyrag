$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$toolsDir = Join-Path $root "tools"
$pythonDir = Join-Path $toolsDir "python-portable"
$zipPath = Join-Path $toolsDir "python-portable.zip"

New-Item -ItemType Directory -Force -Path $toolsDir | Out-Null
New-Item -ItemType Directory -Force -Path $pythonDir | Out-Null

$url = "https://www.python.org/ftp/python/3.13.3/python-3.13.3-embed-amd64.zip"

Write-Host "Downloading portable Python from $url"
Invoke-WebRequest -Uri $url -OutFile $zipPath

Write-Host "Extracting into $pythonDir"
Expand-Archive -Path $zipPath -DestinationPath $pythonDir -Force

Write-Host "Portable Python ready at $pythonDir"
