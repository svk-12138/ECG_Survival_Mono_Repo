#!/usr/bin/env pwsh
# Windows 用户入口。可以通过 -PythonBin 指定 conda/python.exe。
param(
    [string]$Config = "$(Split-Path $PSScriptRoot -Parent)/configs/pipeline.default.yaml",
    [string]$PythonBin = $env:PYTHON_BIN
)

if (-not $PythonBin) {
    $PythonBin = "python"
}

Write-Host "[pipeline] Using config: $Config"
& $PythonBin "$(Split-Path $PSScriptRoot -Parent)/scripts/run_pipeline.py" --config $Config
