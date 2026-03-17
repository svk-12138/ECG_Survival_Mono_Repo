#!/usr/bin/env pwsh
# 单阶段入口：-Stages 支持单个或逗号分隔多个阶段
param(
    [string]$Stages,
    [string]$Config = "$(Split-Path $PSScriptRoot -Parent)/configs/pipeline.default.yaml",
    [string]$PythonBin = $env:PYTHON_BIN
)

if (-not $Stages) {
    Write-Host "Usage: ./scripts/run_stage.ps1 -Stages vae,linear [-Config <path>] [-PythonBin <python>]"
    exit 1
}

if (-not $PythonBin) {
    $PythonBin = "python"
}

$root = Split-Path $PSScriptRoot -Parent
Write-Host "[stage] Using config: $Config"
Write-Host "[stage] Running stages: $Stages"
& $PythonBin "$root/scripts/run_pipeline.py" --config $Config --stages $Stages
