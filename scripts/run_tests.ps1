#!/usr/bin/env pwsh
param(
    [string]$PythonBin = $env:PYTHON_BIN,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Passthru
)

if (-not $PythonBin) {
    $PythonBin = "python"
}

& $PythonBin "$(Split-Path $PSScriptRoot -Parent)/scripts/run_tests.py" @Passthru
