@echo off
REM Windows launcher for stroke thesis training.
REM Edit configs\train_stroke_thesis.env instead of this file.

chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

set SCRIPT_DIR=%~dp0
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%train_stroke_thesis.ps1"
set EXIT_CODE=%ERRORLEVEL%

if not "%EXIT_CODE%"=="0" (
  echo.
  echo [error] Training failed. Exit code=%EXIT_CODE%
  pause
)

exit /b %EXIT_CODE%
