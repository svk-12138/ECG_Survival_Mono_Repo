@echo off
REM 卒中论文训练 Windows 启动入口。
REM 请修改 configs\train_stroke_thesis.env，不要修改本文件。

chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

set SCRIPT_DIR=%~dp0
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%train_stroke_thesis.ps1"
set EXIT_CODE=%ERRORLEVEL%

if not "%EXIT_CODE%"=="0" (
  echo.
  echo [error] 训练启动失败，退出码=%EXIT_CODE%
  echo [hint] 请先确认是否已经创建 configs\train_stroke_thesis.env，
  echo [hint] 并且只修改 .env 文件，不要再改 .ps1 或 .bat。
  pause
)

exit /b %EXIT_CODE%
