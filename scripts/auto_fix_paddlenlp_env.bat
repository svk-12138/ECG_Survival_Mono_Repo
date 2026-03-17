@echo off
REM 自动修复 PaddleNLP 环境脚本

REM 1. 升级 paddlepaddle 和 paddlenlp
pip install --upgrade paddlepaddle paddlenlp

REM 2. 删除 paddlenlp 缓存目录
set "PNLP_CACHE=%USERPROFILE%\.paddlenlp"
if exist "%PNLP_CACHE%" (
    echo 正在删除 PaddleNLP 缓存目录：%PNLP_CACHE%
    rmdir /s /q "%PNLP_CACHE%"
) else (
    echo 未找到 PaddleNLP 缓存目录：%PNLP_CACHE%
)

REM 3. 提示用户重启 Python 环境

echo.
echo 操作完成！请重启你的 Python 解释器或 Jupyter/IDE，然后重新运行 gliner-x-large.py。
pause 