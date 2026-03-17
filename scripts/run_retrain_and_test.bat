@echo off
chcp 65001 >nul
echo ======================================================================
echo   Retraining model with enhanced features...
echo ======================================================================

python train_improved.py > retrain_log.txt 2>&1

echo.
echo Training complete. Checking results...
echo.

python check_model.py > model_check_log.txt 2>&1

if exist model_check_result.txt (
    type model_check_result.txt
) else (
    echo ERROR: model_check_result.txt not created
    type model_check_log.txt
)

echo.
echo ======================================================================
echo   Testing colloquial expressions...
echo ======================================================================

python test_model_final.py > test_log.txt 2>&1
type test_log.txt

pause
