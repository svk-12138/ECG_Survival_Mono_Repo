@echo off
echo ======================================================================
echo   RETRAINING MODEL WITH ENHANCED FEATURES
echo ======================================================================
echo.
echo This will retrain the model using:
echo - Expanded keywords (+42 colloquial terms)
echo - Symptom phrase matching (x3 weight)
echo.
echo Estimated time: 1-2 minutes
echo.

python train_improved.py

if errorlevel 1 (
    echo.
    echo ERROR: Training failed
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo   TESTING COLLOQUIAL EXPRESSIONS
echo ======================================================================
echo.

python check_model.py

echo.
if exist model_check_result.txt (
    type model_check_result.txt
    del model_check_result.txt
) else (
    echo WARNING: Could not generate model check results
)

echo.
pause
