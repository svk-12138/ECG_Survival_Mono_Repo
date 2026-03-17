@echo off
echo Retraining model...
python -u train_improved.py > retrain_log.txt 2>&1
echo Done. Checking log size:
dir retrain_log.txt | find "retrain"

echo.
echo Checking model...
python -u check_model.py
if exist model_check_result.txt (type model_check_result.txt) else (echo No result file)

echo.
echo Testing...
python -u test_model_final.py
pause
