@echo off
echo Activating conda environment...
call conda activate NewVisionModel

echo.
echo Python version:
python --version

echo.
echo Current directory:
cd

echo.
echo Retraining model (this may take 1-2 minutes)...
python train_improved.py

pause
