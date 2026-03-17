@echo off
python test_simple.py > test_result.txt 2>&1
type test_result.txt
