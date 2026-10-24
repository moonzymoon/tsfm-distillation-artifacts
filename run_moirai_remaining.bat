@echo off
chcp 65001 >nul 2>&1
cd /d "D:\0科研\工作1\第20篇SCI_A3TSFM蒸馏\code"
echo [%%time%%] starting Moirai scoring + training pipeline
"D:\0科研\工作1\第20篇SCI_A3TSFM蒸馏\moirai_venv\Scripts\python.exe" -u _moirai_remaining.py >> "D:\0科研\工作1\第20篇SCI_A3TSFM蒸馏\results\moirai_remaining_run.log" 2>&1
echo [%%time%%] MOIRAI_PIPELINE_COMPLETE
pause