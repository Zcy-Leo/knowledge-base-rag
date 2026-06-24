@echo off
title Knowledge Base System

cd /d "%~dp0"

echo Starting Knowledge Base System...

start "Streamlit Server" cmd /k ""%~dp0bge_env\Scripts\python.exe" -m streamlit run app_v2.py --server.port 8501"

timeout /t 10 /nobreak > nul

start http://localhost:8501

echo Server started. UI is ready at: http://localhost:8501
echo If the page shows "can't reach", wait 5 seconds and refresh.

pause > nul