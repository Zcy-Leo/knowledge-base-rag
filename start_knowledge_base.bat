@echo off
echo Starting Knowledge Base RAG System...
echo.

cd /d "%~dp0"

if exist bge_env\Scripts\activate.bat (
    echo Activating virtual environment...
    call bge_env\Scripts\activate.bat
) else if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo Starting Streamlit application...
streamlit run app_v2.py

pause