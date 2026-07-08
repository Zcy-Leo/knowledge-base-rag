#!/bin/bash
echo "Starting Knowledge Base RAG System..."
echo ""

cd "$(dirname "$0")"

if [ -f bge_env/bin/activate ]; then
    echo "Activating virtual environment..."
    source bge_env/bin/activate
elif [ -f venv/bin/activate ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

echo "Starting Streamlit application..."
streamlit run app_v2.py