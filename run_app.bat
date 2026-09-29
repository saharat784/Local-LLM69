@echo off
cd /d "%~dp0"
title Ollama RAG Web Studio
echo ===================================================
echo   Starting Ollama RAG Studio (Streamlit Web UI)
echo ===================================================
echo.
uv run python -m streamlit run app.py
pause
