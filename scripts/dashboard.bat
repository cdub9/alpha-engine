@echo off
REM Launch the AlphaEngine local dashboard.
REM Opens http://localhost:8501 in your default browser.

setlocal
cd /d "C:\Users\Coleman\Projects\AlphaEngine"
call .venv\Scripts\activate.bat
streamlit run dashboard\app.py
