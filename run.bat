@echo off
echo ============================================
echo  Casting Yard - Mould Progress Tracker
echo ============================================
echo.

cd /d %~dp0
call venv\Scripts\activate

echo Starting dashboard...
echo Open http://localhost:8501 in your browser
echo.
streamlit run dashboard.py
