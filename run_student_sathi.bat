@echo off
cd /d "%~dp0"
echo ========================================
echo        STUDENT SATHI - STARTING APP
echo ========================================
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.11+ and try again.
  pause
  exit /b 1
)
if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
pause
