@echo off
REM Start AI Server Only
REM This script starts the local AI runtime server

echo ========================================
echo Starting AI Runtime Server
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv venv
    echo Then: venv\Scripts\activate
    echo Then: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Start AI server
echo Starting server at http://127.0.0.1:8000
echo Press Ctrl+C to stop
echo.

python -m server.main

pause
