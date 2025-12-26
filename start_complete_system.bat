@echo off
REM Start Complete System: AI Server + Trading + GUI
REM This is the full-featured launch

echo ========================================
echo Starting Complete Trading System
echo  - AI Server
echo  - Trading Bot
echo  - GUI Interface
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

REM Check configurations
if not exist "config\config.yaml" (
    echo ERROR: config\config.yaml not found!
    pause
    exit /b 1
)

if not exist "config\ai-coder.yaml" (
    echo WARNING: config\ai-coder.yaml not found!
    echo AI features will be disabled
    timeout /t 3
)

echo Starting complete system...
echo AI Server will start first, then GUI
echo.

REM Start with AI server and paper trading
python launcher.py --with-ai --paper

pause
