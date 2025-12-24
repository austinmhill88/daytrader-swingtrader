@echo off
REM Start Trading System with GUI
REM This script starts the trading bot with graphical interface

echo ========================================
echo Starting Trading System with GUI
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

REM Check if config exists
if not exist "config\config.yaml" (
    echo ERROR: Configuration file not found!
    echo Please create config\config.yaml
    pause
    exit /b 1
)

echo Starting trading system...
echo GUI will open shortly
echo.

REM Start with paper trading mode by default
python launcher.py --paper

pause
