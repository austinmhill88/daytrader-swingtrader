@echo off
REM GUI Only - Connect to Running System
REM Use this if trading system is already running

echo ========================================
echo Starting GUI Only
echo ========================================
echo.
echo This will start the GUI interface only.
echo Make sure the trading system is already running!
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

echo Starting GUI...
echo.

python launcher.py --gui-only

pause
