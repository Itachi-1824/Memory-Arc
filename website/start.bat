@echo off
echo ============================================
echo Memory-Arc Web Chat
echo ============================================
echo.

REM Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found!
    echo Please run setup.bat first.
    pause
    exit /b 1
)

REM Activate venv
call venv\Scripts\activate.bat

REM Start server
echo Starting server...
echo.
python server.py

pause
