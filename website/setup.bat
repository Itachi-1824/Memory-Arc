@echo off
echo ============================================
echo Memory-Arc Web Chat Setup
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo Step 1: Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo.
echo Step 2: Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Step 3: Installing core dependencies...
cd ..
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install core dependencies
    pause
    exit /b 1
)

echo.
echo Step 4: Installing web server dependencies...
cd website
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install web dependencies
    pause
    exit /b 1
)

echo.
echo ============================================
echo Setup Complete!
echo ============================================
echo.
echo To start the server:
echo   1. Run: venv\Scripts\activate.bat
echo   2. Run: python server.py
echo   3. Open index.html in your browser
echo.
echo API Configuration:
echo   Base URL: https://text.pollinations.ai/
echo   Model: gemini
echo   You'll be prompted for API key when you open the website
echo.
pause
