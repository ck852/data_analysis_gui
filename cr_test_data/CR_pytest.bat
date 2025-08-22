@echo off
REM CR_pytest.bat - Windows batch file to run Concentration Response Dialog tests
REM
REM Usage:
REM   CR_pytest           - Run tests with default settings
REM   CR_pytest -v        - Run tests with verbose output
REM   CR_pytest -v -s     - Run tests with verbose output and print statements
REM   CR_pytest --help    - Show pytest help

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0

REM Run the Python test runner script with all passed arguments
python "%SCRIPT_DIR%CR_pytest.py" %*

REM Check if Python command was successful
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error: Failed to run CR_pytest.py
    echo Please ensure Python is installed and in your PATH
    echo You can also try running directly: python CR_pytest.py
    pause
    exit /b %ERRORLEVEL%
)