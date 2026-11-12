@echo off
cd /d "%~dp0"
if defined MOBILEIE_PYTHON (
  "%MOBILEIE_PYTHON%" "%~dp0desktop.py"
) else (
  python "%~dp0desktop.py"
)
if errorlevel 1 pause
