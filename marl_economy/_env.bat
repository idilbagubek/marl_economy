@echo off
REM Shared env setup for all batch files. Sets PYTHON and changes to repo root.
REM Sourced via `call _env.bat` from the other .bat scripts.

cd /d "%~dp0"

set "PYTHON=C:\miniconda\envs\marltest\python.exe"
if not exist "%PYTHON%" (
    echo WARNING: marltest python not found at %PYTHON%.
    echo Falling back to plain 'python' on PATH. Ray/RLlib may be missing.
    set "PYTHON=python"
)
