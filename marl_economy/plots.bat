@echo off
REM ============================================================================
REM Regenerate plots from a run directory's metrics_step.csv.
REM
REM Usage:
REM   plots.bat <RUN_DIR>
REM
REM Example:
REM   plots.bat runs\LME_stage4_seed0
REM   plots.bat "runs\episode_LME_seed0_20260514_153022"
REM ============================================================================
setlocal

call "%~dp0_env.bat"

set "RUN_DIR=%~1"
if "%RUN_DIR%"=="" (
    echo Usage: plots.bat ^<run_dir^>
    echo.
    echo Available runs:
    if exist runs dir /b runs
    pause
    exit /b 1
)

if not exist "%RUN_DIR%\metrics_step.csv" (
    echo ERROR: %RUN_DIR%\metrics_step.csv not found.
    echo Did you point at the right directory?
    pause
    exit /b 1
)

echo Regenerating plots from %RUN_DIR%...
"%PYTHON%" -m marl_economy.eval.plot_run "%RUN_DIR%"
if errorlevel 1 (
    echo Plot generation FAILED. Exit code: %errorlevel%
    pause
    exit /b %errorlevel%
)

echo.
echo Plots saved to %RUN_DIR%\plots\
exit /b 0
