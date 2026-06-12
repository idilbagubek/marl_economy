@echo off
REM ============================================================================
REM Export a Unity playback JSON from an evaluation run directory.
REM Requires snapshots.json in the run directory (produced by
REM evaluate_checkpoint with snapshots enabled).
REM
REM Usage:
REM   unity.bat <RUN_DIR>
REM   unity.bat <RUN_DIR> <OUTPUT_PATH>
REM
REM Example:
REM   unity.bat runs\LME_stage4_seed0_eval
REM ============================================================================
setlocal

call "%~dp0_env.bat"

set "RUN_DIR=%~1"
set "OUTPUT=%~2"

if "%RUN_DIR%"=="" (
    echo Usage: unity.bat ^<run_dir^> [output_path]
    echo.
    echo Available eval-style runs ^(those that should contain snapshots.json^):
    if exist runs dir /b runs ^| findstr /i eval
    pause
    exit /b 1
)

if not exist "%RUN_DIR%\snapshots.json" (
    echo ERROR: %RUN_DIR%\snapshots.json not found.
    echo Generate it first by running:
    echo   evaluate.bat ^<checkpoint^> ^<regime^>
    pause
    exit /b 1
)

echo Exporting Unity playback from %RUN_DIR%...
if "%OUTPUT%"=="" (
    "%PYTHON%" -m marl_economy.viz.export_unity "%RUN_DIR%"
) else (
    "%PYTHON%" -m marl_economy.viz.export_unity "%RUN_DIR%" --output "%OUTPUT%"
)
if errorlevel 1 (
    echo Unity export FAILED. Exit code: %errorlevel%
    pause
    exit /b %errorlevel%
)
exit /b 0
