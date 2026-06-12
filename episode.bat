@echo off
REM ============================================================================
REM Run a single scripted episode (random actions) and write a tidy run dir
REM with metrics_step.csv + summary.json + plots + README.md.
REM
REM Usage:
REM   episode.bat                   -> regime=LME, weeks=156, seed=0
REM   episode.bat SDWE              -> regime SDWE
REM   episode.bat CME 80 7          -> regime CME, 80 weeks, seed 7
REM
REM Output dir naming: runs\episode_<REGIME>_seed<SEED>_<timestamp>\
REM ============================================================================
setlocal

call "%~dp0_env.bat"

set "REGIME=%~1"
if "%REGIME%"=="" set "REGIME=LME"

set "WEEKS=%~2"
if "%WEEKS%"=="" set "WEEKS=156"

set "SEED=%~3"
if "%SEED%"=="" set "SEED=0"

echo Running episode: regime=%REGIME%  weeks=%WEEKS%  seed=%SEED%
echo Started: %DATE% %TIME%
echo.

"%PYTHON%" -m marl_economy.scripts.run_one_episode --regime %REGIME% --weeks %WEEKS% --seed %SEED%
if errorlevel 1 (
    echo Episode FAILED. Exit code: %errorlevel%
    pause
    exit /b %errorlevel%
)

echo.
echo Episode complete. Finished: %DATE% %TIME%
echo ^(Look in runs\episode_%REGIME%_seed%SEED%_<timestamp>\^)
exit /b 0
