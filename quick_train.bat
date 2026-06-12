@echo off
REM ============================================================================
REM Quick sanity training: stage 1 only, short iter count. Use to verify the
REM training pipeline works end-to-end without committing to the full 1600
REM iter run.
REM
REM Usage:
REM   quick_train.bat               -> regime=LME, iters=10
REM   quick_train.bat SDWE          -> regime SDWE, iters=10
REM   quick_train.bat CME 25        -> regime CME, iters=25
REM ============================================================================
setlocal

call "%~dp0_env.bat"

set "REGIME=%~1"
if "%REGIME%"=="" set "REGIME=LME"

set "ITERS=%~2"
if "%ITERS%"=="" set "ITERS=10"

echo Sanity training: regime=%REGIME%  iters=%ITERS%  seed=0
echo Started: %DATE% %TIME%
echo.

"%PYTHON%" -m marl_economy.train.train_rllib --stage 1 --regime %REGIME% --iterations %ITERS% --seed 0 --checkpoint "runs\sanity_%REGIME%_stage1"
if errorlevel 1 (
    echo Sanity training FAILED. Exit code: %errorlevel%
    pause
    exit /b %errorlevel%
)

echo.
echo Sanity training complete. Checkpoint: runs\sanity_%REGIME%_stage1\final
echo Finished: %DATE% %TIME%
exit /b 0
