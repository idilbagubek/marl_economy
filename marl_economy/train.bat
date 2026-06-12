@echo off
REM ============================================================================
REM Train ONE regime through all 4 curriculum stages.
REM
REM Usage:
REM   train.bat                  -> defaults: regime=LME, seed=0
REM   train.bat CME              -> regime CME, seed 0
REM   train.bat SDWE 42          -> regime SDWE, seed 42
REM
REM Per-stage iteration counts match run_full_training_fixed.cmd:
REM   stage 1 = 50,  stage 2 = 50,  stage 3 = 100,  stage 4 = 200 iters.
REM ============================================================================
setlocal EnableDelayedExpansion

call "%~dp0_env.bat"

set "REGIME=%~1"
if "%REGIME%"=="" set "REGIME=LME"

set "SEED=%~2"
if "%SEED%"=="" set "SEED=0"

echo PYTHON=%PYTHON%
echo Regime=%REGIME%  Seed=%SEED%
echo Started: %DATE% %TIME%
echo.

echo [%REGIME%] Stage 1 ^(50 iters^)...
"%PYTHON%" -m marl_economy.train.train_rllib --stage 1 --regime %REGIME% --iterations 50 --seed %SEED% --checkpoint "runs\%REGIME%_stage1_seed%SEED%"
if errorlevel 1 goto :failed

echo [%REGIME%] Stage 2 ^(50 iters^)...
"%PYTHON%" -m marl_economy.train.train_rllib --stage 2 --regime %REGIME% --iterations 50 --seed %SEED% --load-from "runs\%REGIME%_stage1_seed%SEED%\final" --checkpoint "runs\%REGIME%_stage2_seed%SEED%"
if errorlevel 1 goto :failed

echo [%REGIME%] Stage 3 ^(100 iters^)...
"%PYTHON%" -m marl_economy.train.train_rllib --stage 3 --regime %REGIME% --iterations 100 --seed %SEED% --load-from "runs\%REGIME%_stage2_seed%SEED%\final" --checkpoint "runs\%REGIME%_stage3_seed%SEED%"
if errorlevel 1 goto :failed

echo [%REGIME%] Stage 4 ^(200 iters^)...
"%PYTHON%" -m marl_economy.train.train_rllib --stage 4 --regime %REGIME% --iterations 200 --seed %SEED% --load-from "runs\%REGIME%_stage3_seed%SEED%\final" --checkpoint "runs\%REGIME%_stage4_seed%SEED%"
if errorlevel 1 goto :failed

echo.
echo ============================================================
echo Training complete for %REGIME% seed %SEED%.
echo Final checkpoint: runs\%REGIME%_stage4_seed%SEED%\final
echo Finished: %DATE% %TIME%
echo ============================================================
exit /b 0

:failed
echo ============================================================
echo Training FAILED. Exit code: %errorlevel%
echo ============================================================
pause
exit /b %errorlevel%
