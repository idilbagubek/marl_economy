@echo off
REM ============================================================================
REM Evaluate a trained RLlib checkpoint and produce metrics + snapshots + plots
REM in an evaluation directory. After this, run unity.bat on the output dir to
REM build the Unity playback JSON.
REM
REM Usage:
REM   evaluate.bat <CHECKPOINT_PATH> <REGIME>
REM   evaluate.bat <CHECKPOINT_PATH> <REGIME> <STAGE>
REM
REM Example:
REM   evaluate.bat runs\LME_stage4_seed0\final LME 4
REM ============================================================================
setlocal

call "%~dp0_env.bat"

set "CHECKPOINT=%~1"
set "REGIME=%~2"
set "STAGE=%~3"

if "%CHECKPOINT%"=="" (
    echo Usage: evaluate.bat ^<checkpoint_path^> ^<regime^> [stage]
    echo.
    echo Example:
    echo   evaluate.bat runs\LME_stage4_seed0\final LME 4
    pause
    exit /b 1
)
if "%REGIME%"=="" (
    echo Regime required ^(LME, CME, SDWE, or SME^).
    pause
    exit /b 1
)

if "%STAGE%"=="" (
    echo Evaluating %CHECKPOINT% on regime %REGIME%...
    "%PYTHON%" -m marl_economy.eval.evaluate_checkpoint --checkpoint "%CHECKPOINT%" --regime %REGIME%
) else (
    echo Evaluating %CHECKPOINT% on regime %REGIME% ^(stage %STAGE%^)...
    "%PYTHON%" -m marl_economy.eval.evaluate_checkpoint --checkpoint "%CHECKPOINT%" --regime %REGIME% --stage %STAGE%
)
if errorlevel 1 (
    echo Evaluation FAILED. Exit code: %errorlevel%
    pause
    exit /b %errorlevel%
)
exit /b 0
