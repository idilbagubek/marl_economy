@echo off
setlocal EnableDelayedExpansion

REM ============================================================================
REM Full curriculum training for ALL four regimes (LME, CME, SDWE, SME).
REM
REM For each regime: stage 1 (50 iters) -> 2 (50) -> 3 (100) -> 4 (200)
REM = 400 iters/regime x 4 regimes = 1600 iterations total.
REM Then evaluate the final checkpoint and regenerate plots.
REM
REM Fixed 2026-05-14:
REM   * Use the marltest conda env's Python (where Ray + RLlib live).
REM   * Use the correct regime ids from `marl_economy/config/constants.py`.
REM   * Print start / end timestamps per stage so progress is visible.
REM
REM Edit PYTHON below if your conda marltest env lives elsewhere.
REM ============================================================================

cd /d "%~dp0"

set "PYTHON=C:\miniconda\envs\marltest\python.exe"
if not exist "%PYTHON%" (
    echo WARNING: marltest python not found at %PYTHON%.
    echo Falling back to plain 'python' on PATH. Ray/RLlib may be missing.
    set "PYTHON=python"
)

echo PYTHON=%PYTHON%
echo Started: %DATE% %TIME%
echo.

for %%R in (LME CME SDWE SME) do (
    echo ============================================================
    echo Training regime %%R   (started %TIME%^)
    echo ============================================================

    echo [%%R] Stage 1 ^(50 iters^)...
    "%PYTHON%" -m marl_economy.train.train_rllib --stage 1 --regime %%R --iterations 50 --checkpoint "runs\%%R_stage1_reward_fixed"
    if errorlevel 1 goto :failed

    echo [%%R] Stage 2 ^(50 iters^)...
    "%PYTHON%" -m marl_economy.train.train_rllib --stage 2 --regime %%R --iterations 50 --load-from "runs\%%R_stage1_reward_fixed\final" --checkpoint "runs\%%R_stage2_reward_fixed"
    if errorlevel 1 goto :failed

    echo [%%R] Stage 3 ^(100 iters^)...
    "%PYTHON%" -m marl_economy.train.train_rllib --stage 3 --regime %%R --iterations 100 --load-from "runs\%%R_stage2_reward_fixed\final" --checkpoint "runs\%%R_stage3_reward_fixed"
    if errorlevel 1 goto :failed

    echo [%%R] Stage 4 ^(200 iters^)...
    "%PYTHON%" -m marl_economy.train.train_rllib --stage 4 --regime %%R --iterations 200 --load-from "runs\%%R_stage3_reward_fixed\final" --checkpoint "runs\%%R_stage4_reward_fixed"
    if errorlevel 1 goto :failed

    echo [%%R] Evaluating final checkpoint...
    "%PYTHON%" -m marl_economy.eval.evaluate_checkpoint --checkpoint "runs\%%R_stage4_reward_fixed\final" --regime %%R --output "runs\%%R_stage4_reward_fixed_eval"
    if errorlevel 1 goto :failed

    echo [%%R] Generating plots from eval...
    "%PYTHON%" -m marl_economy.eval.plot_run "runs\%%R_stage4_reward_fixed_eval"
    if errorlevel 1 goto :failed

    echo [%%R] Done.   ^(finished %TIME%^)
    echo.
)

echo ============================================================
echo All four regimes (LME, CME, SDWE, SME) completed successfully.
echo Finished: %DATE% %TIME%
echo ============================================================
exit /b 0

:failed
echo ============================================================
echo Training pipeline FAILED. Inspect the error above.
echo Exit code: %errorlevel%
echo Failed at: %DATE% %TIME%
echo ============================================================
pause
exit /b %errorlevel%
