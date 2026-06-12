@echo off
REM ============================================================================
REM Run all four Stage 3 validations + the legacy single-world invariants
REM test. Exits non-zero on any failure. Total wall time: ~30s.
REM
REM Use this:
REM   * before launching long training to confirm the codebase is healthy
REM   * after pulling new commits
REM   * after editing anything in marl_economy/env/
REM ============================================================================
setlocal

call "%~dp0_env.bat"

set FAIL=0

echo ============================================================
echo [1/5] K=1 bit-equivalence ^(BatchedWorld vs single World^)
echo ============================================================
"%PYTHON%" scripts\stage3_test1_k1_equivalence.py || set FAIL=1

echo.
echo ============================================================
echo [2/5] K=8 smoke test ^(all four regimes^)
echo ============================================================
"%PYTHON%" scripts\stage3_test3_smoke.py || set FAIL=1

echo.
echo ============================================================
echo [3/5] K=16 distributional equivalence ^(KS tests^)
echo ============================================================
"%PYTHON%" scripts\stage3_test2_distribution.py || set FAIL=1

echo.
echo ============================================================
echo [4/5] Stage 1 invariants suite ^(pytest^)
echo ============================================================
"%PYTHON%" -c "import pytest" 2>nul
if errorlevel 1 (
    echo pytest not in marltest env, falling back to base python on PATH...
    python -m pytest tests/test_invariants.py -q || set FAIL=1
) else (
    "%PYTHON%" -m pytest tests/test_invariants.py -q || set FAIL=1
)

echo.
echo ============================================================
echo [5/5] Benchmark ^(informational^)
echo ============================================================
"%PYTHON%" scripts\benchmark_stage3.py

echo.
if "%FAIL%"=="0" (
    echo ============================================================
    echo ALL VALIDATIONS PASSED.
    echo ============================================================
    exit /b 0
) else (
    echo ============================================================
    echo ONE OR MORE VALIDATIONS FAILED. See output above.
    echo ============================================================
    pause
    exit /b 1
)
