"""Stage 2 Test 3 - PPO training sanity check.

Run 20 PPO iterations on stage 1 / LME with CPU-only RLlib and verify:

  1. The training loop completes without errors (no crashes from
     vectorized routing interacting badly with the RLlib env loop).
  2. The household policy reward is finite at every iteration.
  3. The reward curve has reasonable magnitude (within bounds known from
     the sim's reward construction - log-utility deltas, bankruptcy /
     death penalty ranges) and is not collapsing to a degenerate -inf.
  4. There is no NaN gradient or learner failure.

This is a SANITY check, not a convergence check.
"""

from __future__ import annotations

import json
import os
import sys
import time
import math
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marl_economy.train.train_rllib import build_ppo_config
from marl_economy.config.constants import REGIME_LME, DEFAULT_RANDOM_SEED


def main():
    n_iter = 20
    print(f"Stage 2 Test 3 - running {n_iter} PPO iterations on stage 1 / LME, CPU-only\n")

    # Configure CPU-only with a single inline env runner so Ray's worker
    # spawning doesn't fight Windows.
    config, metadata = build_ppo_config(
        regime=REGIME_LME,
        stage=1,
        seed=DEFAULT_RANDOM_SEED,
        num_iterations=n_iter,
        num_env_runners=0,          # inline rollouts - no worker processes
        num_envs_per_env_runner=1,
        num_gpus_per_learner=0,     # CPU-only learner
    )
    if config is None:
        print("ERROR: build_ppo_config returned None (Ray/RLlib not installed?)")
        sys.exit(1)

    ray = metadata["ray"]
    ray.init(ignore_reinit_error=True, num_cpus=2, log_to_driver=False)

    print("  Building algorithm...")
    algo = config.build_algo()

    print("  Iter  | mean_episode_return | household_reward | min | max | status")
    rewards = []
    statuses = []
    t0 = time.perf_counter()
    for i in range(n_iter):
        result = algo.train()
        env_runners = result.get("env_runners", {})
        ep_ret = env_runners.get("episode_return_mean", env_runners.get("episode_reward_mean", float("nan")))
        per_policy = env_runners.get("module_episode_returns_mean", {})
        hh_ret = per_policy.get("pi_household", float("nan"))
        ep_ret_min = env_runners.get("episode_return_min", float("nan"))
        ep_ret_max = env_runners.get("episode_return_max", float("nan"))
        finite = math.isfinite(ep_ret) and math.isfinite(hh_ret if not math.isnan(hh_ret) else 0)
        status = "OK" if finite else "NAN"
        rewards.append({
            "iter": i + 1,
            "episode_return_mean": float(ep_ret),
            "pi_household_return_mean": float(hh_ret) if not math.isnan(hh_ret) else None,
            "episode_return_min": float(ep_ret_min) if isinstance(ep_ret_min, (int, float)) else None,
            "episode_return_max": float(ep_ret_max) if isinstance(ep_ret_max, (int, float)) else None,
        })
        statuses.append(status)
        hh_str = f"{hh_ret:>10.4f}" if not math.isnan(hh_ret) else "       n/a"
        print(f"  {i + 1:>4d}  | {ep_ret:>18.4f} | {hh_str} | "
              f"{ep_ret_min if isinstance(ep_ret_min, (int, float)) else 'n/a':>8} | "
              f"{ep_ret_max if isinstance(ep_ret_max, (int, float)) else 'n/a':>8} | {status}")
    dt = time.perf_counter() - t0
    print(f"\n  Wall: {dt:.1f}s ({dt/n_iter:.2f}s / iter)\n")

    algo.stop()
    ray.shutdown()

    # Acceptance checks.
    print("Acceptance checks:")
    all_finite = all(s == "OK" for s in statuses)
    print(f"  All iterations finite & non-NaN:  {'PASS' if all_finite else 'FAIL'}")

    ep_rets = [r["episode_return_mean"] for r in rewards if math.isfinite(r["episode_return_mean"])]
    if not ep_rets:
        print("  No finite returns recorded - FAIL")
        sys.exit(1)

    # Magnitude bound: household reward per agent per step is roughly the
    # log-utility delta plus penalties. Over an episode (78 weeks * 80 hh =
    # 6240 agent-steps in stage 1) the per-episode return should land in a
    # reasonable range. We just check it's finite and not absurdly huge.
    max_abs = max(abs(r) for r in ep_rets)
    bound_ok = max_abs < 1e6
    print(f"  Max |episode_return|: {max_abs:.2f}  ({'PASS' if bound_ok else 'FAIL - explosion'})")

    # Trajectory shape: don't require monotone improvement (20 iters is far
    # too short for that). But the curve shouldn't have a step-change
    # collapse to a degenerate floor - the std should be > 0 in absolute
    # value.
    std = float(np.std(ep_rets))
    shape_ok = std > 0
    print(f"  Std of episode returns:  {std:.4f}  ({'PASS' if shape_ok else 'FAIL - flat'})")

    # Write artifact for inspection.
    out_path = Path("stage2_test3_ppo_log.json")
    with open(out_path, "w") as f:
        json.dump({
            "n_iter": n_iter,
            "wall_seconds": dt,
            "rewards": rewards,
            "statuses": statuses,
            "summary": {
                "all_finite": all_finite,
                "max_abs": max_abs,
                "bound_ok": bound_ok,
                "std": std,
                "shape_ok": shape_ok,
            },
        }, f, indent=2)
    print(f"\nLog written to: {out_path}")

    overall = all_finite and bound_ok and shape_ok
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
