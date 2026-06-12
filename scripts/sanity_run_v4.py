"""Sanity v4 - Windows GPU training via manual module relocation.

NCCL isn't built into Windows torch wheels, and Ray Train hardcodes NCCL when
torch.cuda is available, so `num_learners=1` is unusable on Windows.

This script uses `num_learners=0` (no distributed, no process group) and then
manually moves each policy module to CUDA after build. Optimizers are
recreated so their state tensors live on GPU too.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from marl_economy.train.train_rllib import build_ppo_config
from marl_economy.config.constants import (
    REGIME_LME, DEFAULT_RANDOM_SEED,
    RLLIB_TRAIN_NUM_CPUS, RLLIB_TRAIN_NUM_ENV_RUNNERS,
    RLLIB_NUM_ENVS_PER_ENV_RUNNER,
)


def force_module_to_cuda(algo, module_ids):
    """Move each RLModule in `module_ids` to CUDA and rebuild its optimizer."""
    learner_group = algo.learner_group
    # With num_learners=0, the LearnerGroup has a single local Learner.
    learner = getattr(learner_group, "_learner", None)
    if learner is None:
        # API path varies across Ray versions.
        learners = getattr(learner_group, "_learners", None)
        if learners:
            learner = learners[0]
    if learner is None:
        print("  ERROR: could not locate the local Learner inside LearnerGroup")
        return False

    print(f"  Learner type: {type(learner).__name__}")
    module = learner.module  # MultiAgentRLModule
    moved = 0
    for mid in module_ids:
        try:
            sub_module = module[mid]
            sub_module.to("cuda")
            moved += 1
            for name, p in sub_module.named_parameters():
                print(f"    {mid}: param '{name}' now on {p.device}")
                break
        except (KeyError, AttributeError) as e:
            print(f"    skipped {mid}: {e}")

    # Rebuild the optimizers so their state lives on GPU too.
    if hasattr(learner, "_make_module_optimizer"):
        for mid in module_ids:
            try:
                learner._make_module_optimizer(mid, module[mid])
            except Exception as e:
                print(f"    optimizer rebuild skipped for {mid}: {e}")
    elif hasattr(learner, "configure_optimizers_for_module"):
        for mid in module_ids:
            try:
                learner.configure_optimizers_for_module(mid, learner.config)
            except Exception as e:
                print(f"    optimizer rebuild skipped for {mid}: {e}")
    return moved > 0


def main():
    print(f"=== Sanity v4 (num_learners=0 + manual .to('cuda')) ===")
    print(f"torch={torch.__version__}, cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device={torch.cuda.get_device_name(0)}")
    print()

    config, metadata = build_ppo_config(
        regime=REGIME_LME,
        stage=1,
        seed=DEFAULT_RANDOM_SEED,
        num_iterations=5,
        num_env_runners=RLLIB_TRAIN_NUM_ENV_RUNNERS,
        num_envs_per_env_runner=RLLIB_NUM_ENVS_PER_ENV_RUNNER,
        num_gpus_per_learner=0,  # we'll move to CUDA manually
    )
    if config is None:
        sys.exit(1)

    ray = metadata["ray"]
    ray.init(
        ignore_reinit_error=True,
        num_cpus=RLLIB_TRAIN_NUM_CPUS,
        log_to_driver=False,
    )

    print("  Building algorithm...")
    t0 = time.perf_counter()
    algo = config.build_algo()
    print(f"  Build took {time.perf_counter() - t0:.1f}s")

    print("  Relocating RLModules to CUDA...")
    ok = force_module_to_cuda(algo, ["pi_household", "pi_firm", "pi_gov", "pi_cb"])
    print(f"  Relocation OK: {ok}")

    # Verify the module locations.
    try:
        m = algo.get_module("pi_household")
        for name, p in m.named_parameters():
            print(f"  driver-visible pi_household: '{name}' device={p.device}")
            break
    except Exception:
        pass

    print(f"\n  Iter  |  wall   | mean_episode_return | households | num_episodes")
    iter_times = []
    for i in range(5):
        t0 = time.perf_counter()
        result = algo.train()
        dt = time.perf_counter() - t0
        iter_times.append(dt)
        env_runners = result.get("env_runners", {})
        ep_ret = env_runners.get("episode_return_mean", env_runners.get("episode_reward_mean", float("nan")))
        per_policy = env_runners.get("module_episode_returns_mean", {})
        hh = per_policy.get("pi_household", float("nan"))
        n_ep = env_runners.get("num_episodes", 0)
        print(f"  {i + 1:>4d}  | {dt:>6.1f}s | {ep_ret:>18.4f} | {hh:>10.4f} | {n_ep:>12d}")

    algo.stop()
    ray.shutdown()

    print(f"\n  Mean iter wall: {sum(iter_times)/len(iter_times):.1f}s")
    print(f"  Post-warmup mean (iters 3-5): {sum(iter_times[2:])/3:.1f}s")


if __name__ == "__main__":
    main()
