"""Directly inspect which device the LEARNER's training module is on.

`algo.get_module(mid)` returns the driver's inference copy. The learner has
its own copy that does the actual training. They can be on different devices.
"""

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


def main():
    print(f"=== Inspect learner device ===")
    config, metadata = build_ppo_config(
        regime=REGIME_LME, stage=1,
        seed=DEFAULT_RANDOM_SEED, num_iterations=1,
        num_env_runners=RLLIB_TRAIN_NUM_ENV_RUNNERS,
        num_envs_per_env_runner=RLLIB_NUM_ENVS_PER_ENV_RUNNER,
        num_gpus_per_learner=1,
    )
    ray = metadata["ray"]
    ray.init(ignore_reinit_error=True, num_cpus=RLLIB_TRAIN_NUM_CPUS, log_to_driver=False)
    algo = config.build_algo()

    print()
    print("--- algo.get_module (driver inference copy) ---")
    m = algo.get_module("pi_household")
    for name, p in m.named_parameters():
        print(f"  driver pi_household.{name}: device={p.device}, dtype={p.dtype}")
        break

    print()
    print("--- algo.learner_group._learner (training copy) ---")
    lg = algo.learner_group
    inner_learner = getattr(lg, "_learner", None) or getattr(lg, "_learners", [None])[0]
    print(f"  Learner type: {type(inner_learner).__name__}")
    print(f"  Learner._device: {getattr(inner_learner, '_device', '<missing>')}")
    train_module = inner_learner.module["pi_household"]
    for name, p in train_module.named_parameters():
        print(f"  learner pi_household.{name}: device={p.device}, dtype={p.dtype}")
        break

    # Inspect optimizer state device too.
    print()
    print("--- optimizer state device ---")
    optims = getattr(inner_learner, "_module_optimizers_dict", None)
    if optims:
        print(f"  module optimizers: {list(optims.keys())}")
        for mid, opt in list(optims.items())[:1]:
            sample_state = next(iter(opt.state.values()), {})
            for sk, sv in sample_state.items():
                if hasattr(sv, "device"):
                    print(f"  optimizer state '{sk}' device: {sv.device}")
                    break
    else:
        for attr in ["optimizers", "_optimizers", "_optimizer_params"]:
            v = getattr(inner_learner, attr, None)
            if v is not None:
                print(f"  Found {attr}: {type(v).__name__}")
                break

    # Run one quick train() to see what happens.
    print()
    print("--- Running 1 train iteration to confirm GPU is actually computing ---")
    t0 = time.perf_counter()
    result = algo.train()
    dt = time.perf_counter() - t0
    print(f"  Iter wall: {dt:.1f}s")
    er = result.get("env_runners", {})
    print(f"  episode_return_mean: {er.get('episode_return_mean', 'n/a')}")

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
