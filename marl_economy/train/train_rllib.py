"""
RLlib training for MARL Economy simulation.
Requires: pip install ray[rllib] torch

Usage:
    python -m marl_economy.train.train_rllib --stage 1 --regime <regime_name>
    python -m marl_economy.train.train_rllib --stage 2 --regime <regime_name> --load-from <checkpoint_path>
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from marl_economy.config.constants import (
    DEFAULT_RANDOM_SEED, DEFAULT_REGIME_NAME,
    RLLIB_NUM_ENVS_PER_ENV_RUNNER,
    RLLIB_TRAIN_NUM_CPUS,
    RLLIB_TRAIN_NUM_ENV_RUNNERS, RLLIB_TRAIN_NUM_GPUS_PER_LEARNER,
    RLLIB_LOG_INTERVAL_ITERATIONS, RLLIB_CHECKPOINT_INTERVAL_ITERATIONS,
    PPO_TRAIN_BATCH_SIZE, PPO_MINIBATCH_SIZE, PPO_NUM_SGD_ITER,
    PPO_NETWORK_HIDDENS,
)


def import_rllib():
    """Import RLlib lazily so non-RL workflows do not require Ray."""
    try:
        import ray
        from ray.rllib.algorithms.ppo import PPOConfig
        from ray.rllib.policy.policy import PolicySpec
    except ImportError:
        print("ERROR: Ray/RLlib not installed.")
        print("Install with: pip install 'ray[rllib]' torch")
        return None, None, None
    return ray, PPOConfig, PolicySpec


def modules_for_stage(stage: int) -> list[str]:
    """Return the policy modules expected to exist at a given curriculum stage."""
    return {
        1: ["pi_household"],
        2: ["pi_household", "pi_firm"],
        3: ["pi_household", "pi_firm", "pi_gov", "pi_cb"],
        4: ["pi_household", "pi_firm", "pi_gov", "pi_cb"],
    }.get(stage, [])


def resolve_checkpoint_dir(checkpoint_path: str | os.PathLike) -> Path:
    """Resolve a run root or checkpoint label path to the actual RLlib checkpoint dir."""
    path = Path(checkpoint_path).resolve()
    if path.is_file():
        path = path.parent
    if (path / "rllib_checkpoint.json").exists():
        return path
    for label in ("final", "best"):
        candidate = path / label
        if (candidate / "rllib_checkpoint.json").exists():
            return candidate
    raise FileNotFoundError(f"No RLlib checkpoint found at or under {path}")


def restore_modules_from_checkpoint(algo, checkpoint_path: str, stage: int) -> Path:
    """Restore per-policy RLModules from a saved checkpoint directory."""
    checkpoint_dir = resolve_checkpoint_dir(checkpoint_path)
    print(f"  Loading weights from: {checkpoint_dir}")

    expected = list(modules_for_stage(stage))
    loaded: list[str] = []
    for module_id in expected:
        last_err: Exception | None = None
        for base in [checkpoint_dir, checkpoint_dir / "learner_group" / "learner" / "rl_module"]:
            module_path = base / module_id
            if module_path.exists():
                try:
                    algo.restore_from_path(
                        path=str(module_path),
                        component=f"learner_group/learner/rl_module/{module_id}",
                    )
                    print(f"    Loaded {module_id}")
                    loaded.append(module_id)
                    last_err = None
                    break
                except Exception as e:  # try the next candidate base before failing
                    last_err = e
        if module_id not in loaded:
            raise RuntimeError(
                f"Checkpoint restore FAILED for required module '{module_id}' "
                f"(stage {stage}) from {checkpoint_dir}: "
                + (f"{last_err}" if last_err is not None
                   else "module directory not found in any candidate path")
                + ". Refusing to train from a partially/randomly-initialized "
                  "policy - fix the checkpoint path/layout."
            )

    return checkpoint_dir


def load_checkpoint_run_config(checkpoint_path: str | os.PathLike) -> dict:
    """Load the plain JSON run config saved next to RLlib checkpoints, if present."""
    checkpoint_dir = resolve_checkpoint_dir(checkpoint_path)
    for base in (checkpoint_dir, checkpoint_dir.parent):
        config_path = base / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
    return {}


def build_ppo_config(regime: str = None, stage: int = 1, seed: int = DEFAULT_RANDOM_SEED,
                      num_iterations: int = 100, num_env_runners: int = RLLIB_TRAIN_NUM_ENV_RUNNERS,
                     num_envs_per_env_runner: int = RLLIB_NUM_ENVS_PER_ENV_RUNNER,
                     num_gpus_per_learner: int = RLLIB_TRAIN_NUM_GPUS_PER_LEARNER,
                     batched_k: int = 1):
    """Build the shared RLlib PPO config used by training and evaluation.

    When `batched_k > 1`, uses `BatchedMARLEconomyEnv` which wraps
    `BatchedWorld(K=batched_k)` and exposes K parallel worlds as one env
    with world-prefixed agent IDs (e.g. `h_w0_5`, `f_w3_8`, `gov_w2`,
    `cb_w7`). All worlds train shared policies.
    """
    ray, PPOConfig, PolicySpec = import_rllib()
    if ray is None:
        return None, None

    from gymnasium import spaces
    from marl_economy.config.constants import (
        DEFAULT_REGIME_NAME,
        DEFAULT_NUM_HOUSEHOLDS, DEFAULT_NUM_FIRMS, DEFAULT_WARMUP_WEEKS,
        CURRICULUM, HOUSEHOLD_OBS_DIM, FIRM_OBS_DIM, GOV_OBS_DIM, CB_OBS_DIM,
        NUM_HOUSEHOLD_ACTIONS, HOUSEHOLD_GAMMA, POLICY_GAMMAS,
        FIRM_ACTION_DIMS, GOV_ACTION_DIMS, CB_RATE_DELTAS,
        PPO_LR, PPO_CLIP, PPO_ENTROPY_COEFF, PPO_NUM_SGD_ITER,
        PPO_GAE_LAMBDA, PPO_VF_LOSS_COEFF, PPO_GRAD_CLIP,
        PPO_TRAIN_BATCH_SIZE, PPO_MINIBATCH_SIZE,
    )
    if batched_k > 1:
        from marl_economy.env.marl_env_rllib_batched import (
            BatchedMARLEconomyEnv as _EnvCls,
            policy_mapping_fn_batched as _policy_map_fn,
        )
        _MARLEnv = _EnvCls
        _policy_mapping_fn = _policy_map_fn
    else:
        from marl_economy.env.marl_env_rllib import (
            MARLEconomyEnv as _MARLEnv,
            policy_mapping_fn as _policy_mapping_fn,
        )

    regime = regime or DEFAULT_REGIME_NAME
    phase = CURRICULUM.get(stage, CURRICULUM[1])
    episode_weeks = phase["episode_weeks"]
    trainable = phase.get("policies_to_train", ["pi_household"])
    policy_lr_scales = phase.get("policy_lr_scales", {})

    base_lr = PPO_LR
    if "lr_scale" in phase:
        base_lr *= phase["lr_scale"]

    env_config = {
        "regime": regime,
        "num_households": DEFAULT_NUM_HOUSEHOLDS,
        "num_firms": DEFAULT_NUM_FIRMS,
        "episode_weeks": episode_weeks,
        "warmup_weeks": DEFAULT_WARMUP_WEEKS,
        "seed": seed,
        "curriculum_phase": stage,
        "firms_fixed": phase.get("firms_fixed", False),
        "gov_scripted": phase.get("gov_scripted", False),
        "cb_scripted": phase.get("cb_scripted", False),
        "households_learning": phase.get("households_learning", True),
        "batched_k": batched_k,
    }

    total_env_steps = max(1, num_iterations) * PPO_TRAIN_BATCH_SIZE
    gov_entropy_schedule = [
        (0, 0.05),
        (total_env_steps * 0.3, 0.03),
        (total_env_steps * 0.7, 0.01),
        (total_env_steps, 0.005),
    ]
    cb_entropy_schedule = [
        (0, 0.03),
        (total_env_steps * 0.5, 0.01),
        (total_env_steps, 0.005),
    ]

    config = (
        PPOConfig()
        .environment(
            env=_MARLEnv,
            env_config=env_config,
        )
        .multi_agent(
            policies={
                "pi_household": PolicySpec(
                    observation_space=spaces.Box(-np.inf, np.inf, shape=(HOUSEHOLD_OBS_DIM,), dtype=np.float32),
                    action_space=spaces.Discrete(NUM_HOUSEHOLD_ACTIONS),
                ),
                "pi_firm": PolicySpec(
                    observation_space=spaces.Box(-np.inf, np.inf, shape=(FIRM_OBS_DIM,), dtype=np.float32),
                    action_space=spaces.MultiDiscrete(FIRM_ACTION_DIMS),
                ),
                "pi_gov": PolicySpec(
                    observation_space=spaces.Box(-np.inf, np.inf, shape=(GOV_OBS_DIM,), dtype=np.float32),
                    action_space=spaces.MultiDiscrete(GOV_ACTION_DIMS),
                ),
                "pi_cb": PolicySpec(
                    observation_space=spaces.Box(-np.inf, np.inf, shape=(CB_OBS_DIM,), dtype=np.float32),
                    action_space=spaces.Discrete(len(CB_RATE_DELTAS)),
                ),
            },
            policy_mapping_fn=_policy_mapping_fn,
            policies_to_train=trainable,
            count_steps_by="env_steps",
        )
        .rl_module(
            algorithm_config_overrides_per_module={
                "pi_household": PPOConfig.overrides(
                    lr=base_lr * policy_lr_scales.get("pi_household", 1.0),
                    gamma=POLICY_GAMMAS["pi_household"],
                    entropy_coeff=PPO_ENTROPY_COEFF,
                ),
                "pi_firm": PPOConfig.overrides(
                    lr=base_lr * 0.5 * policy_lr_scales.get("pi_firm", 1.0),
                    gamma=POLICY_GAMMAS["pi_firm"],
                    entropy_coeff=0.02,
                ),
                "pi_gov": PPOConfig.overrides(
                    lr=base_lr * 0.3 * policy_lr_scales.get("pi_gov", 1.0),
                    gamma=POLICY_GAMMAS["pi_gov"],
                    entropy_coeff=0.05,
                    entropy_coeff_schedule=gov_entropy_schedule if "pi_gov" in trainable else None,
                ),
                "pi_cb": PPOConfig.overrides(
                    lr=base_lr * policy_lr_scales.get("pi_cb", 1.0),
                    gamma=POLICY_GAMMAS["pi_cb"],
                    entropy_coeff=0.03,
                    entropy_coeff_schedule=cb_entropy_schedule if "pi_cb" in trainable else None,
                ),
            },
        )
        .training(
            lr=base_lr,
            gamma=HOUSEHOLD_GAMMA,
            lambda_=PPO_GAE_LAMBDA,
            clip_param=PPO_CLIP,
            entropy_coeff=PPO_ENTROPY_COEFF,
            vf_loss_coeff=PPO_VF_LOSS_COEFF,
            num_epochs=PPO_NUM_SGD_ITER,
            train_batch_size_per_learner=PPO_TRAIN_BATCH_SIZE,
            minibatch_size=PPO_MINIBATCH_SIZE,
            grad_clip=PPO_GRAD_CLIP,
        )
        .framework("torch")
        .learners(num_learners=0, num_gpus_per_learner=num_gpus_per_learner)
        .env_runners(num_env_runners=num_env_runners, num_envs_per_env_runner=num_envs_per_env_runner)
    )

    metadata = {
        "ray": ray,
        "phase": phase,
        "trainable": trainable,
        "base_lr": base_lr,
        "episode_weeks": episode_weeks,
        "env_config": env_config,
        "curriculum": CURRICULUM,
    }
    return config, metadata


def train_with_rllib(regime: str = None, stage: int = 1,
                     num_iterations: int = 100, checkpoint_dir: str = None,
                     seed: int = DEFAULT_RANDOM_SEED, load_from: str = None,
                     batched_k: int = 1):
    """Train using RLlib PPO with new API stack.

    `batched_k > 1` switches to `BatchedMARLEconomyEnv` (BatchedWorld(K) per
    env runner with world-prefixed agent IDs).
    """
    config, metadata = build_ppo_config(
        regime=regime,
        stage=stage,
        seed=seed,
        num_iterations=num_iterations,
        num_env_runners=RLLIB_TRAIN_NUM_ENV_RUNNERS,
        num_envs_per_env_runner=RLLIB_NUM_ENVS_PER_ENV_RUNNER,
        num_gpus_per_learner=RLLIB_TRAIN_NUM_GPUS_PER_LEARNER,
        batched_k=batched_k,
    )
    if config is None:
        return

    ray = metadata["ray"]
    trainable = metadata["trainable"]
    base_lr = metadata["base_lr"]
    episode_weeks = metadata["episode_weeks"]
    curriculum = metadata["curriculum"]

    regime = regime or DEFAULT_REGIME_NAME

    if checkpoint_dir is None:
        # Naming convention: `rllib_<regime>_s<stage>_seed<N>_<ts>/` so the
        # directory name alone identifies regime + stage + seed.
        checkpoint_dir = str(
            Path.cwd()
            / f"runs/rllib_{regime}_s{stage}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    checkpoint_dir = str(Path(checkpoint_dir).resolve())
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Auto-generated README at the top of the run dir so it's self-describing.
    try:
        from marl_economy.eval.run_metadata import write_run_readme
        write_run_readme(
            checkpoint_dir,
            {
                "regime": regime,
                "curriculum_phase": stage,
                "stage": stage,
                "seed": seed,
                "episode_weeks": episode_weeks,
                "num_iterations": num_iterations,
                "trainable": trainable,
                "load_from": load_from,
                "base_lr": base_lr,
                "batched_k": batched_k,
            },
            kind="RLlib Training",
        )
    except Exception as e:
        print(f"  (README generation skipped: {e})")

    config_path = str(Path(checkpoint_dir) / "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "regime": regime,
            "stage": stage,
            "seed": seed,
            "episode_weeks": episode_weeks,
            "num_iterations": num_iterations,
            "trainable": trainable,
            "load_from": load_from,
            "base_lr": base_lr,
            "curriculum_phase": metadata["phase"],
            "env_config": metadata["env_config"],
            "runtime": {
                "num_cpus": RLLIB_TRAIN_NUM_CPUS,
                "num_env_runners": RLLIB_TRAIN_NUM_ENV_RUNNERS,
                "num_envs_per_env_runner": RLLIB_NUM_ENVS_PER_ENV_RUNNER,
                "num_gpus_per_learner": RLLIB_TRAIN_NUM_GPUS_PER_LEARNER,
            },
            "ppo": {
                "train_batch_size": PPO_TRAIN_BATCH_SIZE,
                "minibatch_size": PPO_MINIBATCH_SIZE,
                "num_sgd_iter": PPO_NUM_SGD_ITER,
                "network_hiddens": PPO_NETWORK_HIDDENS,
            },
            "checkpointing": {
                "log_interval_iterations": RLLIB_LOG_INTERVAL_ITERATIONS,
                "checkpoint_interval_iterations": RLLIB_CHECKPOINT_INTERVAL_ITERATIONS,
            },
        }, f, indent=2)

    print(f"{'=' * 60}")
    print(f"  RLlib Training Stage {stage}: {regime}")
    print(f"  Trainable policies: {trainable}")
    print(f"  Base LR: {base_lr}")
    print(f"  Episode weeks: {episode_weeks}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    if load_from:
        print(f"  Loading weights from: {load_from}")
    print(f"{'=' * 60}")

    ray.init(ignore_reinit_error=True, num_cpus=RLLIB_TRAIN_NUM_CPUS)

    print("  Building algorithm...")
    algo = config.build_algo()

    if load_from and os.path.isdir(load_from):
        restore_modules_from_checkpoint(algo, load_from, stage)

    print("  Starting training...\n")

    def safe_save(label):
        """Save checkpoint with full error handling."""
        save_path = str(Path(checkpoint_dir) / label)
        try:
            os.makedirs(save_path, exist_ok=True)
            algo.save(save_path)
            print(f"    Checkpoint saved: {label}")
            return save_path
        except Exception as e:
            print(f"    WARNING: Checkpoint save failed ({label}): {e}")
            return save_path

    best_reward = -np.inf
    final_path = checkpoint_dir
    # FIX E (2026-05-18 v2): the old "best" gate trusted episode_return_mean
    # (an episode SUM) alone - a policy that scores high while the population
    # mass-starves still got crowned "best" (exactly how the +350-561 vs
    # eval-collapse fooling happened). Add a SURVIVAL gate using ONLY standard
    # RLlib metrics (zero new-callback / API risk before a multi-day retrain):
    # only crown "best" if episodes ran ~=full horizon (episode_len_mean >=
    # SURVIVAL_FRAC*episode_weeks) - an early-collapsing economy mass-deaths ->
    # SHORTER episodes, so it can no longer be crowned. "final" (and iter_N)
    # are ALWAYS saved, so a usable checkpoint exists even if the gate never
    # passes (the documented fallback). The real success verdict remains the
    # post-train eval gated on survival+health+residual (this is defense-in-
    # depth). NOTE: a true population-HEALTH-gated "best" needs an RLlib
    # on_episode_end callback - deferred as an isolated follow-up (API-risk;
    # must not destabilize this structural retrain). Per-step-normalized
    # reward is now also printed so the SUM can't mislead at a glance.
    SURVIVAL_FRAC = 0.95
    _survival_min_len = SURVIVAL_FRAC * float(episode_weeks)
    best_survival_saved = False

    for i in range(num_iterations):
        result = algo.train()

        env_runners = result.get("env_runners", {})
        avg_reward = env_runners.get("episode_return_mean", env_runners.get("episode_reward_mean", 0.0))
        num_episodes = env_runners.get("num_episodes", 0)
        ep_len = float(env_runners.get("episode_len_mean",
                                       env_runners.get("episode_length_mean", 0.0)))
        survived = ep_len >= _survival_min_len

        if (i + 1) % RLLIB_LOG_INTERVAL_ITERATIONS == 0:
            policy_rewards = env_runners.get("module_episode_returns_mean", {})
            per_wk = avg_reward / ep_len if ep_len > 0 else 0.0
            parts = [f"Iter {i + 1:4d} | Total: {avg_reward:.4f} "
                     f"(/wk {per_wk:+.4f}, len {ep_len:.0f}/{episode_weeks}"
                     f"{'' if survived else ' EARLY-COLLAPSE'})"]
            for pid in trainable:
                pr = policy_rewards.get(pid, None)
                if pr is not None:
                    parts.append(f"{pid}: {pr:.4f}")
            print(f"  {' | '.join(parts)}  (eps: {num_episodes})")

        if (i + 1) % RLLIB_CHECKPOINT_INTERVAL_ITERATIONS == 0:
            safe_save(f"iter_{i + 1}")

        # Survival-gated "best": improved reward AND episodes not early-
        # collapsing. (final/iter_N always saved -> fallback if never gated.)
        if survived and avg_reward > best_reward and avg_reward != 0.0:
            best_reward = avg_reward
            best_survival_saved = True
            safe_save("best")

    final_path = safe_save("final")

    algo.stop()
    ray.shutdown()

    print(f"\n{'=' * 60}")
    print("  Training complete.")
    if best_survival_saved:
        print(f"  Best reward (survival-gated): {best_reward:.4f}")
    else:
        print("  Best reward: NONE survival-gated (every iter early-collapsed) "
              "- use 'final'/'iter_N'. This itself signals the policy never "
              "kept the population alive a full episode.")
    print(f"  Final checkpoint: {final_path}")
    print(f"{'=' * 60}")
    if stage < max(curriculum):
        print("\n  To continue with next stage:")
        print(f"  python -m marl_economy.train.train_rllib --stage {stage + 1} "
              f"--regime {regime} --load-from \"{final_path}\"")


def main():
    import argparse
    from marl_economy.config.constants import DEFAULT_REGIME_NAME, REGIME_NAMES, DEFAULT_RANDOM_SEED

    parser = argparse.ArgumentParser(description="RLlib MARL Economy Training")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--regime", default=DEFAULT_REGIME_NAME, choices=REGIME_NAMES)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--load-from", default=None)
    parser.add_argument("--batched-k", type=int, default=1,
                        help="K worlds per RLlib env runner via BatchedMARLEconomyEnv. "
                             "Default 1 (uses legacy MARLEconomyEnv). Set >1 to use "
                             "BatchedWorld(K) parallelism inside each env runner.")
    args = parser.parse_args()

    train_with_rllib(args.regime, args.stage, args.iterations,
                     args.checkpoint, args.seed, args.load_from,
                     batched_k=args.batched_k)


if __name__ == "__main__":
    main()
