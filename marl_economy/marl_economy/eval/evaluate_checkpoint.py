"""
Evaluate a trained RLlib checkpoint by running a frozen inference episode.

Outputs:
- metrics_step.csv compatible with plot_run.py
- summary.json
- config.json
- optional snapshots.json for external visualization (Unity, web, etc.)
- optional plots generated from the evaluation run
"""

import sys
import os
import json
import csv
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from gymnasium import spaces
from ray.rllib.core.columns import Columns

from marl_economy.config.constants import (
    DEFAULT_REGIME_NAME, REGIME_NAMES, CURRICULUM,
    DEFAULT_RANDOM_SEED, RUN_PROGRESS_INTERVAL_WEEKS,
    RLLIB_EVAL_NUM_CPUS, RLLIB_EVAL_NUM_ENV_RUNNERS, RLLIB_EVAL_NUM_GPUS_PER_LEARNER,
)
from marl_economy.env.marl_env_rllib import MARLEconomyEnv, policy_mapping_fn
from marl_economy.eval.plot_run import (
    generate_all_plots, plot_wealth_distribution, plot_lorenz_curve,
)
from marl_economy.scripts.run_one_episode import (
    EPISODE_METRICS_CSV_FIELDS,
    build_metrics_row,
    build_episode_summary,
    summarize_flat_rewards,
    write_debug_log,
)
from marl_economy.train.train_rllib import (
    build_ppo_config,
    load_checkpoint_run_config,
    resolve_checkpoint_dir,
    restore_modules_from_checkpoint,
)
from marl_economy.viz.export_unity import export_unity_playback


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically safe probabilities for stochastic checkpoint evaluation."""
    logits = np.asarray(logits, dtype=np.float64)
    if not np.all(np.isfinite(logits)):
        logits = np.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)

    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    total = np.sum(exp_vals)
    if not np.isfinite(total) or total <= 0.0:
        return np.full_like(logits, 1.0 / len(logits), dtype=np.float64)
    return exp_vals / total


def decode_inference_action(module, obs: np.ndarray, action_space, explore: bool, rng: np.random.Generator):
    """Run one policy module in inference mode and decode logits into an action."""
    obs_tensor = torch.as_tensor(np.expand_dims(obs, 0), dtype=torch.float32)
    with torch.no_grad():
        outputs = module.forward_inference({Columns.OBS: obs_tensor})

    logits = outputs.get(Columns.ACTION_DIST_INPUTS, outputs.get("action_dist_inputs"))
    if logits is None:
        raise KeyError("RLModule inference output missing action distribution inputs")

    logits = logits[0].detach().cpu().numpy()

    if isinstance(action_space, spaces.Discrete):
        if explore:
            probs = _stable_softmax(logits)
            return int(rng.choice(len(probs), p=probs))
        return int(np.argmax(logits))

    if isinstance(action_space, spaces.MultiDiscrete):
        actions = []
        start = 0
        for dim in action_space.nvec:
            head_logits = logits[start:start + dim]
            if explore:
                probs = _stable_softmax(head_logits)
                actions.append(int(rng.choice(dim, p=probs)))
            else:
                actions.append(int(np.argmax(head_logits)))
            start += dim
        return np.array(actions, dtype=np.int64)

    raise TypeError(f"Unsupported action space for evaluation: {action_space}")


def evaluate_checkpoint(checkpoint_path: str, stage: int = None, regime: str = None,
                        weeks: int = None, seed: int = DEFAULT_RANDOM_SEED, output_dir: str = None,
                        explore: bool = False, export_snapshots: bool = True,
                        make_plots: bool = True) -> dict:
    """Run one frozen-policy evaluation episode from an RLlib checkpoint."""
    checkpoint_dir = resolve_checkpoint_dir(checkpoint_path)
    checkpoint_config = load_checkpoint_run_config(checkpoint_dir)

    stage = stage or int(checkpoint_config.get("stage", max(CURRICULUM)))
    regime = regime or checkpoint_config.get("regime", DEFAULT_REGIME_NAME)
    weeks = weeks or int(checkpoint_config.get(
        "episode_weeks",
        CURRICULUM.get(stage, CURRICULUM[1])["episode_weeks"],
    ))

    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"runs/eval_checkpoint_stage{stage}_{regime}_{stamp}"

    output_dir = str(Path(output_dir).resolve())
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/plots", exist_ok=True)

    config, metadata = build_ppo_config(
        regime=regime,
        stage=stage,
        seed=seed,
        num_iterations=1,
        num_env_runners=RLLIB_EVAL_NUM_ENV_RUNNERS,
        num_gpus_per_learner=RLLIB_EVAL_NUM_GPUS_PER_LEARNER,
    )
    if config is None:
        return {}

    ray = metadata["ray"]
    env_config = dict(metadata["env_config"])
    env_config["episode_weeks"] = weeks
    env_config["seed"] = seed

    ray.init(ignore_reinit_error=True, num_cpus=RLLIB_EVAL_NUM_CPUS)
    algo = config.build_algo()
    restore_modules_from_checkpoint(algo, str(checkpoint_dir), stage)

    env = MARLEconomyEnv(env_config)
    obs, _ = env.reset(seed=seed)
    modules = {module_id: algo.get_module(module_id) for module_id in set(policy_mapping_fn(aid) for aid in env.possible_agents)}
    rng = np.random.default_rng(seed)

    all_metrics = []
    reward_history = []
    snapshots = []
    firm_entries = 0
    firm_exits = 0
    prev_firm_count = env.world.f.num_alive

    with open(f"{output_dir}/config.json", "w") as f:
        json.dump({
            "mode": "checkpoint_evaluation",
            "checkpoint_path": str(checkpoint_dir),
            "regime": regime,
            "stage": stage,
            "seed": seed,
            "episode_weeks": weeks,
            "explore": explore,
            "export_snapshots": export_snapshots,
            "plots": make_plots,
        }, f, indent=2)

    csv_path = f"{output_dir}/metrics_step.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=EPISODE_METRICS_CSV_FIELDS)
        writer.writeheader()

        for step in range(weeks):
            action_dict = {}
            for agent_id, agent_obs in obs.items():
                policy_id = policy_mapping_fn(agent_id)
                action = decode_inference_action(
                    modules[policy_id],
                    agent_obs,
                    env.action_spaces[agent_id],
                    explore,
                    rng,
                )
                action_dict[agent_id] = action

            obs, rewards, terminateds, truncateds, infos = env.step(action_dict)

            metrics = env.world.metrics.history[-1]
            current_firm_count = metrics["num_firms"]
            if current_firm_count > prev_firm_count:
                firm_entries += current_firm_count - prev_firm_count
            elif current_firm_count < prev_firm_count:
                firm_exits += prev_firm_count - current_firm_count
            prev_firm_count = current_firm_count

            reward_stats = summarize_flat_rewards(rewards)
            writer.writerow(build_metrics_row(step, regime, metrics, firm_entries, firm_exits, reward_stats))
            all_metrics.append(metrics)
            reward_history.append(reward_stats)

            if export_snapshots:
                snapshots.append(env.world.get_snapshot())

            if step % RUN_PROGRESS_INTERVAL_WEEKS == 0:
                print(f"  Week {step:4d}/{weeks} | "
                      f"GDP={metrics['gdp']:.3f} Gini={metrics['gini']:.3f} "
                      f"Health={metrics['avg_health']:.3f} Firms={metrics['num_firms']}")

            done = terminateds.get("__all__", False) or truncateds.get("__all__", False)
            if done:
                print(f"  Terminated at week {step}: {env.world.termination_reason}")
                break

    total_weeks = step + 1
    summary = build_episode_summary(
        regime, env.world, all_metrics, firm_entries, firm_exits, total_weeks, reward_history
    )
    summary["checkpoint_path"] = str(checkpoint_dir)
    summary["stage"] = stage
    summary["explore"] = explore

    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if export_snapshots:
        with open(f"{output_dir}/snapshots.json", "w") as f:
            json.dump({
                "mode": "checkpoint_evaluation",
                "checkpoint_path": str(checkpoint_dir),
                "regime": regime,
                "stage": stage,
                "seed": seed,
                "snapshots": snapshots,
            }, f)
        try:
            summary["unity_playback_path"] = export_unity_playback(output_dir)
            with open(f"{output_dir}/summary.json", "w") as f:
                json.dump(summary, f, indent=2)
        except Exception as exc:
            print(f"  WARNING: Unity playback export failed: {exc}")

    write_debug_log(env.world.debug_log, output_dir)

    if make_plots:
        generate_all_plots(output_dir)
        if not export_snapshots:
            final_wealths = env.world.h.wealth.astype(float)
            plot_wealth_distribution(final_wealths, f"{output_dir}/plots", label="final")
            plot_lorenz_curve(final_wealths, f"{output_dir}/plots")

    algo.stop()
    ray.shutdown()

    print(f"\nEvaluation saved to {output_dir}/")
    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate an RLlib MARL checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to an RLlib checkpoint dir or run root")
    parser.add_argument("--stage", type=int, default=None, choices=[1, 2, 3, 4])
    parser.add_argument("--regime", default=None, choices=REGIME_NAMES)
    parser.add_argument("--weeks", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--output", default=None)
    parser.add_argument("--explore", action="store_true", help="Sample stochastically instead of greedy inference")
    parser.add_argument("--no-snapshots", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    summary = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        stage=args.stage,
        regime=args.regime,
        weeks=args.weeks,
        seed=args.seed,
        output_dir=args.output,
        explore=args.explore,
        export_snapshots=not args.no_snapshots,
        make_plots=not args.no_plots,
    )
    print(f"\nSummary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
