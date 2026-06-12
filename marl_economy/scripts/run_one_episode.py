"""
Run one full episode with random or scripted policies.
Logs metrics to CSV and config to JSON (Section 14.1-14.2).
"""

import sys
import os
import json
import csv
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from marl_economy.config.constants import (
    DEFAULT_REGIME_NAME, REGIME_NAMES,
    DEFAULT_EPISODE_WEEKS, DEFAULT_RANDOM_SEED,
    FIRM_UPDATE_PERIOD, GOV_UPDATE_PERIOD, CB_UPDATE_PERIOD,
    RUN_PROGRESS_INTERVAL_WEEKS,
    NUM_HOUSEHOLD_ACTIONS, FIRM_PRICE_DELTAS, FIRM_WAGE_DELTAS,
    FIRM_CAPACITY_DELTAS, GOV_TAX_DELTAS, GOV_WELFARE_DELTAS,
    GOV_POLICING_DELTAS, GOV_HEALTHCARE_DELTAS, GOV_EDUCATION_DELTAS,
    GOV_MINWAGE_DELTAS, CB_RATE_DELTAS,
)
from marl_economy.env.world import World

from marl_economy.env.events import ALL_EVENT_TYPES

# Event-derived columns (one per typed event, this week's count).
EVENT_CSV_FIELDS = [f"events_{t}" for t in ALL_EVENT_TYPES]

# Cumulative running totals retained for backward compatibility with old plots.
EPISODE_METRICS_CSV_FIELDS = [
    "t_week", "regime", "num_households", "num_firms",
    "gdp_week", "mean_price", "mean_wage", "inflation_proxy",
    "gini", "poverty_rate", "avg_health", "crime_rate",
    "crime_attempts_rate",
    "unemp_proxy",
    # `firm_entries` / `firm_exits` are running totals derived from net
    # firm-count change (preserved for backward compatibility with old
    # plot scripts). Per-week deltas live in `firm_entries_week` /
    # `firm_entries` / `firm_exits` are running totals; per-week deltas live in
    # `firm_entries_week`, `firm_exits_week`, and the typed `events_*` columns.
    "firm_entries", "firm_exits",
    "firm_entries_week", "firm_exits_week",
    # Fatigue.
    "mean_fatigue", "max_fatigue", "frac_fatigued", "frac_exhausted",
    # Illness state distribution.
    "frac_healthy", "frac_exposed", "frac_mild", "frac_severe", "frac_recovering",
    # Clinic capacity.
    "clinic_capacity", "clinic_used",
    # Disasters.
    "disaster_signal", "supply_shock_active", "epidemic_wave_active", "firms_in_downtime",
    # Firm-side financial diagnostics.
    "mean_firm_cash", "min_firm_cash", "max_firm_cash",
    "frac_insolvent_firms", "total_inventory", "total_money",
    "money_minted", "money_destroyed", "money_residual",
    "treasury", "tax_collected_wk", "welfare_paid_wk", "dividend_paid_wk",
    # Household-side hunger diagnostics.
    "mean_hunger", "max_hunger", "frac_hungry", "frac_crisis",
    # Action attempt/success counters.
    "shop_attempts", "shop_matched",
    "clinic_attempts", "clinic_matched",
    "steal_success", "work_refused_solvency",
    "weekly_unmet_demand", "weekly_total_demand",
    # Per-week deaths/displacements/hunger crises (also exposed as events).
    "weekly_deaths", "weekly_displacements", "weekly_hunger_crisis",
    # Reward streams.
    "reward_households_mean", "reward_firms_mean",
    "reward_gov", "reward_cb", "reward_total",
] + EVENT_CSV_FIELDS


def summarize_world_rewards(rewards: dict) -> dict:
    """Summarize World.step reward payloads into plottable scalar series."""
    household_rewards = list(rewards.get("households", {}).values())
    firm_rewards = list(rewards.get("firms", {}).values())
    gov_reward = float(rewards.get("gov", 0.0))
    cb_reward = float(rewards.get("cb", 0.0))

    household_mean = float(np.mean(household_rewards)) if household_rewards else 0.0
    firm_mean = float(np.mean(firm_rewards)) if firm_rewards else 0.0
    total_reward = float(np.sum(household_rewards) + np.sum(firm_rewards) + gov_reward + cb_reward)

    return {
        "reward_households_mean": household_mean,
        "reward_firms_mean": firm_mean,
        "reward_gov": gov_reward,
        "reward_cb": cb_reward,
        "reward_total": total_reward,
    }


def summarize_flat_rewards(rewards: dict) -> dict:
    """Summarize RLlib-style flat reward dicts into the same scalar schema."""
    household_rewards = [reward for agent_id, reward in rewards.items() if agent_id.startswith("h_")]
    firm_rewards = [reward for agent_id, reward in rewards.items() if agent_id.startswith("f_")]
    gov_reward = float(rewards.get("gov", 0.0))
    cb_reward = float(rewards.get("cb", 0.0))

    household_mean = float(np.mean(household_rewards)) if household_rewards else 0.0
    firm_mean = float(np.mean(firm_rewards)) if firm_rewards else 0.0
    total_reward = float(np.sum(household_rewards) + np.sum(firm_rewards) + gov_reward + cb_reward)

    return {
        "reward_households_mean": household_mean,
        "reward_firms_mean": firm_mean,
        "reward_gov": gov_reward,
        "reward_cb": cb_reward,
        "reward_total": total_reward,
    }


def build_metrics_row(step: int, regime: str, metrics: dict,
                      firm_entries: int, firm_exits: int,
                      reward_stats: dict | None = None) -> dict:
    """Build one metrics_step.csv row using the canonical plot schema."""
    reward_stats = reward_stats or summarize_world_rewards({})
    row = {
        "t_week": step,
        "regime": regime,
        "num_households": metrics["num_households"],
        "num_firms": metrics["num_firms"],
        "gdp_week": metrics["gdp"],
        "mean_price": metrics["mean_price"],
        "mean_wage": metrics["mean_wage"],
        "inflation_proxy": metrics["inflation"],
        "gini": metrics["gini"],
        "poverty_rate": metrics["poverty"],
        "avg_health": metrics["avg_health"],
        "crime_rate": metrics["crime_rate"],
        "unemp_proxy": metrics["unemployment"],
        "crime_attempts_rate": metrics.get("crime_attempts_rate", 0.0),
        "firm_entries": firm_entries,
        "firm_exits": firm_exits,
        # Per-week deltas derived from the event ledger. Entries sum normal +
        # emergency; exits exclude blocked attempts.
        "firm_entries_week": (
            metrics.get("events_firm_entry_normal", 0)
            + metrics.get("events_firm_entry_emergency", 0)
        ),
        "firm_exits_week": metrics.get("events_firm_exit", 0),
        # Firm-side diagnostics
        "mean_firm_cash": metrics.get("mean_firm_cash", 0.0),
        "min_firm_cash": metrics.get("min_firm_cash", 0.0),
        "max_firm_cash": metrics.get("max_firm_cash", 0.0),
        "frac_insolvent_firms": metrics.get("frac_insolvent_firms", 0.0),
        "total_inventory": metrics.get("total_inventory", 0.0),
        "total_money": metrics.get("total_money", 0.0),
        "money_minted": metrics.get("money_minted", 0.0),
        "money_destroyed": metrics.get("money_destroyed", 0.0),
        "money_residual": metrics.get("money_residual", 0.0),
        "treasury": metrics.get("treasury", 0.0),
        "tax_collected_wk": metrics.get("tax_collected_wk", 0.0),
        "welfare_paid_wk": metrics.get("welfare_paid_wk", 0.0),
        "dividend_paid_wk": metrics.get("dividend_paid_wk", 0.0),
        # Fatigue.
        "mean_fatigue": metrics.get("mean_fatigue", 0.0),
        "max_fatigue": metrics.get("max_fatigue", 0.0),
        "frac_fatigued": metrics.get("frac_fatigued", 0.0),
        "frac_exhausted": metrics.get("frac_exhausted", 0.0),
        # Illness state distribution.
        "frac_healthy": metrics.get("frac_healthy", 1.0),
        "frac_exposed": metrics.get("frac_exposed", 0.0),
        "frac_mild": metrics.get("frac_mild", 0.0),
        "frac_severe": metrics.get("frac_severe", 0.0),
        "frac_recovering": metrics.get("frac_recovering", 0.0),
        # Clinic capacity.
        "clinic_capacity": metrics.get("clinic_capacity", 0),
        "clinic_used": metrics.get("clinic_used", 0),
        # Disasters.
        "disaster_signal": metrics.get("disaster_signal", 0),
        "supply_shock_active": metrics.get("supply_shock_active", 0),
        "epidemic_wave_active": metrics.get("epidemic_wave_active", 0),
        "firms_in_downtime": metrics.get("firms_in_downtime", 0),
        # Hunger diagnostics
        "mean_hunger": metrics.get("mean_hunger", 0.0),
        "max_hunger": metrics.get("max_hunger", 0.0),
        "frac_hungry": metrics.get("frac_hungry", 0.0),
        "frac_crisis": metrics.get("frac_crisis", 0.0),
        # Attempt/success counters
        "shop_attempts": metrics.get("shop_attempts", 0),
        "shop_matched": metrics.get("shop_matched", 0),
        "clinic_attempts": metrics.get("clinic_attempts", 0),
        "clinic_matched": metrics.get("clinic_matched", 0),
        "steal_success": metrics.get("steal_success", 0),
        "work_refused_solvency": metrics.get("work_refused_solvency", 0),
        "weekly_unmet_demand": metrics.get("weekly_unmet_demand", 0),
        "weekly_total_demand": metrics.get("weekly_total_demand", 0),
        # Per-week event-style counters
        "weekly_deaths": metrics.get("weekly_deaths", 0),
        "weekly_displacements": metrics.get("weekly_displacements", 0),
        "weekly_hunger_crisis": metrics.get("weekly_hunger_crisis", 0),
        # Rewards
        "reward_households_mean": reward_stats["reward_households_mean"],
        "reward_firms_mean": reward_stats["reward_firms_mean"],
        "reward_gov": reward_stats["reward_gov"],
        "reward_cb": reward_stats["reward_cb"],
        "reward_total": reward_stats["reward_total"],
    }
    # Per-event counters (events_firm_entry_normal, events_firm_exit, etc.)
    for col in EVENT_CSV_FIELDS:
        row[col] = metrics.get(col, 0)
    return row


def build_episode_summary(regime: str, world: World, all_metrics: list[dict],
                          firm_entries: int, firm_exits: int,
                          total_weeks: int,
                          reward_history: list[dict] | None = None) -> dict:
    """Build the common summary payload used by episode/evaluation runners."""
    reward_history = reward_history or []

    summary = {
        "regime": regime,
        "total_weeks": total_weeks,
        "terminated_early": world.terminated_early,
        "termination_reason": world.termination_reason,
        "final_gdp": all_metrics[-1]["gdp"] if all_metrics else 0.0,
        "mean_gini": float(np.mean([m["gini"] for m in all_metrics])) if all_metrics else 0.0,
        "mean_health": float(np.mean([m["avg_health"] for m in all_metrics])) if all_metrics else 0.0,
        "mean_inflation": float(np.mean([m["inflation"] for m in all_metrics])) if all_metrics else 0.0,
        "mean_poverty": float(np.mean([m["poverty"] for m in all_metrics])) if all_metrics else 0.0,
        "mean_crime": float(np.mean([m["crime_rate"] for m in all_metrics])) if all_metrics else 0.0,
        # Inferred entries/exits - kept for backward compatibility but are
        # superseded by the event-ledger totals below.
        "firm_entries_inferred": firm_entries,
        "firm_exits_inferred": firm_exits,
        "firm_entries": firm_entries,
        "firm_exits": firm_exits,
        # Diagnostic averages.
        "mean_firm_cash": (
            float(np.mean([m.get("mean_firm_cash", 0.0) for m in all_metrics]))
            if all_metrics else 0.0
        ),
        "final_min_firm_cash": (
            float(all_metrics[-1].get("min_firm_cash", 0.0))
            if all_metrics else 0.0
        ),
        "mean_hunger": (
            float(np.mean([m.get("mean_hunger", 0.0) for m in all_metrics]))
            if all_metrics else 0.0
        ),
        "final_frac_crisis": (
            float(all_metrics[-1].get("frac_crisis", 0.0))
            if all_metrics else 0.0
        ),
        "mean_reward_households": (
            float(np.mean([r["reward_households_mean"] for r in reward_history]))
            if reward_history else 0.0
        ),
        "mean_reward_firms": (
            float(np.mean([r["reward_firms_mean"] for r in reward_history]))
            if reward_history else 0.0
        ),
        "mean_reward_gov": (
            float(np.mean([r["reward_gov"] for r in reward_history]))
            if reward_history else 0.0
        ),
        "mean_reward_cb": (
            float(np.mean([r["reward_cb"] for r in reward_history]))
            if reward_history else 0.0
        ),
        # `mean_reward_total` is the per-week sum across all agents averaged
        # over weeks (a per-week aggregate, not a per-agent average).
        # `mean_total_reward_per_week` is the same number under a clearer name.
        "mean_reward_total": (
            float(np.mean([r["reward_total"] for r in reward_history]))
            if reward_history else 0.0
        ),
        "mean_total_reward_per_week": (
            float(np.mean([r["reward_total"] for r in reward_history]))
            if reward_history else 0.0
        ),
    }

    # Event-ledger totals supersede inferred counts.
    if hasattr(world, "event_ledger") and world.event_ledger is not None:
        summary.update(world.event_ledger.to_summary())

    # Collapse detector outputs (None when not triggered).
    if hasattr(world, "collapse_detector") and world.collapse_detector is not None:
        summary.update(world.collapse_detector.to_summary())

    return summary


def write_debug_log(debug_log: list[str], output_dir: str) -> None:
    """Persist the world debug log if any entries were emitted."""
    if not debug_log:
        return
    with open(f"{output_dir}/debug.log", "w") as f:
        for entry in debug_log:
            f.write(entry + "\n")


def run_episode(regime: str = DEFAULT_REGIME_NAME, episode_weeks: int = DEFAULT_EPISODE_WEEKS,
                seed: int = DEFAULT_RANDOM_SEED, output_dir: str = None) -> dict:
    """Run one episode, log everything, return summary."""
    rng = np.random.default_rng(seed)

    world = World(
        regime_name=regime,
        episode_weeks=episode_weeks,
        seed=seed,
        curriculum_phase=1,
    )

    # Setup output dir. Naming convention: `episode_<regime>_seed<N>_<ts>/`
    # so every run is identifiable by its directory name alone.
    if output_dir is None:
        run_id = f"episode_{regime}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = f"runs/{run_id}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/plots", exist_ok=True)

    # Save config
    config = world.get_config()
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Auto-generated README so the directory contents are self-describing.
    try:
        from marl_economy.eval.run_metadata import write_run_readme, write_plots_info
        write_run_readme(output_dir, config, kind="Episode")
        write_plots_info(f"{output_dir}/plots", config)
    except Exception as e:
        print(f"  (README generation skipped: {e})")

    # Prepare CSV
    csv_path = f"{output_dir}/metrics_step.csv"
    obs = world.reset()
    all_metrics = []
    reward_history = []
    firm_entries = 0
    firm_exits = 0
    prev_firm_count = world.f.num_alive

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=EPISODE_METRICS_CSV_FIELDS)
        writer.writeheader()

        for step in range(episode_weeks):
            # Random actions
            household_actions = {
                int(world.h.ids[i]): rng.integers(0, NUM_HOUSEHOLD_ACTIONS)
                for i in range(world.h.size)
            }

            firm_actions = None
            if step > 0 and step % FIRM_UPDATE_PERIOD == 0:
                alive_slots = world.f.alive_slots()
                firm_actions = {
                    int(world.f.id[slot]): (
                        rng.integers(0, len(FIRM_PRICE_DELTAS)),
                        rng.integers(0, len(FIRM_WAGE_DELTAS)),
                        rng.integers(0, len(FIRM_CAPACITY_DELTAS)),
                    )
                    for slot in alive_slots.tolist()
                }

            gov_action = None
            if step > 0 and step % GOV_UPDATE_PERIOD == 0:
                gov_action = (
                    rng.integers(0, len(GOV_TAX_DELTAS)),
                    rng.integers(0, len(GOV_WELFARE_DELTAS)),
                    rng.integers(0, len(GOV_POLICING_DELTAS)),
                    rng.integers(0, len(GOV_HEALTHCARE_DELTAS)),
                    rng.integers(0, len(GOV_EDUCATION_DELTAS)),
                    rng.integers(0, len(GOV_MINWAGE_DELTAS)),
                )

            cb_action = None
            if step > 0 and step % CB_UPDATE_PERIOD == 0:
                cb_action = rng.integers(0, len(CB_RATE_DELTAS))

            obs, rewards, done, info = world.step(
                household_actions, firm_actions, gov_action, cb_action
            )

            m = info["metrics"]

            # Track entries/exits
            current_firm_count = m["num_firms"]
            if current_firm_count > prev_firm_count:
                firm_entries += (current_firm_count - prev_firm_count)
            elif current_firm_count < prev_firm_count:
                firm_exits += (prev_firm_count - current_firm_count)
            prev_firm_count = current_firm_count

            reward_stats = summarize_world_rewards(rewards)
            row = build_metrics_row(step, regime, m, firm_entries, firm_exits, reward_stats)
            writer.writerow(row)
            all_metrics.append(m)
            reward_history.append(reward_stats)

            if step % RUN_PROGRESS_INTERVAL_WEEKS == 0:
                print(f"  Week {step:4d}/{episode_weeks} | "
                      f"GDP={m['gdp']:.3f} Gini={m['gini']:.3f} "
                      f"Health={m['avg_health']:.3f} Firms={m['num_firms']}")

            if done:
                print(f"  Terminated at week {step}: {info.get('termination_reason', 'normal end')}")
                break

    write_debug_log(world.debug_log, output_dir)

    summary = build_episode_summary(
        regime, world, all_metrics, firm_entries, firm_exits, step + 1, reward_history
    )

    with open(f"{output_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Re-write README at the end so the file index reflects all artifacts.
    try:
        from marl_economy.eval.run_metadata import write_run_readme
        write_run_readme(output_dir, config, kind="Episode")
    except Exception:
        pass
    print(f"\n  Output saved to {output_dir}/")
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run one MARL Economy episode")
    parser.add_argument("--regime", default=DEFAULT_REGIME_NAME, choices=REGIME_NAMES)
    parser.add_argument("--weeks", type=int, default=DEFAULT_EPISODE_WEEKS)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"Running episode: {args.regime} for {args.weeks} weeks")
    summary = run_episode(args.regime, args.weeks, args.seed, args.output)
    print(f"\nSummary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
