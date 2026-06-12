"""Collect distribution-level summary stats across many seeds.

Stage 2 changes the order of RNG consumption inside the routing loop, so per-seed
trajectories will differ - but the STATISTICAL distribution of outcomes should
be unchanged. This script:

  * Captures `stage1` distribution: 50 seeds x 3 regimes, full 156-week episodes
    with random policies. Saves a flat list of per-episode summary dicts.
  * Diffs two captures via KS test on each metric, reporting p-values.

Usage:
  python distribution_baseline.py --out dist_stage1.json
  python distribution_baseline.py --diff dist_stage1.json dist_stage2.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marl_economy.config.constants import (
    REGIME_LME, REGIME_SDWE, REGIME_SME,
    NUM_HOUSEHOLD_ACTIONS,
    FIRM_UPDATE_PERIOD, GOV_UPDATE_PERIOD, CB_UPDATE_PERIOD,
    FIRM_PRICE_DELTAS, FIRM_WAGE_DELTAS, FIRM_CAPACITY_DELTAS,
    GOV_TAX_DELTAS, GOV_WELFARE_DELTAS, GOV_POLICING_DELTAS,
    GOV_HEALTHCARE_DELTAS, GOV_EDUCATION_DELTAS, GOV_MINWAGE_DELTAS,
    CB_RATE_DELTAS,
)
from marl_economy.env.world import World


SEEDS = list(range(50))
REGIMES = [REGIME_LME, REGIME_SDWE, REGIME_SME]
WEEKS = 156


def run_one(regime: str, seed: int, weeks: int = WEEKS) -> dict:
    """Run one episode with random actions and return summary stats."""
    action_rng = np.random.default_rng(seed)
    world = World(
        regime_name=regime,
        episode_weeks=weeks,
        seed=seed,
        curriculum_phase=1,
    )
    world.reset()

    for step in range(weeks):
        hh_actions = {
            int(world.h.ids[i]): int(action_rng.integers(0, NUM_HOUSEHOLD_ACTIONS))
            for i in range(world.h.size)
        }
        firm_actions = None
        if step > 0 and step % FIRM_UPDATE_PERIOD == 0:
            alive_slots = world.f.alive_slots()
            firm_actions = {
                int(world.f.id[slot]): (
                    int(action_rng.integers(0, len(FIRM_PRICE_DELTAS))),
                    int(action_rng.integers(0, len(FIRM_WAGE_DELTAS))),
                    int(action_rng.integers(0, len(FIRM_CAPACITY_DELTAS))),
                )
                for slot in alive_slots.tolist()
            }
        gov_action = None
        if step > 0 and step % GOV_UPDATE_PERIOD == 0:
            gov_action = (
                int(action_rng.integers(0, len(GOV_TAX_DELTAS))),
                int(action_rng.integers(0, len(GOV_WELFARE_DELTAS))),
                int(action_rng.integers(0, len(GOV_POLICING_DELTAS))),
                int(action_rng.integers(0, len(GOV_HEALTHCARE_DELTAS))),
                int(action_rng.integers(0, len(GOV_EDUCATION_DELTAS))),
                int(action_rng.integers(0, len(GOV_MINWAGE_DELTAS))),
            )
        cb_action = None
        if step > 0 and step % CB_UPDATE_PERIOD == 0:
            cb_action = int(action_rng.integers(0, len(CB_RATE_DELTAS)))

        _, _, done, _ = world.step(hh_actions, firm_actions, gov_action, cb_action)
        if done:
            break

    # Episode-level summary.
    metrics = world.metrics.history
    if not metrics:
        return None
    final = metrics[-1]
    weeks_run = len(metrics)

    # Means over episode.
    mean_gdp = float(np.mean([m["gdp"] for m in metrics]))
    mean_gini = float(np.mean([m["gini"] for m in metrics]))
    mean_avg_health = float(np.mean([m["avg_health"] for m in metrics]))
    mean_poverty = float(np.mean([m["poverty"] for m in metrics]))
    mean_num_firms = float(np.mean([m["num_firms"] for m in metrics]))
    total_deaths = int(sum(m.get("weekly_deaths", 0) for m in metrics))
    total_displacements = int(sum(m.get("weekly_displacements", 0) for m in metrics))
    total_hunger_crises = int(sum(m.get("weekly_hunger_crisis", 0) for m in metrics))
    total_steal_success = int(sum(m.get("steal_success", 0) for m in metrics))
    total_work_attempts = int(sum(m.get("work_attempts", 0) for m in metrics))
    total_work_matched = int(sum(m.get("work_matched", 0) for m in metrics))
    total_work_refused = int(sum(m.get("work_refused_solvency", 0) for m in metrics))
    total_shop_matched = int(sum(m.get("shop_matched", 0) for m in metrics))
    total_clinic_matched = int(sum(m.get("clinic_matched", 0) for m in metrics))

    # Final population state.
    final_wealth_mean = float(np.mean(world.h.wealth))
    final_health_mean = float(np.mean(world.h.health))
    final_hunger_mean = float(np.mean(world.h.hunger))
    final_fatigue_mean = float(np.mean(world.h.fatigue))
    final_alive_firms = int(world.f.num_alive)
    final_firm_cash_mean = float(np.mean(world.f.cash[world.f.alive_slots()])) if world.f.num_alive > 0 else 0.0

    return {
        "regime": regime,
        "seed": seed,
        "weeks_run": weeks_run,
        "terminated_early": bool(world.terminated_early),
        "termination_reason": world.termination_reason,
        "mean_gdp": mean_gdp,
        "mean_gini": mean_gini,
        "mean_avg_health": mean_avg_health,
        "mean_poverty": mean_poverty,
        "mean_num_firms": mean_num_firms,
        "total_deaths": total_deaths,
        "total_displacements": total_displacements,
        "total_hunger_crises": total_hunger_crises,
        "total_steal_success": total_steal_success,
        "total_work_attempts": total_work_attempts,
        "total_work_matched": total_work_matched,
        "total_work_refused": total_work_refused,
        "total_shop_matched": total_shop_matched,
        "total_clinic_matched": total_clinic_matched,
        "final_wealth_mean": final_wealth_mean,
        "final_health_mean": final_health_mean,
        "final_hunger_mean": final_hunger_mean,
        "final_fatigue_mean": final_fatigue_mean,
        "final_alive_firms": final_alive_firms,
        "final_firm_cash_mean": final_firm_cash_mean,
        # Final-state GDP can be 0 if firms had no sales last week - keep as a
        # smoothing-free episode-end snapshot for those who want it.
        "final_gdp": float(final["gdp"]),
        "final_gini": float(final["gini"]),
    }


def capture(out_path: Path) -> None:
    rows = []
    t0 = time.perf_counter()
    for regime in REGIMES:
        regime_t0 = time.perf_counter()
        for seed in SEEDS:
            data = run_one(regime, seed)
            if data is not None:
                rows.append(data)
        print(f"  {regime}: {len(SEEDS)} seeds, {time.perf_counter() - regime_t0:.2f}s")
    print(f"Total: {len(rows)} episodes, {time.perf_counter() - t0:.2f}s")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)


# Metrics we KS-test (continuous-valued) vs frequency-test (binary/count).
KS_METRICS = (
    "mean_gdp", "mean_gini", "mean_avg_health", "mean_poverty",
    "mean_num_firms", "final_wealth_mean", "final_health_mean",
    "final_hunger_mean", "final_fatigue_mean", "final_firm_cash_mean",
    "final_gdp", "final_gini",
)
COUNT_METRICS = (
    "weeks_run", "total_deaths", "total_displacements", "total_hunger_crises",
    "total_steal_success", "total_work_attempts", "total_work_matched",
    "total_work_refused", "total_shop_matched", "total_clinic_matched",
    "final_alive_firms",
)


def diff(a_path: str, b_path: str, alpha: float = 0.01) -> int:
    a_rows = json.load(open(a_path))
    b_rows = json.load(open(b_path))

    fail = 0
    for regime in REGIMES:
        a_r = [r for r in a_rows if r["regime"] == regime]
        b_r = [r for r in b_rows if r["regime"] == regime]
        print(f"\n=== {regime}: {len(a_r)} vs {len(b_r)} seeds ===")
        if len(a_r) == 0 or len(b_r) == 0:
            print("  (skipping - empty sample)")
            continue

        # Termination reasons should match in proportion (same simulation
        # behavior).
        a_term = {r["termination_reason"] or "<none>" for r in a_r}
        b_term = {r["termination_reason"] or "<none>" for r in b_r}
        print(f"  termination_reasons:  A={sorted(a_term)} B={sorted(b_term)}")

        for metric in KS_METRICS + COUNT_METRICS:
            a_vals = np.array([r[metric] for r in a_r], dtype=float)
            b_vals = np.array([r[metric] for r in b_r], dtype=float)
            # Two-sample KS test (robust to ties for count data).
            try:
                ks_stat, p = stats.ks_2samp(a_vals, b_vals)
            except Exception as exc:
                print(f"  {metric:<25} ks test failed: {exc}")
                fail += 1
                continue
            a_mean = float(np.mean(a_vals))
            b_mean = float(np.mean(b_vals))
            flag = "OK" if p > alpha else "FAIL"
            if p <= alpha:
                fail += 1
            print(f"  {metric:<25} mean A={a_mean:10.4f} B={b_mean:10.4f} "
                  f"diff={b_mean-a_mean:+9.4f}  KS={ks_stat:.3f} p={p:.3g} [{flag}]")

    print(f"\nSummary: {fail} metric-regime pairs failed KS test at alpha={alpha}")
    return 0 if fail == 0 else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, help="Capture into this JSON file")
    p.add_argument("--diff", nargs=2, metavar=("A", "B"))
    args = p.parse_args()
    if args.diff:
        sys.exit(diff(args.diff[0], args.diff[1]))
    if args.out:
        capture(args.out)
        return
    p.error("specify --out or --diff")


if __name__ == "__main__":
    main()
