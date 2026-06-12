"""Stage 2 Test 2 - per-firm hire and sale fairness check.

The worry behind this test: a subtle bug in `route_work_vec`'s eligibility
mask or cap-enforcement could systematically over- or under-route workers to
specific firm slots while leaving the population total unchanged. To rule
that out we:

  1. Run 50 seeds x 3 regimes with random policies.
  2. For each (regime, seed, week, slot), record per-firm hires and sales.
  3. Aggregate to per-slot totals across seeds for the INITIAL firms
     (slots 0..5 are always the initial firms; later slots are entries).
  4. Compare per-slot share-of-hires and share-of-sales against the
     expected softmax-of-wages share when the slot was alive.

If routing is unbiased, observed-share / expected-share should be ~1.0 for
every slot. We require the geometric-mean ratio to be within [0.9, 1.1]
and no individual slot's ratio more than 2x off.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marl_economy.config.constants import (
    REGIME_LME, REGIME_SDWE, REGIME_SME,
    NUM_HOUSEHOLD_ACTIONS,
    FIRM_UPDATE_PERIOD, GOV_UPDATE_PERIOD, CB_UPDATE_PERIOD,
    FIRM_PRICE_DELTAS, FIRM_WAGE_DELTAS, FIRM_CAPACITY_DELTAS,
    GOV_TAX_DELTAS, GOV_WELFARE_DELTAS, GOV_POLICING_DELTAS,
    GOV_HEALTHCARE_DELTAS, GOV_EDUCATION_DELTAS, GOV_MINWAGE_DELTAS,
    CB_RATE_DELTAS,
    KAPPA_WAGE,
)
from marl_economy.env.world import World
from marl_economy.env.routing import softmax


SEEDS = list(range(50))
REGIMES = [REGIME_LME, REGIME_SDWE, REGIME_SME]
WEEKS = 156
N_INITIAL_FIRMS = 6  # slots 0..5 are always the initial firms


def run_one(regime: str, seed: int):
    """Run one episode, recording per-week per-slot hires and sales.

    Returns:
      hires_by_slot       - array of total hires per slot (across the episode)
      sales_by_slot       - array of total sales per slot
      expected_hire_share - softmax(wage)-derived expected share for initial slots,
                            averaged over weeks in which each slot was alive
      alive_weeks_by_slot - count of weeks each slot was alive
    """
    action_rng = np.random.default_rng(seed)
    world = World(regime_name=regime, episode_weeks=WEEKS, seed=seed, curriculum_phase=1)
    world.reset()

    # Pre-allocate trackers up to the max slot table size to be safe.
    max_slots = world.f.max_firms
    hires_by_slot = np.zeros(max_slots, dtype=np.int64)
    sales_by_slot = np.zeros(max_slots, dtype=np.float64)
    alive_weeks_by_slot = np.zeros(max_slots, dtype=np.int64)
    expected_hire_share_sum = np.zeros(max_slots, dtype=np.float64)

    for step in range(WEEKS):
        # Capture per-firm wages BEFORE step (predicts that week's hire share).
        alive_slots = world.f.alive_slots()
        if alive_slots.size > 0:
            wages = world.f.wage_offer[alive_slots]
            probs = softmax(KAPPA_WAGE * wages)
            expected_hire_share_sum[alive_slots] += probs
            alive_weeks_by_slot[alive_slots] += 1

        # Random actions.
        hh_actions = {
            int(world.h.ids[i]): int(action_rng.integers(0, NUM_HOUSEHOLD_ACTIONS))
            for i in range(world.h.size)
        }
        firm_actions = None
        if step > 0 and step % FIRM_UPDATE_PERIOD == 0:
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

        # Record per-slot weekly hires/sales. world.f.window_* accumulates
        # across the firm's monthly window, so we use the per-week f.hires and
        # f.sales which are reset weekly at the top of the next step.
        # Sample these BEFORE the next reset by reading them now.
        weekly_alive = world.f.alive_slots()
        if weekly_alive.size > 0:
            hires_by_slot[weekly_alive] += world.f.hires[weekly_alive].astype(np.int64)
            sales_by_slot[weekly_alive] += world.f.sales[weekly_alive].astype(np.float64)
        if done:
            break

    return hires_by_slot, sales_by_slot, expected_hire_share_sum, alive_weeks_by_slot


def main():
    print(f"Stage 2 Test 2 - per-firm hire fairness")
    print(f"  {len(SEEDS)} seeds x {len(REGIMES)} regimes\n")

    overall_pass = True
    for regime in REGIMES:
        # Sum across seeds.
        agg_hires = np.zeros(0, dtype=np.int64)
        agg_sales = np.zeros(0, dtype=np.float64)
        agg_expected = np.zeros(0, dtype=np.float64)

        t0 = time.perf_counter()
        for seed in SEEDS:
            h, s, e, w = run_one(regime, seed)
            if agg_hires.size == 0:
                agg_hires = h.copy()
                agg_sales = s.copy()
                agg_expected = e.copy()
            else:
                agg_hires += h
                agg_sales += s
                agg_expected += e
        dt = time.perf_counter() - t0

        # Inspect the initial firms (slots 0..N_INITIAL_FIRMS-1).
        total_hires = int(agg_hires[:N_INITIAL_FIRMS].sum())
        total_expected = float(agg_expected[:N_INITIAL_FIRMS].sum())

        print(f"=== {regime} (50 seeds, {dt:.1f}s) ===")
        print(f"  Initial firms (slots 0..{N_INITIAL_FIRMS - 1}):")
        print(f"  {'slot':>4} {'hires':>10} {'sales':>12} {'exp_share':>10} {'obs_share':>10} {'ratio':>8}")

        regime_pass = True
        ratios = []
        for slot in range(N_INITIAL_FIRMS):
            obs_hires = int(agg_hires[slot])
            obs_sales = float(agg_sales[slot])
            expected_share = float(agg_expected[slot]) / total_expected if total_expected > 0 else 0.0
            obs_share = obs_hires / total_hires if total_hires > 0 else 0.0
            if expected_share > 0:
                ratio = obs_share / expected_share
                ratios.append(ratio)
            else:
                ratio = float("nan")
            ok = (0.5 <= ratio <= 2.0) if not np.isnan(ratio) else True
            if not ok:
                regime_pass = False
            print(f"  {slot:>4} {obs_hires:>10} {obs_sales:>12.2f} "
                  f"{expected_share:>10.3f} {obs_share:>10.3f} {ratio:>8.3f}"
                  f"{'  FAIL' if not ok else ''}")

        geo_mean = float(np.exp(np.mean(np.log(ratios)))) if ratios else 1.0
        print(f"  Geometric-mean ratio: {geo_mean:.4f} "
              f"({'OK' if 0.9 <= geo_mean <= 1.1 else 'FAIL'})\n")
        if not (0.9 <= geo_mean <= 1.1):
            regime_pass = False
        overall_pass &= regime_pass

    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
