"""Benchmark a long-running mixed-action episode so the inner loop dominates wall time."""

import time
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marl_economy.config.constants import (
    REGIME_LME,
    ACTION_WORK, ACTION_SHOP, ACTION_REST, ACTION_INVEST, ACTION_CLINIC,
    FIRM_UPDATE_PERIOD, GOV_UPDATE_PERIOD, CB_UPDATE_PERIOD,
    FIRM_PRICE_DELTAS, FIRM_WAGE_DELTAS, FIRM_CAPACITY_DELTAS,
    GOV_TAX_DELTAS, GOV_WELFARE_DELTAS, GOV_POLICING_DELTAS,
    GOV_HEALTHCARE_DELTAS, GOV_EDUCATION_DELTAS, GOV_MINWAGE_DELTAS,
    CB_RATE_DELTAS, FIRM_NEUTRAL_ACTION,
)
from marl_economy.env.world import World


def run_long_episode(seed: int, weeks: int = 156) -> float:
    """Run a long episode with a mixed action policy that mostly keeps agents alive."""
    rng = np.random.default_rng(seed)
    # Heavy bias toward shop+work, no theft/rest-only sequences that kill the pop fast.
    mix = [ACTION_WORK, ACTION_WORK, ACTION_SHOP, ACTION_SHOP, ACTION_SHOP,
           ACTION_REST, ACTION_INVEST, ACTION_CLINIC]
    mix = np.array(mix, dtype=np.int32)

    w = World(regime_name=REGIME_LME, episode_weeks=weeks, seed=seed)
    w.reset()

    t0 = time.perf_counter()
    for step in range(weeks):
        # Bias toward survival-friendly actions.
        actions_arr = rng.choice(mix, size=w.h.size)
        hh_actions = {int(w.h.ids[i]): int(actions_arr[i]) for i in range(w.h.size)}

        firm_actions = None
        if step > 0 and step % FIRM_UPDATE_PERIOD == 0:
            alive_slots = w.f.alive_slots()
            firm_actions = {int(w.f.id[s]): FIRM_NEUTRAL_ACTION for s in alive_slots.tolist()}

        gov_action = None
        cb_action = None
        if step > 0 and step % GOV_UPDATE_PERIOD == 0:
            gov_action = (2, 2, 2, 2, 2, 1)  # neutral / no-change indices
        if step > 0 and step % CB_UPDATE_PERIOD == 0:
            cb_action = 2  # neutral

        _, _, done, _ = w.step(hh_actions, firm_actions, gov_action, cb_action)
        if done:
            break
    dt = time.perf_counter() - t0
    return dt, w.week, w.terminated_early, w.termination_reason


if __name__ == "__main__":
    times = []
    for seed in range(5):
        dt, weeks, term, reason = run_long_episode(seed)
        print(f"seed={seed} weeks={weeks:<4} term={term} reason={reason!s:<30} t={dt:.3f}s")
        times.append(dt)
    mean = sum(times) / len(times)
    print(f"\nMean: {mean:.3f}s per episode")
    print(f"Steps/sec (assuming 156 weeks): {156 / mean:.1f}")
