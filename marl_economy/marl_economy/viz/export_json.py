"""
Export simulation snapshots to JSON for external visualization.
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from marl_economy.config.constants import (
    DEFAULT_REGIME_NAME, REGIME_NAMES,
    DEFAULT_EPISODE_WEEKS, DEFAULT_RANDOM_SEED,
    FIRM_UPDATE_PERIOD, GOV_UPDATE_PERIOD, CB_UPDATE_PERIOD,
    NUM_HOUSEHOLD_ACTIONS, FIRM_PRICE_DELTAS, FIRM_WAGE_DELTAS,
    FIRM_CAPACITY_DELTAS, GOV_TAX_DELTAS, GOV_WELFARE_DELTAS,
    GOV_POLICING_DELTAS, GOV_HEALTHCARE_DELTAS, GOV_EDUCATION_DELTAS,
    GOV_MINWAGE_DELTAS, CB_RATE_DELTAS,
)
from marl_economy.env.world import World


def _json_default(value):
    """Convert NumPy scalar values before writing snapshots to JSON."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def export_episode_json(regime: str = DEFAULT_REGIME_NAME, weeks: int = DEFAULT_EPISODE_WEEKS,
                        seed: int = DEFAULT_RANDOM_SEED, output_path: str = None) -> str:
    """Export full episode as JSON snapshots."""
    rng = np.random.default_rng(seed)
    world = World(regime_name=regime, episode_weeks=weeks, seed=seed)
    world.reset()

    snapshots = []
    for step in range(weeks):
        actions = {int(world.h.ids[i]): rng.integers(0, NUM_HOUSEHOLD_ACTIONS)
                   for i in range(world.h.size)}

        firm_actions = None
        if step > 0 and step % FIRM_UPDATE_PERIOD == 0:
            alive_slots = world.f.alive_slots()
            firm_actions = {int(world.f.id[slot]): (
                rng.integers(0, len(FIRM_PRICE_DELTAS)),
                rng.integers(0, len(FIRM_WAGE_DELTAS)),
                rng.integers(0, len(FIRM_CAPACITY_DELTAS)),
            ) for slot in alive_slots.tolist()}

        gov_action = None
        cb_action = None
        if step > 0 and step % GOV_UPDATE_PERIOD == 0:
            gov_action = tuple(rng.integers(0, len(d)) for d in [
                GOV_TAX_DELTAS, GOV_WELFARE_DELTAS, GOV_POLICING_DELTAS,
                GOV_HEALTHCARE_DELTAS, GOV_EDUCATION_DELTAS, GOV_MINWAGE_DELTAS])
        if step > 0 and step % CB_UPDATE_PERIOD == 0:
            cb_action = rng.integers(0, len(CB_RATE_DELTAS))

        world.step(actions, firm_actions, gov_action, cb_action)
        snapshots.append(world.get_snapshot())

        if world.done:
            break

    if output_path is None:
        output_path = f"runs/{regime}_export.json"

    with open(output_path, "w") as f:
        json.dump({"regime": regime, "seed": seed, "snapshots": snapshots}, f, default=_json_default)

    print(f"Exported {len(snapshots)} snapshots to {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", default=DEFAULT_REGIME_NAME, choices=REGIME_NAMES)
    parser.add_argument("--weeks", type=int, default=DEFAULT_EPISODE_WEEKS)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    export_episode_json(args.regime, args.weeks, output_path=args.output)
