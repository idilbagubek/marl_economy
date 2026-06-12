"""Capture per-step and end-of-episode metrics for a fixed seed x regime grid
so the Stage 1 refactor can be diffed bit-for-bit afterwards.

Run BEFORE refactor:   python baseline_capture.py --out baseline_pre_refactor
Run AFTER refactor:    python baseline_capture.py --out baseline_post_refactor
Then compare:          python baseline_capture.py --diff baseline_pre_refactor baseline_post_refactor
"""

import argparse
import json
import os
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
)
from marl_economy.env.world import World


SEEDS = [0, 1, 2, 3, 4]
REGIMES = [REGIME_LME, REGIME_SDWE, REGIME_SME]
WEEKS = 156


def run_one(regime: str, seed: int, weeks: int = WEEKS) -> dict:
    """Run one episode with random actions, return a dict suitable for diffing.

    The action sampler uses a SEPARATE rng seeded with the same seed so that
    refactor diffs only have to worry about world-internal RNG order. The
    world's own RNG is constructed from `seed` inside World.__init__.
    """
    action_rng = np.random.default_rng(seed)
    world = World(
        regime_name=regime,
        episode_weeks=weeks,
        seed=seed,
        curriculum_phase=1,
    )
    world.reset()

    per_week_metrics = []

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
        m = world.metrics.history[-1]
        # Keep a small fixed-key payload so diffs are stable across schema changes.
        per_week_metrics.append({
            "week": int(m["week"]),
            "gdp": float(m["gdp"]),
            "mean_price": float(m["mean_price"]),
            "mean_wage": float(m["mean_wage"]),
            "gini": float(m["gini"]),
            "poverty": float(m["poverty"]),
            "avg_health": float(m["avg_health"]),
            "crime_rate": float(m["crime_rate"]),
            "unemployment": float(m["unemployment"]),
            "num_firms": int(m["num_firms"]),
            "weekly_deaths": int(m.get("weekly_deaths", 0)),
            "weekly_displacements": int(m.get("weekly_displacements", 0)),
        })
        if done:
            break

    # Final population-level state for sanity.
    final_wealths = [float(world.h.wealth[i]) for i in range(world.h.size)]
    final_healths = [float(world.h.health[i]) for i in range(world.h.size)]
    final_hungers = [float(world.h.hunger[i]) for i in range(world.h.size)]
    final_fatigues = [float(world.h.fatigue[i]) for i in range(world.h.size)]
    final_illnesses = [int(world.h.illness[i]) for i in range(world.h.size)]
    final_firm_cash = [float(world.f.cash[s]) for s in range(world.f.size)]
    final_firm_alive = [bool(world.f.alive[s]) for s in range(world.f.size)]
    final_firm_ids = [int(world.f.id[s]) for s in range(world.f.size)]

    return {
        "regime": regime,
        "seed": seed,
        "weeks_run": len(per_week_metrics),
        "terminated_early": bool(world.terminated_early),
        "termination_reason": world.termination_reason,
        "per_week": per_week_metrics,
        "final_wealths": final_wealths,
        "final_healths": final_healths,
        "final_hungers": final_hungers,
        "final_fatigues": final_fatigues,
        "final_illnesses": final_illnesses,
        "final_firm_cash": final_firm_cash,
        "final_firm_alive": final_firm_alive,
        "final_firm_ids": final_firm_ids,
        "event_totals": world.event_ledger.to_summary(),
        "collapse": world.collapse_detector.to_summary(),
    }


def capture(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    total_t = 0.0
    for regime in REGIMES:
        for seed in SEEDS:
            label = f"{regime}_seed{seed}"
            t0 = time.perf_counter()
            data = run_one(regime, seed)
            dt = time.perf_counter() - t0
            total_t += dt
            path = out_dir / f"{label}.json"
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"  {label:<25} weeks={data['weeks_run']:<4} term={data['termination_reason']!s:<35} t={dt:.2f}s")
    print(f"Total baselines: {len(REGIMES) * len(SEEDS)}; total wall {total_t:.2f}s")
    timing = {
        "total_seconds": total_t,
        "num_runs": len(REGIMES) * len(SEEDS),
        "weeks_each": WEEKS,
    }
    with open(out_dir / "_timing.json", "w") as f:
        json.dump(timing, f, indent=2)


def diff(a_dir: Path, b_dir: Path) -> int:
    """Compare two baseline directories key-for-key, value-for-value."""
    a_dir = Path(a_dir)
    b_dir = Path(b_dir)
    a_files = sorted(p for p in a_dir.iterdir() if p.suffix == ".json" and not p.name.startswith("_"))
    b_files = sorted(p for p in b_dir.iterdir() if p.suffix == ".json" and not p.name.startswith("_"))
    a_names = {p.name for p in a_files}
    b_names = {p.name for p in b_files}
    if a_names != b_names:
        print(f"FAIL: file sets differ. only_in_a={a_names - b_names}, only_in_b={b_names - a_names}")
        return 1

    mismatch = 0
    for name in sorted(a_names):
        with open(a_dir / name) as f:
            a = json.load(f)
        with open(b_dir / name) as f:
            b = json.load(f)
        diffs = _diff_payload(a, b, prefix=name)
        if diffs:
            mismatch += 1
            for d in diffs[:20]:
                print(f"  DIFF {d}")
            if len(diffs) > 20:
                print(f"  ... +{len(diffs)-20} more diffs in {name}")
        else:
            print(f"  OK   {name}")
    print()
    if mismatch == 0:
        print(f"ALL {len(a_names)} baselines match bit-exact.")
        return 0
    print(f"FAIL: {mismatch}/{len(a_names)} baselines differ.")
    return 1


def _diff_payload(a, b, prefix: str) -> list[str]:
    """Recursive deep-equal diff. Returns paths that disagree."""
    diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(a.keys()) | set(b.keys())
        for k in sorted(keys):
            if k not in a:
                diffs.append(f"{prefix}.{k}: only in b")
                continue
            if k not in b:
                diffs.append(f"{prefix}.{k}: only in a")
                continue
            diffs.extend(_diff_payload(a[k], b[k], f"{prefix}.{k}"))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{prefix}: list length {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diffs.extend(_diff_payload(x, y, f"{prefix}[{i}]"))
    elif isinstance(a, float) or isinstance(b, float):
        if not (a == b):
            diffs.append(f"{prefix}: {a!r} != {b!r}")
    else:
        if a != b:
            diffs.append(f"{prefix}: {a!r} != {b!r}")
    return diffs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, help="Capture into this dir")
    parser.add_argument("--diff", nargs=2, metavar=("A_DIR", "B_DIR"),
                        help="Diff two captured dirs")
    args = parser.parse_args()
    if args.diff:
        sys.exit(diff(args.diff[0], args.diff[1]))
    if args.out:
        capture(args.out)
        return
    parser.error("specify --out or --diff")


if __name__ == "__main__":
    main()
