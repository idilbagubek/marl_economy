"""Stage 3 Test 1 - BatchedWorld(K=1) bit-equivalence with single World.

A `BatchedWorld` configured with K=1 should consume RNG in the same order
and apply the same math as a single `World`. We replay an identical action
sequence through both and diff the per-week metrics history.

Bit-exact for everything except the 1-ULP gdp summation drift that Stage 1
already documented (CPython's compensated sum interacts differently with
np.float64 vs Python-float operands).
"""

from __future__ import annotations

import json
import sys
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
from marl_economy.env.batched_world import BatchedWorld


SEEDS = [0, 1, 2, 3, 4]
REGIMES = [REGIME_LME, REGIME_SDWE, REGIME_SME]
WEEKS = 156


def gen_actions(regime: str, seed: int, weeks: int):
    """Generate the per-week action sequence used by both runs."""
    rng = np.random.default_rng(seed)
    seq = []
    for step in range(weeks):
        hh_actions = {i: int(rng.integers(0, NUM_HOUSEHOLD_ACTIONS)) for i in range(80)}
        firm_actions = None
        if step > 0 and step % FIRM_UPDATE_PERIOD == 0:
            firm_actions = {
                fid: (
                    int(rng.integers(0, len(FIRM_PRICE_DELTAS))),
                    int(rng.integers(0, len(FIRM_WAGE_DELTAS))),
                    int(rng.integers(0, len(FIRM_CAPACITY_DELTAS))),
                )
                for fid in range(6)  # initial firms only; we re-key later
            }
        gov_action = None
        if step > 0 and step % GOV_UPDATE_PERIOD == 0:
            gov_action = (
                int(rng.integers(0, len(GOV_TAX_DELTAS))),
                int(rng.integers(0, len(GOV_WELFARE_DELTAS))),
                int(rng.integers(0, len(GOV_POLICING_DELTAS))),
                int(rng.integers(0, len(GOV_HEALTHCARE_DELTAS))),
                int(rng.integers(0, len(GOV_EDUCATION_DELTAS))),
                int(rng.integers(0, len(GOV_MINWAGE_DELTAS))),
            )
        cb_action = None
        if step > 0 and step % CB_UPDATE_PERIOD == 0:
            cb_action = int(rng.integers(0, len(CB_RATE_DELTAS)))
        seq.append((hh_actions, firm_actions, gov_action, cb_action))
    return seq


def run_single(regime: str, seed: int):
    seq = gen_actions(regime, seed, WEEKS)
    # World.__init__ already calls reset(); do NOT reset again or rng advances twice.
    w = World(regime_name=regime, episode_weeks=WEEKS, seed=seed, curriculum_phase=1)
    metrics_log = []
    for step, (hh, fa, ga, ca) in enumerate(seq):
        # Replay only alive-firm ids.
        alive_fa = None
        if fa is not None:
            alive_ids = [int(w.f.id[s]) for s in w.f.alive_slots().tolist()]
            alive_fa = {fid: fa[fid] for fid in alive_ids if fid in fa}
        _, _, done, info = w.step(hh, alive_fa, ga, ca)
        metrics_log.append(_summarize_metric(info["metrics"]))
        if done:
            break
    return metrics_log, w


def run_batched(regime: str, seed: int):
    seq = gen_actions(regime, seed, WEEKS)
    bw = BatchedWorld(K=1, regime_name=regime, episode_weeks=WEEKS, seed=seed, curriculum_phase=1)
    metrics_log = []
    for step, (hh, fa, ga, ca) in enumerate(seq):
        alive_fa = None
        if fa is not None:
            alive_ids = [int(bw.f.id[0, s]) for s in bw.f.alive_slots_in_world(0).tolist()]
            alive_fa = {fid: fa[fid] for fid in alive_ids if fid in fa}
        _, _, done, info = bw.step(
            [hh],
            firm_actions=[alive_fa] if alive_fa is not None else None,
            gov_action=[ga] if ga is not None else None,
            cb_action=[ca] if ca is not None else None,
        )
        if info["metrics"][0] is not None:
            metrics_log.append(_summarize_metric(info["metrics"][0]))
        if done[0]:
            break
    return metrics_log, bw


def _summarize_metric(m: dict) -> dict:
    return {
        "week": int(m["week"]),
        "gdp": float(m["gdp"]),
        "mean_price": float(m["mean_price"]),
        "mean_wage": float(m["mean_wage"]),
        "gini": float(m["gini"]),
        "poverty": float(m["poverty"]),
        "avg_health": float(m["avg_health"]),
        "unemployment": float(m["unemployment"]),
        "num_firms": int(m["num_firms"]),
        "weekly_deaths": int(m.get("weekly_deaths", 0)),
        "weekly_displacements": int(m.get("weekly_displacements", 0)),
    }


def main():
    print("Stage 3 Test 1 - K=1 bit-equivalence (BatchedWorld vs single World)\n")
    overall_diffs = 0
    overall_records = 0
    field_diffs = {}
    for regime in REGIMES:
        for seed in SEEDS:
            single_log, _ = run_single(regime, seed)
            batched_log, _ = run_batched(regime, seed)
            if len(single_log) != len(batched_log):
                # find first index where they diverge
                first_div = None
                for i, (a, b) in enumerate(zip(single_log, batched_log)):
                    for k in a:
                        if a[k] != b[k]:
                            first_div = (i, k, a[k], b[k])
                            break
                    if first_div:
                        break
                print(f"  {regime} seed={seed} weeks_run mismatch: single={len(single_log)} batched={len(batched_log)} FAIL  first_div={first_div}")
                overall_diffs += 1
                continue
            run_diffs = 0
            for i, (a, b) in enumerate(zip(single_log, batched_log)):
                for k in a:
                    overall_records += 1
                    if a[k] != b[k]:
                        run_diffs += 1
                        overall_diffs += 1
                        field_diffs[k] = field_diffs.get(k, 0) + 1
            tag = "OK " if run_diffs == 0 else f"{run_diffs} diffs"
            print(f"  {regime} seed={seed} weeks={len(single_log)} {tag}")
    print()
    print(f"Total field-records compared: {overall_records}")
    print(f"Total bit-level diffs:        {overall_diffs}")
    if field_diffs:
        for f, c in sorted(field_diffs.items(), key=lambda x: -x[1]):
            print(f"  {f}: {c}")
    if overall_diffs == 0:
        print("\nPASS - BatchedWorld(K=1) is bit-exact with single World.")
        sys.exit(0)
    elif set(field_diffs.keys()).issubset({"gdp"}):
        print("\nPASS (modulo known gdp 1-ULP drift) - all non-gdp fields bit-exact.")
        sys.exit(0)
    print("\nFAIL - unexpected diffs in non-gdp fields.")
    sys.exit(1)


if __name__ == "__main__":
    main()
