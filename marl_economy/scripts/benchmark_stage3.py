"""Stage 3 benchmark - single-world sequential vs BatchedWorld parallel.

Two equivalent workloads compared:

  A. Single-world serial:  K independent `World` instances, looped through
     one at a time, K * 156 weeks total.
  B. Batched parallel:     One `BatchedWorld(K=K)`, 156 weeks (K trajectories
     in parallel, all stepped per call).

Reports per-K wall time and effective speedup. Uses random LME actions.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marl_economy.config.constants import (
    REGIME_LME, NUM_HOUSEHOLD_ACTIONS,
    FIRM_UPDATE_PERIOD, GOV_UPDATE_PERIOD, CB_UPDATE_PERIOD,
    FIRM_PRICE_DELTAS, FIRM_WAGE_DELTAS, FIRM_CAPACITY_DELTAS,
    GOV_TAX_DELTAS, GOV_WELFARE_DELTAS, GOV_POLICING_DELTAS,
    GOV_HEALTHCARE_DELTAS, GOV_EDUCATION_DELTAS, GOV_MINWAGE_DELTAS,
    CB_RATE_DELTAS,
)
from marl_economy.env.world import World
from marl_economy.env.batched_world import BatchedWorld


WEEKS = 156


def run_single(seed: int) -> float:
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    w = World(regime_name=REGIME_LME, episode_weeks=WEEKS, seed=seed, curriculum_phase=1)
    for step in range(WEEKS):
        hh = {i: int(rng.integers(0, NUM_HOUSEHOLD_ACTIONS)) for i in range(80)}
        fa = None
        if step > 0 and step % FIRM_UPDATE_PERIOD == 0:
            alive_ids = [int(w.f.id[s]) for s in w.f.alive_slots().tolist()]
            fa = {fid: (
                int(rng.integers(0, len(FIRM_PRICE_DELTAS))),
                int(rng.integers(0, len(FIRM_WAGE_DELTAS))),
                int(rng.integers(0, len(FIRM_CAPACITY_DELTAS))),
            ) for fid in alive_ids}
        ga = None
        if step > 0 and step % GOV_UPDATE_PERIOD == 0:
            ga = tuple(int(rng.integers(0, len(d))) for d in [
                GOV_TAX_DELTAS, GOV_WELFARE_DELTAS, GOV_POLICING_DELTAS,
                GOV_HEALTHCARE_DELTAS, GOV_EDUCATION_DELTAS, GOV_MINWAGE_DELTAS,
            ])
        ca = None
        if step > 0 and step % CB_UPDATE_PERIOD == 0:
            ca = int(rng.integers(0, len(CB_RATE_DELTAS)))
        _, _, done, _ = w.step(hh, fa, ga, ca)
        if done:
            break
    return time.perf_counter() - t0


def run_batched(K: int, base_seed: int = 0) -> float:
    t0 = time.perf_counter()
    rng = np.random.default_rng(base_seed + 1_000_000)
    bw = BatchedWorld(K=K, regime_name=REGIME_LME, episode_weeks=WEEKS,
                      seed=base_seed, curriculum_phase=1)
    for step in range(WEEKS):
        if np.all(bw.done):
            break
        hh_per = [
            {i: int(rng.integers(0, NUM_HOUSEHOLD_ACTIONS)) for i in range(80)}
            for _ in range(K)
        ]
        fa = None
        if step > 0 and step % FIRM_UPDATE_PERIOD == 0:
            fa = []
            for k in range(K):
                alive_ids = [int(bw.f.id[k, s]) for s in bw.f.alive_slots_in_world(k).tolist()]
                fa.append({fid: (
                    int(rng.integers(0, len(FIRM_PRICE_DELTAS))),
                    int(rng.integers(0, len(FIRM_WAGE_DELTAS))),
                    int(rng.integers(0, len(FIRM_CAPACITY_DELTAS))),
                ) for fid in alive_ids})
        ga = None
        if step > 0 and step % GOV_UPDATE_PERIOD == 0:
            ga = [tuple(int(rng.integers(0, len(d))) for d in [
                GOV_TAX_DELTAS, GOV_WELFARE_DELTAS, GOV_POLICING_DELTAS,
                GOV_HEALTHCARE_DELTAS, GOV_EDUCATION_DELTAS, GOV_MINWAGE_DELTAS,
            ]) for _ in range(K)]
        ca = None
        if step > 0 and step % CB_UPDATE_PERIOD == 0:
            ca = [int(rng.integers(0, len(CB_RATE_DELTAS))) for _ in range(K)]
        bw.step(hh_per, fa, ga, ca)
    return time.perf_counter() - t0


def main():
    Ks = [1, 4, 8, 16, 32, 64]
    print(f"Stage 3 benchmark - LME, {WEEKS} weeks per episode, random actions\n")

    # Per-K timing.
    print(f"  {'K':>4} | {'single x K total':>15} | {'batched K-parallel':>20} | {'speedup':>8}")
    print(f"  {'-'*4}-+-{'-'*15}-+-{'-'*20}-+-{'-'*8}")

    for K in Ks:
        # Single: run K episodes sequentially with seeds 0..K-1
        single_total = sum(run_single(s) for s in range(K))
        # Batched: K parallel worlds in one BatchedWorld instance
        batched_t = run_batched(K, base_seed=0)
        speedup = single_total / max(batched_t, 1e-9)
        print(f"  {K:>4} | {single_total:>14.2f}s | {batched_t:>19.2f}s | {speedup:>7.2f}x")


if __name__ == "__main__":
    main()
