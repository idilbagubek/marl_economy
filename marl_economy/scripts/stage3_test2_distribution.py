"""Stage 3 Test 2 - K=10 batched runs match single-world distribution.

Setup
-----
* 50 single-world runs: regimes in {LME, SDWE, SME} x seeds 0..49 (150 episodes).
* 50 batched-world trajectories: 5 BatchedWorld(K=10) instances per regime,
  with base seeds 0..4. The 10 worlds within each batch share the rng so
  individual per-world trajectories are correlated WITHIN a batch but not
  ACROSS batches.

KS-test the per-trajectory summary distributions for each regime. Stage 2's
acceptance criterion (alpha=0.01) carries over.

The randomness pattern of BatchedWorld(K>1) is intentionally different from
running K single Worlds: rng draws for K worlds happen in a single batched
call, so any one world inside the batch consumes a different rng sub-
sequence than a same-seed single World would. The DISTRIBUTION of outcomes
is what should match, not per-seed trajectories.
"""

from __future__ import annotations

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
from marl_economy.env.batched_world import BatchedWorld


SINGLE_SEEDS = list(range(50))
BATCH_SEEDS = list(range(5))   # 5 BatchedWorld instances
K_PER_BATCH = 16              # K=16 worlds each -> 80 trajectories total
REGIMES = [REGIME_LME, REGIME_SDWE, REGIME_SME]
WEEKS = 156


def _summarize_single(w: World, metrics_history: list[dict]) -> dict:
    final_mh = metrics_history[-1] if metrics_history else {}
    return _common_summary(
        weeks_run=len(metrics_history),
        mh=metrics_history,
        final_h_wealth=w.h.wealth,
        final_h_health=w.h.health,
        final_h_hunger=w.h.hunger,
        final_h_fatigue=w.h.fatigue,
        final_f_cash=w.f.cash[w.f.alive_slots()] if w.f.num_alive > 0 else np.zeros(0),
        n_alive_firms=w.f.num_alive,
        terminated_early=bool(w.terminated_early),
        termination_reason=w.termination_reason,
    )


def _summarize_batched(bw: BatchedWorld, k: int) -> dict:
    mh = bw.metrics[k].history
    alive_slots = bw.f.alive_slots_in_world(k)
    return _common_summary(
        weeks_run=len(mh),
        mh=mh,
        final_h_wealth=bw.h.wealth[k],
        final_h_health=bw.h.health[k],
        final_h_hunger=bw.h.hunger[k],
        final_h_fatigue=bw.h.fatigue[k],
        final_f_cash=bw.f.cash[k, alive_slots] if alive_slots.size > 0 else np.zeros(0),
        n_alive_firms=int(bw.f.num_alive(k)),
        terminated_early=bool(bw.terminated_early[k]),
        termination_reason=bw.termination_reason[k],
    )


def _common_summary(
    *, weeks_run, mh, final_h_wealth, final_h_health, final_h_hunger,
    final_h_fatigue, final_f_cash, n_alive_firms, terminated_early,
    termination_reason,
):
    mean_gdp = float(np.mean([m["gdp"] for m in mh])) if mh else 0.0
    mean_gini = float(np.mean([m["gini"] for m in mh])) if mh else 0.0
    mean_health = float(np.mean([m["avg_health"] for m in mh])) if mh else 0.0
    mean_poverty = float(np.mean([m["poverty"] for m in mh])) if mh else 0.0
    mean_num_firms = float(np.mean([m["num_firms"] for m in mh])) if mh else 0.0
    total_deaths = int(sum(m.get("weekly_deaths", 0) for m in mh))
    total_disp = int(sum(m.get("weekly_displacements", 0) for m in mh))
    total_hung = int(sum(m.get("weekly_hunger_crisis", 0) for m in mh))
    total_steal = int(sum(m.get("steal_success", 0) for m in mh))
    total_work_atts = int(sum(m.get("work_attempts", 0) for m in mh))
    total_work_matched = int(sum(m.get("work_matched", 0) for m in mh))
    total_work_refused = int(sum(m.get("work_refused_solvency", 0) for m in mh))
    total_shop_matched = int(sum(m.get("shop_matched", 0) for m in mh))
    total_clinic_matched = int(sum(m.get("clinic_matched", 0) for m in mh))
    return {
        "weeks_run": weeks_run,
        "terminated_early": terminated_early,
        "termination_reason": termination_reason,
        "mean_gdp": mean_gdp,
        "mean_gini": mean_gini,
        "mean_avg_health": mean_health,
        "mean_poverty": mean_poverty,
        "mean_num_firms": mean_num_firms,
        "total_deaths": total_deaths,
        "total_displacements": total_disp,
        "total_hunger_crises": total_hung,
        "total_steal_success": total_steal,
        "total_work_attempts": total_work_atts,
        "total_work_matched": total_work_matched,
        "total_work_refused": total_work_refused,
        "total_shop_matched": total_shop_matched,
        "total_clinic_matched": total_clinic_matched,
        "final_wealth_mean": float(np.mean(final_h_wealth)),
        "final_health_mean": float(np.mean(final_h_health)),
        "final_hunger_mean": float(np.mean(final_h_hunger)),
        "final_fatigue_mean": float(np.mean(final_h_fatigue)),
        "final_alive_firms": int(n_alive_firms),
        "final_firm_cash_mean": float(np.mean(final_f_cash)) if final_f_cash.size > 0 else 0.0,
    }


def run_single_episode(regime: str, seed: int) -> dict:
    action_rng = np.random.default_rng(seed)
    w = World(regime_name=regime, episode_weeks=WEEKS, seed=seed, curriculum_phase=1)
    metrics_history = []
    for step in range(WEEKS):
        hh = {i: int(action_rng.integers(0, NUM_HOUSEHOLD_ACTIONS)) for i in range(80)}
        fa = None
        if step > 0 and step % FIRM_UPDATE_PERIOD == 0:
            alive_ids = [int(w.f.id[s]) for s in w.f.alive_slots().tolist()]
            fa = {fid: (
                int(action_rng.integers(0, len(FIRM_PRICE_DELTAS))),
                int(action_rng.integers(0, len(FIRM_WAGE_DELTAS))),
                int(action_rng.integers(0, len(FIRM_CAPACITY_DELTAS))),
            ) for fid in alive_ids}
        ga = None
        if step > 0 and step % GOV_UPDATE_PERIOD == 0:
            ga = tuple(int(action_rng.integers(0, len(d))) for d in [
                GOV_TAX_DELTAS, GOV_WELFARE_DELTAS, GOV_POLICING_DELTAS,
                GOV_HEALTHCARE_DELTAS, GOV_EDUCATION_DELTAS, GOV_MINWAGE_DELTAS,
            ])
        ca = None
        if step > 0 and step % CB_UPDATE_PERIOD == 0:
            ca = int(action_rng.integers(0, len(CB_RATE_DELTAS)))
        _, _, done, info = w.step(hh, fa, ga, ca)
        metrics_history.append(info["metrics"])
        if done:
            break
    return _summarize_single(w, metrics_history)


def run_batched_instance(regime: str, base_seed: int, K: int) -> list[dict]:
    action_rng = np.random.default_rng(base_seed + 1_000_000)  # separate stream from world
    bw = BatchedWorld(K=K, regime_name=regime, episode_weeks=WEEKS,
                      seed=base_seed, curriculum_phase=1)
    for step in range(WEEKS):
        if np.all(bw.done):
            break
        hh_per_world = [
            {i: int(action_rng.integers(0, NUM_HOUSEHOLD_ACTIONS)) for i in range(80)}
            for _ in range(K)
        ]
        fa_list = [None] * K
        if step > 0 and step % FIRM_UPDATE_PERIOD == 0:
            fa_list = []
            for k in range(K):
                alive_ids = [int(bw.f.id[k, s]) for s in bw.f.alive_slots_in_world(k).tolist()]
                fa_list.append({fid: (
                    int(action_rng.integers(0, len(FIRM_PRICE_DELTAS))),
                    int(action_rng.integers(0, len(FIRM_WAGE_DELTAS))),
                    int(action_rng.integers(0, len(FIRM_CAPACITY_DELTAS))),
                ) for fid in alive_ids})
        ga_list = [None] * K
        if step > 0 and step % GOV_UPDATE_PERIOD == 0:
            ga_list = [tuple(int(action_rng.integers(0, len(d))) for d in [
                GOV_TAX_DELTAS, GOV_WELFARE_DELTAS, GOV_POLICING_DELTAS,
                GOV_HEALTHCARE_DELTAS, GOV_EDUCATION_DELTAS, GOV_MINWAGE_DELTAS,
            ]) for _ in range(K)]
        ca_list = [None] * K
        if step > 0 and step % CB_UPDATE_PERIOD == 0:
            ca_list = [int(action_rng.integers(0, len(CB_RATE_DELTAS))) for _ in range(K)]
        bw.step(hh_per_world,
                firm_actions=fa_list if any(x is not None for x in fa_list) else None,
                gov_action=ga_list if any(x is not None for x in ga_list) else None,
                cb_action=ca_list if any(x is not None for x in ca_list) else None)
    return [_summarize_batched(bw, k) for k in range(K)]


KS_METRICS = (
    "mean_gdp", "mean_gini", "mean_avg_health", "mean_poverty",
    "mean_num_firms", "final_wealth_mean", "final_health_mean",
    "final_hunger_mean", "final_fatigue_mean", "final_firm_cash_mean",
)
COUNT_METRICS = (
    "weeks_run", "total_deaths", "total_displacements", "total_hunger_crises",
    "total_steal_success", "total_work_attempts", "total_work_matched",
    "total_work_refused", "total_shop_matched", "total_clinic_matched",
    "final_alive_firms",
)


def main():
    print(f"Stage 3 Test 2 - distribution match: {len(SINGLE_SEEDS)} single vs "
          f"{len(BATCH_SEEDS)} x K={K_PER_BATCH} batched trajectories per regime\n")
    overall_pass = True
    alpha = 0.01
    for regime in REGIMES:
        t0 = time.perf_counter()
        # Single runs.
        single_rows = [run_single_episode(regime, s) for s in SINGLE_SEEDS]
        single_t = time.perf_counter() - t0

        # Batched runs (5 instances x K=10).
        t0 = time.perf_counter()
        batched_rows: list[dict] = []
        for base_seed in BATCH_SEEDS:
            batched_rows.extend(run_batched_instance(regime, base_seed, K_PER_BATCH))
        batched_t = time.perf_counter() - t0

        print(f"=== {regime}: single={len(single_rows)} ({single_t:.1f}s) "
              f"vs batched={len(batched_rows)} ({batched_t:.1f}s, "
              f"speedup={single_t/max(batched_t, 1e-9):.2f}x) ===")
        regime_fail = False
        for metric in KS_METRICS + COUNT_METRICS:
            a_vals = np.array([r[metric] for r in single_rows], dtype=float)
            b_vals = np.array([r[metric] for r in batched_rows], dtype=float)
            ks, p = stats.ks_2samp(a_vals, b_vals)
            a_mean = float(np.mean(a_vals))
            b_mean = float(np.mean(b_vals))
            ok = p > alpha
            if not ok:
                regime_fail = True
            print(f"  {metric:<25} single_mean={a_mean:10.4f} batched_mean={b_mean:10.4f} "
                  f"diff={b_mean - a_mean:+9.4f} KS={ks:.3f} p={p:.3g} "
                  f"[{'OK' if ok else 'FAIL'}]")
        overall_pass &= not regime_fail
        print()
    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
