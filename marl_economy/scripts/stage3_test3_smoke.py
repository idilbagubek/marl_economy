"""Stage 3 Test 3 - BatchedWorld smoke test across all 4 regimes at K=8.

Runs 200 weeks per regime with random actions, asserts invariants on every
world and every step:

  * Household wealth in [0, inf); health in [0, 1]; hunger in [0, 1]
  * Firm cash finite, firm count in [F_MIN, F_MAX]
  * routed_firm_id (when set) references an alive firm in the same world
  * Rewards finite, |reward| under SMOKE_TEST_MAX_ABS_REWARD
  * Per-world metrics contain no NaN

Mirrors `scripts.run_smoke_test` for parity, but exercises BatchedWorld.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marl_economy.config.constants import (
    REGIME_NAMES, NUM_HOUSEHOLD_ACTIONS,
    FIRM_UPDATE_PERIOD, GOV_UPDATE_PERIOD, CB_UPDATE_PERIOD,
    FIRM_PRICE_DELTAS, FIRM_WAGE_DELTAS, FIRM_CAPACITY_DELTAS,
    GOV_TAX_DELTAS, GOV_WELFARE_DELTAS, GOV_POLICING_DELTAS,
    GOV_HEALTHCARE_DELTAS, GOV_EDUCATION_DELTAS, GOV_MINWAGE_DELTAS,
    CB_RATE_DELTAS,
    F_MIN, F_MAX, NUM_ILLNESS_STATES,
    SMOKE_TEST_DEFAULT_STEPS, SMOKE_TEST_EPISODE_BUFFER_WEEKS,
    SMOKE_TEST_MIN_EPISODE_WEEKS, SMOKE_TEST_MAX_ABS_REWARD,
    DEATH_THRESHOLD,
)
from marl_economy.env.batched_world import BatchedWorld


K = 8
STEPS = SMOKE_TEST_DEFAULT_STEPS


def run_smoke(regime: str):
    rng = np.random.default_rng(42)
    bw = BatchedWorld(
        regime_name=regime, K=K,
        episode_weeks=max(STEPS + SMOKE_TEST_EPISODE_BUFFER_WEEKS, SMOKE_TEST_MIN_EPISODE_WEEKS),
        seed=42, curriculum_phase=1,
    )

    passed = 0
    failed = 0

    def check(cond, msg):
        nonlocal passed, failed
        if cond:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: {msg}")

    for step in range(STEPS):
        if np.all(bw.done):
            break
        hh_per = [
            {i: int(rng.integers(0, NUM_HOUSEHOLD_ACTIONS)) for i in range(bw.num_households)}
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
        obs, rewards, done, info = bw.step(hh_per, fa, ga, ca)

        for k in range(K):
            if done[k]:
                continue
            # Household invariants
            check(np.all(bw.h.wealth[k] >= 0),
                  f"week {step} world {k}: negative wealth detected")
            check(np.all((0 <= bw.h.health[k]) & (bw.h.health[k] <= 1.0)),
                  f"week {step} world {k}: health out of [0,1]")
            check(np.all((0 <= bw.h.hunger[k]) & (bw.h.hunger[k] <= 1.0)),
                  f"week {step} world {k}: hunger out of [0,1]")
            check(np.all((0 <= bw.h.fatigue[k]) & (bw.h.fatigue[k] <= 1.0)),
                  f"week {step} world {k}: fatigue out of [0,1]")
            check(np.all((0 <= bw.h.illness[k]) & (bw.h.illness[k] < NUM_ILLNESS_STATES)),
                  f"week {step} world {k}: illness state out of range")
            # Firm invariants
            alive_slots = bw.f.alive_slots_in_world(k)
            check(np.all(np.isfinite(bw.f.cash[k, alive_slots])),
                  f"week {step} world {k}: non-finite firm cash")
            alive_count = int(alive_slots.size)
            check(F_MIN <= alive_count <= F_MAX,
                  f"week {step} world {k}: firm count {alive_count} out of [F_MIN={F_MIN}, F_MAX={F_MAX}]")
            # Routing references alive firms.
            alive_ids = {int(bw.f.id[k, s]) for s in alive_slots.tolist()}
            for i in range(bw.num_households):
                rf = int(bw.h.last_routed_firm_id[k, i])
                if rf >= 0:
                    check(rf in alive_ids,
                          f"week {step} world {k}: household {i} routed to dead firm {rf}")
            # Rewards
            for hid, rv in rewards["households"][k].items():
                check(np.isfinite(rv) and abs(rv) <= SMOKE_TEST_MAX_ABS_REWARD,
                      f"week {step} world {k}: household {hid} reward {rv} unbounded")
            for fid, rv in rewards["firms"][k].items():
                check(np.isfinite(rv) and abs(rv) <= SMOKE_TEST_MAX_ABS_REWARD,
                      f"week {step} world {k}: firm {fid} reward {rv} unbounded")
            # Metric finiteness
            mh = info["metrics"][k]
            if mh:
                for kk, vv in mh.items():
                    if isinstance(vv, float):
                        check(not np.isnan(vv),
                              f"week {step} world {k}: NaN in metric {kk}")

    return passed, failed, bw


def main():
    total_passed = 0
    total_failed = 0
    for regime in REGIME_NAMES:
        print(f"=== {regime} smoke test (K={K}, steps={STEPS}) ===")
        t0 = time.perf_counter()
        p, f, bw = run_smoke(regime)
        dt = time.perf_counter() - t0
        total_passed += p
        total_failed += f
        alive_now = [int(bw.f.num_alive(k)) for k in range(K)]
        term_reasons = [bw.termination_reason[k] for k in range(K)]
        print(f"  assertions passed={p} failed={f}  wall={dt:.1f}s")
        print(f"  per-world alive firms: {alive_now}")
        print(f"  per-world termination_reason: {term_reasons}")
    print()
    print(f"OVERALL: passed={total_passed} failed={total_failed} "
          f"{'PASS' if total_failed == 0 else 'FAIL'}")
    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
