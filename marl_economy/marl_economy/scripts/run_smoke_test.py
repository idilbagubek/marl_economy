"""
Smoke test: run random actions for 1000 steps and assert invariants.
Section 10.1: Before training anything, verify no NaNs, bounded values,
consistent firm updates, stable entry/exit, and computed metrics.
Section 17.7: Implementation checklist assertions.
"""

import sys
import os
import numpy as np
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from marl_economy.config.constants import (
    DEFAULT_REGIME_NAME, REGIME_NAMES,
    DEFAULT_NUM_HOUSEHOLDS, DEFAULT_NUM_FIRMS,
    DEFAULT_RANDOM_SEED, DEFAULT_WARMUP_WEEKS,
    FIRM_UPDATE_PERIOD, GOV_UPDATE_PERIOD, CB_UPDATE_PERIOD,
    F_MIN, F_MAX, DEATH_THRESHOLD, NUM_HOUSEHOLD_ACTIONS,
    CB_RATE_MIN, CB_RATE_MAX,
    SMOKE_TEST_DEFAULT_STEPS, SMOKE_TEST_EPISODE_BUFFER_WEEKS,
    SMOKE_TEST_MIN_EPISODE_WEEKS, SMOKE_TEST_LOG_INTERVAL_STEPS,
    SMOKE_TEST_MAX_ABS_REWARD,
    FIRM_PRICE_DELTAS, FIRM_WAGE_DELTAS, FIRM_CAPACITY_DELTAS,
    GOV_TAX_DELTAS, GOV_WELFARE_DELTAS, GOV_POLICING_DELTAS,
    GOV_HEALTHCARE_DELTAS, GOV_EDUCATION_DELTAS, GOV_MINWAGE_DELTAS,
    CB_RATE_DELTAS,
)
from marl_economy.env.world import World
from marl_economy.env.metrics import compute_top_shares


def run_smoke_test(regime: str = DEFAULT_REGIME_NAME, steps: int = SMOKE_TEST_DEFAULT_STEPS,
                   num_households: int = DEFAULT_NUM_HOUSEHOLDS, num_firms: int = DEFAULT_NUM_FIRMS,
                   seed: int = DEFAULT_RANDOM_SEED, verbose: bool = True):
    """
    Run smoke test with random actions. Returns True if all assertions pass.
    """
    print(f"{'='*60}")
    print(f"SMOKE TEST: regime={regime}, steps={steps}, seed={seed}")
    print(f"{'='*60}")

    rng = np.random.default_rng(seed)

    world = World(
        regime_name=regime,
        num_households=num_households,
        num_firms=num_firms,
        episode_weeks=max(steps + SMOKE_TEST_EPISODE_BUFFER_WEEKS, SMOKE_TEST_MIN_EPISODE_WEEKS),
        warmup_weeks=DEFAULT_WARMUP_WEEKS,
        seed=seed,
        curriculum_phase=1,
    )

    obs = world.reset()
    assertions_passed = 0
    assertions_failed = 0
    total_assertions = 0

    def assert_check(condition, message):
        nonlocal assertions_passed, assertions_failed, total_assertions
        total_assertions += 1
        if condition:
            assertions_passed += 1
        else:
            assertions_failed += 1
            print(f"  FAIL: {message}")

    for step in range(steps):
        # Generate random actions
        household_actions = {
            int(world.h.ids[i]): rng.integers(0, NUM_HOUSEHOLD_ACTIONS)
            for i in range(world.h.size)
        }

        # Firm actions (every 4 weeks)
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

        # Gov action (every 12 weeks)
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

        # CB action (every 12 weeks)
        cb_action = None
        if step > 0 and step % CB_UPDATE_PERIOD == 0:
            cb_action = rng.integers(0, len(CB_RATE_DELTAS))

        obs, rewards, done, info = world.step(
            household_actions, firm_actions, gov_action, cb_action
        )

        # Invariant checks every step.

        # 1. No NaN in household obs
        for hid, ob in obs["households"].items():
            assert_check(not np.any(np.isnan(ob)),
                         f"Step {step}: NaN in household {hid} obs")

        # 2. Wealth and health bounded
        for i in range(world.h.size):
            hid = int(world.h.ids[i])
            wealth = float(world.h.wealth[i])
            health = float(world.h.health[i])
            last_action = int(world.h.last_action[i])
            hunger = float(world.h.hunger[i])
            assert_check(wealth >= 0,
                         f"Step {step}: negative wealth h_{hid}: {wealth}")
            assert_check(0 <= health <= 1.0,
                         f"Step {step}: health out of bounds h_{hid}: {health}")
            assert_check(0 <= last_action < NUM_HOUSEHOLD_ACTIONS,
                         f"Step {step}: invalid household action marker h_{hid}: {last_action}")
            assert_check(0.0 <= hunger <= 1.0,
                         f"Step {step}: hunger out of bounds h_{hid}: {hunger}")

        # 3. Firm cash is finite
        for slot in world.f.alive_slots().tolist():
            fid = int(world.f.id[slot])
            cash = float(world.f.cash[slot])
            assert_check(np.isfinite(cash),
                         f"Step {step}: non-finite firm cash f_{fid}: {cash}")

        # 4. No NaN in metrics
        if info["metrics"]:
            for k, v in info["metrics"].items():
                if isinstance(v, float):
                    assert_check(not np.isnan(v),
                                 f"Step {step}: NaN in metric {k}")

        # 4b. Top-share helper remains finite even for current population size
        top_shares = compute_top_shares(world.h.wealth)
        for k, v in top_shares.items():
            assert_check(np.isfinite(v),
                         f"Step {step}: non-finite top-share metric {k}: {v}")

        # 4c. Rewards must be finite and not silently explode.
        reward_items = []
        reward_items.extend((f"household:{hid}", rew) for hid, rew in rewards.get("households", {}).items())
        reward_items.extend((f"firm:{fid}", rew) for fid, rew in rewards.get("firms", {}).items())
        reward_items.append(("gov", rewards.get("gov", 0.0)))
        reward_items.append(("cb", rewards.get("cb", 0.0)))
        for reward_name, reward_value in reward_items:
            reward_value = float(reward_value)
            assert_check(np.isfinite(reward_value),
                         f"Step {step}: non-finite reward {reward_name}: {reward_value}")
            assert_check(abs(reward_value) <= SMOKE_TEST_MAX_ABS_REWARD,
                         f"Step {step}: reward explosion {reward_name}: {reward_value}")

        # 5. Firm count within bounds (F_min, F_max)
        alive_count = world.f.num_alive
        assert_check(alive_count >= F_MIN,
                      f"Step {step}: firm count {alive_count} below F_MIN={F_MIN}")
        assert_check(alive_count <= F_MAX,
                      f"Step {step}: firm count {alive_count} above F_MAX={F_MAX}")

        # 6. Routing never references dead firm (check last_routed_firm_id)
        alive_ids = {int(world.f.id[s]) for s in world.f.alive_slots().tolist()}
        for i in range(world.h.size):
            rf = int(world.h.last_routed_firm_id[i])
            if rf >= 0:
                assert_check(rf in alive_ids,
                             f"Step {step}: h_{int(world.h.ids[i])} routed to dead firm {rf}")

        # 7. No NaN in firm obs
        for fid, ob in obs.get("firms", {}).items():
            assert_check(not np.any(np.isnan(ob)),
                         f"Step {step}: NaN in firm {fid} obs")

        # 8. Gov/CB obs not NaN
        if obs.get("gov") is not None:
            assert_check(not np.any(np.isnan(obs["gov"])),
                         f"Step {step}: NaN in gov obs")
        if obs.get("cb") is not None:
            assert_check(not np.any(np.isnan(obs["cb"])),
                         f"Step {step}: NaN in cb obs")

        # 9. Central-bank rate remains in realistic configured bounds
        assert_check(CB_RATE_MIN <= world.interest_rate <= CB_RATE_MAX,
                     f"Step {step}: interest rate out of bounds: {world.interest_rate}")

        # Print progress
        if verbose and step % SMOKE_TEST_LOG_INTERVAL_STEPS == 0:
            m = info["metrics"]
            print(f"  Week {step:4d} | GDP={m['gdp']:.4f} Gini={m['gini']:.4f} "
                  f"Health={m['avg_health']:.4f} Infl={m['inflation']:.4f} "
                  f"Crime={m['crime_rate']:.4f} Firms={m['num_firms']}")

        if done:
            print(f"  Episode terminated at step {step}: {info.get('termination_reason', 'normal')}")
            break

    # Final summary.
    print(f"\n{'='*60}")
    print(f"SMOKE TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Total assertions: {total_assertions}")
    print(f"  Passed: {assertions_passed}")
    print(f"  Failed: {assertions_failed}")
    print(f"  Steps completed: {min(step + 1, steps)}")
    print(f"  Final firms alive: {world.f.num_alive}")
    print(f"  Debug log entries: {len(world.debug_log)}")

    if world.debug_log:
        print(f"\n  Last 5 debug entries:")
        for entry in world.debug_log[-5:]:
            print(f"    {entry}")

    success = assertions_failed == 0
    print(f"\n  {'PASSED' if success else 'FAILED'}")
    return success


def main():
    """Run smoke tests for all regimes."""
    import argparse
    parser = argparse.ArgumentParser(description="MARL Economy Smoke Test")
    parser.add_argument("--regime", default=None, choices=REGIME_NAMES, help="Single regime to test")
    parser.add_argument("--steps", type=int, default=SMOKE_TEST_DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()

    regimes = [args.regime] if args.regime else list(REGIME_NAMES)
    all_passed = True

    for regime in regimes:
        passed = run_smoke_test(regime, steps=args.steps, seed=args.seed)
        if not passed:
            all_passed = False
        print()

    if all_passed:
        print("ALL SMOKE TESTS PASSED")
    else:
        print("SOME SMOKE TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
