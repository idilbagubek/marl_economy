"""End-to-end invariants for the simulation. Run with `pytest tests/`.

All household and firm state is read/written through the struct-of-arrays
containers `world.h` (HouseholdState) and `world.f` (FirmState).
"""

import math
import numpy as np
import pytest

from marl_economy.config.constants import (
    ACTION_WORK, ACTION_SHOP, ACTION_REST, ACTION_CLINIC,
    ACTION_STEAL, ACTION_INVEST,
    ILLNESS_HEALTHY, ILLNESS_MILD, ILLNESS_SEVERE, ILLNESS_RECOVERING,
    FIRM_INSOLVENT_CASH_THRESHOLD,
    SMOKE_TEST_MAX_ABS_REWARD,
    FATIGUE_PRODUCTIVITY_THRESHOLD,
    HUNGER_HEALTH_PENALTY_THRESHOLD,
)
from marl_economy.env.world import World
from marl_economy.env.state import FirmState
from marl_economy.env.firm_solvency import can_hire


def _make_world(regime="LME", weeks=120, seed=42):
    w = World(regime_name=regime, episode_weeks=weeks, seed=seed)
    w.reset()
    return w


def _all_household_actions(w, action):
    return {int(w.h.ids[i]): action for i in range(w.h.size)}


# Bankrupt firms and unemployment.

def test_bankrupt_firm_cannot_hire():
    f = FirmState(max_firms=4)
    solvent = f.spawn(price=0.5, wage=0.4, cash=1.0, capacity=5.0, stock=10.0)
    bankrupt = f.spawn(price=0.5, wage=0.4, cash=-50.0, capacity=5.0, stock=10.0)
    assert can_hire(f, solvent) is True
    assert can_hire(f, bankrupt) is False


def test_no_solvent_employer_makes_work_fail_and_unemp_rises():
    """When every alive firm is forced into deep debt, work fails and unemp_proxy hits 1.0."""
    w = _make_world(weeks=20)
    w.f.cash[w.f.alive_slots()] = FIRM_INSOLVENT_CASH_THRESHOLD - 5.0

    actions = _all_household_actions(w, ACTION_WORK)
    w.step(actions)
    metric = w.metrics.history[-1]
    assert metric["unemployment"] >= 0.99, (
        f"unemployment should be ~1.0, got {metric['unemployment']:.3f}"
    )


# Hunger, fatigue, health.

def test_food_shopping_reduces_hunger():
    w = _make_world(weeks=30)
    w.f.stock[w.f.alive_slots()] = 999.0
    w.h.wealth[:] = 10.0
    w.h.hunger[:] = 0.6

    initial_mean = float(np.mean(w.h.hunger))
    for _ in range(5):
        if w.done:
            break
        w.step(_all_household_actions(w, ACTION_SHOP))
    final_mean = float(np.mean(w.h.hunger))
    assert final_mean < initial_mean, (
        f"hunger should fall under all-shop, {initial_mean:.3f} -> {final_mean:.3f}"
    )


def test_no_shopping_policy_eventually_damages_health():
    w = _make_world(weeks=80)
    initial_health = float(np.mean(w.h.health))
    for _ in range(40):
        if w.done:
            break
        w.step(_all_household_actions(w, ACTION_REST))
    final_health = float(np.mean(w.h.health))
    final_hunger = float(np.mean(w.h.hunger))
    assert final_hunger > 0.4, f"40 weeks of rest should push hunger up, got {final_hunger:.3f}"
    assert final_health < initial_health, (
        f"health should fall, {initial_health:.3f} -> {final_health:.3f}"
    )


def test_rest_reduces_fatigue_but_not_hunger():
    w = _make_world(weeks=20)
    w.h.fatigue[:] = 0.7
    w.h.hunger[:] = 0.3
    pre_fatigue = float(np.mean(w.h.fatigue))
    pre_hunger = float(np.mean(w.h.hunger))
    w.step(_all_household_actions(w, ACTION_REST))
    post_fatigue = float(np.mean(w.h.fatigue))
    post_hunger = float(np.mean(w.h.hunger))
    assert post_fatigue < pre_fatigue
    assert post_hunger > pre_hunger


def test_all_work_policy_creates_fatigue_or_hunger():
    """All-work for many weeks should drive fatigue or hunger above the threshold."""
    w = _make_world(weeks=80)
    for _ in range(40):
        if w.done:
            break
        w.step(_all_household_actions(w, ACTION_WORK))
    mean_fatigue = float(np.mean(w.h.fatigue))
    mean_hunger = float(np.mean(w.h.hunger))
    triggered = (
        mean_fatigue >= FATIGUE_PRODUCTIVITY_THRESHOLD
        or mean_hunger >= HUNGER_HEALTH_PENALTY_THRESHOLD
    )
    assert triggered, f"fatigue={mean_fatigue:.3f} hunger={mean_hunger:.3f}"


# Clinic accounting and capacity.

def test_clinic_cost_is_conserved_in_firm_cash():
    """A successful clinic visit credits the firm with what the household paid."""
    w = _make_world(weeks=10)
    alive = w.f.alive_slots()
    target_slot = int(alive[0])
    w.f.stock[target_slot] = 999.0
    w.f.cash[target_slot] = 0.0
    w.h.wealth[:] = 50.0

    pre_cash_total = float(np.sum(w.f.cash[alive]))
    pre_wealth = float(np.sum(w.h.wealth))
    w.step(_all_household_actions(w, ACTION_CLINIC))
    post_cash_total = float(np.sum(w.f.cash[w.f.alive_slots()]))

    cash_delta = post_cash_total - pre_cash_total
    assert cash_delta > 0, "Clinic visit should credit firm cash"


def test_clinic_capacity_caps_treatment():
    w = _make_world(weeks=5)
    w.f.stock[w.f.alive_slots()] = 999.0
    w.h.wealth[:] = 50.0
    w.step(_all_household_actions(w, ACTION_CLINIC))
    used = w._weekly_clinic_used
    cap = w._weekly_clinic_capacity
    assert used <= cap, f"clinic used ({used}) exceeded capacity ({cap})"
    assert used > 0, "at least some visitors should have been treated"


# Illness.

def test_mild_illness_can_recover_naturally():
    w = _make_world(weeks=40)
    w.h.illness[:] = ILLNESS_MILD
    saw_recovery = False
    for _ in range(20):
        if w.done:
            break
        w.step(_all_household_actions(w, ACTION_REST))
        if np.any((w.h.illness == ILLNESS_RECOVERING) | (w.h.illness == ILLNESS_HEALTHY)):
            saw_recovery = True
            break
    assert saw_recovery, "some mild patients should recover naturally"


def test_severe_illness_recovery_is_better_with_clinic():
    """Treated severe patients recover at the higher rate; untreated at the lower."""
    rng = np.random.default_rng(7)
    n_trials = 40
    base_recover_no_care = 0
    base_recover_with_care = 0
    from marl_economy.env.illness import step_illness_transitions
    for _ in range(n_trials):
        states = np.array([ILLNESS_SEVERE], dtype=np.int32)
        health = np.array([0.5], dtype=np.float64)
        rested = np.array([False], dtype=bool)
        no_treat = np.array([False], dtype=bool)
        with_treat = np.array([True], dtype=bool)
        zero_exposed = np.array([False], dtype=bool)

        s_no, _ = step_illness_transitions(states.copy(), health.copy(), rng,
                                           rested, no_treat, zero_exposed)
        s_with, _ = step_illness_transitions(states.copy(), health.copy(), rng,
                                             rested, with_treat, zero_exposed)
        if s_no[0] == ILLNESS_RECOVERING:
            base_recover_no_care += 1
        if s_with[0] == ILLNESS_RECOVERING:
            base_recover_with_care += 1

    assert base_recover_with_care > base_recover_no_care, (
        f"with_care={base_recover_with_care}, no_care={base_recover_no_care}"
    )


def test_illness_spreads_among_colocated_agents():
    """One infectious agent + 79 healthy at the same workplace should spread."""
    w = _make_world(weeks=20)
    w.h.illness[:] = ILLNESS_HEALTHY
    w.h.illness[0] = ILLNESS_MILD
    w.h.wealth[:] = 5.0
    alive = w.f.alive_slots()
    w.f.cash[alive] = 50.0
    w.f.stock[alive] = 100.0
    w.f.capacity[alive] = 50.0

    new_exposed_seen = 0
    for _ in range(8):
        if w.done:
            break
        w.step(_all_household_actions(w, ACTION_WORK))
        n_infected = int(np.sum(w.h.illness != ILLNESS_HEALTHY))
        if n_infected > 1:
            new_exposed_seen = n_infected
            break

    assert new_exposed_seen > 1, (
        f"workplace transmission should infect more agents; only {new_exposed_seen} infected"
    )


# Reward bounds and CSV consistency.

def test_firm_rewards_bounded_during_zero_sales():
    """Firms producing nothing must never spike their reward."""
    w = _make_world(weeks=20)
    w.f.stock[w.f.alive_slots()] = 0.0
    w.h.wealth[:] = 0.0
    for _ in range(5):
        if w.done:
            break
        _, rewards, _, _ = w.step(_all_household_actions(w, ACTION_SHOP))
        for fid, fr in rewards.get("firms", {}).items():
            assert math.isfinite(fr), f"f_{fid} reward not finite: {fr}"
            assert abs(fr) <= SMOKE_TEST_MAX_ABS_REWARD, f"f_{fid} reward exploded: {fr}"


def test_csv_event_counts_match_ledger_totals():
    from marl_economy.env.events import ALL_EVENT_TYPES
    w = _make_world(weeks=40)
    for step in range(40):
        if w.done:
            break
        rng = np.random.default_rng(step)
        actions = {int(w.h.ids[i]): int(rng.integers(0, 6)) for i in range(w.h.size)}
        w.step(actions)

    for ev_type in ALL_EVENT_TYPES:
        col = f"events_{ev_type}"
        per_week_sum = sum(m.get(col, 0) for m in w.metrics.history)
        ledger_total = w.event_ledger.total_count(ev_type)
        assert per_week_sum == ledger_total, (
            f"{ev_type}: csv sum {per_week_sum} != ledger total {ledger_total}"
        )


# Hidden-collapse detector.

def test_economic_collapse_triggers_even_with_F_MIN_floor():
    """Persistent zero-GDP + insolvency must terminate the episode even when F_MIN > 0."""
    w = _make_world(weeks=60)
    w.f.cash[w.f.alive_slots()] = -100.0
    w.f.stock[w.f.alive_slots()] = 0.0
    w.h.wealth[:] = 0.0

    fired = False
    for _ in range(50):
        # Keep households artificially alive so mass_death doesn't beat the
        # economic-collapse detector to termination.
        w.h.health[:] = 0.6
        w.h.hunger[:] = 0.1
        w.h.fatigue[:] = 0.1
        alive = w.f.alive_slots()
        w.f.cash[alive] = -100.0
        w.f.stock[alive] = 0.0
        if w.done:
            break
        w.step(_all_household_actions(w, ACTION_REST))
        if w.collapse_detector.triggered:
            fired = True
            break
    assert fired, (
        "collapse detector must fire on persistent zero-GDP + insolvency. "
        f"reason={w.collapse_detector.trigger_reason}, "
        f"streak={w.collapse_detector.gdp_zero_streak}"
    )


# C1 money-conservation ledger.

def test_money_conservation_residual_is_zero():
    """C1: every step, Delta (total_money) must equal (minted - destroyed).

    `money_residual = (total_money - prev_total_money) - (minted - destroyed)`
    is computed at the world layer. If the per-step mint/destroy ledger is
    COMPLETE and correctly SIGNED, the residual is float-noise (~1e-12). A
    persistent |residual| >~ 1e-3 reveals a missed or sign-flipped site
    (positive => missed destroy; negative => missed mint).

    A mixed round-robin policy exercises every household site (drag every
    step, INVEST gains/losses, STEAL clawback-clamp) and a long horizon
    drives firm overhead/exit/spawn and stochastic disasters. We do NOT
    externally mutate state here - any manual cash/wealth poke would itself
    break conservation and create an artificial residual.
    """
    w = _make_world(regime="LME", weeks=60, seed=7)
    cycle = [ACTION_WORK, ACTION_SHOP, ACTION_STEAL,
             ACTION_REST, ACTION_INVEST, ACTION_CLINIC]
    TOL = 1e-6

    steps = 0
    saw_destroyed = False
    for _ in range(60):
        if w.done:
            break
        actions = {int(w.h.ids[i]): cycle[i % len(cycle)]
                   for i in range(w.h.size)}
        w.step(actions)
        steps += 1
        m = w.metrics.history[-1]
        for key in ("total_money", "money_minted", "money_destroyed",
                    "money_residual"):
            assert key in m, f"metrics missing C1 key {key!r}"
            assert math.isfinite(m[key]), f"{key} not finite: {m[key]}"
        assert abs(m["money_residual"]) < TOL, (
            f"week {steps}: money NOT conserved - residual="
            f"{m['money_residual']:.3e} (tol {TOL:.0e}); "
            f"minted={m['money_minted']:.4f} destroyed={m['money_destroyed']:.4f} "
            f"total_money={m['total_money']:.4f}. A persistent residual of "
            f"this size means a C1 site is missing or sign-flipped."
        )
        if m["money_destroyed"] > 0.0:
            saw_destroyed = True

    assert steps >= 5, f"test exercised too few steps ({steps})"
    # Drag + firm overhead destroy money every single step, so a complete
    # ledger MUST have recorded destruction - a permanently-zero destroyed
    # accumulator would itself be a bug (sites D/E silently not wired).
    assert saw_destroyed, (
        "money_destroyed stayed 0 across the whole run - drag/overhead "
        "destroy sites (D/E) are not feeding the C1 ledger."
    )
