"""Firm-side dynamics: production, sales, overhead, capacity, reward.

All functions operate on `FirmState` arrays. Per-firm functions take a `slot`
index; population-wide functions take a `slots` array (typically from
`FirmState.alive_slots()`).
"""

from __future__ import annotations

import numpy as np

from marl_economy.config.constants import (
    EPSILON,
    FIRM_PRODUCTIVITY_DEFAULT, FIRM_PRODUCTIVITY_MIN,
    FIRM_FIXED_COST, FIRM_CAP_COST, FIRM_CAPACITY_MIN, FIRM_CAPACITY_BETA,
    FIRM_CHI, FIRM_ETA, FIRM_DEFAULT_SMOOTHNESS_SCALE, FIRM_BANKRUPTCY_PENALTY,
    FIRM_BANKRUPTCY_GRACE_WEEKS, FIRM_BANKRUPTCY_FULL_WEEKS,
    FIRM_ACTIVITY_BONUS, FIRM_ACTIVITY_REF_SALES,
    FIRM_DEAD_ECONOMY_PENALTY, FIRM_DEAD_SALES_RAMP,
    FIRM_PRICE_FLOOR, FIRM_WAGE_FLOOR, FIRM_STOCK_TRANSACTION_UNITS,
    FIRM_REWARD_MIN_SALES_NORMALIZER, FIRM_UPDATE_PERIOD,
    FIRM_PRICE_DELTAS, FIRM_WAGE_DELTAS, FIRM_CAPACITY_DELTAS,
)
from marl_economy.env.state import FirmState


def firm_produce_stock_slot(f: FirmState, slot: int) -> float:
    """FIX A (2026-05-18 v2): the STOCK half of weekly production, SPLIT out so
    it can run BEFORE the household step's `_ration_stock` snapshot (the real
    food-supply throttle was that the ration saw only last-week leftover stock,
    not this-week production). q = min(productivity*sqrt(hires+eps), capacity).

    Needs ONLY hires (set in phase-4 routing) / productivity / capacity - NO
    wage, NO cash -> money-conservation-neutral (goods only). Returns q_prod.
    The matching wage debit is `firm_pay_wage_slot`, called later (it needs the
    Option-B gross wage computed in the phase-5 household step).
    """
    hires = int(f.hires[slot])
    raw_production = float(f.productivity[slot]) * float(np.sqrt(hires + EPSILON))
    q_prod = min(raw_production, max(0.0, float(f.capacity[slot])))
    f.stock[slot] += q_prod
    f.window_hires[slot] += hires
    return q_prod


def firm_pay_wage_slot(f: FirmState, slot: int,
                       wage_bill_override: float | None = None) -> float:
    """FIX A: the WAGE-PAYMENT half of weekly production, SPLIT from
    `firm_produce_stock_slot`. Runs AFTER the household step (it needs the
    Option-B GROSS effective wage = `wage_bill_override` = net_wage_by_firm).
    Conserves money - firm pays exactly what households receive net + what the
    treasury collects in tax. Default None = legacy nominal wage_offer*hires.
    Returns the wage bill paid.
    """
    hires = int(f.hires[slot])
    if wage_bill_override is None:
        wage_bill = float(f.wage_offer[slot]) * hires
    else:
        wage_bill = float(wage_bill_override)
    f.cash[slot] -= wage_bill
    f.window_wage_bill[slot] += wage_bill
    return wage_bill


def firm_sell_slot(
    f: FirmState,
    slot: int,
    quantity: float = FIRM_STOCK_TRANSACTION_UNITS,
    unit_price: float | None = None,
) -> bool:
    """Process one sale at `slot`. Returns True iff stock was available.

    `unit_price` overrides the firm's posted price so callers can credit the
    actual amount paid (e.g. clinic at price * multiplier).
    """
    if float(f.stock[slot]) >= quantity:
        per_unit = float(f.price[slot] if unit_price is None else unit_price)
        revenue = per_unit * quantity
        f.stock[slot] -= quantity
        f.cash[slot] += revenue
        f.sales[slot] += quantity
        f.window_sales[slot] += quantity
        f.window_revenue[slot] += revenue
        return True
    else:
        f.unmet_demand[slot] += quantity
        f.window_unmet[slot] += quantity
        return False


def firm_pay_overhead_slot(f: FirmState, slot: int) -> float:
    """Weekly overhead: FIRM_FIXED_COST + FIRM_CAP_COST * capacity."""
    overhead = FIRM_FIXED_COST + FIRM_CAP_COST * float(f.capacity[slot])
    f.cash[slot] -= overhead
    return overhead


def firm_update_capacity_slot(f: FirmState, slot: int) -> None:
    """Monthly capacity smoothing: cap = (1-beta) * cap + beta * target."""
    new_cap = ((1.0 - FIRM_CAPACITY_BETA) * float(f.capacity[slot])
               + FIRM_CAPACITY_BETA * float(f.target_capacity[slot]))
    f.capacity[slot] = max(FIRM_CAPACITY_MIN, new_cap)


def firm_apply_action_slot(
    f: FirmState,
    slot: int,
    action: tuple,
    min_wage_floor: float = 0.0,
) -> None:
    """Apply firm action (discrete deltas) to price, wage, target_capacity."""
    price_delta = FIRM_PRICE_DELTAS[action[0]]
    wage_delta = FIRM_WAGE_DELTAS[action[1]]
    cap_delta = FIRM_CAPACITY_DELTAS[action[2]]

    f.prev_price[slot] = f.price[slot]
    f.prev_wage[slot] = f.wage_offer[slot]

    f.price[slot] = max(FIRM_PRICE_FLOOR, float(f.price[slot]) + price_delta)
    f.wage_offer[slot] = max(FIRM_WAGE_FLOOR, min_wage_floor,
                              float(f.wage_offer[slot]) + wage_delta)
    f.target_capacity[slot] = max(FIRM_CAPACITY_MIN,
                                   float(f.target_capacity[slot]) + cap_delta)


def compute_firm_reward_slot(
    f: FirmState,
    slot: int,
    smoothness_scale: float = FIRM_DEFAULT_SMOOTHNESS_SCALE,
) -> float:
    """Monthly firm reward (normalized).

    R = profit_norm - chi * unmet_rate - eta * smoothness_scale * (dp^2 + dw^2)
        - bankruptcy_penalty + activity_reward - dead_economy_penalty

    `activity_reward` rewards real monthly sales volume (saturating) and
    `dead_economy_penalty` explicitly punishes a firm that sells ~nothing, so
    "shrink to F_MIN and idle" is no longer reward-indistinguishable from a
    healthy firm (which it was under profit/max(sales,1) alone).
    """
    revenue = float(f.window_revenue[slot])
    wage_bill = float(f.window_wage_bill[slot])
    overhead = (FIRM_FIXED_COST + FIRM_CAP_COST * float(f.capacity[slot])) * FIRM_UPDATE_PERIOD
    profit = revenue - wage_bill - overhead

    sales = float(f.window_sales[slot])
    unmet = float(f.window_unmet[slot])

    profit_denominator = max(sales, FIRM_REWARD_MIN_SALES_NORMALIZER)
    profit_norm = profit / profit_denominator

    unmet_rate = unmet / (sales + unmet + EPSILON)

    dp = float(f.price[slot]) - float(f.prev_price[slot])
    dw = float(f.wage_offer[slot]) - float(f.prev_wage[slot])
    smoothness_penalty = FIRM_ETA * smoothness_scale * (dp ** 2 + dw ** 2)

    # S1 FIX (+ re-review hardening): MONOTONE RAMP on SUSTAINED insolvency,
    # NOT an instantaneous cash<0 snapshot and NOT a hard step. negative_cash_
    # weeks is maintained every step for every alive firm by check_firm_exit_
    # slot (phase-3): +1 per consecutive cash<0 week, reset 0 otherwise, exit
    # at FIRM_EXIT_CONSECUTIVE_WEEKS(8). Penalty ramps linearly 0 -> full over
    # ncw in [GRACE(3), FULL(7)]: 0 at/below the benign <=3-wk monthly-wage
    # trough; full just before the 8-wk exit. A ramp (vs a hard step) has no
    # cliff for a neg,neg,neg,pos "sawtooth" to straddle (which would have
    # evaded BOTH penalty and exit), removes monthly-sampling aliasing, and
    # gives a smooth solvency gradient in sync with the exit countdown.
    # Reward-only -> money-conservation-neutral, deterministic.
    _ncw = int(f.negative_cash_weeks[slot])
    _ramp = (_ncw - FIRM_BANKRUPTCY_GRACE_WEEKS) / max(
        1, FIRM_BANKRUPTCY_FULL_WEEKS - FIRM_BANKRUPTCY_GRACE_WEEKS
    )
    bankruptcy_penalty = FIRM_BANKRUPTCY_PENALTY * min(1.0, max(0.0, _ramp))

    # Dead-economy shaping: reward real sales volume (saturating at the
    # reference) and explicitly penalize a firm that sold ~nothing this month.
    activity_reward = FIRM_ACTIVITY_BONUS * min(sales / FIRM_ACTIVITY_REF_SALES, 1.0)
    # Smooth ramp (not a binary <1-sale cliff): full penalty at 0 sales, fading
    # linearly to 0 at FIRM_DEAD_SALES_RAMP. A token sale can no longer zero it.
    dead_economy_penalty = FIRM_DEAD_ECONOMY_PENALTY * max(
        0.0, 1.0 - sales / FIRM_DEAD_SALES_RAMP
    )

    reward = (profit_norm - FIRM_CHI * unmet_rate - smoothness_penalty
              - bankruptcy_penalty + activity_reward - dead_economy_penalty)

    # Stash profit so callers can log it (parity with the old dict field).
    f.profit[slot] = profit
    return float(reward)


def firm_reset_weekly_vec(f: FirmState, slots: np.ndarray) -> None:
    """Reset per-week accumulators on the given slots."""
    f.sales[slots] = 0.0
    f.hires[slots] = 0
    f.unmet_demand[slots] = 0.0


def firm_reset_window_slot(f: FirmState, slot: int) -> None:
    """Reset monthly window accumulators for one firm."""
    f.window_sales[slot] = 0.0
    f.window_hires[slot] = 0
    f.window_unmet[slot] = 0.0
    f.window_revenue[slot] = 0.0
    f.window_wage_bill[slot] = 0.0


def firm_get_observation_slot(
    f: FirmState,
    slot: int,
    mean_price: float,
    mean_wage: float,
    tax_rate: float,
    min_wage: float,
) -> np.ndarray:
    """Firm observation (10D).

    [price, wage, capacity, avg_weekly_sales, avg_weekly_hires, avg_weekly_unmet,
     tax_rate, min_wage, mean_price, mean_wage]
    """
    return np.array([
        float(f.price[slot]),
        float(f.wage_offer[slot]),
        float(f.capacity[slot]),
        float(f.window_sales[slot]) / max(1, FIRM_UPDATE_PERIOD),
        float(f.window_hires[slot]) / max(1, FIRM_UPDATE_PERIOD),
        float(f.window_unmet[slot]) / max(1, FIRM_UPDATE_PERIOD),
        float(tax_rate),
        float(min_wage),
        float(mean_price),
        float(mean_wage),
    ], dtype=np.float32)
