"""Solvency predicates for firm hiring (slot-indexed against FirmState)."""

from __future__ import annotations

import numpy as np

from marl_economy.config.constants import (
    FIRM_HIRE_CREDIT_LIMIT,
    FIRM_HIRE_MIN_CAPACITY,
    FIRM_INSOLVENT_CASH_THRESHOLD,
)
from marl_economy.env.state import FirmState


def can_hire(f: FirmState, slot: int) -> bool:
    """True if the firm at `slot` can afford to add one more worker this week.

    Bankrupt firms, firms in disaster downtime, and firms below the minimum
    capacity floor are all rejected.
    """
    if not f.alive[slot]:
        return False
    if f.disaster_downtime[slot] > 0:
        return False
    cash = float(f.cash[slot])
    if cash < FIRM_INSOLVENT_CASH_THRESHOLD:
        return False
    expected_total_bill = (int(f.hires[slot]) + 1) * float(f.wage_offer[slot])
    if expected_total_bill > cash + FIRM_HIRE_CREDIT_LIMIT:
        return False
    if float(f.capacity[slot]) < FIRM_HIRE_MIN_CAPACITY:
        return False
    return True


def is_insolvent(f: FirmState, slot: int) -> bool:
    return float(f.cash[slot]) < FIRM_INSOLVENT_CASH_THRESHOLD


def firm_solvency_diagnostics(f: FirmState) -> dict:
    """Population-level firm finance numbers for the metrics CSV."""
    alive_slots = f.alive_slots()
    if alive_slots.size == 0:
        return {
            "mean_firm_cash": 0.0,
            "min_firm_cash": 0.0,
            "max_firm_cash": 0.0,
            "frac_insolvent_firms": 0.0,
            "total_inventory": 0.0,
        }
    cashes = f.cash[alive_slots]
    stocks = f.stock[alive_slots]
    insolvent = int(np.sum(cashes < FIRM_INSOLVENT_CASH_THRESHOLD))
    return {
        "mean_firm_cash": float(np.mean(cashes)),
        "min_firm_cash": float(np.min(cashes)),
        "max_firm_cash": float(np.max(cashes)),
        "frac_insolvent_firms": insolvent / int(alive_slots.size),
        "total_inventory": float(np.sum(stocks)),
    }
