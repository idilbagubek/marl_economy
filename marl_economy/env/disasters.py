"""Disaster generator.

Five exogenous shocks: earthquake, fire, flood, supply shock, epidemic wave.
Each rolls independently per week. Effects mutate `HouseholdState`,
`FirmState`, and `illness_states` directly.
"""

from __future__ import annotations

import numpy as np

from marl_economy.config.constants import (
    DISASTER_TYPES,
    DISASTER_PROB_PER_WEEK,
    EARTHQUAKE_FIRM_CAPACITY_LOSS_FRAC,
    EARTHQUAKE_FIRM_INVENTORY_LOSS_FRAC,
    EARTHQUAKE_FIRM_CASH_LOSS,
    EARTHQUAKE_HOUSEHOLD_INJURY_PROB,
    EARTHQUAKE_HOUSEHOLD_HEALTH_LOSS,
    FIRE_FIRMS_AFFECTED_RANGE,
    FIRE_FIRM_DOWNTIME_WEEKS,
    FLOOD_FIRMS_AFFECTED_RANGE,
    FLOOD_FIRM_DOWNTIME_WEEKS,
    FLOOD_HOUSEHOLD_DISPLACE_PROB,
    SUPPLY_SHOCK_DURATION_WEEKS,
    SUPPLY_SHOCK_PRICE_MULT,
    SUPPLY_SHOCK_CAPACITY_PENALTY,
    EPIDEMIC_WAVE_DURATION_WEEKS,
    EPIDEMIC_WAVE_INITIAL_INFECTED_FRAC,
    EPIDEMIC_WAVE_INFECTION_BOOST,
    ILLNESS_EXPOSED,
    ILLNESS_HEALTHY,
    FIRM_CAPACITY_MIN,
)
from marl_economy.env.state import FirmState, HouseholdState


def init_disaster_state() -> dict:
    return {
        "supply_shock_remaining": 0,
        "epidemic_wave_remaining": 0,
        "fire_active": 0,
        "flood_active": 0,
    }


def disaster_signal(disaster_state: dict) -> int:
    """Number of currently-active disasters."""
    count = 0
    for key in ("supply_shock_remaining", "epidemic_wave_remaining",
                "fire_active", "flood_active"):
        if disaster_state.get(key, 0) > 0:
            count += 1
    return count


def supply_shock_active(disaster_state: dict) -> bool:
    return disaster_state.get("supply_shock_remaining", 0) > 0


def epidemic_wave_active(disaster_state: dict) -> bool:
    return disaster_state.get("epidemic_wave_remaining", 0) > 0


def epidemic_wave_boost(disaster_state: dict) -> float:
    return EPIDEMIC_WAVE_INFECTION_BOOST if epidemic_wave_active(disaster_state) else 1.0


def supply_shock_price_mult(disaster_state: dict) -> float:
    return SUPPLY_SHOCK_PRICE_MULT if supply_shock_active(disaster_state) else 1.0


def supply_shock_capacity_mult(disaster_state: dict) -> float:
    return SUPPLY_SHOCK_CAPACITY_PENALTY if supply_shock_active(disaster_state) else 1.0


def _apply_earthquake(f: FirmState, h: HouseholdState, rng: np.random.Generator) -> dict:
    alive_slots = f.alive_slots()
    cash_destroyed = 0.0
    if alive_slots.size > 0:
        cap_loss = f.capacity[alive_slots] * EARTHQUAKE_FIRM_CAPACITY_LOSS_FRAC
        f.capacity[alive_slots] = np.maximum(FIRM_CAPACITY_MIN, f.capacity[alive_slots] - cap_loss)
        f.target_capacity[alive_slots] = np.maximum(
            FIRM_CAPACITY_MIN, f.target_capacity[alive_slots] - cap_loss
        )
        f.stock[alive_slots] = np.maximum(
            0.0, f.stock[alive_slots] * (1.0 - EARTHQUAKE_FIRM_INVENTORY_LOSS_FRAC)
        )
        f.cash[alive_slots] = f.cash[alive_slots] - EARTHQUAKE_FIRM_CASH_LOSS
        # C1 G: every still-ALIVE firm loses a flat EARTHQUAKE_FIRM_CASH_LOSS to
        # nowhere (no clamp; the firm stays alive so the loss stays in the
        # alive-firm cash sum) -> DESTROYED, exactly loss * #alive.
        cash_destroyed = float(EARTHQUAKE_FIRM_CASH_LOSS) * int(alive_slots.size)

    # One rng.random per household (matching original per-household roll order).
    rolls = rng.random(h.size)
    injured_mask = rolls < EARTHQUAKE_HOUSEHOLD_INJURY_PROB
    if np.any(injured_mask):
        h.health[injured_mask] = np.maximum(
            0.0, h.health[injured_mask] - EARTHQUAKE_HOUSEHOLD_HEALTH_LOSS
        )

    return {
        "affected_firms": int(alive_slots.size),
        "injured_households": int(injured_mask.sum()),
        "cash_destroyed": cash_destroyed,
    }


def _apply_fire(f: FirmState, rng: np.random.Generator) -> dict:
    alive_slots = f.alive_slots()
    if alive_slots.size == 0:
        return {"affected_firms": 0}
    lo, hi = FIRE_FIRMS_AFFECTED_RANGE
    n_affected = min(int(rng.integers(lo, hi)), int(alive_slots.size))
    targets = rng.choice(int(alive_slots.size), size=n_affected, replace=False)
    target_slots = alive_slots[targets]
    f.disaster_downtime[target_slots] = np.maximum(
        f.disaster_downtime[target_slots], FIRE_FIRM_DOWNTIME_WEEKS
    )
    return {"affected_firms": int(n_affected)}


def _apply_flood(f: FirmState, h: HouseholdState, rng: np.random.Generator) -> dict:
    alive_slots = f.alive_slots()
    n_affected = 0
    if alive_slots.size > 0:
        lo, hi = FLOOD_FIRMS_AFFECTED_RANGE
        n_affected = min(int(rng.integers(lo, hi)), int(alive_slots.size))
        targets = rng.choice(int(alive_slots.size), size=n_affected, replace=False)
        target_slots = alive_slots[targets]
        f.disaster_downtime[target_slots] = np.maximum(
            f.disaster_downtime[target_slots], FLOOD_FIRM_DOWNTIME_WEEKS
        )

    # One rng.random per household.
    rolls = rng.random(h.size)
    displaced_mask = rolls < FLOOD_HOUSEHOLD_DISPLACE_PROB
    if np.any(displaced_mask):
        h.displaced[displaced_mask] = True

    return {
        "affected_firms": int(n_affected),
        "displaced_households": int(displaced_mask.sum()),
    }


def _start_supply_shock(disaster_state: dict) -> dict:
    disaster_state["supply_shock_remaining"] = SUPPLY_SHOCK_DURATION_WEEKS
    return {"duration_weeks": SUPPLY_SHOCK_DURATION_WEEKS}


def _start_epidemic_wave(
    disaster_state: dict,
    illness_states: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    disaster_state["epidemic_wave_remaining"] = EPIDEMIC_WAVE_DURATION_WEEKS
    healthy_idxs = np.where(illness_states == ILLNESS_HEALTHY)[0]
    n_seed = min(
        int(round(len(illness_states) * EPIDEMIC_WAVE_INITIAL_INFECTED_FRAC)),
        int(healthy_idxs.size),
    )
    if n_seed > 0:
        chosen = rng.choice(healthy_idxs, size=n_seed, replace=False)
        illness_states[chosen] = ILLNESS_EXPOSED
    return {"duration_weeks": EPIDEMIC_WAVE_DURATION_WEEKS, "seeded": int(n_seed)}


def maybe_trigger(
    week: int,
    disaster_state: dict,
    f: FirmState,
    h: HouseholdState,
    illness_states: np.ndarray,
    rng: np.random.Generator,
) -> list[dict]:
    """Roll for new disasters, apply effects, decay ongoing counters.

    Mutates `f`, `h`, and `illness_states` in place. Returns a list of
    triggered-disaster info dicts so the world layer can record them in the
    event ledger and append Unity visual events.

    NOTE: RNG call ORDER must match the legacy version exactly to keep bit-
    exact equivalence for the same seed.
    """
    triggered: list[dict] = []

    # Decay counters first.
    for key in ("supply_shock_remaining", "epidemic_wave_remaining"):
        if disaster_state.get(key, 0) > 0:
            disaster_state[key] -= 1

    # Firm disaster_downtime decay. Match the old iteration: loop firms in
    # slot order over [0, f.size); decrement positive entries, force-zero
    # otherwise (defensive - the field starts at 0 and only the spawn path
    # writes anything else).
    if f.size > 0:
        active_slots = np.arange(f.size)
        downtimes = f.disaster_downtime[active_slots]
        positive = downtimes > 0
        f.disaster_downtime[active_slots[positive]] = downtimes[positive] - 1
        f.disaster_downtime[active_slots[~positive]] = 0

    # Trigger rolls. One rng.random() per disaster type, in DISASTER_TYPES order.
    for dtype in DISASTER_TYPES:
        prob = DISASTER_PROB_PER_WEEK.get(dtype, 0.0)
        if rng.random() >= prob:
            continue
        info: dict = {"type": dtype, "week": week}
        if dtype == "earthquake":
            info.update(_apply_earthquake(f, h, rng))
        elif dtype == "fire":
            info.update(_apply_fire(f, rng))
            disaster_state["fire_active"] = max(disaster_state.get("fire_active", 0),
                                                FIRE_FIRM_DOWNTIME_WEEKS)
        elif dtype == "flood":
            info.update(_apply_flood(f, h, rng))
            disaster_state["flood_active"] = max(disaster_state.get("flood_active", 0),
                                                 FLOOD_FIRM_DOWNTIME_WEEKS)
        elif dtype == "supply_shock":
            info.update(_start_supply_shock(disaster_state))
        elif dtype == "epidemic_wave":
            info.update(_start_epidemic_wave(disaster_state, illness_states, rng))
        triggered.append(info)

    # Decay fire/flood-active counters at the end of the tick.
    for key in ("fire_active", "flood_active"):
        if disaster_state.get(key, 0) > 0:
            disaster_state[key] -= 1

    return triggered
