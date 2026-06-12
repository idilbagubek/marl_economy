"""Firm entry, exit, and emergency replacement on FirmState slots."""

from __future__ import annotations

import numpy as np

from marl_economy.config.constants import (
    F_MIN, F_MAX, FIRM_EXIT_CONSECUTIVE_WEEKS, FIRM_ENTRY_WINDOW_WEEKS,
    FIRM_ENTRY_THRESHOLD, FIRM_ENTRY_PROB, FIRM_INIT_CASH,
    FIRM_INIT_CAPACITY, FIRM_INIT_STOCK,
    FIRM_PRODUCTIVITY_MIN, FIRM_PRODUCTIVITY_LOGNORMAL_MEAN, FIRM_PRODUCTIVITY_LOGNORMAL_SIGMA,
    FIRM_MARKET_NOISE_STD, FIRM_PRICE_FLOOR, FIRM_WAGE_FLOOR,
    MARKET_DEFAULT_PRICE, MARKET_DEFAULT_WAGE,
    EMERGENCY_ENTRY_BUFFER, EPSILON,
)
from marl_economy.env.events import (
    EventLedger,
    FIRM_EXIT,
    FIRM_EXIT_BLOCKED,
    FIRM_ENTRY_EMERGENCY,
)
from marl_economy.env.state import FirmState, HouseholdState


def check_firm_exit_slot(f: FirmState, slot: int) -> bool:
    """Update negative-cash counter and return True if the firm should exit."""
    if f.cash[slot] < 0:
        f.negative_cash_weeks[slot] += 1
    else:
        f.negative_cash_weeks[slot] = 0
    return int(f.negative_cash_weeks[slot]) >= FIRM_EXIT_CONSECUTIVE_WEEKS


def process_firm_exit(
    f: FirmState,
    exiting_slot: int,
    h: HouseholdState,
    debug_log: list,
    event_ledger: EventLedger | None = None,
    week: int = 0,
) -> dict:
    """Mark a firm exited, displace its workers, and redistribute its cash."""
    firm_id = int(f.id[exiting_slot])
    cash_at_exit = float(f.cash[exiting_slot])
    exit_info = {
        "firm_id": firm_id,
        "cash_at_exit": cash_at_exit,
        "displaced_workers": 0,
        "money_minted": 0.0,
        "money_destroyed": 0.0,
    }

    # Mark as dead.
    f.kill(exiting_slot)

    # Worker displacement - households whose last routed firm was the exiter.
    displaced_mask = h.last_routed_firm_id == firm_id
    exit_info["displaced_workers"] = int(displaced_mask.sum())
    if exit_info["displaced_workers"] > 0:
        h.displaced[displaced_mask] = True

    # Asset redistribution to surviving firms.
    surviving = f.alive_slots()
    if surviving.size > 0 and cash_at_exit > 0:
        share = cash_at_exit / int(surviving.size)
        f.cash[surviving] += share
    # Negative cash: debt written off. Remaining stock: destroyed.

    # C1 conservation ledger. total_money sums ALIVE-firm cash only (the dead
    # slot's cash is excluded the instant `f.kill` flips `alive`). So:
    #  - cash>0 redistributed to >=1 survivor -> conserved TRANSFER (no ledger).
    #  - cash>0 but NO survivor      -> that money leaves the live economy with
    #       no offset -> DESTROYED (cash_at_exit).
    #  - cash<0 (debt written off)   -> removing a NEGATIVE from the alive-only
    #       sum RAISES total_money by |cash|. Economically the firm already
    #       deficit-spent that money into households/treasury and it stays
    #       there; the write-off just drops the unbacked IOU -> this is a MINT
    #       of |cash_at_exit| (NOT a destroy - the "writeoff=destroy" shorthand
    #       is sign-wrong against an alive-only total; verified by residual~=0).
    if cash_at_exit > 0.0:
        if surviving.size == 0:
            exit_info["money_destroyed"] = cash_at_exit
    elif cash_at_exit < 0.0:
        exit_info["money_minted"] = -cash_at_exit

    debug_log.append(
        f"FIRM_EXIT: id={firm_id}, cash={cash_at_exit:.4f}, "
        f"displaced={exit_info['displaced_workers']}"
    )
    if event_ledger is not None:
        event_ledger.record(
            FIRM_EXIT, week=week, actor_id=firm_id,
            cash_at_exit=cash_at_exit,
            displaced_workers=exit_info["displaced_workers"],
        )
    return exit_info


def process_mass_exit(
    f: FirmState,
    h: HouseholdState,
    unmet_demand_history: list,
    rng: np.random.Generator,
    debug_log: list,
    event_ledger: EventLedger | None = None,
    week: int = 0,
) -> dict:
    """Process all queued firm exits this week, respecting F_MIN.

    Weakest firms (lowest cash) exit first when the floor is binding. If exits
    leave the market below the buffer target, emergency-spawn replacements.

    Returns a C1 ledger dict ``{"money_minted", "money_destroyed"}`` aggregating
    every exit's write-off/destroy (see ``process_firm_exit``) plus the
    FIRM_INIT_CASH minted by each emergency replacement spawn.
    """
    ledger = {"money_minted": 0.0, "money_destroyed": 0.0}
    alive_before = f.alive_slots()
    # Decide candidates: alive AND triggered the exit predicate this week.
    exit_candidates: list[int] = []
    for slot in alive_before.tolist():
        if check_firm_exit_slot(f, slot):
            exit_candidates.append(int(slot))

    # Weakest first (lowest cash). Match the old `sorted(..., key=lambda i: firms[i]["cash"])`.
    exit_candidates.sort(key=lambda s: float(f.cash[s]))

    exits_processed = 0
    for slot in exit_candidates:
        alive_count = f.num_alive
        if alive_count <= F_MIN:
            firm_id = int(f.id[slot])
            debug_log.append(
                f"FIRM_EXIT_BLOCKED: id={firm_id}, alive_count={alive_count} <= F_MIN={F_MIN}"
            )
            if event_ledger is not None:
                event_ledger.record(
                    FIRM_EXIT_BLOCKED, week=week, actor_id=firm_id,
                    alive_count=alive_count, cash=float(f.cash[slot]),
                )
            continue

        _ei = process_firm_exit(f, slot, h, debug_log, event_ledger=event_ledger, week=week)
        ledger["money_minted"] += float(_ei.get("money_minted", 0.0))
        ledger["money_destroyed"] += float(_ei.get("money_destroyed", 0.0))
        exits_processed += 1

    # Emergency entry: spawn replacements if a single exit pushed us below the
    # F_MIN + buffer target.
    if exits_processed >= 1:
        alive_count = f.num_alive
        surviving = f.alive_slots()
        mean_price = (float(np.mean(f.price[surviving]))
                      if surviving.size > 0 else MARKET_DEFAULT_PRICE)
        mean_wage = (float(np.mean(f.wage_offer[surviving]))
                     if surviving.size > 0 else MARKET_DEFAULT_WAGE)

        target = min(F_MIN + EMERGENCY_ENTRY_BUFFER, F_MAX)
        while alive_count < target:
            new_slot = spawn_firm(f, mean_price, mean_wage, rng)
            new_id = int(f.id[new_slot])
            alive_count += 1
            # C1: a fresh firm enters the alive set with cash=FIRM_INIT_CASH -
            # money created from nowhere -> MINT.
            ledger["money_minted"] += float(FIRM_INIT_CASH)
            debug_log.append(f"EMERGENCY_ENTRY: id={new_id} after mass exit")
            if event_ledger is not None:
                event_ledger.record(
                    FIRM_ENTRY_EMERGENCY, week=week, actor_id=new_id,
                    mean_price=mean_price, mean_wage=mean_wage,
                )

    return ledger


def check_firm_entry(
    f: FirmState,
    unmet_demand_history: list,
    rng: np.random.Generator,
) -> bool:
    """Entry condition: unmet_rate over the recent window above threshold AND room under F_MAX."""
    if f.num_alive >= F_MAX:
        return False

    if len(unmet_demand_history) < FIRM_ENTRY_WINDOW_WEEKS:
        return False

    recent = unmet_demand_history[-FIRM_ENTRY_WINDOW_WEEKS:]
    total_unmet = sum(r["unmet"] for r in recent)
    total_demand = sum(r["demand"] for r in recent)

    if total_demand < EPSILON:
        return False

    unmet_rate = total_unmet / total_demand
    if unmet_rate > FIRM_ENTRY_THRESHOLD:
        return rng.random() < FIRM_ENTRY_PROB

    return False


def spawn_firm(
    f: FirmState,
    mean_price: float,
    mean_wage: float,
    rng: np.random.Generator,
) -> int:
    """Spawn a new firm near market averages. Returns the new slot index."""
    price = max(FIRM_PRICE_FLOOR, mean_price + rng.normal(0, FIRM_MARKET_NOISE_STD))
    wage = max(FIRM_WAGE_FLOOR, mean_wage + rng.normal(0, FIRM_MARKET_NOISE_STD))
    productivity = max(
        FIRM_PRODUCTIVITY_MIN,
        rng.lognormal(FIRM_PRODUCTIVITY_LOGNORMAL_MEAN, FIRM_PRODUCTIVITY_LOGNORMAL_SIGMA),
    )
    return f.spawn(
        price=price,
        wage=wage,
        cash=FIRM_INIT_CASH,
        capacity=FIRM_INIT_CAPACITY,
        stock=FIRM_INIT_STOCK,
        productivity=productivity,
    )


def recompute_market_averages(f: FirmState) -> tuple[float, float]:
    """Mean price and mean wage across alive firms only."""
    surviving = f.alive_slots()
    if surviving.size == 0:
        return MARKET_DEFAULT_PRICE, MARKET_DEFAULT_WAGE
    return float(np.mean(f.price[surviving])), float(np.mean(f.wage_offer[surviving]))
