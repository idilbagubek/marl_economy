"""Struct-of-arrays storage for household and firm state.

Replaces the old `list[dict]` layout. Every per-step field lives as a NumPy
array indexed by slot. Vectorized dynamics read and write these arrays
directly without dict pack/unpack overhead.

Slot semantics:
    HouseholdState - fixed size N. Slot index equals household ID.
                     No births, so size never changes within an episode.

    FirmState      - fixed-capacity slot table. Slot index equals firm ID
                     (matches the old `self.next_firm_id` counter). Slots are
                     never reused, even after a firm exits, so the iteration
                     order over slots [0, 1, ..., size-1] mirrors the old
                     `for f in self.firms` insertion order. The `alive` mask
                     filters dead slots.
"""

from __future__ import annotations

import numpy as np

from marl_economy.config.constants import (
    ACTION_REST,
    ILLNESS_HEALTHY,
    FIRM_PRODUCTIVITY_DEFAULT,
)

# Sentinel for HouseholdState.last_routed_firm_id when no firm was routed.
NO_ROUTED_FIRM_ID = -1


class HouseholdState:
    """Parallel-array storage for N households.

    Slot index equals household ID for the lifetime of the episode (no births).
    """

    __slots__ = (
        "size",
        "wealth", "health", "hunger", "fatigue", "illness",
        "experience", "age",
        "last_action", "last_routed_firm_id", "last_action_success",
        "last_reward", "last_wealth_change", "last_health_change",
        "last_crime_success", "displaced",
    )

    def __init__(self, n: int):
        self.size = n

        # Per-step physical state.
        self.wealth     = np.zeros(n, dtype=np.float64)
        self.health     = np.zeros(n, dtype=np.float64)
        self.hunger     = np.zeros(n, dtype=np.float64)
        self.fatigue    = np.zeros(n, dtype=np.float64)
        self.illness    = np.full(n, ILLNESS_HEALTHY, dtype=np.int32)
        self.experience = np.zeros(n, dtype=np.float64)
        self.age        = np.zeros(n, dtype=np.int64)

        # Per-step bookkeeping consumed by observations, events, and Unity.
        self.last_action          = np.full(n, ACTION_REST, dtype=np.int32)
        self.last_routed_firm_id  = np.full(n, NO_ROUTED_FIRM_ID, dtype=np.int32)
        self.last_action_success  = np.ones(n, dtype=bool)
        self.last_reward          = np.zeros(n, dtype=np.float64)
        self.last_wealth_change   = np.zeros(n, dtype=np.float64)
        self.last_health_change   = np.zeros(n, dtype=np.float64)
        self.last_crime_success   = np.zeros(n, dtype=bool)
        self.displaced            = np.zeros(n, dtype=bool)

    @property
    def ids(self) -> np.ndarray:
        """Household IDs as an array (ID == slot for Stage 1)."""
        return np.arange(self.size, dtype=np.int32)

    def __len__(self) -> int:
        return self.size


class FirmState:
    """Slot-based storage for firms. Slot index equals firm ID.

    Slots are filled by `spawn()` in sequential order and never reused, mirroring
    the old `self.firms.append(new_firm)` semantics so that iteration order is
    preserved bit-for-bit across the Stage 1 refactor.
    """

    __slots__ = (
        "max_firms", "size",
        "alive", "id",
        "price", "wage_offer", "cash",
        "capacity", "target_capacity", "stock",
        "productivity",
        "sales", "hires", "unmet_demand",
        "profit", "prev_price", "prev_wage",
        "negative_cash_weeks",
        "window_sales", "window_hires", "window_unmet",
        "window_revenue", "window_wage_bill",
        "disaster_downtime",
    )

    def __init__(self, max_firms: int):
        self.max_firms = int(max_firms)
        self.size = 0  # number of slots ever spawned (alive or dead)

        self.alive             = np.zeros(self.max_firms, dtype=bool)
        # Firm ID per slot (== slot index after spawn). -1 in unallocated slots.
        self.id                = np.full(self.max_firms, -1, dtype=np.int32)

        self.price             = np.zeros(self.max_firms, dtype=np.float64)
        self.wage_offer        = np.zeros(self.max_firms, dtype=np.float64)
        self.cash              = np.zeros(self.max_firms, dtype=np.float64)
        self.capacity          = np.zeros(self.max_firms, dtype=np.float64)
        self.target_capacity   = np.zeros(self.max_firms, dtype=np.float64)
        self.stock             = np.zeros(self.max_firms, dtype=np.float64)
        self.productivity      = np.zeros(self.max_firms, dtype=np.float64)
        self.sales             = np.zeros(self.max_firms, dtype=np.float64)
        self.hires             = np.zeros(self.max_firms, dtype=np.int32)
        self.unmet_demand      = np.zeros(self.max_firms, dtype=np.float64)
        self.profit            = np.zeros(self.max_firms, dtype=np.float64)
        self.prev_price        = np.zeros(self.max_firms, dtype=np.float64)
        self.prev_wage         = np.zeros(self.max_firms, dtype=np.float64)
        self.negative_cash_weeks = np.zeros(self.max_firms, dtype=np.int32)
        self.window_sales      = np.zeros(self.max_firms, dtype=np.float64)
        self.window_hires      = np.zeros(self.max_firms, dtype=np.int32)
        self.window_unmet      = np.zeros(self.max_firms, dtype=np.float64)
        self.window_revenue    = np.zeros(self.max_firms, dtype=np.float64)
        self.window_wage_bill  = np.zeros(self.max_firms, dtype=np.float64)
        self.disaster_downtime = np.zeros(self.max_firms, dtype=np.int32)

    @property
    def num_alive(self) -> int:
        return int(self.alive[: self.size].sum())

    @property
    def next_id(self) -> int:
        """Next firm ID to assign (matches `size` since IDs equal slot indices)."""
        return self.size

    def alive_slots(self) -> np.ndarray:
        """Slot indices of currently alive firms, in slot (== insertion) order."""
        # Only consider slots 0..size-1; slots past size are unallocated.
        return np.where(self.alive[: self.size])[0]

    def slot_of(self, firm_id: int) -> int:
        """Return slot for a given firm ID. -1 if out of range or dead.

        Because IDs equal slot indices, this is just a bounds check.
        """
        fid = int(firm_id)
        if 0 <= fid < self.size and self.alive[fid]:
            return fid
        return -1

    def spawn(
        self,
        *,
        price: float,
        wage: float,
        cash: float,
        capacity: float,
        stock: float,
        productivity: float = FIRM_PRODUCTIVITY_DEFAULT,
    ) -> int:
        """Allocate the next slot, write initial state, return slot index.

        Slot index equals the assigned firm ID. Raises if the slot table is full.
        """
        if self.size >= self.max_firms:
            raise RuntimeError(
                f"FirmState capacity {self.max_firms} exhausted; raise max_firms"
            )
        slot = self.size
        self.size += 1

        self.alive[slot] = True
        self.id[slot] = slot

        self.price[slot] = price
        self.wage_offer[slot] = wage
        self.cash[slot] = cash
        self.capacity[slot] = capacity
        self.target_capacity[slot] = capacity
        self.stock[slot] = stock
        self.productivity[slot] = max(productivity, 0.0)

        self.sales[slot] = 0.0
        self.hires[slot] = 0
        self.unmet_demand[slot] = 0.0
        self.profit[slot] = 0.0
        self.prev_price[slot] = price
        self.prev_wage[slot] = wage
        self.negative_cash_weeks[slot] = 0
        self.window_sales[slot] = 0.0
        self.window_hires[slot] = 0
        self.window_unmet[slot] = 0.0
        self.window_revenue[slot] = 0.0
        self.window_wage_bill[slot] = 0.0
        self.disaster_downtime[slot] = 0

        return slot

    def kill(self, slot: int) -> None:
        """Mark a slot dead. Iteration past `size` is unaffected; the slot stays
        allocated forever (matches the old append-only list semantics).
        """
        slot = int(slot)
        if 0 <= slot < self.size:
            self.alive[slot] = False
