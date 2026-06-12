"""Batched struct-of-arrays storage for K parallel worlds.

Stage 3 vectorizes the hot path (household batch step, routing) across a
leading K dimension while leaving per-world cold-path operations
(disasters, firm entry/exit, event recording, metrics) as Python loops over
worlds. The shape contract:

  BatchedHouseholdState     N households per world, K worlds
    every field is shape (K, N), dtype as in single-world `HouseholdState`.

  BatchedFirmState          Slot-table with capacity F_max per world
    every field is shape (K, F_max). `size[k]` is the count of allocated
    slots in world k (slots in [size[k], F_max) are unallocated, NOT just
    dead). `alive[k, slot]` is True iff slot is currently producing.

The slot semantics match the single-world Stage-1 implementation: slots are
never reused even after a firm exits, so iteration order over alive slots
(in ascending slot index) preserves the legacy `for f in self.firms` order
for each world.

`np.random.SeedSequence.spawn(K)` (called by `BatchedWorld`) gives K
independent rng streams so per-world disaster rolls, firm spawns, and
softmax samples are statistically decoupled.
"""

from __future__ import annotations

import numpy as np

from marl_economy.config.constants import (
    ACTION_REST,
    ILLNESS_HEALTHY,
    FIRM_PRODUCTIVITY_DEFAULT,
)
from marl_economy.env.state import NO_ROUTED_FIRM_ID  # share the -1 sentinel


class BatchedHouseholdState:
    """Parallel-array storage for K worlds x N households per world."""

    __slots__ = (
        "K", "N",
        "wealth", "health", "hunger", "fatigue", "illness",
        "experience", "age",
        "last_action", "last_routed_firm_id", "last_action_success",
        "last_reward", "last_wealth_change", "last_health_change",
        "last_crime_success", "displaced", "tainted_wealth",
    )

    def __init__(self, K: int, N: int):
        self.K = int(K)
        self.N = int(N)
        shape = (self.K, self.N)

        # Per-step physical state.
        self.wealth     = np.zeros(shape, dtype=np.float64)
        self.health     = np.zeros(shape, dtype=np.float64)
        self.hunger     = np.zeros(shape, dtype=np.float64)
        self.fatigue    = np.zeros(shape, dtype=np.float64)
        self.illness    = np.full(shape, ILLNESS_HEALTHY, dtype=np.int32)
        self.experience = np.zeros(shape, dtype=np.float64)
        self.age        = np.zeros(shape, dtype=np.int64)

        # Per-step bookkeeping fields.
        self.last_action          = np.full(shape, ACTION_REST, dtype=np.int32)
        self.last_routed_firm_id  = np.full(shape, NO_ROUTED_FIRM_ID, dtype=np.int32)
        self.last_action_success  = np.ones(shape, dtype=bool)
        self.last_reward          = np.zeros(shape, dtype=np.float64)
        self.last_wealth_change   = np.zeros(shape, dtype=np.float64)
        self.last_health_change   = np.zeros(shape, dtype=np.float64)
        self.last_crime_success   = np.zeros(shape, dtype=bool)
        self.displaced            = np.zeros(shape, dtype=bool)
        # A3: wealth acquired via theft, not yet cleared. Excluded from the
        # REWARD utility/transition computation (the act of stealing yields no
        # wealth-channel reward, ever) while remaining fully spendable IN-WORLD.
        # Grows only on a successful steal; each step capped to min(tainted,
        # wealth) so spending the stolen money draws the taint down (clears it)
        # while legitimate earnings (wealth above taint) are rewarded normally.
        self.tainted_wealth       = np.zeros(shape, dtype=np.float64)

    @property
    def size(self) -> int:
        """Number of households per world (uniform across K)."""
        return self.N

    @property
    def ids(self) -> np.ndarray:
        """Household IDs as a (K, N) array. ID == column index for Stage 1."""
        return np.broadcast_to(
            np.arange(self.N, dtype=np.int32), (self.K, self.N),
        )


class BatchedFirmState:
    """Slot-based storage for K worlds x F_max firm slots per world.

    Slots are filled by `spawn(world_idx, ...)` in sequential per-world order
    and never reused. `size[k]` tracks how many slots are allocated in world k.
    """

    __slots__ = (
        "K", "max_firms", "size",
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

    def __init__(self, K: int, max_firms: int):
        self.K = int(K)
        self.max_firms = int(max_firms)
        shape = (self.K, self.max_firms)

        # Per-world count of slots ever spawned (alive or dead).
        self.size = np.zeros(self.K, dtype=np.int32)

        self.alive             = np.zeros(shape, dtype=bool)
        self.id                = np.full(shape, -1, dtype=np.int32)

        self.price             = np.zeros(shape, dtype=np.float64)
        self.wage_offer        = np.zeros(shape, dtype=np.float64)
        self.cash              = np.zeros(shape, dtype=np.float64)
        self.capacity          = np.zeros(shape, dtype=np.float64)
        self.target_capacity   = np.zeros(shape, dtype=np.float64)
        self.stock             = np.zeros(shape, dtype=np.float64)
        self.productivity      = np.zeros(shape, dtype=np.float64)
        self.sales             = np.zeros(shape, dtype=np.float64)
        self.hires             = np.zeros(shape, dtype=np.int32)
        self.unmet_demand      = np.zeros(shape, dtype=np.float64)
        self.profit            = np.zeros(shape, dtype=np.float64)
        self.prev_price        = np.zeros(shape, dtype=np.float64)
        self.prev_wage         = np.zeros(shape, dtype=np.float64)
        self.negative_cash_weeks = np.zeros(shape, dtype=np.int32)
        self.window_sales      = np.zeros(shape, dtype=np.float64)
        self.window_hires      = np.zeros(shape, dtype=np.int32)
        self.window_unmet      = np.zeros(shape, dtype=np.float64)
        self.window_revenue    = np.zeros(shape, dtype=np.float64)
        self.window_wage_bill  = np.zeros(shape, dtype=np.float64)
        self.disaster_downtime = np.zeros(shape, dtype=np.int32)

    def num_alive(self, world_idx: int) -> int:
        return int(self.alive[world_idx, : int(self.size[world_idx])].sum())

    def num_alive_per_world(self) -> np.ndarray:
        """Return shape (K,) int - count of alive firms in each world."""
        return self.alive.sum(axis=1).astype(np.int64)

    def alive_mask(self) -> np.ndarray:
        """Full (K, F_max) bool mask. Note that slots beyond `size[k]` are
        unallocated; those positions are guaranteed False (the spawn path is
        the only writer of `alive=True`)."""
        return self.alive

    def alive_slots_in_world(self, world_idx: int) -> np.ndarray:
        """Slot indices of alive firms in one world, ascending order."""
        return np.where(self.alive[world_idx, : int(self.size[world_idx])])[0]

    def slot_of(self, world_idx: int, firm_id: int) -> int:
        """For Stage 1 / Stage 2 semantics, ID == slot index per world.
        Returns -1 if out of range or dead."""
        fid = int(firm_id)
        k = int(world_idx)
        if 0 <= fid < int(self.size[k]) and self.alive[k, fid]:
            return fid
        return -1

    def spawn(
        self,
        world_idx: int,
        *,
        price: float,
        wage: float,
        cash: float,
        capacity: float,
        stock: float,
        productivity: float = FIRM_PRODUCTIVITY_DEFAULT,
    ) -> int:
        """Allocate the next slot in `world_idx`. Returns the slot index, which
        equals the firm's per-world ID. Raises if the slot table is full.
        """
        k = int(world_idx)
        slot = int(self.size[k])
        if slot >= self.max_firms:
            raise RuntimeError(
                f"BatchedFirmState capacity {self.max_firms} exhausted in world {k}"
            )
        self.size[k] = slot + 1

        self.alive[k, slot] = True
        self.id[k, slot] = slot

        self.price[k, slot] = price
        self.wage_offer[k, slot] = wage
        self.cash[k, slot] = cash
        self.capacity[k, slot] = capacity
        self.target_capacity[k, slot] = capacity
        self.stock[k, slot] = stock
        self.productivity[k, slot] = max(productivity, 0.0)

        self.sales[k, slot] = 0.0
        self.hires[k, slot] = 0
        self.unmet_demand[k, slot] = 0.0
        self.profit[k, slot] = 0.0
        self.prev_price[k, slot] = price
        self.prev_wage[k, slot] = wage
        self.negative_cash_weeks[k, slot] = 0
        self.window_sales[k, slot] = 0.0
        self.window_hires[k, slot] = 0
        self.window_unmet[k, slot] = 0.0
        self.window_revenue[k, slot] = 0.0
        self.window_wage_bill[k, slot] = 0.0
        self.disaster_downtime[k, slot] = 0

        return slot

    def kill(self, world_idx: int, slot: int) -> None:
        """Mark a slot dead in one world. Slot stays allocated."""
        k = int(world_idx)
        s = int(slot)
        if 0 <= s < int(self.size[k]):
            self.alive[k, s] = False
