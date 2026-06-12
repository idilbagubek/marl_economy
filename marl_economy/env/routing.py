"""Market routing for Work/Shop/Clinic actions.

Two layers:

* `route_*_vec` - vectorized batch routing. One softmax per action type,
  one `rng.choice` per action type producing a slot index per household in
  one call. Used by the world loop. Returns -1 for refused households.

* `route_steal_target` - per-thief scalar routing. Called once per successful
  thief, keeps its rng draw scalar to match the legacy semantics.

The vectorized variants preserve the legacy semantic that work routing caps
hires per firm by their `cash + FIRM_HIRE_CREDIT_LIMIT` budget: any worker
who lands on a firm beyond its hire-budget is refused, mirroring the way the
scalar `route_work` + `firm_can_hire` interaction dropped firms from the
eligibility set as hires accumulated through the per-household loop.

RNG note (Stage 2): batch `rng.choice(..., size=K, p=probs)` consumes RNG in
a different order than K separate scalar calls. Per-seed trajectories will
differ from the post-Stage-1 baseline, but the statistical distributions of
all outcomes are preserved.
"""

from __future__ import annotations

import numpy as np

from marl_economy.config.constants import (
    KAPPA_WAGE, KAPPA_PRICE, ROUTING_MIN_STOCK,
    FIRM_HIRE_CREDIT_LIMIT, FIRM_HIRE_MIN_CAPACITY,
    FIRM_INSOLVENT_CASH_THRESHOLD,
)
from marl_economy.env.state import FirmState
from marl_economy.env.batched_state import BatchedFirmState


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax that guarantees sum to 1."""
    logits = np.asarray(logits, dtype=np.float64)
    if len(logits) == 0:
        return logits
    if not np.all(np.isfinite(logits)):
        logits = np.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)

    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    total = np.sum(exp_vals)
    if not np.isfinite(total) or total <= 0.0:
        return np.full_like(logits, 1.0 / len(logits), dtype=np.float64)

    probs = exp_vals / total
    prob_sum = probs.sum()
    if not np.isfinite(prob_sum) or prob_sum <= 0.0:
        return np.full_like(logits, 1.0 / len(logits), dtype=np.float64)
    return probs / prob_sum


def route_work_vec(
    f: FirmState,
    work_mask: np.ndarray,
    alive_slots: np.ndarray,
    min_wage_floor: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Batch-route every WORK household to a firm slot in one pass.

    Parameters
    ----------
    f
        FirmState. `f.hires[slot]` is MUTATED in-place: each accepted match
        increments the firm's hire count.
    work_mask
        Shape (N,) bool. True for households whose action is WORK this week.
    alive_slots
        Slot indices of currently alive firms, in slot order.
    min_wage_floor
        Households reject firms paying less than this wage.
    rng
        numpy Generator.

    Returns
    -------
    np.ndarray, shape (N,), dtype int32
        For each household: firm slot index, or -1 if the household tried to
        work but no firm could take them (refused).

    Semantics
    ---------
    The scalar `route_work` re-checked `firm_can_hire` for every household,
    so firms dropped out of the eligibility set as they filled. Here we
    approximate that by:
      1. Filtering `alive_slots` to firms eligible at the start of the
         tick (`firm_can_hire` predicate, vectorized).
      2. Softmax-sampling one firm per worker by `wage_offer`.
      3. Computing each firm's remaining hire budget
         `floor((cash + credit_limit - hires*wage) / wage)` and refusing
         any worker beyond that quota, in worker (slot) order.
    """
    n_total = int(work_mask.shape[0])
    n_workers = int(work_mask.sum())
    out = np.full(n_total, -1, dtype=np.int32)
    if n_workers == 0 or alive_slots.size == 0:
        return out

    # Vectorized firm_can_hire at the start of the tick.
    cash = f.cash[alive_slots]
    wage = f.wage_offer[alive_slots]
    hires = f.hires[alive_slots]
    capacity = f.capacity[alive_slots]
    downtime = f.disaster_downtime[alive_slots]

    elig_mask = (
        (downtime == 0)
        & (wage >= min_wage_floor)
        & (cash >= FIRM_INSOLVENT_CASH_THRESHOLD)
        & (capacity >= FIRM_HIRE_MIN_CAPACITY)
        & (((hires + 1) * wage) <= (cash + FIRM_HIRE_CREDIT_LIMIT))
    )
    eligible_slots = alive_slots[elig_mask]
    if eligible_slots.size == 0:
        return out

    # Softmax over wage_offer of eligible firms.
    wages_elig = f.wage_offer[eligible_slots]
    probs = softmax(KAPPA_WAGE * wages_elig)

    # One batch RNG draw: pick a firm-among-eligible index per worker.
    chosen_idx = rng.choice(eligible_slots.size, size=n_workers, p=probs)

    # Per-firm hire budget: max additional workers each firm can accept.
    cash_elig = f.cash[eligible_slots]
    wage_elig = f.wage_offer[eligible_slots]
    hires_elig = f.hires[eligible_slots].astype(np.int64)
    # remaining_budget = cash + credit - hires * wage
    # max_extra = floor(remaining_budget / wage)
    remaining_budget = cash_elig + FIRM_HIRE_CREDIT_LIMIT - hires_elig * wage_elig
    max_extra = np.maximum(0, np.floor(remaining_budget / wage_elig).astype(np.int64))

    # Compute each worker's RANK within its assigned firm (in worker-slot order).
    # Workers are processed in slot order (the order they appear in work_mask),
    # so within-firm rank = number of earlier workers (lower slot index) that
    # also chose this firm.
    order = np.argsort(chosen_idx, kind="stable")
    sorted_firms = chosen_idx[order]
    group_change = np.concatenate(([True], sorted_firms[1:] != sorted_firms[:-1]))
    group_starts = np.where(group_change)[0]
    group_id_at_pos = np.cumsum(group_change) - 1
    rank_in_sorted = np.arange(chosen_idx.size) - group_starts[group_id_at_pos]
    ranks = np.empty(chosen_idx.size, dtype=np.int64)
    ranks[order] = rank_in_sorted

    accepted = ranks < max_extra[chosen_idx]

    # Write accepted slot assignments back into the (N,) output array.
    worker_idxs = np.where(work_mask)[0]  # household indices, in slot order
    accepted_slots = eligible_slots[chosen_idx[accepted]]
    out[worker_idxs[accepted]] = accepted_slots.astype(np.int32)

    # Bulk-increment f.hires for each firm by its accepted count.
    accepted_chosen = chosen_idx[accepted]
    if accepted_chosen.size > 0:
        counts = np.bincount(accepted_chosen, minlength=eligible_slots.size)
        # Add to f.hires[eligible_slots] in-place via fancy indexing.
        np.add.at(f.hires, eligible_slots, counts.astype(f.hires.dtype))

    return out


def route_shop_vec(
    f: FirmState,
    shop_mask: np.ndarray,
    alive_slots: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Batch-route every SHOP household.

    Returns shape (N,) int32 - firm slot per household, -1 if no eligible firm.
    """
    n_total = int(shop_mask.shape[0])
    n_shoppers = int(shop_mask.sum())
    out = np.full(n_total, -1, dtype=np.int32)
    if n_shoppers == 0 or alive_slots.size == 0:
        return out

    elig_mask = (
        (f.disaster_downtime[alive_slots] == 0)
        & (f.stock[alive_slots] >= ROUTING_MIN_STOCK)
    )
    eligible_slots = alive_slots[elig_mask]
    if eligible_slots.size == 0:
        return out

    prices_elig = f.price[eligible_slots]
    probs = softmax(-KAPPA_PRICE * prices_elig)
    chosen_idx = rng.choice(eligible_slots.size, size=n_shoppers, p=probs)

    shopper_idxs = np.where(shop_mask)[0]
    out[shopper_idxs] = eligible_slots[chosen_idx].astype(np.int32)
    return out


def route_clinic_vec(
    f: FirmState,
    clinic_mask: np.ndarray,
    alive_slots: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Batch-route every CLINIC household. Same predicate as shop."""
    return route_shop_vec(f, clinic_mask, alive_slots, rng)


def route_steal_target(
    f: FirmState,
    alive_slots: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """Pick a firm to steal from. Uniform among alive firms with positive cash.

    Kept scalar - called once per successful thief, which is rare per week.
    """
    eligible = [int(slot) for slot in alive_slots.tolist() if f.cash[slot] > 0]
    if not eligible:
        return -1
    return int(rng.choice(eligible))


# ---------------------------------------------------------------------------
# K-vectorized routing (Stage 3, refactored 2026-05-14 for per-world RNGs).
#
# These functions operate on the whole BatchedFirmState (K worlds at once)
# and accept a per-world list of `np.random.Generator` instances. Each call
# delegates per-world to the single-world `route_*_vec` implementations with
# that world's rng - preserving per-seed reproducibility across different K
# values (Cost 3 fix from the "better design" refactor).
#
# Why a per-world loop, not pad-and-mask K-vec: per-seed reproducibility
# requires each world's rng to be advanced independently in its own draw
# order (work_routing -> shop_routing -> clinic_routing -> household_step).
# Sharing one batched `rng.random(size=K*M)` across worlds would mean world
# 0's routing draws depend on K, breaking the property
# `BatchedWorld(K, seed)[world 0] == BatchedWorld(1, seed)[world 0]`.
# ---------------------------------------------------------------------------


class _BatchedFirmSlice:
    """Single-world view of one row of a BatchedFirmState.

    Numpy basic indexing (`fb.arr[k]` for integer k) returns a view, so any
    in-place mutation by the delegated routing (e.g. `np.add.at(f.hires,...)`)
    propagates back to the batched state.
    """

    __slots__ = (
        "alive", "id", "price", "wage_offer", "cash", "capacity",
        "target_capacity", "stock", "productivity", "hires",
        "disaster_downtime",
    )

    def __init__(self, fb: BatchedFirmState, k: int):
        self.alive = fb.alive[k]
        self.id = fb.id[k]
        self.price = fb.price[k]
        self.wage_offer = fb.wage_offer[k]
        self.cash = fb.cash[k]
        self.capacity = fb.capacity[k]
        self.target_capacity = fb.target_capacity[k]
        self.stock = fb.stock[k]
        self.productivity = fb.productivity[k]
        self.hires = fb.hires[k]
        self.disaster_downtime = fb.disaster_downtime[k]


def route_shop_vec_kn(
    f: BatchedFirmState,
    shop_mask_kn: np.ndarray,
    rngs: list[np.random.Generator],
) -> np.ndarray:
    """K-vectorized shop routing. Returns (K, N) int32 - firm slot per
    household, or -1 where the household did not shop or no firm was eligible.

    Each world's draws come from its own `rngs[k]`, in shop-routing slot
    order, so the per-world rng state matches what a single `World` with the
    same seed would have produced.
    """
    shop_mask_kn = np.asarray(shop_mask_kn, dtype=bool)
    K, N = shop_mask_kn.shape
    out = np.full((K, N), -1, dtype=np.int32)
    for k in range(K):
        view = _BatchedFirmSlice(f, k)
        alive_slots = np.where(view.alive[: int(f.size[k])])[0]
        out[k] = route_shop_vec(view, shop_mask_kn[k], alive_slots, rngs[k])
    return out


def route_clinic_vec_kn(
    f: BatchedFirmState,
    clinic_mask_kn: np.ndarray,
    rngs: list[np.random.Generator],
) -> np.ndarray:
    """K-vectorized clinic routing. Same predicate / softmax as shop routing.
    Per-world rng draws consumed identically to a single `World` call.
    """
    return route_shop_vec_kn(f, clinic_mask_kn, rngs)


def route_work_vec_kn(
    f: BatchedFirmState,
    work_mask_kn: np.ndarray,
    min_wage_floor_k: np.ndarray,
    rngs: list[np.random.Generator],
) -> np.ndarray:
    """K-vectorized work routing. Returns (K, N) int32 - firm slot assigned
    to each worker, or -1 if the household did not work or was refused.

    Mutates `f.hires` in place: each accepted worker adds 1 to its firm's
    hire counter for the world.

    Each world's draws come from its own `rngs[k]`. Per-world hire-budget
    semantics match single-world `route_work_vec` exactly.
    """
    work_mask_kn = np.asarray(work_mask_kn, dtype=bool)
    min_wage_floor_k = np.asarray(min_wage_floor_k, dtype=np.float64)
    K, N = work_mask_kn.shape
    out = np.full((K, N), -1, dtype=np.int32)
    for k in range(K):
        view = _BatchedFirmSlice(f, k)
        alive_slots = np.where(view.alive[: int(f.size[k])])[0]
        out[k] = route_work_vec(
            view, work_mask_kn[k], alive_slots,
            float(min_wage_floor_k[k]), rngs[k],
        )
    return out
