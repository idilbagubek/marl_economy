"""Household dynamics: vectorized per-week step over the full N-household
population stored in `HouseholdState`.
"""

from __future__ import annotations

import numpy as np

from marl_economy.config.constants import (
    EPSILON,
    ACTION_WORK, ACTION_SHOP, ACTION_STEAL, ACTION_REST, ACTION_INVEST, ACTION_CLINIC,
    HEALTH_DECAY_BASE, HEALTH_DECAY_HEALTHCARE_FACTOR,
    WEALTH_DRAG_BASE, WEALTH_DRAG_LOW_COEFF, WEALTH_DRAG_HIGH_FIXED,
    WEALTH_DRAG_HIGH_COEFF, WEALTH_DRAG_HIGH_EXP, WEALTH_DRAG_THRESHOLD,
    EXPERIENCE_CAP, WORK_EXPERIENCE_INCREMENT, MOTIVATION_WEALTH_FACTOR, WAGE_GAP_EXP_FACTOR,
    WAGE_GAP_DEFAULT,
    WORK_NOISE_MIN, WORK_NOISE_MAX,
    INVEST_RETURN_BASE, INVEST_RETURN_EDU_COEFF, INVEST_RETURN_WEALTH_FACTOR,
    INVEST_CRASH_PROB_BASE, INVEST_CRASH_PROB_MIN, INVEST_CRASH_PROB_MAX, INVEST_CRASH_LOSS,
    CRIME_LOGISTIC_SCALE, CRIME_LOGISTIC_CENTER, CRIME_WELFARE_FACTOR,
    CRIME_STEAL_ATTEMPT_MIN, CRIME_STEAL_ATTEMPT_MAX,
    WELFARE_THRESHOLD, WELFARE_MULTIPLIER, HEALTH_SAFETY_NET_THRESHOLD,
    SHOP_COST_FRACTION, SHOP_HEALTH_GAIN, CLINIC_HEALTH_GAIN_SCALE, HEALTH_SAFETY_NET_BOOST_SCALE,
    FAILED_STEAL_HEALTH_PENALTY, REST_HEALTH_RECOVERY, LOG_UTILITY_HEALTH_WEIGHT,
    HOUSEHOLD_HEALTH_LEVEL_WEIGHT, HEALTH_LEVEL_NEUTRAL,
    HOUSEHOLD_FED_WEIGHT, HUNGER_NEUTRAL,
    DEATH_THRESHOLD, DEATH_TRANSITION_PENALTY, DEAD_STATE_REWARD,
    BANKRUPTCY_THRESHOLD, BANKRUPTCY_TRANSITION_PENALTY, STEAL_TAINT_CARRY_PENALTY,
    CLINIC_COST_MULTIPLIER, FIRM_STOCK_TRANSACTION_UNITS,
    CB_ALPHA_R, CB_BETA_R,
)
from marl_economy.env.dynamics_hunger import (
    advance_hunger_vec,
    hunger_health_decay_bonus_vec,
    hunger_productivity_multiplier_vec,
    is_hunger_crisis_vec,
)
from marl_economy.env.dynamics_fatigue import (
    advance_fatigue_vec,
    fatigue_health_decay_bonus_vec,
    fatigue_productivity_multiplier_vec,
)
from marl_economy.env.illness import (
    illness_health_decay_bonus_vec,
    illness_productivity_multiplier_vec,
)
from marl_economy.env.state import HouseholdState, FirmState, NO_ROUTED_FIRM_ID
from marl_economy.env.batched_state import BatchedHouseholdState, BatchedFirmState

# Money-conservation load-bearing invariant. SHOP/CLINIC conservation holds
# because the household is charged `price * fraction` while the firm is
# credited `unit_price * FIRM_STOCK_TRANSACTION_UNITS` (firm_sell_slot), and
# the `_ration_stock` budget is `floor(stock / FIRM_STOCK_TRANSACTION_UNITS)`.
# These line up EXACTLY only at unit == 1.0. If this constant is ever retuned,
# the charge/credit symmetry breaks and money is silently destroyed/minted -
# fail loudly here instead so the symmetry is fixed deliberately (see the
# 2026-05-17 SHOP/CLINIC conservation fix worklog).
assert FIRM_STOCK_TRANSACTION_UNITS == 1.0, (
    "SHOP/CLINIC money conservation + stock rationing assume "
    f"FIRM_STOCK_TRANSACTION_UNITS == 1.0, got {FIRM_STOCK_TRANSACTION_UNITS!r}. "
    "Re-deriving the charge/credit symmetry is required before changing it."
)


def compute_crime_success_prob(policing: float, welfare: float) -> float:
    """Logistic in policing, mildly damped by welfare."""
    logit = CRIME_LOGISTIC_SCALE * (CRIME_LOGISTIC_CENTER - policing)
    sigmoid = 1.0 / (1.0 + np.exp(-logit))
    return sigmoid * (1.0 - CRIME_WELFARE_FACTOR * welfare)


def _wealth_drag_vec(wealth: np.ndarray) -> np.ndarray:
    low = wealth < WEALTH_DRAG_THRESHOLD
    out = np.empty_like(wealth, dtype=np.float64)
    out[low] = WEALTH_DRAG_BASE + WEALTH_DRAG_LOW_COEFF * wealth[low]
    high = ~low
    if np.any(high):
        excess = wealth[high] - WEALTH_DRAG_THRESHOLD
        out[high] = (WEALTH_DRAG_BASE + WEALTH_DRAG_HIGH_FIXED
                     + WEALTH_DRAG_HIGH_COEFF * np.power(excess, WEALTH_DRAG_HIGH_EXP))
    return out


def _ration_stock(
    k_idx: np.ndarray, slots: np.ndarray,
    wealth_ok: np.ndarray, stock_budget: np.ndarray,
) -> np.ndarray:
    """SHOP/CLINIC stock-rationing for money conservation.

    Decide which requesters in a SHOP-or-CLINIC block are actually served,
    bounded by the firm's deliverable stock so the household-step charge can
    never exceed what the firm-sell phase will credit (the pre-existing leak:
    households were charged on a stale, never-decremented stock read, then the
    firm-sell phase silently failed once stock ran out -> money destroyed).

    `k_idx`, `slots`, `wealth_ok` are aligned 1-D arrays in the SAME order
    `np.where(<action>_with_firm)` produced them - ascending world, then
    ascending household index - which is exactly the order the scalar
    firm-sell phase (`for k: for i in range(num_households)`) consumes stock.
    `stock_budget` is a mutable (K, F_max) int array of the per-(world,firm)
    remaining deliverable-transaction count (= floor(stock / unit)); it is
    decremented in place so a later CLINIC pass draws from the SHOP remainder
    (SHOP-priority, deterministic, no RNG -> per-world reproducibility intact).

    A request is granted iff it is wealth-affordable AND its firm still has
    budget. Granted count per firm <= floor(stock) => every granted household
    gets a successful sale in the firm-sell phase => firm-credit ==
    household-charge exactly => Delta money = 0 (conserved by construction). Floor
    on the FP stock only ever UNDER-serves by <=1 on a rare tie - the
    conservation-safe side (never re-introduces an over-charge).
    """
    granted = np.zeros(k_idx.shape[0], dtype=bool)
    for t in range(k_idx.shape[0]):
        if not wealth_ok[t]:
            continue
        kk = int(k_idx[t])
        ss = int(slots[t])
        if stock_budget[kk, ss] > 0:
            stock_budget[kk, ss] -= 1
            granted[t] = True
    return granted


def _household_reward_vec(wealth_old: np.ndarray, health_old: np.ndarray,
                          wealth_new: np.ndarray, health_new: np.ndarray,
                          hunger_new: np.ndarray,
                          episode_progress: float,
                          tainted_new=0.0) -> np.ndarray:
    """Phase-3 household reward: clamped log-utility delta + dense wellbeing
    LEVEL term, then one-time bankruptcy/death transition penalties and a
    bounded persistent dead-state, then the A3 tainted-wealth carry penalty.

    - log args clamped at BANKRUPTCY_THRESHOLD / DEATH_THRESHOLD (kills the
      log(1e-6)~=-13.8 spike that swamped the delta near zero).
    - LEVEL term bounded [-0.30, +0.30]/wk - the signal PPO must learn.
    - Death is a ONE-TIME transition penalty + a bounded persistent dead-state
      (not "-8 every week"); no-suicide-incentive: DEAD_STATE_REWARD <
      worst sustainable-alive. `episode_progress` retained for signature
      compatibility but no longer used (scaling dropped).
    """
    util_old = (np.log(np.maximum(wealth_old, BANKRUPTCY_THRESHOLD))
                + LOG_UTILITY_HEALTH_WEIGHT * np.log(np.maximum(health_old, DEATH_THRESHOLD)))
    util_new = (np.log(np.maximum(wealth_new, BANKRUPTCY_THRESHOLD))
                + LOG_UTILITY_HEALTH_WEIGHT * np.log(np.maximum(health_new, DEATH_THRESHOLD)))
    rewards = util_new - util_old
    rewards = (
        rewards
        + HOUSEHOLD_HEALTH_LEVEL_WEIGHT * (health_new - HEALTH_LEVEL_NEUTRAL)
        + HOUSEHOLD_FED_WEIGHT * (HUNGER_NEUTRAL - hunger_new)
    )

    # One-time bankruptcy transition (wealth crosses below the threshold).
    newly_bankrupt = (wealth_old > BANKRUPTCY_THRESHOLD) & (wealth_new <= BANKRUPTCY_THRESHOLD)
    rewards = np.where(newly_bankrupt, BANKRUPTCY_TRANSITION_PENALTY, rewards)

    # Death dominates bankruptcy: one-time transition + bounded persistent
    # dead-state (death not absorbing - safety net can revive; re-death re-fires
    # the transition). newly_dead / already_dead are mutually exclusive.
    newly_dead = (health_old > DEATH_THRESHOLD) & (health_new <= DEATH_THRESHOLD)
    already_dead = (health_old <= DEATH_THRESHOLD) & (health_new <= DEATH_THRESHOLD)
    rewards = np.where(newly_dead, DEATH_TRANSITION_PENALTY, rewards)
    rewards = np.where(already_dead, DEAD_STATE_REWARD, rewards)
    # A3 carry penalty: claw back lambda*(currently-held tainted/stolen wealth) every
    # step it is still held. Stolen money is already wealth-channel-excluded,
    # but it still buys real SHOP/CLINIC consumption (health+hunger reward) over
    # the weeks it is held; this prices that multi-week value over exactly those
    # weeks (taint falls only via genuine consumption_outflow). Applied AFTER
    # the cliffs/dead-state so death/bankruptcy cannot launder held stolen money
    # (no suicide-to-escape). Non-thieves have tainted_new==0 -> zero effect.
    rewards = rewards - STEAL_TAINT_CARRY_PENALTY * tainted_new
    return rewards


def compute_household_reward(
    wealth_old: float, health_old: float,
    wealth_new: float, health_new: float,
    hunger_new: float,
    episode_progress: float = 0.5,
    tainted_new: float = 0.0,
) -> float:
    """Scalar reward computation (used by the theft-clawback recompute).

    Kept as a thin scalar wrapper around the vectorized version so the
    callsite path remains a single, audited implementation. `hunger_new` is the
    household's post-step hunger (unchanged by the theft clawback, which only
    adjusts wealth) so the recompute stays consistent with the per-step reward.
    `tainted_new` is the household's post-clawback tainted (stolen) wealth so
    the A3 carry penalty applies on the theft step too (defaults 0 for any
    caller without taint state - same as the pre-A3-carry behavior).
    """
    out = _household_reward_vec(
        np.array([wealth_old], dtype=np.float64),
        np.array([health_old], dtype=np.float64),
        np.array([wealth_new], dtype=np.float64),
        np.array([health_new], dtype=np.float64),
        np.array([hunger_new], dtype=np.float64),
        episode_progress,
        np.array([tainted_new], dtype=np.float64),
    )
    return float(out[0])


def step_households_batch(
    h: HouseholdState,
    f: FirmState,
    actions: np.ndarray,
    routed_firm_slots: np.ndarray,
    policy_state: dict,
    interest_rate: float,
    rng: np.random.Generator,
    episode_progress: float = 0.5,
) -> dict:
    """Advance every household one week and write the results into `h` in place.

    `actions`            - int32 (N,) per-household action enum.
    `routed_firm_slots`  - int32 (N,) firm slot index, or NO_ROUTED_FIRM_ID (-1)
                           for actions that do not consume a firm.
    `policy_state`       - current regime policy dict.
    `interest_rate`      - central bank rate (scalar).
    `rng`                - numpy Generator.
    `episode_progress`   - float in [0, 1] for early-death penalty scaling.

    Returns a small diagnostic dict the world layer uses for per-step
    bookkeeping (event counts, firm-side effects). Per-agent fields like
    wealth, hunger, etc. are written directly to `h`.
    """
    n = h.size
    if n == 0:
        empty = np.zeros(0, dtype=bool)
        empty_int = np.zeros(0, dtype=np.int32)
        return {
            "work_matched": empty,
            "shop_matched": empty,
            "clinic_matched": empty,
            "crime_occurred": empty,
            "crime_success": empty,
            "steal_amount": np.zeros(0, dtype=np.float64),
            "hunger_crisis": empty,
            "action_success": empty,
            "routed_firm_slots": empty_int,
        }

    actions_arr = np.asarray(actions, dtype=np.int32)
    routed_firm_slots = np.asarray(routed_firm_slots, dtype=np.int32)
    has_firm = routed_firm_slots != NO_ROUTED_FIRM_ID

    # Snapshot pre-step values so we can compute reward deltas at the end.
    w_old = h.wealth.copy()
    h_old = h.health.copy()
    hunger_old = h.hunger.copy()
    fatigue_old = h.fatigue.copy()
    illness_old = h.illness.copy()  # int32

    # Mutable working copies for the step.
    w_new = w_old.copy()
    h_new = h_old.copy()
    experience = h.experience.copy()

    action_success = np.ones(n, dtype=bool)
    crime_occurred = np.zeros(n, dtype=bool)
    crime_success = np.zeros(n, dtype=bool)
    work_matched = np.zeros(n, dtype=bool)
    shop_matched = np.zeros(n, dtype=bool)
    clinic_matched = np.zeros(n, dtype=bool)
    steal_amount = np.zeros(n, dtype=np.float64)

    # 1) Health decay (baseline + hunger + fatigue + illness) and wealth drag.
    base_decay = HEALTH_DECAY_BASE * max(
        0.0, HEALTH_DECAY_HEALTHCARE_FACTOR - policy_state["healthcare_access"]
    )
    h_new = (
        h_new
        - base_decay
        - hunger_health_decay_bonus_vec(hunger_old)
        - fatigue_health_decay_bonus_vec(fatigue_old)
        - illness_health_decay_bonus_vec(illness_old)
    )

    drag = _wealth_drag_vec(w_old)
    w_new = np.maximum(0.0, w_new - drag)

    # Helper: read a per-household firm-slot field, defaulting to a fill value
    # where no firm was routed (the mask filters those out before use anyway).
    def _firm_field(field: np.ndarray, mask: np.ndarray, fill: float = 0.0) -> np.ndarray:
        idxs = routed_firm_slots[mask]
        return field[idxs] if idxs.size > 0 else np.zeros(0, dtype=field.dtype)

    # 2) Per-action effects.

    # WORK.
    work_mask = (actions_arr == ACTION_WORK)
    work_with_firm = work_mask & has_firm
    work_no_firm = work_mask & ~has_firm
    if np.any(work_no_firm):
        action_success[work_no_firm] = False
    if np.any(work_with_firm):
        idxs = np.where(work_with_firm)[0]
        wage_offers = f.wage_offer[routed_firm_slots[idxs]]
        wage_gap = WAGE_GAP_DEFAULT
        exp_mult = np.minimum(
            np.exp(wage_gap * experience[idxs] * WAGE_GAP_EXP_FACTOR),
            EXPERIENCE_CAP,
        )
        motivation = 1.0 / (1.0 + MOTIVATION_WEALTH_FACTOR * w_new[idxs])
        noise = rng.uniform(WORK_NOISE_MIN, WORK_NOISE_MAX, size=idxs.size)
        wages = wage_offers * exp_mult * motivation * noise * (1.0 - policy_state["tax_rate"])
        wages = np.maximum(0.0, wages)
        # Hunger / fatigue / illness damp wages multiplicatively (pre-step values).
        wages *= hunger_productivity_multiplier_vec(hunger_old[idxs])
        wages *= fatigue_productivity_multiplier_vec(fatigue_old[idxs])
        wages *= illness_productivity_multiplier_vec(illness_old[idxs])
        w_new[idxs] += wages
        work_matched[idxs] = True
        experience[idxs] = np.minimum(
            experience[idxs] + WORK_EXPERIENCE_INCREMENT, EXPERIENCE_CAP
        )

    # SHOP. Charges SHOP_COST_FRACTION * posted_price.
    shop_mask = (actions_arr == ACTION_SHOP)
    shop_with_firm = shop_mask & has_firm
    shop_no_firm = shop_mask & ~has_firm
    if np.any(shop_no_firm):
        action_success[shop_no_firm] = False
    if np.any(shop_with_firm):
        idxs = np.where(shop_with_firm)[0]
        prices = f.price[routed_firm_slots[idxs]]
        shop_costs = prices * SHOP_COST_FRACTION
        stocks = f.stock[routed_firm_slots[idxs]]
        affordable = (w_new[idxs] >= shop_costs) & (stocks >= FIRM_STOCK_TRANSACTION_UNITS)
        succ_local = idxs[affordable]
        fail_local = idxs[~affordable]
        if succ_local.size > 0:
            paid = shop_costs[affordable]
            w_new[succ_local] -= paid
            h_new[succ_local] += SHOP_HEALTH_GAIN
            shop_matched[succ_local] = True
        if fail_local.size > 0:
            action_success[fail_local] = False

    # STEAL: roll outcomes here; firm-side cash transfer happens in world.step.
    steal_mask = (actions_arr == ACTION_STEAL)
    if np.any(steal_mask):
        idxs = np.where(steal_mask)[0]
        crime_occurred[idxs] = True
        success_prob = compute_crime_success_prob(
            policy_state["policing"], policy_state["welfare"]
        )
        attempts = rng.uniform(
            CRIME_STEAL_ATTEMPT_MIN, CRIME_STEAL_ATTEMPT_MAX, size=idxs.size
        )
        rolls = rng.random(size=idxs.size)
        succ = rolls < success_prob
        succ_local = idxs[succ]
        fail_local = idxs[~succ]
        if succ_local.size > 0:
            steal_amount[succ_local] = attempts[succ]
            w_new[succ_local] += attempts[succ]
            crime_success[succ_local] = True
        if fail_local.size > 0:
            h_new[fail_local] -= FAILED_STEAL_HEALTH_PENALTY
            action_success[fail_local] = False

    # REST.
    rest_mask = (actions_arr == ACTION_REST)
    if np.any(rest_mask):
        h_new[rest_mask] += REST_HEALTH_RECOVERY

    # INVEST.
    invest_mask = (actions_arr == ACTION_INVEST)
    if np.any(invest_mask):
        idxs = np.where(invest_mask)[0]
        edu = float(policy_state["education_quality"])
        r_eff_base = INVEST_RETURN_BASE + INVEST_RETURN_EDU_COEFF * edu
        r_eff = (
            r_eff_base * (1.0 - CB_ALPHA_R * interest_rate)
            / (1.0 + INVEST_RETURN_WEALTH_FACTOR * w_new[idxs])
        )
        crash_prob = float(np.clip(
            INVEST_CRASH_PROB_BASE + CB_BETA_R * interest_rate,
            INVEST_CRASH_PROB_MIN, INVEST_CRASH_PROB_MAX,
        ))
        rolls = rng.random(size=idxs.size)
        crashed = rolls < crash_prob
        returns = np.where(crashed, -INVEST_CRASH_LOSS * w_new[idxs], r_eff * w_new[idxs])
        w_new[idxs] += returns

    # CLINIC. Charges price * CLINIC_COST_MULTIPLIER.
    clinic_mask = (actions_arr == ACTION_CLINIC)
    clinic_with_firm = clinic_mask & has_firm
    clinic_no_firm = clinic_mask & ~has_firm
    if np.any(clinic_no_firm):
        action_success[clinic_no_firm] = False
    if np.any(clinic_with_firm):
        idxs = np.where(clinic_with_firm)[0]
        prices = f.price[routed_firm_slots[idxs]]
        stocks = f.stock[routed_firm_slots[idxs]]
        clinic_prices = prices * CLINIC_COST_MULTIPLIER
        affordable = (w_new[idxs] >= clinic_prices) & (stocks >= FIRM_STOCK_TRANSACTION_UNITS)
        succ_local = idxs[affordable]
        fail_local = idxs[~affordable]
        if succ_local.size > 0:
            paid = clinic_prices[affordable]
            w_new[succ_local] -= paid
            h_new[succ_local] += CLINIC_HEALTH_GAIN_SCALE * policy_state["healthcare_access"]
            clinic_matched[succ_local] = True
        if fail_local.size > 0:
            action_success[fail_local] = False

    # 3) Welfare transfer and health safety net.
    welfare_lvl = float(policy_state["welfare"])
    welfare_amount = np.where(
        w_new < WELFARE_THRESHOLD,
        welfare_lvl * (WELFARE_THRESHOLD - w_new) / WELFARE_THRESHOLD * WELFARE_MULTIPLIER,
        0.0,
    )
    w_new = w_new + welfare_amount

    healthcare = float(policy_state["healthcare_access"])
    health_boost = np.where(
        h_new < HEALTH_SAFETY_NET_THRESHOLD,
        HEALTH_SAFETY_NET_BOOST_SCALE * healthcare,
        0.0,
    )
    h_new = h_new + health_boost

    # 4) Hunger and fatigue update.
    hunger_new = advance_hunger_vec(hunger_old, shop_matched)
    hunger_crisis = is_hunger_crisis_vec(hunger_new)

    rested_mask = (actions_arr == ACTION_REST)
    fatigue_new = advance_fatigue_vec(
        fatigue_old, work_matched, rested_mask, clinic_matched,
    )

    # 5) Clamp and finalize.
    w_new = np.maximum(0.0, w_new)
    h_new = np.clip(h_new, 0.0, 1.0)
    age_new = h.age + 1

    rewards = _household_reward_vec(w_old, h_old, w_new, h_new, hunger_new, episode_progress)

    # 6) Write back into HouseholdState. Illness is intentionally NOT updated
    # here - the world layer advances it after place-based transmission and
    # the clinic-capacity triage.
    h.wealth[:] = w_new
    h.health[:] = h_new
    h.hunger[:] = hunger_new
    h.fatigue[:] = fatigue_new
    h.experience[:] = experience
    h.age[:] = age_new
    h.last_action[:] = actions_arr
    h.last_action_success[:] = action_success
    h.last_reward[:] = rewards
    h.last_wealth_change[:] = w_new - w_old
    h.last_health_change[:] = h_new - h_old
    h.last_crime_success[:] = crime_success
    # last_routed_firm_id is written by the world layer (it has the firm-id mapping).

    return {
        "work_matched": work_matched,
        "shop_matched": shop_matched,
        "clinic_matched": clinic_matched,
        "crime_occurred": crime_occurred,
        "crime_success": crime_success,
        "steal_amount": steal_amount,
        "hunger_crisis": hunger_crisis,
        "action_success": action_success,
        "routed_firm_slots": routed_firm_slots,
    }


def step_households_batch_kn(
    h: BatchedHouseholdState,
    f: BatchedFirmState,
    actions: np.ndarray,
    routed_firm_slots: np.ndarray,
    tax_rate_k: np.ndarray,
    welfare_k: np.ndarray,
    policing_k: np.ndarray,
    healthcare_k: np.ndarray,
    education_k: np.ndarray,
    interest_rate_k: np.ndarray,
    rngs: list,
    episode_progress_k: np.ndarray,
) -> dict:
    """Advance every (world, household) one week in one vectorized pass.

    All per-world policy values are shape (K,) arrays; firm/household state
    is read/written through (K, N) and (K, F_max) views. Math mirrors
    `step_households_batch` exactly, but operations broadcast over a leading
    K dimension.

    Parameters
    ----------
    h, f
        BatchedHouseholdState, BatchedFirmState - mutated in place.
    actions
        (K, N) int32. Per-(world, household) action enum.
    routed_firm_slots
        (K, N) int32. -1 where no firm was routed.
    tax_rate_k, welfare_k, ...
        (K,) float64 per-world policy values.
    interest_rate_k
        (K,) float64 per-world central-bank rate.
    rngs
        List of K `np.random.Generator` instances - one per world. Each
        world's randomness is drawn from its own rng in a fixed order:
        WORK noise -> STEAL attempts -> STEAL rolls -> INVEST rolls. This
        matches the order single-world `step_households_batch` consumes its
        rng, so per-world reproducibility holds:
        `BatchedWorld(K, seed)[world i]` produces the same household
        trajectories as `BatchedWorld(1, seed_spawn_i)`.
    episode_progress_k
        (K,) float64 in [0, 1] for early-failure penalty scaling.

    Returns
    -------
    dict with keys identical to `step_households_batch`, but each value has
    shape (K, N) instead of (N,). Illness is intentionally NOT advanced - the
    world layer drives the illness state machine after place-based exposure.
    """
    K, N = h.K, h.N
    if K == 0 or N == 0:
        empty = np.zeros((K, N), dtype=bool)
        empty_int = np.zeros((K, N), dtype=np.int32)
        empty_f = np.zeros((K, N), dtype=np.float64)
        return {
            "work_matched": empty, "shop_matched": empty, "clinic_matched": empty,
            "crime_occurred": empty, "crime_success": empty,
            "steal_amount": empty_f, "hunger_crisis": empty,
            "action_success": empty, "routed_firm_slots": empty_int,
            "net_wage_by_firm": np.zeros_like(f.cash, dtype=np.float64),
            "tax_collected_k": np.zeros(K, dtype=np.float64),
            "welfare_paid_k": np.zeros(K, dtype=np.float64),
            "tainted_old": np.zeros((K, N), dtype=np.float64),
            "drag_destroyed_k": np.zeros(K, dtype=np.float64),       # C1 D
            "invest_minted_k": np.zeros(K, dtype=np.float64),        # C1 F
            "invest_destroyed_k": np.zeros(K, dtype=np.float64),     # C1 F
            "shop_clinic_unmet_k": np.zeros(K, dtype=np.int64),
        }

    # C-contiguity is a load-bearing determinism invariant: every
    # `np.where(<(K,N) mask>)` here (RNG-alignment via per-world count cumsum
    # for WORK/STEAL/INVEST, AND `_ration_stock`) requires ROW-major
    # (world-major then household-major) traversal. Forcing the base arrays
    # C-contiguous makes all derived elementwise masks C-contiguous too, so
    # the order is correct BY CONSTRUCTION regardless of caller layout
    # (no-op when already C-contiguous - the normal path - so zero behavior
    # change; only a non-contiguous caller, which would silently misattribute
    # per-world RNG, gets corrected).
    actions = np.ascontiguousarray(actions, dtype=np.int32)
    routed_firm_slots = np.ascontiguousarray(routed_firm_slots, dtype=np.int32)
    has_firm = routed_firm_slots != NO_ROUTED_FIRM_ID

    # Snapshot pre-step values.
    w_old = h.wealth.copy()
    h_old = h.health.copy()
    hunger_old = h.hunger.copy()
    fatigue_old = h.fatigue.copy()
    illness_old = h.illness.copy()
    tainted_old = h.tainted_wealth.copy()   # A3: theft-tainted wealth at step start

    w_new = w_old.copy()
    h_new = h_old.copy()
    experience = h.experience.copy()

    action_success = np.ones((K, N), dtype=bool)
    crime_occurred = np.zeros((K, N), dtype=bool)
    crime_success = np.zeros((K, N), dtype=bool)
    work_matched = np.zeros((K, N), dtype=bool)
    shop_matched = np.zeros((K, N), dtype=bool)
    clinic_matched = np.zeros((K, N), dtype=bool)
    steal_amount = np.zeros((K, N), dtype=np.float64)
    # A3 (harsh-review redesign): wealth spent on CONSUMPTION (SHOP+CLINIC) this
    # step. Tainted wealth is drawn down ONLY by genuine consumption outflow -
    # NOT by a wealth-level min-cap (the old `min(tainted, wealth)` let a
    # transient dip / drag / welfare-floor permanently launder the taint and
    # paid a crash-recover Delta log spike). Drag/welfare/earnings do NOT clear taint.
    consumption_outflow = np.zeros((K, N), dtype=np.float64)
    # Wage conservation (Option B): per-(world,firm) GROSS effective wage the
    # firm must pay, and per-world tax collected to the treasury.
    net_wage_by_firm = np.zeros_like(f.cash, dtype=np.float64)
    tax_collected_k = np.zeros(K, dtype=np.float64)
    # C1 conservation ledger (per-world). Two household mint/destroy sites:
    #  D) wealth drag - money leaves the economy to nowhere -> DESTROYED. The
    #     drag clamp `max(0, w_old-drag)` floors at 0, so the actually-removed
    #     amount is min(w_old, drag) (drag can exceed w_old at tiny wealth).
    #  F) INVEST returns - signed: gains MINT money from nowhere, losses
    #     DESTROY it. INVEST_CRASH_LOSS=0.15 <= 1 and SHOP/CLINIC are
    #     affordability-gated, so w_new never goes negative before the final
    #     `max(0, w_new)` clamp -> that clamp is a conservation no-op (no
    #     clamp-mint term needed). Split by sign of `returns`, not by `crashed`
    #     (robust if r_eff itself is negative at very high CB rates).
    drag_destroyed_k = np.zeros(K, dtype=np.float64)
    invest_minted_k = np.zeros(K, dtype=np.float64)
    invest_destroyed_k = np.zeros(K, dtype=np.float64)

    # 1) Per-world health decay (uses healthcare_k, broadcasts over N).
    base_decay_k = HEALTH_DECAY_BASE * np.maximum(
        0.0, HEALTH_DECAY_HEALTHCARE_FACTOR - healthcare_k,
    )  # shape (K,)
    h_new = (
        h_new
        - base_decay_k[:, None]
        - hunger_health_decay_bonus_vec(hunger_old)
        - fatigue_health_decay_bonus_vec(fatigue_old)
        - illness_health_decay_bonus_vec(illness_old)
    )

    drag = _wealth_drag_vec(w_old)
    # C1 D: w_new == w_old here (drag is the first wealth modification), so the
    # amount the clamp `max(0, w_old-drag)` actually removes is min(w_old, drag)
    # per (world,hh) - drag can exceed w_old at tiny wealth. -> DESTROYED.
    drag_destroyed_k = np.minimum(w_old, drag).sum(axis=1)
    w_new = np.maximum(0.0, w_new - drag)

    # Pre-compute crime success prob per world.
    crime_success_prob_k = np.vectorize(
        compute_crime_success_prob, otypes=[np.float64]
    )(policing_k, welfare_k)
    # Pre-compute invest crash prob per world.
    invest_crash_prob_k = np.clip(
        INVEST_CRASH_PROB_BASE + CB_BETA_R * interest_rate_k,
        INVEST_CRASH_PROB_MIN, INVEST_CRASH_PROB_MAX,
    )  # shape (K,)

    # 2) Per-action effects, in the same RNG-consumption order as the
    # single-world step: WORK -> SHOP -> STEAL -> REST -> INVEST -> CLINIC.

    # WORK.
    work_mask = (actions == ACTION_WORK)
    work_with_firm = work_mask & has_firm
    work_no_firm = work_mask & ~has_firm
    if np.any(work_no_firm):
        action_success[work_no_firm] = False
    if np.any(work_with_firm):
        k_idx, n_idx = np.where(work_with_firm)
        slots = routed_firm_slots[k_idx, n_idx]
        wage_offers = f.wage_offer[k_idx, slots]
        exp_here = experience[k_idx, n_idx]
        w_here = w_new[k_idx, n_idx]
        exp_mult = np.minimum(
            np.exp(WAGE_GAP_DEFAULT * exp_here * WAGE_GAP_EXP_FACTOR),
            EXPERIENCE_CAP,
        )
        motivation = 1.0 / (1.0 + MOTIVATION_WEALTH_FACTOR * w_here)
        # Per-world noise draws: np.where returned k_idx in row-major order,
        # so cumsum of per-world counts gives the boundaries that align noise
        # to (k_idx, n_idx).
        work_counts_k = work_with_firm.sum(axis=1).astype(np.int64)
        noise = np.empty(k_idx.size, dtype=np.float64)
        _cur = 0
        for _k in range(K):
            _nk = int(work_counts_k[_k])
            if _nk > 0:
                noise[_cur:_cur + _nk] = rngs[_k].uniform(
                    WORK_NOISE_MIN, WORK_NOISE_MAX, size=_nk,
                )
                _cur += _nk
        tax_here = tax_rate_k[k_idx]
        wages = wage_offers * exp_mult * motivation * noise * (1.0 - tax_here)
        wages = np.maximum(0.0, wages)
        wages *= hunger_productivity_multiplier_vec(hunger_old[k_idx, n_idx])
        wages *= fatigue_productivity_multiplier_vec(fatigue_old[k_idx, n_idx])
        wages *= illness_productivity_multiplier_vec(illness_old[k_idx, n_idx])
        # Scatter the per-worker contributions back to (K, N).
        np.add.at(w_new, (k_idx, n_idx), wages)
        # Wage conservation (Option B): household keeps `wages` (net, unchanged);
        # the firm must pay the GROSS effective wage g = wages/(1-tax); the tax
        # wedge (g - wages) funds the per-world treasury instead of vanishing.
        keep_frac = np.maximum(1.0 - tax_here, 1e-6)
        gross_wages = wages / keep_frac
        np.add.at(net_wage_by_firm, (k_idx, slots), gross_wages)
        np.add.at(tax_collected_k, k_idx, gross_wages - wages)
        work_matched[k_idx, n_idx] = True
        new_exp = np.minimum(exp_here + WORK_EXPERIENCE_INCREMENT, EXPERIENCE_CAP)
        experience[k_idx, n_idx] = new_exp

    # SHOP/CLINIC stock-rationing conservation fix (shared budget). f.stock is
    # NOT mutated anywhere in the household step, so this snapshot equals the
    # stock the firm-sell phase will start from. Per-(world,firm) count of
    # deliverable transactions = floor(stock / unit). SHOP draws first, CLINIC
    # from the remainder (see `_ration_stock`). int64 floor only ever
    # under-serves by <=1 on an FP tie - the conservation-safe side.
    stock_budget = np.floor(
        f.stock / FIRM_STOCK_TRANSACTION_UNITS
    ).astype(np.int64)
    # Preserve the firm-entry stock-shortage signal. Pre-fix, weekly_unmet_demand
    # was incremented by the firm-sell phase failing on stock-out - but that only
    # happened BECAUSE of the over-charge bug (households were matched the firm
    # couldn't serve). Post-fix those are rationed out here instead, so we count
    # the genuine unmet demand (wealth-affordable but stock-denied SHOP/CLINIC)
    # per world. Money-neutral: this feeds firm-entry/metrics ONLY, no wealth.
    shop_clinic_unmet_k = np.zeros(K, dtype=np.int64)

    # SHOP.
    shop_mask = (actions == ACTION_SHOP)
    shop_with_firm = shop_mask & has_firm
    shop_no_firm = shop_mask & ~has_firm
    if np.any(shop_no_firm):
        action_success[shop_no_firm] = False
    if np.any(shop_with_firm):
        k_idx, n_idx = np.where(shop_with_firm)
        slots = routed_firm_slots[k_idx, n_idx]
        prices = f.price[k_idx, slots]
        shop_costs = prices * SHOP_COST_FRACTION
        # Conservation fix: charge only if wealth-affordable AND the firm has
        # un-reserved deliverable stock (rationed in firm-sell order). Replaces
        # the old stale `f.stock >= unit` per-household read that let every
        # shopper be charged while the firm-sell phase delivered to only a few.
        wealth_ok = w_new[k_idx, n_idx] >= shop_costs
        affordable = _ration_stock(k_idx, slots, wealth_ok, stock_budget)
        # Genuine stock-denied demand (could afford, firm out of budget) -> the
        # firm-entry unmet signal (replaces the old bug-driven failed-sell).
        _denied = wealth_ok & ~affordable
        if np.any(_denied):
            np.add.at(shop_clinic_unmet_k, k_idx[_denied], 1)
        if np.any(affordable):
            k_ok = k_idx[affordable]
            n_ok = n_idx[affordable]
            np.add.at(w_new, (k_ok, n_ok), -shop_costs[affordable])
            np.add.at(consumption_outflow, (k_ok, n_ok), shop_costs[affordable])
            np.add.at(h_new, (k_ok, n_ok), SHOP_HEALTH_GAIN)
            shop_matched[k_ok, n_ok] = True
        if np.any(~affordable):
            k_no = k_idx[~affordable]
            n_no = n_idx[~affordable]
            action_success[k_no, n_no] = False

    # STEAL. Firm-side cash transfer happens in the world layer.
    steal_mask = (actions == ACTION_STEAL)
    if np.any(steal_mask):
        k_idx, n_idx = np.where(steal_mask)
        crime_occurred[k_idx, n_idx] = True
        # Per-world rng order: uniform(attempt) -> random(roll), matches
        # single-world `step_households_batch`. cumsum-of-per-world-counts
        # gives the boundaries that align attempts/rolls to (k_idx, n_idx).
        steal_counts_k = steal_mask.sum(axis=1).astype(np.int64)
        attempts = np.empty(k_idx.size, dtype=np.float64)
        rolls = np.empty(k_idx.size, dtype=np.float64)
        _cur = 0
        for _k in range(K):
            _nk = int(steal_counts_k[_k])
            if _nk > 0:
                attempts[_cur:_cur + _nk] = rngs[_k].uniform(
                    CRIME_STEAL_ATTEMPT_MIN, CRIME_STEAL_ATTEMPT_MAX, size=_nk,
                )
                rolls[_cur:_cur + _nk] = rngs[_k].random(size=_nk)
                _cur += _nk
        success_prob_per_worker = crime_success_prob_k[k_idx]
        succ = rolls < success_prob_per_worker
        if np.any(succ):
            k_ok = k_idx[succ]
            n_ok = n_idx[succ]
            steal_amount[k_ok, n_ok] = attempts[succ]
            np.add.at(w_new, (k_ok, n_ok), attempts[succ])
            crime_success[k_ok, n_ok] = True
        if np.any(~succ):
            k_no = k_idx[~succ]
            n_no = n_idx[~succ]
            np.add.at(h_new, (k_no, n_no), -FAILED_STEAL_HEALTH_PENALTY)
            action_success[k_no, n_no] = False

    # REST.
    rest_mask = (actions == ACTION_REST)
    if np.any(rest_mask):
        h_new[rest_mask] = h_new[rest_mask] + REST_HEALTH_RECOVERY

    # INVEST.
    invest_mask = (actions == ACTION_INVEST)
    if np.any(invest_mask):
        k_idx, n_idx = np.where(invest_mask)
        edu_here = education_k[k_idx]
        interest_here = interest_rate_k[k_idx]
        r_eff_base = INVEST_RETURN_BASE + INVEST_RETURN_EDU_COEFF * edu_here
        w_here = w_new[k_idx, n_idx]
        r_eff = (
            r_eff_base * (1.0 - CB_ALPHA_R * interest_here)
            / (1.0 + INVEST_RETURN_WEALTH_FACTOR * w_here)
        )
        crash_prob_here = invest_crash_prob_k[k_idx]
        # Per-world rolls; boundaries from cumsum of per-world invest counts.
        invest_counts_k = invest_mask.sum(axis=1).astype(np.int64)
        rolls = np.empty(k_idx.size, dtype=np.float64)
        _cur = 0
        for _k in range(K):
            _nk = int(invest_counts_k[_k])
            if _nk > 0:
                rolls[_cur:_cur + _nk] = rngs[_k].random(size=_nk)
                _cur += _nk
        crashed = rolls < crash_prob_here
        returns = np.where(crashed, -INVEST_CRASH_LOSS * w_here, r_eff * w_here)
        np.add.at(w_new, (k_idx, n_idx), returns)
        # C1 F: signed INVEST returns - gains MINT money from nowhere, losses
        # DESTROY it. Split by sign of `returns` (not by `crashed`: r_eff can be
        # negative at very high CB rates, which is still a destroy). k_idx is the
        # per-investor world index from np.where(invest_mask), aligned to returns.
        _pos = returns > 0.0
        _neg = returns < 0.0
        if np.any(_pos):
            np.add.at(invest_minted_k, k_idx[_pos], returns[_pos])
        if np.any(_neg):
            np.add.at(invest_destroyed_k, k_idx[_neg], -returns[_neg])

    # CLINIC.
    clinic_mask = (actions == ACTION_CLINIC)
    clinic_with_firm = clinic_mask & has_firm
    clinic_no_firm = clinic_mask & ~has_firm
    if np.any(clinic_no_firm):
        action_success[clinic_no_firm] = False
    if np.any(clinic_with_firm):
        k_idx, n_idx = np.where(clinic_with_firm)
        slots = routed_firm_slots[k_idx, n_idx]
        prices = f.price[k_idx, slots]
        clinic_prices = prices * CLINIC_COST_MULTIPLIER
        # Conservation fix: same stock-rationing as SHOP, drawing from the
        # SHARED `stock_budget` AFTER SHOP has consumed its grants (SHOP
        # priority per firm; both compete for the same firm stock the firm-
        # sell phase will draw down).
        wealth_ok = w_new[k_idx, n_idx] >= clinic_prices
        affordable = _ration_stock(k_idx, slots, wealth_ok, stock_budget)
        _denied = wealth_ok & ~affordable
        if np.any(_denied):
            np.add.at(shop_clinic_unmet_k, k_idx[_denied], 1)
        if np.any(affordable):
            k_ok = k_idx[affordable]
            n_ok = n_idx[affordable]
            np.add.at(w_new, (k_ok, n_ok), -clinic_prices[affordable])
            np.add.at(consumption_outflow, (k_ok, n_ok), clinic_prices[affordable])
            np.add.at(
                h_new, (k_ok, n_ok),
                CLINIC_HEALTH_GAIN_SCALE * healthcare_k[k_ok],
            )
            clinic_matched[k_ok, n_ok] = True
        if np.any(~affordable):
            k_no = k_idx[~affordable]
            n_no = n_idx[~affordable]
            action_success[k_no, n_no] = False

    # 3) Welfare and health safety net (per-world thresholds).
    welfare_amount = np.where(
        w_new < WELFARE_THRESHOLD,
        welfare_k[:, None] * (WELFARE_THRESHOLD - w_new) / WELFARE_THRESHOLD * WELFARE_MULTIPLIER,
        0.0,
    )
    w_new = w_new + welfare_amount
    # Treasury funds welfare (Option B): household receipt unchanged; the
    # per-world total is drawn from the treasury at the world layer.
    welfare_paid_k = welfare_amount.sum(axis=1)

    health_boost = np.where(
        h_new < HEALTH_SAFETY_NET_THRESHOLD,
        HEALTH_SAFETY_NET_BOOST_SCALE * healthcare_k[:, None],
        0.0,
    )
    h_new = h_new + health_boost

    # 4) Hunger / fatigue update.
    hunger_new = advance_hunger_vec(hunger_old, shop_matched)
    hunger_crisis = is_hunger_crisis_vec(hunger_new)

    fatigue_new = advance_fatigue_vec(
        fatigue_old, work_matched, rest_mask, clinic_matched,
    )

    # 5) Clamp / age.
    w_new = np.maximum(0.0, w_new)
    h_new = np.clip(h_new, 0.0, 1.0)
    age_new = h.age + 1

    # A3: reward uses EFFECTIVE (untainted) wealth. Tainted wealth decreases
    # ONLY by genuine consumption outflow (SHOP+CLINIC) - drag/welfare/earnings
    # do NOT clear it (that was the laundering ratchet). No wealth-level cap, so
    # eff-wealth moves smoothly (no crash-recover Delta log spike). Stolen money is
    # spendable in-world but never earns a wealth-channel reward; legit earnings
    # above the taint reward normally. This step's steal (if any) is added to
    # tainted and the thief's reward recomputed in the world theft phase, so the
    # provisional per-step value for a thief here is intentionally overwritten.
    tainted_new = np.maximum(0.0, tainted_old - consumption_outflow)
    eff_w_old = np.maximum(0.0, w_old - tainted_old)
    eff_w_new = np.maximum(0.0, w_new - tainted_new)

    # Reward: vectorized over (K, N). `tainted_new` (post-step held stolen
    # wealth) drives the A3 carry penalty. For a thief THIS step this per-step
    # value is provisional and gets fully overwritten by the world theft-phase
    # recompute (which passes the freshly-stolen tainted), so no double-count.
    rewards = _household_reward_kn(
        eff_w_old, h_old, eff_w_new, h_new, hunger_new, episode_progress_k,
        tainted_new,
    )

    # 6) Write back into the batched state.
    h.wealth[:] = w_new
    h.health[:] = h_new
    h.hunger[:] = hunger_new
    h.fatigue[:] = fatigue_new
    h.experience[:] = experience
    h.age[:] = age_new
    h.last_action[:] = actions
    h.last_action_success[:] = action_success
    h.last_reward[:] = rewards
    h.last_wealth_change[:] = w_new - w_old
    h.last_health_change[:] = h_new - h_old
    h.tainted_wealth[:] = tainted_new   # A3: persist cleared/decayed taint
    h.last_crime_success[:] = crime_success

    return {
        "work_matched": work_matched,
        "shop_matched": shop_matched,
        "clinic_matched": clinic_matched,
        "crime_occurred": crime_occurred,
        "crime_success": crime_success,
        "steal_amount": steal_amount,
        "hunger_crisis": hunger_crisis,
        "action_success": action_success,
        "routed_firm_slots": routed_firm_slots,
        "net_wage_by_firm": net_wage_by_firm,
        "tax_collected_k": tax_collected_k,
        "welfare_paid_k": welfare_paid_k,
        "tainted_old": tainted_old,   # A3: pre-step taint, for theft recompute
        "drag_destroyed_k": drag_destroyed_k,       # C1 D
        "invest_minted_k": invest_minted_k,         # C1 F (gains)
        "invest_destroyed_k": invest_destroyed_k,   # C1 F (losses)
        "shop_clinic_unmet_k": shop_clinic_unmet_k, # stock-rationed demand
    }


def _household_reward_kn(
    wealth_old: np.ndarray, health_old: np.ndarray,
    wealth_new: np.ndarray, health_new: np.ndarray,
    hunger_new: np.ndarray,
    episode_progress_k: np.ndarray,
    tainted_new=0.0,
) -> np.ndarray:
    """Phase-3 (K,N) household reward - must stay identical in logic to
    `_household_reward_vec`. Clamped log-utility delta + dense LEVEL term, then
    one-time bankruptcy/death transitions + bounded persistent dead-state, then
    the A3 tainted-wealth carry penalty.
    `episode_progress_k` retained for signature compatibility, no longer used.
    """
    util_old = (np.log(np.maximum(wealth_old, BANKRUPTCY_THRESHOLD))
                + LOG_UTILITY_HEALTH_WEIGHT * np.log(np.maximum(health_old, DEATH_THRESHOLD)))
    util_new = (np.log(np.maximum(wealth_new, BANKRUPTCY_THRESHOLD))
                + LOG_UTILITY_HEALTH_WEIGHT * np.log(np.maximum(health_new, DEATH_THRESHOLD)))
    rewards = util_new - util_old
    rewards = (
        rewards
        + HOUSEHOLD_HEALTH_LEVEL_WEIGHT * (health_new - HEALTH_LEVEL_NEUTRAL)
        + HOUSEHOLD_FED_WEIGHT * (HUNGER_NEUTRAL - hunger_new)
    )

    newly_bankrupt = (wealth_old > BANKRUPTCY_THRESHOLD) & (wealth_new <= BANKRUPTCY_THRESHOLD)
    rewards = np.where(newly_bankrupt, BANKRUPTCY_TRANSITION_PENALTY, rewards)

    newly_dead = (health_old > DEATH_THRESHOLD) & (health_new <= DEATH_THRESHOLD)
    already_dead = (health_old <= DEATH_THRESHOLD) & (health_new <= DEATH_THRESHOLD)
    rewards = np.where(newly_dead, DEATH_TRANSITION_PENALTY, rewards)
    rewards = np.where(already_dead, DEAD_STATE_REWARD, rewards)
    # A3 carry penalty - identical in logic to `_household_reward_vec`
    # (claw back lambda*held-tainted/stolen wealth every step; applied after the
    # cliffs so death/bankruptcy can't launder it; 0 for non-thieves).
    rewards = rewards - STEAL_TAINT_CARRY_PENALTY * tainted_new
    return rewards
