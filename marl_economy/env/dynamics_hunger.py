"""Hunger dynamics: rises every week, only Shop reduces it."""

import numpy as np

from marl_economy.config.constants import (
    HUNGER_INIT_MIN,
    HUNGER_INIT_MAX,
    HUNGER_RISE_PER_WEEK,
    HUNGER_DROP_FROM_FOOD,
    HUNGER_HEALTH_PENALTY_THRESHOLD,
    HUNGER_HEALTH_PENALTY_RATE,
    HUNGER_PRODUCTIVITY_THRESHOLD,
    HUNGER_PRODUCTIVITY_FLOOR,
    HUNGER_FATAL_THRESHOLD,
    HUNGER_FATAL_HEALTH_DECAY,
    HUNGER_CRISIS_THRESHOLD,
)


# FIX C (2026-05-18 v2, 5-Opus-reviewed): continuous fatal-hunger health-decay.
# These module constants are the SINGLE source for the ramp so the scalar and
# vec paths CANNOT desync (review binding requirement - do NOT hard-code the
# 0.95-join value; it must track HUNGER_HEALTH_PENALTY_RATE/THRESHOLD).
_HUNGER_PENALTY_SPAN = max(1e-9, 1.0 - HUNGER_HEALTH_PENALTY_THRESHOLD)
_HUNGER_FATAL_SPAN = max(1e-9, 1.0 - HUNGER_FATAL_THRESHOLD)
# Decay value AT the fatal threshold on the linear penalty segment (= the
# continuous join point between the penalty segment and the fatal ramp).
_HUNGER_DECAY_AT_FATAL = (
    HUNGER_HEALTH_PENALTY_RATE
    * (HUNGER_FATAL_THRESHOLD - HUNGER_HEALTH_PENALTY_THRESHOLD)
    / _HUNGER_PENALTY_SPAN
)


def init_hunger(rng: np.random.Generator) -> float:
    return float(rng.uniform(HUNGER_INIT_MIN, HUNGER_INIT_MAX))


def advance_hunger(hunger_old: float, ate_this_week: bool) -> float:
    new_hunger = hunger_old + HUNGER_RISE_PER_WEEK
    if ate_this_week:
        new_hunger -= HUNGER_DROP_FROM_FOOD
    return float(np.clip(new_hunger, 0.0, 1.0))


def hunger_health_decay_bonus(hunger: float) -> float:
    """Extra weekly health decay as hunger climbs past the penalty threshold.

    FIX C: CONTINUOUS (no discontinuity). 0 for h<=PENALTY_THR; linear penalty
    segment up to `_HUNGER_DECAY_AT_FATAL` at h=FATAL_THR; then a CONTINUOUS
    linear ramp from that join up to HUNGER_FATAL_HEALTH_DECAY at h=1.0.
    Removes the old absorbing-state cliff (was a 28x jump at 0.95) so a
    re-fed household can recover; chronic starvation still lethal (~7 wk).
    """
    if hunger <= HUNGER_HEALTH_PENALTY_THRESHOLD:
        return 0.0
    if hunger <= HUNGER_FATAL_THRESHOLD:
        return (HUNGER_HEALTH_PENALTY_RATE
                * (hunger - HUNGER_HEALTH_PENALTY_THRESHOLD)
                / _HUNGER_PENALTY_SPAN)
    frac = min(1.0, (hunger - HUNGER_FATAL_THRESHOLD) / _HUNGER_FATAL_SPAN)
    return _HUNGER_DECAY_AT_FATAL + (HUNGER_FATAL_HEALTH_DECAY - _HUNGER_DECAY_AT_FATAL) * frac


def hunger_productivity_multiplier(hunger: float) -> float:
    """Work-income multiplier in [FLOOR, 1.0]."""
    if hunger <= HUNGER_PRODUCTIVITY_THRESHOLD:
        return 1.0
    span = max(1e-6, 1.0 - HUNGER_PRODUCTIVITY_THRESHOLD)
    severity = (hunger - HUNGER_PRODUCTIVITY_THRESHOLD) / span
    severity = float(np.clip(severity, 0.0, 1.0))
    return float(1.0 - severity * (1.0 - HUNGER_PRODUCTIVITY_FLOOR))


def is_hunger_crisis(hunger: float) -> bool:
    return hunger >= HUNGER_CRISIS_THRESHOLD


def advance_hunger_vec(hungers: np.ndarray, ate: np.ndarray) -> np.ndarray:
    new_hunger = hungers + HUNGER_RISE_PER_WEEK
    new_hunger = new_hunger - ate.astype(np.float64) * HUNGER_DROP_FROM_FOOD
    return np.clip(new_hunger, 0.0, 1.0)


def hunger_health_decay_bonus_vec(hungers: np.ndarray) -> np.ndarray:
    """Vectorized - MUST be byte-identical to `hunger_health_decay_bonus`
    (same constants, same branch boundaries: mid = PENALTY_THR<h<=FATAL_THR,
    ramp = h>FATAL_THR; continuous at both joins)."""
    out = np.zeros_like(hungers, dtype=np.float64)
    mid = (hungers > HUNGER_HEALTH_PENALTY_THRESHOLD) & (hungers <= HUNGER_FATAL_THRESHOLD)
    if np.any(mid):
        out[mid] = (HUNGER_HEALTH_PENALTY_RATE
                    * (hungers[mid] - HUNGER_HEALTH_PENALTY_THRESHOLD)
                    / _HUNGER_PENALTY_SPAN)
    ramp = hungers > HUNGER_FATAL_THRESHOLD
    if np.any(ramp):
        frac = np.minimum(1.0, (hungers[ramp] - HUNGER_FATAL_THRESHOLD) / _HUNGER_FATAL_SPAN)
        out[ramp] = _HUNGER_DECAY_AT_FATAL + (HUNGER_FATAL_HEALTH_DECAY - _HUNGER_DECAY_AT_FATAL) * frac
    return out


def hunger_productivity_multiplier_vec(hungers: np.ndarray) -> np.ndarray:
    out = np.ones_like(hungers, dtype=np.float64)
    above = hungers > HUNGER_PRODUCTIVITY_THRESHOLD
    if np.any(above):
        span = max(1e-6, 1.0 - HUNGER_PRODUCTIVITY_THRESHOLD)
        severity = np.clip((hungers[above] - HUNGER_PRODUCTIVITY_THRESHOLD) / span, 0.0, 1.0)
        out[above] = 1.0 - severity * (1.0 - HUNGER_PRODUCTIVITY_FLOOR)
    return out


def is_hunger_crisis_vec(hungers: np.ndarray) -> np.ndarray:
    return hungers >= HUNGER_CRISIS_THRESHOLD


def hunger_summary(hungers: np.ndarray) -> dict:
    if len(hungers) == 0:
        return {"mean_hunger": 0.0, "max_hunger": 0.0,
                "frac_hungry": 0.0, "frac_crisis": 0.0}
    arr = np.asarray(hungers, dtype=float)
    return {
        "mean_hunger": float(np.mean(arr)),
        "max_hunger": float(np.max(arr)),
        "frac_hungry": float(np.mean(arr >= HUNGER_HEALTH_PENALTY_THRESHOLD)),
        "frac_crisis": float(np.mean(arr >= HUNGER_CRISIS_THRESHOLD)),
    }
