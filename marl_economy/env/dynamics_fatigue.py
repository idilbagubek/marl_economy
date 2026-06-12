"""Fatigue dynamics: rises with work, falls with rest, scales work productivity."""

import numpy as np

from marl_economy.config.constants import (
    FATIGUE_INIT_MIN,
    FATIGUE_INIT_MAX,
    FATIGUE_RISE_FROM_WORK,
    FATIGUE_DROP_FROM_REST,
    FATIGUE_DROP_FROM_CLINIC,
    FATIGUE_PRODUCTIVITY_THRESHOLD,
    FATIGUE_PRODUCTIVITY_FLOOR,
    FATIGUE_HEALTH_PENALTY_THRESHOLD,
    FATIGUE_HEALTH_PENALTY_RATE,
)


def init_fatigue(rng: np.random.Generator) -> float:
    return float(rng.uniform(FATIGUE_INIT_MIN, FATIGUE_INIT_MAX))


def advance_fatigue_vec(fatigue_old: np.ndarray,
                        worked: np.ndarray,
                        rested: np.ndarray,
                        clinic_treated: np.ndarray) -> np.ndarray:
    """Advance fatigue one week for N households. All inputs are length-N arrays."""
    new_fatigue = fatigue_old.copy()
    new_fatigue += worked.astype(np.float64) * FATIGUE_RISE_FROM_WORK
    new_fatigue -= rested.astype(np.float64) * FATIGUE_DROP_FROM_REST
    new_fatigue -= clinic_treated.astype(np.float64) * FATIGUE_DROP_FROM_CLINIC
    return np.clip(new_fatigue, 0.0, 1.0)


def fatigue_health_decay_bonus_vec(fatigue: np.ndarray) -> np.ndarray:
    """Extra weekly health decay above the penalty threshold."""
    out = np.zeros_like(fatigue, dtype=np.float64)
    above = fatigue > FATIGUE_HEALTH_PENALTY_THRESHOLD
    if np.any(above):
        denom = max(1e-9, 1.0 - FATIGUE_HEALTH_PENALTY_THRESHOLD)
        excess = fatigue[above] - FATIGUE_HEALTH_PENALTY_THRESHOLD
        out[above] = FATIGUE_HEALTH_PENALTY_RATE * excess / denom
    return out


def fatigue_productivity_multiplier_vec(fatigue: np.ndarray) -> np.ndarray:
    """Work-income multiplier in [FLOOR, 1.0]; ramps down past the threshold."""
    out = np.ones_like(fatigue, dtype=np.float64)
    above = fatigue > FATIGUE_PRODUCTIVITY_THRESHOLD
    if np.any(above):
        span = max(1e-6, 1.0 - FATIGUE_PRODUCTIVITY_THRESHOLD)
        severity = np.clip((fatigue[above] - FATIGUE_PRODUCTIVITY_THRESHOLD) / span, 0.0, 1.0)
        out[above] = 1.0 - severity * (1.0 - FATIGUE_PRODUCTIVITY_FLOOR)
    return out


def fatigue_summary(fatigues: np.ndarray) -> dict:
    if len(fatigues) == 0:
        return {"mean_fatigue": 0.0, "max_fatigue": 0.0,
                "frac_fatigued": 0.0, "frac_exhausted": 0.0}
    arr = np.asarray(fatigues, dtype=float)
    return {
        "mean_fatigue": float(np.mean(arr)),
        "max_fatigue": float(np.max(arr)),
        "frac_fatigued": float(np.mean(arr >= FATIGUE_PRODUCTIVITY_THRESHOLD)),
        "frac_exhausted": float(np.mean(arr >= FATIGUE_HEALTH_PENALTY_THRESHOLD)),
    }
