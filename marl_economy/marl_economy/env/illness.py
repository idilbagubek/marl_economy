"""Illness state machine.

States: HEALTHY -> EXPOSED -> MILD -> SEVERE -> RECOVERING -> HEALTHY.
Severe untreated illness can also push health past the death threshold.

All transitions are vectorized over the N household population per week.
Risk modifiers from hunger, fatigue, and low health are applied multiplicatively
to the per-place infection probability in `epidemics.py`.
"""

import numpy as np

from marl_economy.config.constants import (
    ILLNESS_HEALTHY,
    ILLNESS_EXPOSED,
    ILLNESS_MILD,
    ILLNESS_SEVERE,
    ILLNESS_RECOVERING,
    ILLNESS_EXPOSED_TO_MILD,
    ILLNESS_EXPOSED_TO_RECOVERED,
    ILLNESS_MILD_TO_SEVERE,
    ILLNESS_MILD_TO_RECOVERING,
    ILLNESS_SEVERE_TO_RECOVERING_NO_CARE,
    ILLNESS_SEVERE_TO_RECOVERING_WITH_CARE,
    ILLNESS_SEVERE_DEATH_NO_CARE,
    ILLNESS_SEVERE_DEATH_WITH_CARE,
    ILLNESS_RECOVERING_TO_HEALTHY,
    ILLNESS_REST_RECOVERY_BONUS,
    ILLNESS_MILD_PRODUCTIVITY,
    ILLNESS_SEVERE_PRODUCTIVITY,
    ILLNESS_RECOVERING_PRODUCTIVITY,
    ILLNESS_MILD_HEALTH_DECAY,
    ILLNESS_SEVERE_HEALTH_DECAY,
    ILLNESS_HUNGER_RISK_MULT,
    ILLNESS_FATIGUE_RISK_MULT,
    ILLNESS_LOW_HEALTH_RISK_MULT,
    ILLNESS_LOW_HEALTH_THRESHOLD,
    HUNGER_HEALTH_PENALTY_THRESHOLD,
    FATIGUE_HEALTH_PENALTY_THRESHOLD,
    DEATH_THRESHOLD,
)


def init_illness() -> int:
    return ILLNESS_HEALTHY


def illness_health_decay_bonus_vec(states: np.ndarray) -> np.ndarray:
    """Extra weekly health decay from current illness state."""
    out = np.zeros_like(states, dtype=np.float64)
    out[states == ILLNESS_MILD] = ILLNESS_MILD_HEALTH_DECAY
    out[states == ILLNESS_SEVERE] = ILLNESS_SEVERE_HEALTH_DECAY
    return out


def illness_productivity_multiplier_vec(states: np.ndarray) -> np.ndarray:
    """Work-income multiplier from current illness state."""
    out = np.ones_like(states, dtype=np.float64)
    out[states == ILLNESS_MILD] = ILLNESS_MILD_PRODUCTIVITY
    out[states == ILLNESS_SEVERE] = ILLNESS_SEVERE_PRODUCTIVITY
    out[states == ILLNESS_RECOVERING] = ILLNESS_RECOVERING_PRODUCTIVITY
    return out


def per_household_susceptibility_vec(hunger: np.ndarray,
                                     fatigue: np.ndarray,
                                     health: np.ndarray) -> np.ndarray:
    """Multiplicative infection-risk modifier per household."""
    mult = np.ones_like(hunger, dtype=np.float64)
    mult = np.where(hunger >= HUNGER_HEALTH_PENALTY_THRESHOLD,
                    mult * ILLNESS_HUNGER_RISK_MULT, mult)
    mult = np.where(fatigue >= FATIGUE_HEALTH_PENALTY_THRESHOLD,
                    mult * ILLNESS_FATIGUE_RISK_MULT, mult)
    mult = np.where(health < ILLNESS_LOW_HEALTH_THRESHOLD,
                    mult * ILLNESS_LOW_HEALTH_RISK_MULT, mult)
    return mult


def is_infectious_vec(states: np.ndarray) -> np.ndarray:
    """MILD and SEVERE are infectious. EXPOSED is incubating only."""
    return np.isin(states, (ILLNESS_MILD, ILLNESS_SEVERE))


def step_illness_transitions(states: np.ndarray,
                             health: np.ndarray,
                             rng: np.random.Generator,
                             rested: np.ndarray,
                             clinic_treated: np.ndarray,
                             newly_exposed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Advance the state machine by one week.

    Returns (new_states, illness_health_loss). Severe-untreated mortality is
    surfaced as an extra health loss the caller applies to the health array,
    so it cascades through the existing death threshold.
    """
    n = len(states)
    new_states = states.copy()
    health_loss = np.zeros(n, dtype=np.float64)
    rolls = rng.random(n)

    just_exposed = (states == ILLNESS_HEALTHY) & newly_exposed
    new_states[just_exposed] = ILLNESS_EXPOSED

    exposed = states == ILLNESS_EXPOSED
    if np.any(exposed):
        ex_idx = np.where(exposed)[0]
        r = rolls[ex_idx]
        becomes_mild = r < ILLNESS_EXPOSED_TO_MILD
        becomes_recovered = (r >= ILLNESS_EXPOSED_TO_MILD) & (
            r < ILLNESS_EXPOSED_TO_MILD + ILLNESS_EXPOSED_TO_RECOVERED
        )
        new_states[ex_idx[becomes_mild]] = ILLNESS_MILD
        new_states[ex_idx[becomes_recovered]] = ILLNESS_HEALTHY

    mild = states == ILLNESS_MILD
    if np.any(mild):
        m_idx = np.where(mild)[0]
        r = rolls[m_idx]
        rest_bonus = rested[m_idx].astype(np.float64) * ILLNESS_REST_RECOVERY_BONUS
        prog = r < ILLNESS_MILD_TO_SEVERE
        rec = (r >= ILLNESS_MILD_TO_SEVERE) & (
            r < ILLNESS_MILD_TO_SEVERE + ILLNESS_MILD_TO_RECOVERING + rest_bonus
        )
        new_states[m_idx[prog]] = ILLNESS_SEVERE
        new_states[m_idx[rec]] = ILLNESS_RECOVERING

    severe = states == ILLNESS_SEVERE
    if np.any(severe):
        s_idx = np.where(severe)[0]
        r = rolls[s_idx]
        treated = clinic_treated[s_idx]
        rec_prob = np.where(treated,
                            ILLNESS_SEVERE_TO_RECOVERING_WITH_CARE,
                            ILLNESS_SEVERE_TO_RECOVERING_NO_CARE)
        death_prob = np.where(treated,
                              ILLNESS_SEVERE_DEATH_WITH_CARE,
                              ILLNESS_SEVERE_DEATH_NO_CARE)
        recovers = r < rec_prob
        dies = (r >= rec_prob) & (r < rec_prob + death_prob)
        new_states[s_idx[recovers]] = ILLNESS_RECOVERING
        died_idx = s_idx[dies]
        # Force health below the death threshold so the existing death detection fires.
        health_loss[died_idx] = np.maximum(
            0.0, health[died_idx] - DEATH_THRESHOLD * 0.5
        )

    recovering = states == ILLNESS_RECOVERING
    if np.any(recovering):
        rc_idx = np.where(recovering)[0]
        r = rolls[rc_idx]
        rest_bonus = rested[rc_idx].astype(np.float64) * ILLNESS_REST_RECOVERY_BONUS
        becomes_healthy = r < ILLNESS_RECOVERING_TO_HEALTHY + rest_bonus
        new_states[rc_idx[becomes_healthy]] = ILLNESS_HEALTHY

    return new_states, health_loss


def illness_summary(states: np.ndarray) -> dict:
    if len(states) == 0:
        return {f"frac_{label}": 0.0 for label in
                ("healthy", "exposed", "mild", "severe", "recovering")}
    return {
        "frac_healthy": float(np.mean(states == ILLNESS_HEALTHY)),
        "frac_exposed": float(np.mean(states == ILLNESS_EXPOSED)),
        "frac_mild": float(np.mean(states == ILLNESS_MILD)),
        "frac_severe": float(np.mean(states == ILLNESS_SEVERE)),
        "frac_recovering": float(np.mean(states == ILLNESS_RECOVERING)),
    }
