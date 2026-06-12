"""Clinic capacity allocator.

Capacity per week scales with healthcare_access. When demand exceeds capacity,
the sickest are treated first; everyone else still pays the firm but uses the
no-treatment branch of the illness state machine.
"""

import numpy as np

from marl_economy.config.constants import (
    CLINIC_CAPACITY_BASE,
    CLINIC_CAPACITY_HEALTHCARE_BONUS,
)


def weekly_capacity(healthcare_access: float) -> int:
    capacity = CLINIC_CAPACITY_BASE + float(healthcare_access) * CLINIC_CAPACITY_HEALTHCARE_BONUS
    return max(0, int(round(capacity)))


def allocate_treatment(clinic_visit_idxs: np.ndarray,
                       capacity: int,
                       priority_severity: np.ndarray | None = None,
                       rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Allocate weekly clinic slots to a set of visitors.

    Returns (treated_mask, used_slots). When demand exceeds capacity we
    triage by `priority_severity` (descending) with random tie-breaks.
    """
    n = len(clinic_visit_idxs)
    if n == 0:
        return np.zeros(0, dtype=bool), 0
    if capacity <= 0:
        return np.zeros(n, dtype=bool), 0
    if n <= capacity:
        return np.ones(n, dtype=bool), n

    if priority_severity is not None and rng is not None:
        jitter = rng.random(n) * 1e-6
        score = priority_severity.astype(np.float64) + jitter
        order = np.argsort(-score)
    elif rng is not None:
        order = rng.permutation(n)
    else:
        order = np.arange(n)

    treated_mask = np.zeros(n, dtype=bool)
    treated_mask[order[:capacity]] = True
    return treated_mask, int(treated_mask.sum())
