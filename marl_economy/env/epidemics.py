"""Place-based disease transmission.

Each week, every household is "at" a place determined by its action:

    WORK   -> workplace (the routed firm)
    SHOP   -> market    (the routed firm)
    CLINIC -> clinic    (the routed firm)
    REST   -> home
    INVEST, STEAL -> public

Per-place infection probability: P = 1 - (1 - alpha)^k, where k is the count
of infectious attendees at that place. Healthcare access damps alpha;
clinic overcrowding boosts it.
"""

from collections import defaultdict
import numpy as np

from marl_economy.config.constants import (
    ACTION_WORK, ACTION_SHOP, ACTION_STEAL, ACTION_INVEST, ACTION_CLINIC,
    ILLNESS_BASE_INFECTION_RATE,
    EPIDEMIC_PLACE_HOME, EPIDEMIC_PLACE_WORKPLACE, EPIDEMIC_PLACE_MARKET,
    EPIDEMIC_PLACE_CLINIC, EPIDEMIC_PLACE_PUBLIC,
    EPIDEMIC_TRANSMISSION_ALPHA,
    EPIDEMIC_HEALTHCARE_DAMP,
    CLINIC_OVERCROWD_TRANSMISSION_MULT,
    CLINIC_OVERCROWD_THRESHOLD,
)
from marl_economy.env.illness import is_infectious_vec, per_household_susceptibility_vec


def _place_for_action(action: int, routed_firm_id: int) -> tuple[str, str | int]:
    """Map (action, routed firm id) to a (place_type, place_id) pair.

    `routed_firm_id` is -1 (or any negative value) when no firm was routed.
    """
    if action == ACTION_WORK and routed_firm_id >= 0:
        return EPIDEMIC_PLACE_WORKPLACE, int(routed_firm_id)
    if action == ACTION_SHOP and routed_firm_id >= 0:
        return EPIDEMIC_PLACE_MARKET, int(routed_firm_id)
    if action == ACTION_CLINIC and routed_firm_id >= 0:
        return EPIDEMIC_PLACE_CLINIC, int(routed_firm_id)
    if action in (ACTION_INVEST, ACTION_STEAL):
        return EPIDEMIC_PLACE_PUBLIC, "public"
    return EPIDEMIC_PLACE_HOME, "home"


def transmit_at_places(actions: np.ndarray,
                       routed_firm_ids: np.ndarray,
                       illness_states: np.ndarray,
                       hunger: np.ndarray,
                       fatigue: np.ndarray,
                       health: np.ndarray,
                       rng: np.random.Generator,
                       healthcare_access: float,
                       clinic_capacity: int,
                       clinic_used: int,
                       infection_boost: float = 1.0) -> tuple[np.ndarray, dict]:
    """Compute newly-exposed households this week from co-location.

    `actions` and `routed_firm_ids` are length-N int arrays. `routed_firm_ids`
    uses -1 to mean "no firm". Returns (newly_exposed_bool, place_diagnostics).
    """
    n = int(len(actions))
    if n == 0:
        return np.zeros(0, dtype=bool), {}

    infectious = is_infectious_vec(illness_states)
    susceptibility = per_household_susceptibility_vec(hunger, fatigue, health)

    place_members: dict[tuple[str, object], list[int]] = defaultdict(list)
    actions_list = actions.tolist() if hasattr(actions, "tolist") else list(actions)
    firm_ids_list = (routed_firm_ids.tolist() if hasattr(routed_firm_ids, "tolist")
                     else list(routed_firm_ids))
    for i, (a, fid) in enumerate(zip(actions_list, firm_ids_list)):
        ptype, pid = _place_for_action(int(a), int(fid))
        place_members[(ptype, pid)].append(i)

    healthcare_damp = max(0.0, 1.0 - EPIDEMIC_HEALTHCARE_DAMP * float(healthcare_access))
    clinic_overcrowded = (
        clinic_capacity > 0
        and clinic_used > clinic_capacity * CLINIC_OVERCROWD_THRESHOLD
    )

    newly_exposed = np.zeros(n, dtype=bool)
    place_stats: dict[str, dict] = {}

    for (ptype, _), members in place_members.items():
        attendees = np.array(members, dtype=np.int64)
        infect_count = int(infectious[attendees].sum())
        susceptible = ~infectious[attendees]
        if infect_count == 0 or not susceptible.any():
            continue

        alpha = EPIDEMIC_TRANSMISSION_ALPHA.get(ptype, 0.005)
        alpha = alpha * healthcare_damp * infection_boost
        if ptype == EPIDEMIC_PLACE_CLINIC and clinic_overcrowded:
            alpha *= CLINIC_OVERCROWD_TRANSMISSION_MULT
        alpha = max(0.0, min(0.95, alpha))

        p_base = 1.0 - (1.0 - alpha) ** infect_count
        agent_p = np.clip(p_base * susceptibility[attendees], 0.0, 0.99)

        rolls = rng.random(len(attendees))
        infect_mask = rolls < agent_p
        newly_exposed[attendees[infect_mask & susceptible]] = True

        ps = place_stats.setdefault(ptype, {"attendees": 0, "infectious": 0,
                                            "newly_exposed": 0})
        ps["attendees"] += len(attendees)
        ps["infectious"] += infect_count
        ps["newly_exposed"] += int((infect_mask & susceptible).sum())

    # Background infection rate covers contacts not modeled by named places.
    if ILLNESS_BASE_INFECTION_RATE > 0:
        bg_rolls = rng.random(n)
        bg_infected = (
            bg_rolls < ILLNESS_BASE_INFECTION_RATE
            * float(healthcare_damp) * infection_boost
        ) & ~infectious & ~newly_exposed
        newly_exposed[bg_infected] = True

    return newly_exposed, place_stats
