"""Typed event ledger for the simulation.

Every meaningful state transition (firm entry/exit, blocked exit, hire refused,
household death, hunger or fatigue crisis, illness onset, clinic overcrowding,
disasters, etc.) appends one record here. CSV writers and plot tooling read
the per-week counts from this ledger instead of inferring from net counts.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


# Firm lifecycle
FIRM_ENTRY_NORMAL = "firm_entry_normal"
FIRM_ENTRY_EMERGENCY = "firm_entry_emergency"
FIRM_EXIT = "firm_exit"
FIRM_EXIT_BLOCKED = "firm_exit_blocked"
FIRM_INSOLVENT = "firm_insolvent"
FIRM_HIRE_REFUSED = "firm_hire_refused"

# Household state changes
HOUSEHOLD_DEATH = "household_death"
HOUSEHOLD_HUNGER_CRISIS = "household_hunger_crisis"
HOUSEHOLD_DISPLACED = "household_displaced"
HOUSEHOLD_FATIGUE_CRISIS = "household_fatigue_crisis"
HOUSEHOLD_INFECTED = "household_infected"
HOUSEHOLD_SEVERE_ILLNESS = "household_severe_illness"
HOUSEHOLD_RECOVERED = "household_recovered"

# Healthcare bottleneck
CLINIC_OVERCROWDED = "clinic_overcrowded"
CLINIC_TREATMENT_DENIED = "clinic_treatment_denied"

# Disasters
DISASTER_EARTHQUAKE = "disaster_earthquake"
DISASTER_FIRE = "disaster_fire"
DISASTER_FLOOD = "disaster_flood"
DISASTER_SUPPLY_SHOCK = "disaster_supply_shock"
DISASTER_EPIDEMIC_WAVE = "disaster_epidemic_wave"


# Iteration order matters for CSV column stability.
ALL_EVENT_TYPES = (
    FIRM_ENTRY_NORMAL,
    FIRM_ENTRY_EMERGENCY,
    FIRM_EXIT,
    FIRM_EXIT_BLOCKED,
    FIRM_INSOLVENT,
    FIRM_HIRE_REFUSED,
    HOUSEHOLD_DEATH,
    HOUSEHOLD_HUNGER_CRISIS,
    HOUSEHOLD_DISPLACED,
    HOUSEHOLD_FATIGUE_CRISIS,
    HOUSEHOLD_INFECTED,
    HOUSEHOLD_SEVERE_ILLNESS,
    HOUSEHOLD_RECOVERED,
    CLINIC_OVERCROWDED,
    CLINIC_TREATMENT_DENIED,
    DISASTER_EARTHQUAKE,
    DISASTER_FIRE,
    DISASTER_FLOOD,
    DISASTER_SUPPLY_SHOCK,
    DISASTER_EPIDEMIC_WAVE,
)


@dataclass
class Event:
    type: str
    week: int
    actor_id: Optional[int] = None
    payload: dict = field(default_factory=dict)


class EventLedger:
    """Append-only ledger with weekly counters."""

    def __init__(self) -> None:
        self.events: list[Event] = []
        self._weekly_counts: dict[int, dict[str, int]] = {}
        self._totals: dict[str, int] = {t: 0 for t in ALL_EVENT_TYPES}

    def record(self, event_type: str, week: int,
               actor_id: Optional[int] = None, **payload: Any) -> Event:
        ev = Event(type=event_type, week=week, actor_id=actor_id, payload=payload)
        self.events.append(ev)
        bucket = self._weekly_counts.setdefault(week, {})
        bucket[event_type] = bucket.get(event_type, 0) + 1
        self._totals[event_type] = self._totals.get(event_type, 0) + 1
        return ev

    def weekly_count(self, week: int, event_type: str) -> int:
        return self._weekly_counts.get(week, {}).get(event_type, 0)

    def total_count(self, event_type: str) -> int:
        return self._totals.get(event_type, 0)

    def recent_count(self, current_week: int, event_type: str,
                     window_weeks: int) -> int:
        if window_weeks <= 0:
            return 0
        start = max(0, current_week - window_weeks + 1)
        return sum(
            self._weekly_counts.get(w, {}).get(event_type, 0)
            for w in range(start, current_week + 1)
        )

    def to_csv_row(self, week: int) -> dict:
        bucket = self._weekly_counts.get(week, {})
        return {f"events_{t}": bucket.get(t, 0) for t in ALL_EVENT_TYPES}

    def csv_columns(self) -> list[str]:
        return [f"events_{t}" for t in ALL_EVENT_TYPES]

    def to_summary(self) -> dict:
        return {f"events_total_{t}": self._totals.get(t, 0) for t in ALL_EVENT_TYPES}
