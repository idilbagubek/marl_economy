"""Multi-signal economic-collapse detector.

The default termination check fires only when every firm has exited, which
the F_MIN floor prevents. This detector watches GDP, firm cash, and blocked
exits so a hidden insolvency cascade still terminates the episode.
"""

from marl_economy.config.constants import (
    COLLAPSE_GDP_ZERO_WEEKS,
    COLLAPSE_NEGATIVE_CASH_THRESHOLD,
    COLLAPSE_BLOCKED_EXIT_RECENT_WEEKS,
    COLLAPSE_BLOCKED_EXIT_THRESHOLD,
    COLLAPSE_LOW_HEALTH_THRESHOLD,
    COLLAPSE_LOW_HEALTH_WEEKS,
    EPSILON,
)
from marl_economy.env.events import EventLedger, FIRM_EXIT_BLOCKED


class CollapseDetector:
    """Call `update` once per week. Becomes a no-op once triggered."""

    def __init__(self) -> None:
        self.gdp_zero_streak = 0
        self.low_health_streak = 0
        self.triggered = False
        self.trigger_reason: str | None = None
        self.trigger_week: int | None = None

    def update(self, week: int, weekly_gdp: float, alive_firms: list,
               event_ledger: EventLedger | None,
               mean_health: float = 1.0) -> bool:
        if self.triggered:
            return False

        if weekly_gdp <= EPSILON:
            self.gdp_zero_streak += 1
        else:
            self.gdp_zero_streak = 0

        # Humanitarian-collapse trigger: independent of GDP/firm state. Catches
        # the "mass-death subsistence" failure mode (firms churn so cash is
        # never deep-negative, intermittent tiny GDP keeps the zero-streak from
        # accumulating) that the triggers below are structurally blind to.
        # mean_health defaults to 1.0 so legacy callers never spuriously fire.
        if mean_health <= COLLAPSE_LOW_HEALTH_THRESHOLD:
            self.low_health_streak += 1
        else:
            self.low_health_streak = 0
        if self.low_health_streak >= COLLAPSE_LOW_HEALTH_WEEKS:
            self._fire("humanitarian_collapse", week)
            return True

        if not alive_firms:
            self._fire("no_firms", week)
            return True

        if self.gdp_zero_streak >= COLLAPSE_GDP_ZERO_WEEKS:
            cashes = [float(f.get("cash", 0.0)) for f in alive_firms]
            if cashes:
                mean_cash = sum(cashes) / len(cashes)
                all_negative = all(c < 0 for c in cashes)
                if all_negative and mean_cash < COLLAPSE_NEGATIVE_CASH_THRESHOLD:
                    self._fire("hidden_insolvency", week)
                    return True

            if event_ledger is not None:
                blocked_recent = event_ledger.recent_count(
                    week, FIRM_EXIT_BLOCKED, COLLAPSE_BLOCKED_EXIT_RECENT_WEEKS
                )
                if blocked_recent >= COLLAPSE_BLOCKED_EXIT_THRESHOLD:
                    self._fire("blocked_exit_cascade", week)
                    return True

        return False

    def _fire(self, reason: str, week: int) -> None:
        self.triggered = True
        self.trigger_reason = reason
        self.trigger_week = week

    def to_summary(self) -> dict:
        return {
            "collapse_triggered": bool(self.triggered),
            "collapse_reason": self.trigger_reason,
            "collapse_week": self.trigger_week,
            "gdp_zero_streak_final": self.gdp_zero_streak,
            "low_health_streak_final": self.low_health_streak,
        }
