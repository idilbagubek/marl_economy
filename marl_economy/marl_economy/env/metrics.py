"""Macro metrics: GDP, inflation, Gini, poverty, unemployment, crime, health."""

from __future__ import annotations

import math

import numpy as np

from marl_economy.config.constants import (
    EPSILON,
    MARKET_DEFAULT_PRICE, MARKET_DEFAULT_WAGE,
    POVERTY_RATE_THRESHOLD,
    METRICS_EMA_ALPHA, METRICS_TOP_SHARE_FRACTION, METRICS_BOTTOM_SHARE_FRACTION,
    MACRO_SUMMARY_WINDOW_WEEKS,
    DEATH_THRESHOLD, MASS_DEATH_THRESHOLD,
)
from marl_economy.env.state import FirmState, HouseholdState


def compute_gini(wealths: np.ndarray) -> float:
    """G = (2 * sum(i*X_i)) / (n * sum(X_i)) - (n+1)/n  over sorted wealths."""
    if len(wealths) == 0:
        return 0.0
    sorted_w = np.sort(wealths)
    n = len(sorted_w)
    total = np.sum(sorted_w)
    if total < EPSILON:
        return 0.0
    indices = np.arange(1, n + 1)
    return float((2.0 * np.sum(indices * sorted_w)) / (n * total) - (n + 1) / n)


def compute_gdp(f: FirmState) -> float:
    """GDP proxy: sum over alive firms of price * sales.

    Uses `math.fsum` (compensated summation) over Python floats in slot order
    to match the legacy `sum(f["price"] * f["sales"] for f in firms if f["alive"])`
    exactly.

    Subtle point: in the legacy code `f["price"]` was a Python float (because
    `rng.normal(0, std)` without a `size=` argument returns a Python float, and
    Python's `__add__` on `anchor_price + py_float` stayed `float`). Python's
    built-in `sum` applies compensated summation to pure-`float` iterables.
    With our struct-of-arrays layout `f.price[slot]` is `np.float64`, so we
    cast to `float` and use `math.fsum` to reproduce the same compensated
    result regardless of operand type.
    """
    alive_slots = f.alive_slots()
    if alive_slots.size == 0:
        return 0.0
    return math.fsum(
        float(f.price[slot]) * float(f.sales[slot])
        for slot in alive_slots.tolist()
    )


def compute_gdp_per_capita(f: FirmState, n_households: int) -> float:
    if n_households <= 0:
        return 0.0
    return compute_gdp(f) / n_households


def compute_inflation(mean_price_current: float, mean_price_prev: float) -> float:
    if mean_price_prev <= EPSILON:
        return 0.0
    return (mean_price_current - mean_price_prev) / mean_price_prev


def compute_poverty_rate(wealths: np.ndarray, threshold: float = POVERTY_RATE_THRESHOLD) -> float:
    if len(wealths) == 0:
        return 0.0
    return float(np.mean(wealths < threshold))


def compute_unemployment(work_attempts: int, work_matched: int) -> float:
    if work_attempts <= 0:
        return 0.0
    return (work_attempts - work_matched) / work_attempts


def compute_crime_rate(crimes: int, n_households: int) -> float:
    if n_households <= 0:
        return 0.0
    return crimes / n_households


def compute_avg_health(healths: np.ndarray) -> float:
    if len(healths) == 0:
        return 0.0
    return float(np.mean(healths))


def compute_gdp_growth(gdp_current: float, gdp_prev: float) -> float:
    if gdp_prev <= EPSILON:
        return 0.0
    return (gdp_current - gdp_prev) / gdp_prev


def compute_top_shares(wealths: np.ndarray) -> dict:
    if len(wealths) == 0:
        return {"top_1_pct": 0.0, "bottom_50_pct": 0.0}
    sorted_w = np.sort(wealths)
    total = np.sum(sorted_w) + EPSILON
    n = len(sorted_w)

    top_count = max(1, int(np.ceil(n * METRICS_TOP_SHARE_FRACTION)))
    top_1_share = float(np.sum(sorted_w[-top_count:]) / total)

    bottom_50_idx = int(np.floor(n * METRICS_BOTTOM_SHARE_FRACTION))
    bottom_50_share = float(np.sum(sorted_w[:bottom_50_idx]) / total)

    return {"top_1_pct": top_1_share, "bottom_50_pct": bottom_50_share}


def compute_lorenz_curve(wealths: np.ndarray) -> tuple:
    if len(wealths) == 0:
        return np.array([0, 1]), np.array([0, 1])
    sorted_w = np.sort(wealths)
    n = len(sorted_w)
    total = np.sum(sorted_w) + EPSILON

    cum_pop = np.arange(1, n + 1) / n
    cum_wealth = np.cumsum(sorted_w) / total

    cum_pop = np.insert(cum_pop, 0, 0.0)
    cum_wealth = np.insert(cum_wealth, 0, 0.0)

    return cum_pop, cum_wealth


class MetricsTracker:
    """Tracks all macro metrics over time with EMA smoothing.

    Provides windowed computation for gov / central-bank quarterly observations.
    """

    def __init__(self, ema_alpha: float = METRICS_EMA_ALPHA):
        self.ema_alpha = ema_alpha
        self.history: list[dict] = []
        self.ema_inflation = 0.0
        self.ema_gdp_growth = 0.0
        self.prev_mean_price: float | None = None
        self.prev_gdp: float | None = None

    def update(
        self,
        h: HouseholdState,
        f: FirmState,
        week: int,
        work_attempts: int,
        work_matched: int,
        crimes: int,
        steal_success: int = 0,
        extra: dict | None = None,
    ) -> dict:
        """Compute and store this week's metrics."""
        alive_slots = f.alive_slots()
        n_h = h.size

        wealths = h.wealth
        healths = h.health

        if alive_slots.size > 0:
            mean_price = float(np.mean(f.price[alive_slots]))
            mean_wage = float(np.mean(f.wage_offer[alive_slots]))
        else:
            mean_price = MARKET_DEFAULT_PRICE
            mean_wage = MARKET_DEFAULT_WAGE

        gdp = compute_gdp(f)
        gdp_pc = compute_gdp_per_capita(f, n_h)

        if self.prev_mean_price is not None:
            inflation = compute_inflation(mean_price, self.prev_mean_price)
        else:
            inflation = 0.0
        self.ema_inflation = self.ema_alpha * inflation + (1 - self.ema_alpha) * self.ema_inflation

        if self.prev_gdp is not None and self.prev_gdp > EPSILON:
            gdp_growth = compute_gdp_growth(gdp, self.prev_gdp)
        else:
            gdp_growth = 0.0
        self.ema_gdp_growth = self.ema_alpha * gdp_growth + (1 - self.ema_alpha) * self.ema_gdp_growth

        self.prev_mean_price = mean_price
        self.prev_gdp = gdp

        gini = compute_gini(wealths)
        poverty = compute_poverty_rate(wealths)
        unemp = compute_unemployment(work_attempts, work_matched)
        # crime_rate counts successful thefts; attempts are tracked separately.
        crime_rate = compute_crime_rate(steal_success, n_h)
        crime_attempts_rate = compute_crime_rate(crimes, n_h)
        avg_health = compute_avg_health(healths)
        num_firms = int(alive_slots.size)

        metrics = {
            "week": week,
            "gdp": gdp,
            "gdp_pc": gdp_pc,
            "gdp_growth": gdp_growth,
            "inflation": inflation,
            "ema_inflation": self.ema_inflation,
            "ema_gdp_growth": self.ema_gdp_growth,
            "gini": gini,
            "poverty": poverty,
            "unemployment": unemp,
            "crime_rate": crime_rate,
            "crime_attempts_rate": crime_attempts_rate,
            "avg_health": avg_health,
            "mean_price": mean_price,
            "mean_wage": mean_wage,
            "num_firms": num_firms,
            "num_households": n_h,
            "work_attempts": work_attempts,
            "work_matched": work_matched,
            "crimes": crimes,
            "steal_success": steal_success,
        }

        if extra:
            for key, value in extra.items():
                if key in metrics:
                    continue
                metrics[key] = value

        self.history.append(metrics)
        return metrics

    def get_window(self, window_weeks: int) -> list:
        return self.history[-window_weeks:]

    def get_quarterly_summary(self, window_weeks: int = MACRO_SUMMARY_WINDOW_WEEKS) -> dict:
        window_weeks = max(1, int(window_weeks))
        window = self.get_window(window_weeks)
        if not window:
            return self._empty_summary()

        return {
            "gdp": np.mean([m["gdp"] for m in window]),
            "gdp_pc": np.mean([m["gdp_pc"] for m in window]),
            "gdp_growth": np.mean([m["gdp_growth"] for m in window]),
            "inflation": np.mean([m["inflation"] for m in window]),
            "gini": np.mean([m["gini"] for m in window]),
            "poverty": np.mean([m["poverty"] for m in window]),
            "unemployment": np.mean([m["unemployment"] for m in window]),
            "crime_rate": np.mean([m["crime_rate"] for m in window]),
            "crime_attempts_rate": np.mean([m.get("crime_attempts_rate", 0.0) for m in window]),
            "avg_health": np.mean([m["avg_health"] for m in window]),
            "mean_price": np.mean([m["mean_price"] for m in window]),
            "mean_wage": np.mean([m["mean_wage"] for m in window]),
        }

    def _empty_summary(self) -> dict:
        return {k: 0.0 for k in [
            "gdp", "gdp_pc", "gdp_growth", "inflation", "gini",
            "poverty", "unemployment", "crime_rate", "crime_attempts_rate",
            "avg_health", "mean_price", "mean_wage",
        ]}

    def check_hyperinflation(self) -> bool:
        from marl_economy.config.constants import (
            HYPERINFLATION_THRESHOLD, HYPERINFLATION_CONSECUTIVE
        )
        if len(self.history) < HYPERINFLATION_CONSECUTIVE:
            return False
        recent = self.history[-HYPERINFLATION_CONSECUTIVE:]
        return all(m["inflation"] > HYPERINFLATION_THRESHOLD for m in recent)

    def check_mass_death(self, h: HouseholdState) -> bool:
        """True iff more than MASS_DEATH_THRESHOLD of households are below DEATH_THRESHOLD."""
        if h.size == 0:
            return True
        dead_frac = float(np.mean(h.health <= DEATH_THRESHOLD))
        return dead_frac > MASS_DEATH_THRESHOLD
