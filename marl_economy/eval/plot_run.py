"""
Matplotlib Analysis Suite (Section 15.2).
Generates static figures from CSV logs:
- Time series (GDP, Gini, health, poverty, crime, inflation, unemployment)
- Price & wage over time
- Firm count + entry/exit markers
- Wealth distribution, Lorenz curve
- Stylized-facts plots (Okun, inflation-output, crime-poverty)
"""

import sys
import os
import csv
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from marl_economy.config.constants import (
    EPSILON,
    PLOT_TIME_SERIES_FIGSIZE, PLOT_HISTOGRAM_FIGSIZE, PLOT_LORENZ_FIGSIZE, PLOT_SCATTER_FIGSIZE,
    PLOT_DPI, PLOT_GRID_ALPHA, PLOT_HISTOGRAM_BINS, PLOT_HISTOGRAM_ALPHA,
    PLOT_REFERENCE_LINE_ALPHA, PLOT_FILL_ALPHA, PLOT_SCATTER_ALPHA, PLOT_SCATTER_SIZE,
)


def load_metrics(csv_path: str) -> dict:
    """Load metrics_step.csv into dict of lists."""
    data = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                if key not in data:
                    data[key] = []
                try:
                    data[key].append(float(val))
                except ValueError:
                    data[key].append(val)
    return data


def load_final_wealths(snapshot_path: str) -> np.ndarray | None:
    """Load final household wealths from a snapshots export if present."""
    if not os.path.exists(snapshot_path):
        return None

    with open(snapshot_path, "r") as f:
        payload = json.load(f)

    snapshots = payload.get("snapshots", [])
    if not snapshots:
        return None

    households = snapshots[-1].get("households", [])
    if not households:
        return None

    return np.array([h["wealth"] for h in households], dtype=float)


def plot_time_series(data: dict, output_dir: str):
    """Generate time series plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    weeks = data["t_week"]

    series = [
        ("gdp_week", "GDP (Weekly Revenue)", "GDP"),
        ("gini", "Gini Coefficient", "Gini"),
        ("avg_health", "Average Health", "Avg Health"),
        ("poverty_rate", "Poverty Rate", "Poverty Rate"),
        ("crime_rate", "Crime Rate", "Crime Rate"),
        ("inflation_proxy", "Inflation Proxy", "Inflation"),
        ("unemp_proxy", "Unemployment Proxy", "Unemployment"),
    ]

    for key, title, ylabel in series:
        if key not in data:
            continue
        fig, ax = plt.subplots(figsize=PLOT_TIME_SERIES_FIGSIZE)
        ax.plot(weeks, data[key], linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("Week")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=PLOT_GRID_ALPHA)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/{key}.png", dpi=PLOT_DPI)
        plt.close(fig)

    # Price & wage on same plot
    if "mean_price" in data and "mean_wage" in data:
        fig, ax = plt.subplots(figsize=PLOT_TIME_SERIES_FIGSIZE)
        ax.plot(weeks, data["mean_price"], label="Mean Price", linewidth=1.0)
        ax.plot(weeks, data["mean_wage"], label="Mean Wage", linewidth=1.0)
        ax.set_title("Price & Wage Over Time")
        ax.set_xlabel("Week")
        ax.legend()
        ax.grid(True, alpha=PLOT_GRID_ALPHA)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/price_wage.png", dpi=PLOT_DPI)
        plt.close(fig)

    # Firm count
    if "num_firms" in data:
        fig, ax = plt.subplots(figsize=PLOT_TIME_SERIES_FIGSIZE)
        ax.plot(weeks, data["num_firms"], linewidth=1.5, color="tab:green")
        ax.set_title("Number of Firms")
        ax.set_xlabel("Week")
        ax.set_ylabel("Firm Count")
        ax.grid(True, alpha=PLOT_GRID_ALPHA)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/firm_count.png", dpi=PLOT_DPI)
        plt.close(fig)

    reward_series = [
        ("reward_households_mean", "Household Reward (Mean)", "Reward"),
        ("reward_firms_mean", "Firm Reward (Mean)", "Reward"),
        ("reward_gov", "Government Reward", "Reward"),
        ("reward_cb", "Central Bank Reward", "Reward"),
        ("reward_total", "Total Reward", "Reward"),
    ]

    for key, title, ylabel in reward_series:
        if key not in data:
            continue
        fig, ax = plt.subplots(figsize=PLOT_TIME_SERIES_FIGSIZE)
        ax.plot(weeks, data[key], linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("Week")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=PLOT_GRID_ALPHA)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/{key}.png", dpi=PLOT_DPI)
        plt.close(fig)

    available_reward_keys = [key for key, _, _ in reward_series if key in data]
    if available_reward_keys:
        fig, ax = plt.subplots(figsize=PLOT_TIME_SERIES_FIGSIZE)
        for key in available_reward_keys:
            ax.plot(weeks, data[key], linewidth=1.0, label=key)
        ax.set_title("Reward Breakdown Over Time")
        ax.set_xlabel("Week")
        ax.set_ylabel("Reward")
        ax.legend()
        ax.grid(True, alpha=PLOT_GRID_ALPHA)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/rewards.png", dpi=PLOT_DPI)
        plt.close(fig)

    print(f"  Time series plots saved to {output_dir}/")


def plot_wealth_distribution(wealths: np.ndarray, output_dir: str, label: str = "final"):
    """Plot wealth histogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=PLOT_HISTOGRAM_FIGSIZE)
    ax.hist(wealths, bins=PLOT_HISTOGRAM_BINS, edgecolor="black", alpha=PLOT_HISTOGRAM_ALPHA)
    ax.set_title(f"Wealth Distribution ({label})")
    ax.set_xlabel("Wealth")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=PLOT_GRID_ALPHA)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/wealth_hist_{label}.png", dpi=PLOT_DPI)
    plt.close(fig)


def plot_lorenz_curve(wealths: np.ndarray, output_dir: str):
    """Plot Lorenz curve with Gini annotation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from marl_economy.env.metrics import compute_lorenz_curve, compute_gini

    os.makedirs(output_dir, exist_ok=True)
    cum_pop, cum_wealth = compute_lorenz_curve(wealths)
    gini = compute_gini(wealths)

    fig, ax = plt.subplots(figsize=PLOT_LORENZ_FIGSIZE)
    ax.plot(cum_pop, cum_wealth, linewidth=2, label="Lorenz Curve")
    ax.plot([0, 1], [0, 1], "k--", alpha=PLOT_REFERENCE_LINE_ALPHA, label="Perfect Equality")
    ax.fill_between(cum_pop, cum_wealth, cum_pop, alpha=PLOT_FILL_ALPHA)
    ax.set_title(f"Lorenz Curve (Gini = {gini:.4f})")
    ax.set_xlabel("Cumulative Population Share")
    ax.set_ylabel("Cumulative Wealth Share")
    ax.legend()
    ax.grid(True, alpha=PLOT_GRID_ALPHA)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/lorenz.png", dpi=PLOT_DPI)
    plt.close(fig)


def plot_stylized_facts(data: dict, output_dir: str):
    """
    ABIDES-style validation plots (Section 14.4):
    - Okun-like: Delta GDP vs unemployment
    - Inflation vs GDP growth
    - Crime vs poverty
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Compute GDP growth.
    # When gdp_prev is small but nonzero the ratio can spike enough to dominate
    # the y-axis. Use a stricter floor and clip; raw values are still in the CSV.
    gdp = np.array(data["gdp_week"])
    gdp_prev = gdp[:-1]
    gdp_growth = np.zeros_like(gdp_prev, dtype=float)
    GDP_PREV_FLOOR = max(EPSILON, 0.05)
    GDP_GROWTH_DISPLAY_CLIP = 3.0
    valid_growth = gdp_prev > GDP_PREV_FLOOR
    gdp_growth[valid_growth] = np.diff(gdp)[valid_growth] / gdp_prev[valid_growth]
    gdp_growth = np.clip(gdp_growth, -GDP_GROWTH_DISPLAY_CLIP, GDP_GROWTH_DISPLAY_CLIP)

    # Okun: GDP growth vs unemployment
    unemp = np.array(data["unemp_proxy"])[1:]
    if len(gdp_growth) == len(unemp):
        fig, ax = plt.subplots(figsize=PLOT_SCATTER_FIGSIZE)
        ax.scatter(unemp, gdp_growth, alpha=PLOT_SCATTER_ALPHA, s=PLOT_SCATTER_SIZE)
        ax.set_title("Okun-like: GDP Growth vs Unemployment")
        ax.set_xlabel("Unemployment Proxy")
        ax.set_ylabel("GDP Growth")
        ax.grid(True, alpha=PLOT_GRID_ALPHA)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/okun.png", dpi=PLOT_DPI)
        plt.close(fig)

    # Inflation vs GDP growth
    inflation = np.array(data["inflation_proxy"])[1:]
    if len(gdp_growth) == len(inflation):
        fig, ax = plt.subplots(figsize=PLOT_SCATTER_FIGSIZE)
        ax.scatter(inflation, gdp_growth, alpha=PLOT_SCATTER_ALPHA, s=PLOT_SCATTER_SIZE)
        ax.set_title("Inflation vs GDP Growth")
        ax.set_xlabel("Inflation Proxy")
        ax.set_ylabel("GDP Growth")
        ax.grid(True, alpha=PLOT_GRID_ALPHA)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/inflation_output.png", dpi=PLOT_DPI)
        plt.close(fig)

    # Crime vs poverty
    crime = np.array(data["crime_rate"])
    poverty = np.array(data["poverty_rate"])
    fig, ax = plt.subplots(figsize=PLOT_SCATTER_FIGSIZE)
    ax.scatter(poverty, crime, alpha=PLOT_SCATTER_ALPHA, s=PLOT_SCATTER_SIZE)
    ax.set_title("Crime Rate vs Poverty Rate")
    ax.set_xlabel("Poverty Rate")
    ax.set_ylabel("Crime Rate")
    ax.grid(True, alpha=PLOT_GRID_ALPHA)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/crime_poverty.png", dpi=PLOT_DPI)
    plt.close(fig)

    print(f"  Stylized-facts plots saved to {output_dir}/")


# Diagnostic plots. Each one silently no-ops if its required columns are
# missing, so old runs still produce as many plots as their CSV supports.

def _has_columns(data: dict, *cols: str) -> bool:
    return all(c in data and len(data[c]) > 0 for c in cols)


def _safe_ratio(num, den):
    """Element-wise num/den with NaN where den == 0, for matplotlib safety."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full_like(num, np.nan)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out


def plot_firm_health(data: dict, output_dir: str):
    """
    Firm financial trajectory: mean/min/max cash with insolvency reference lines.

    This is the diagnostic that exposes the "bankrupt firms hire forever"
    failure mode that pure firm-count plots hide.
    """
    if not _has_columns(data, "t_week", "mean_firm_cash", "min_firm_cash", "max_firm_cash"):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from marl_economy.config.constants import FIRM_INSOLVENT_CASH_THRESHOLD

    os.makedirs(output_dir, exist_ok=True)
    weeks = data["t_week"]

    fig, ax = plt.subplots(figsize=PLOT_TIME_SERIES_FIGSIZE)
    ax.fill_between(weeks, data["min_firm_cash"], data["max_firm_cash"],
                    alpha=PLOT_FILL_ALPHA, color="tab:blue", label="min-max range")
    ax.plot(weeks, data["mean_firm_cash"], linewidth=1.2,
            color="tab:blue", label="mean firm cash")
    ax.plot(weeks, data["min_firm_cash"], linewidth=0.8,
            color="tab:red", alpha=0.6, label="min firm cash")
    ax.axhline(0.0, linestyle="-", color="black", alpha=PLOT_REFERENCE_LINE_ALPHA, linewidth=0.8)
    ax.axhline(FIRM_INSOLVENT_CASH_THRESHOLD, linestyle="--", color="tab:red",
               alpha=PLOT_REFERENCE_LINE_ALPHA, linewidth=0.8,
               label=f"insolvent ({FIRM_INSOLVENT_CASH_THRESHOLD})")
    ax.set_title("Firm Financial Health Over Time")
    ax.set_xlabel("Week")
    ax.set_ylabel("Cash")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=PLOT_GRID_ALPHA)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/firm_cash.png", dpi=PLOT_DPI)
    plt.close(fig)

    # Frac-insolvent companion plot if available.
    if "frac_insolvent_firms" in data:
        fig, ax = plt.subplots(figsize=PLOT_TIME_SERIES_FIGSIZE)
        ax.plot(weeks, data["frac_insolvent_firms"], linewidth=1.0, color="tab:red")
        ax.set_title("Fraction of Insolvent Firms")
        ax.set_xlabel("Week")
        ax.set_ylabel("Fraction")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=PLOT_GRID_ALPHA)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/frac_insolvent_firms.png", dpi=PLOT_DPI)
        plt.close(fig)


def plot_hunger_dynamics(data: dict, output_dir: str):
    """
    Population hunger over time: mean, max, and fraction in crisis.

    Mean and max share a single axis (both in [0, 1]); frac_hungry/frac_crisis
    are plotted on the same axis since they're also fractions.
    """
    if not _has_columns(data, "t_week", "mean_hunger"):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from marl_economy.config.constants import (
        HUNGER_HEALTH_PENALTY_THRESHOLD,
        HUNGER_CRISIS_THRESHOLD,
        HUNGER_FATAL_THRESHOLD,
    )

    os.makedirs(output_dir, exist_ok=True)
    weeks = data["t_week"]

    fig, ax = plt.subplots(figsize=PLOT_TIME_SERIES_FIGSIZE)
    ax.plot(weeks, data["mean_hunger"], linewidth=1.2,
            color="tab:orange", label="mean hunger")
    if "max_hunger" in data:
        ax.plot(weeks, data["max_hunger"], linewidth=0.8,
                color="tab:red", alpha=0.7, label="max hunger")
    if "frac_hungry" in data:
        ax.plot(weeks, data["frac_hungry"], linewidth=0.9,
                color="tab:purple", alpha=0.7, linestyle="--",
                label="frac hungry (>= penalty)")
    if "frac_crisis" in data:
        ax.plot(weeks, data["frac_crisis"], linewidth=0.9,
                color="tab:brown", alpha=0.7, linestyle=":",
                label="frac crisis (>= 0.85)")
    ax.axhline(HUNGER_HEALTH_PENALTY_THRESHOLD, linestyle="--",
               color="tab:purple", alpha=PLOT_REFERENCE_LINE_ALPHA, linewidth=0.6)
    ax.axhline(HUNGER_CRISIS_THRESHOLD, linestyle="--",
               color="tab:brown", alpha=PLOT_REFERENCE_LINE_ALPHA, linewidth=0.6)
    ax.axhline(HUNGER_FATAL_THRESHOLD, linestyle="--",
               color="tab:red", alpha=PLOT_REFERENCE_LINE_ALPHA, linewidth=0.6)
    ax.set_title("Population Hunger Dynamics")
    ax.set_xlabel("Week")
    ax.set_ylabel("Hunger / Fraction")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=PLOT_GRID_ALPHA)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/hunger_dynamics.png", dpi=PLOT_DPI)
    plt.close(fig)


def plot_action_success_rates(data: dict, output_dir: str):
    """
    Per-action match/success rate over time.

    Work success = work_matched / (work_attempts). Shop success similarly.
    Also plots work_refused_solvency to expose how often firms turn workers
    away - the previously-invisible signal that bankrupt-hire blocking is
    actually firing.
    """
    if "t_week" not in data:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    weeks = data["t_week"]

    series = []
    # Work
    if "shop_attempts" in data and "shop_matched" in data:
        rate = _safe_ratio(data["shop_matched"], data["shop_attempts"])
        series.append((rate, "shop match rate", "tab:blue"))
    if "clinic_attempts" in data and "clinic_matched" in data:
        rate = _safe_ratio(data["clinic_matched"], data["clinic_attempts"])
        series.append((rate, "clinic match rate", "tab:green"))
    # `unemp_proxy` is (work_attempts - work_matched) / work_attempts, so
    # work success = 1 - unemp_proxy.
    if "unemp_proxy" in data:
        unemp = np.asarray(data["unemp_proxy"], dtype=float)
        series.append((1.0 - unemp, "work match rate", "tab:orange"))

    if series:
        fig, ax = plt.subplots(figsize=PLOT_TIME_SERIES_FIGSIZE)
        for values, label, color in series:
            ax.plot(weeks, values, linewidth=1.0, label=label, color=color)
        ax.set_title("Action Match / Success Rate")
        ax.set_xlabel("Week")
        ax.set_ylabel("Match rate (matched / attempts)")
        ax.set_ylim(-0.02, 1.05)
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, alpha=PLOT_GRID_ALPHA)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/action_match_rates.png", dpi=PLOT_DPI)
        plt.close(fig)

    # Refused-hire counts: distinct y-axis from match rate, so its own plot.
    if "work_refused_solvency" in data:
        fig, ax = plt.subplots(figsize=PLOT_TIME_SERIES_FIGSIZE)
        ax.plot(weeks, data["work_refused_solvency"], linewidth=1.0,
                color="tab:red", label="work refused (no solvent firm)")
        if "weekly_unmet_demand" in data:
            ax.plot(weeks, data["weekly_unmet_demand"], linewidth=1.0,
                    color="tab:purple", alpha=0.6, label="unmet shop/clinic demand")
        ax.set_title("Hire Refusals & Unmet Demand")
        ax.set_xlabel("Week")
        ax.set_ylabel("Households")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=PLOT_GRID_ALPHA)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/hire_refusals.png", dpi=PLOT_DPI)
        plt.close(fig)


# Event columns that carry per-week counts. Only those present in the CSV
# are plotted, so this is safe across old/new runs.
_FIRM_EVENT_COLS = [
    ("events_firm_entry_normal", "firm entry (normal)", "tab:green"),
    ("events_firm_entry_emergency", "firm entry (emergency)", "tab:olive"),
    ("events_firm_exit", "firm exit", "tab:red"),
    ("events_firm_exit_blocked", "firm exit BLOCKED", "tab:purple"),
    ("events_firm_hire_refused", "firm hire refused", "tab:gray"),
]
_HOUSEHOLD_EVENT_COLS = [
    ("events_household_death", "household death", "black"),
    ("events_household_hunger_crisis", "hunger crisis", "tab:orange"),
    ("events_household_displaced", "displaced", "tab:cyan"),
]


def plot_event_timeline(data: dict, output_dir: str):
    """
    Per-week event counts: separate panels for firm-side and household-side events.
    """
    if "t_week" not in data:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    weeks = data["t_week"]

    firm_present = [(c, label, color) for c, label, color in _FIRM_EVENT_COLS if c in data]
    if firm_present:
        fig, ax = plt.subplots(figsize=PLOT_TIME_SERIES_FIGSIZE)
        for col, label, color in firm_present:
            ax.plot(weeks, data[col], linewidth=1.0, label=label, color=color)
        ax.set_title("Weekly Firm Events")
        ax.set_xlabel("Week")
        ax.set_ylabel("Events this week")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=PLOT_GRID_ALPHA)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/events_firm_weekly.png", dpi=PLOT_DPI)
        plt.close(fig)

    hh_present = [(c, label, color) for c, label, color in _HOUSEHOLD_EVENT_COLS if c in data]
    if hh_present:
        fig, ax = plt.subplots(figsize=PLOT_TIME_SERIES_FIGSIZE)
        for col, label, color in hh_present:
            ax.plot(weeks, data[col], linewidth=1.0, label=label, color=color)
        ax.set_title("Weekly Household Events")
        ax.set_xlabel("Week")
        ax.set_ylabel("Events this week")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=PLOT_GRID_ALPHA)
        fig.tight_layout()
        fig.savefig(f"{output_dir}/events_household_weekly.png", dpi=PLOT_DPI)
        plt.close(fig)


def plot_cumulative_events(data: dict, output_dir: str):
    """
    Cumulative running totals of event counts. Lets a reader see whether
    events are concentrated early, late, or persistent.
    """
    if "t_week" not in data:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    weeks = data["t_week"]

    all_cols = _FIRM_EVENT_COLS + _HOUSEHOLD_EVENT_COLS
    present = [(c, label, color) for c, label, color in all_cols if c in data]
    if not present:
        return

    fig, ax = plt.subplots(figsize=PLOT_TIME_SERIES_FIGSIZE)
    for col, label, color in present:
        cum = np.cumsum(np.asarray(data[col], dtype=float))
        ax.plot(weeks, cum, linewidth=1.0, label=label, color=color)
    ax.set_title("Cumulative Event Totals")
    ax.set_xlabel("Week")
    ax.set_ylabel("Cumulative events")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=PLOT_GRID_ALPHA)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/events_cumulative.png", dpi=PLOT_DPI)
    plt.close(fig)


def plot_inventory_demand(data: dict, output_dir: str):
    """
    Total firm inventory and weekly unmet shop/clinic demand.

    A rising inventory line with rising unmet demand is the "produced but
    nobody bought" signature that pure GDP cannot distinguish from
    "production stopped."
    """
    if "t_week" not in data:
        return
    if not ("total_inventory" in data or "weekly_unmet_demand" in data):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    weeks = data["t_week"]

    fig, ax_left = plt.subplots(figsize=PLOT_TIME_SERIES_FIGSIZE)
    plotted = False
    if "total_inventory" in data:
        ax_left.plot(weeks, data["total_inventory"], linewidth=1.2,
                     color="tab:blue", label="total inventory")
        ax_left.set_ylabel("Total inventory", color="tab:blue")
        ax_left.tick_params(axis="y", labelcolor="tab:blue")
        plotted = True

    if "weekly_unmet_demand" in data:
        ax_right = ax_left.twinx() if plotted else ax_left
        ax_right.plot(weeks, data["weekly_unmet_demand"], linewidth=1.0,
                      color="tab:red", alpha=0.7, label="unmet demand")
        if plotted:
            ax_right.set_ylabel("Unmet demand (weekly)", color="tab:red")
            ax_right.tick_params(axis="y", labelcolor="tab:red")
        else:
            ax_right.set_ylabel("Unmet demand (weekly)")

    ax_left.set_title("Inventory vs. Unmet Demand")
    ax_left.set_xlabel("Week")
    ax_left.grid(True, alpha=PLOT_GRID_ALPHA)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/inventory_demand.png", dpi=PLOT_DPI)
    plt.close(fig)


def plot_economy_dashboard(data: dict, output_dir: str):
    """
    2x3 multi-panel dashboard combining the most informative diagnostic
    series in one figure. Panels silently blank if their column is missing.
    """
    if "t_week" not in data:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    weeks = data["t_week"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    (ax_gdp, ax_firms, ax_cash) = axes[0]
    (ax_hunger, ax_match, ax_deaths) = axes[1]

    # Panel 1: GDP
    if "gdp_week" in data:
        ax_gdp.plot(weeks, data["gdp_week"], linewidth=1.0, color="tab:blue")
        ax_gdp.set_title("GDP (weekly revenue)")
        ax_gdp.grid(True, alpha=PLOT_GRID_ALPHA)

    # Panel 2: Firm count
    if "num_firms" in data:
        ax_firms.plot(weeks, data["num_firms"], linewidth=1.0, color="tab:green")
        ax_firms.set_title("Firms alive")
        ax_firms.grid(True, alpha=PLOT_GRID_ALPHA)

    # Panel 3: Mean firm cash with insolvency line
    if "mean_firm_cash" in data:
        from marl_economy.config.constants import FIRM_INSOLVENT_CASH_THRESHOLD
        ax_cash.plot(weeks, data["mean_firm_cash"], linewidth=1.0, color="tab:purple")
        ax_cash.axhline(0.0, linestyle="-", color="black", alpha=0.4, linewidth=0.8)
        ax_cash.axhline(FIRM_INSOLVENT_CASH_THRESHOLD, linestyle="--",
                        color="tab:red", alpha=0.5, linewidth=0.8)
        ax_cash.set_title("Mean firm cash")
        ax_cash.grid(True, alpha=PLOT_GRID_ALPHA)

    # Panel 4: Hunger
    if "mean_hunger" in data:
        ax_hunger.plot(weeks, data["mean_hunger"], linewidth=1.0,
                       color="tab:orange", label="mean")
        if "frac_crisis" in data:
            ax_hunger.plot(weeks, data["frac_crisis"], linewidth=0.9,
                           color="tab:red", alpha=0.7, linestyle="--",
                           label="frac crisis")
        ax_hunger.set_title("Hunger")
        ax_hunger.set_ylim(-0.02, 1.05)
        ax_hunger.legend(loc="upper left", fontsize=7)
        ax_hunger.grid(True, alpha=PLOT_GRID_ALPHA)

    # Panel 5: Action match rates
    plotted_match = False
    if "unemp_proxy" in data:
        unemp = np.asarray(data["unemp_proxy"], dtype=float)
        ax_match.plot(weeks, 1.0 - unemp, linewidth=1.0,
                      color="tab:orange", label="work")
        plotted_match = True
    if "shop_attempts" in data and "shop_matched" in data:
        rate = _safe_ratio(data["shop_matched"], data["shop_attempts"])
        ax_match.plot(weeks, rate, linewidth=1.0, color="tab:blue", label="shop")
        plotted_match = True
    if plotted_match:
        ax_match.set_title("Action match rate")
        ax_match.set_ylim(-0.02, 1.05)
        ax_match.legend(loc="lower left", fontsize=7)
        ax_match.grid(True, alpha=PLOT_GRID_ALPHA)

    # Panel 6: Cumulative deaths
    if "events_household_death" in data:
        cum_deaths = np.cumsum(np.asarray(data["events_household_death"], dtype=float))
        ax_deaths.plot(weeks, cum_deaths, linewidth=1.2, color="black")
        ax_deaths.set_title("Cumulative household deaths")
        ax_deaths.grid(True, alpha=PLOT_GRID_ALPHA)
    elif "weekly_deaths" in data:
        cum_deaths = np.cumsum(np.asarray(data["weekly_deaths"], dtype=float))
        ax_deaths.plot(weeks, cum_deaths, linewidth=1.2, color="black")
        ax_deaths.set_title("Cumulative household deaths")
        ax_deaths.grid(True, alpha=PLOT_GRID_ALPHA)

    for ax in axes.flat:
        ax.set_xlabel("Week")

    fig.suptitle("Economy Diagnostic Dashboard", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{output_dir}/dashboard.png", dpi=PLOT_DPI)
    plt.close(fig)


def plot_fatigue_dynamics(data: dict, output_dir: str):
    """Mean / max fatigue and the fatigued / exhausted population fractions."""
    if not _has_columns(data, "t_week", "mean_fatigue"):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from marl_economy.config.constants import (
        FATIGUE_PRODUCTIVITY_THRESHOLD, FATIGUE_HEALTH_PENALTY_THRESHOLD,
    )
    os.makedirs(output_dir, exist_ok=True)
    weeks = data["t_week"]
    fig, ax = plt.subplots(figsize=PLOT_TIME_SERIES_FIGSIZE)
    ax.plot(weeks, data["mean_fatigue"], linewidth=1.2, color="tab:purple", label="mean fatigue")
    if "max_fatigue" in data:
        ax.plot(weeks, data["max_fatigue"], linewidth=0.8, color="tab:red", alpha=0.7, label="max fatigue")
    if "frac_fatigued" in data:
        ax.plot(weeks, data["frac_fatigued"], linewidth=0.9, linestyle="--",
                color="tab:orange", alpha=0.7, label="frac fatigued")
    if "frac_exhausted" in data:
        ax.plot(weeks, data["frac_exhausted"], linewidth=0.9, linestyle=":",
                color="tab:brown", alpha=0.7, label="frac exhausted")
    ax.axhline(FATIGUE_PRODUCTIVITY_THRESHOLD, linestyle="--",
               color="tab:purple", alpha=PLOT_REFERENCE_LINE_ALPHA, linewidth=0.6)
    ax.axhline(FATIGUE_HEALTH_PENALTY_THRESHOLD, linestyle="--",
               color="tab:red", alpha=PLOT_REFERENCE_LINE_ALPHA, linewidth=0.6)
    ax.set_title("Population Fatigue Dynamics")
    ax.set_xlabel("Week")
    ax.set_ylabel("Fatigue / Fraction")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=PLOT_GRID_ALPHA)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/fatigue_dynamics.png", dpi=PLOT_DPI)
    plt.close(fig)


def plot_illness_states(data: dict, output_dir: str):
    """Stacked-area chart of illness-state fractions over time."""
    state_cols = ("frac_healthy", "frac_exposed", "frac_mild",
                  "frac_severe", "frac_recovering")
    if not _has_columns(data, "t_week", *state_cols):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)
    weeks = data["t_week"]
    series = np.array([data[c] for c in state_cols], dtype=np.float64)
    fig, ax = plt.subplots(figsize=PLOT_TIME_SERIES_FIGSIZE)
    colors = ["tab:green", "gold", "tab:orange", "tab:red", "tab:cyan"]
    labels = ["healthy", "exposed", "mild", "severe", "recovering"]
    ax.stackplot(weeks, series, labels=labels, colors=colors, alpha=0.85)
    ax.set_title("Illness State Distribution")
    ax.set_xlabel("Week")
    ax.set_ylabel("Population fraction")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=PLOT_GRID_ALPHA)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/illness_states.png", dpi=PLOT_DPI)
    plt.close(fig)


def plot_clinic_capacity(data: dict, output_dir: str):
    """Clinic capacity vs. weekly utilization, with an overcrowd marker."""
    if not _has_columns(data, "t_week", "clinic_capacity", "clinic_used"):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)
    weeks = data["t_week"]
    cap = np.asarray(data["clinic_capacity"], dtype=float)
    used = np.asarray(data["clinic_used"], dtype=float)
    fig, ax = plt.subplots(figsize=PLOT_TIME_SERIES_FIGSIZE)
    ax.plot(weeks, cap, linewidth=1.2, color="tab:green", label="capacity")
    ax.fill_between(weeks, 0, used, alpha=0.4, color="tab:blue", label="used")
    overcrowd = used > cap * 0.95
    if np.any(overcrowd):
        ax.scatter(np.array(weeks)[overcrowd], used[overcrowd],
                   s=10, color="tab:red", label="overcrowded", zorder=5)
    ax.set_title("Clinic Capacity vs. Utilization")
    ax.set_xlabel("Week")
    ax.set_ylabel("Slots / Visits")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=PLOT_GRID_ALPHA)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/clinic_capacity.png", dpi=PLOT_DPI)
    plt.close(fig)


def plot_disasters(data: dict, output_dir: str):
    """Active disasters with shaded supply-shock and epidemic-wave windows."""
    if not _has_columns(data, "t_week", "disaster_signal"):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(output_dir, exist_ok=True)
    weeks = data["t_week"]
    fig, ax = plt.subplots(figsize=PLOT_TIME_SERIES_FIGSIZE)
    ax.plot(weeks, data["disaster_signal"], linewidth=1.2,
            color="tab:red", label="active disaster count")
    if "firms_in_downtime" in data:
        ax.plot(weeks, data["firms_in_downtime"], linewidth=0.9,
                color="tab:gray", alpha=0.7, label="firms in downtime")
    if "supply_shock_active" in data:
        ax.fill_between(weeks, 0, np.asarray(data["supply_shock_active"]) * 4.0,
                        color="tab:orange", alpha=0.15, label="supply shock window")
    if "epidemic_wave_active" in data:
        ax.fill_between(weeks, 0, np.asarray(data["epidemic_wave_active"]) * 4.0,
                        color="tab:purple", alpha=0.15, label="epidemic wave window")
    ax.set_title("Active Disasters Over Time")
    ax.set_xlabel("Week")
    ax.set_ylabel("Count")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=PLOT_GRID_ALPHA)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/disasters.png", dpi=PLOT_DPI)
    plt.close(fig)


def generate_all_plots(run_dir: str):
    """Generate all plots for a given run directory."""
    csv_path = f"{run_dir}/metrics_step.csv"
    if not os.path.exists(csv_path):
        print(f"No metrics file found at {csv_path}")
        return

    plot_dir = f"{run_dir}/plots"
    data = load_metrics(csv_path)

    # Drop an INFO.md alongside the plots so each PNG is traceable back to
    # its source run even if the directory is copied or shared.
    try:
        from marl_economy.eval.run_metadata import write_plots_info, load_run_config
        write_plots_info(plot_dir, load_run_config(run_dir))
    except Exception as e:
        print(f"  (plots INFO generation skipped: {e})")

    plot_time_series(data, plot_dir)
    plot_stylized_facts(data, plot_dir)

    # Diagnostic plots from the extended schema. Each is a no-op if its
    # required columns are absent, so old runs degrade gracefully.
    plot_firm_health(data, plot_dir)
    plot_hunger_dynamics(data, plot_dir)
    plot_fatigue_dynamics(data, plot_dir)
    plot_illness_states(data, plot_dir)
    plot_clinic_capacity(data, plot_dir)
    plot_disasters(data, plot_dir)
    plot_action_success_rates(data, plot_dir)
    plot_event_timeline(data, plot_dir)
    plot_cumulative_events(data, plot_dir)
    plot_inventory_demand(data, plot_dir)
    plot_economy_dashboard(data, plot_dir)

    final_wealths = load_final_wealths(f"{run_dir}/snapshots.json")
    if final_wealths is not None and len(final_wealths) > 0:
        plot_wealth_distribution(final_wealths, plot_dir, label="final")
        plot_lorenz_curve(final_wealths, plot_dir)

    print(f"All plots generated in {plot_dir}/")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate plots from run logs")
    parser.add_argument("run_dir", help="Path to run directory")
    args = parser.parse_args()
    generate_all_plots(args.run_dir)


if __name__ == "__main__":
    main()
