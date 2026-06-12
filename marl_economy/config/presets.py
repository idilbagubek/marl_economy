"""
Regime accessor helpers.

Resolves both the canonical academic names (LME, CME, SDWE, SME) introduced in
the FatBar paper and the older free-form names (FreeMarket, Restricted,
SocialDem, Command) so existing run dirs, Unity playback files, and CLI
commands keep working.
"""

import numpy as np

from marl_economy.config.constants import (
    REGIME_ALIAS_MAP,
    REGIME_CONFIGS,
    REGIME_NAMES,
)


REGIMES = REGIME_CONFIGS


def normalize_regime_name(name: str) -> str:
    """Map any accepted regime label to its canonical key."""
    if name in REGIMES:
        return name
    if name in REGIME_ALIAS_MAP:
        return REGIME_ALIAS_MAP[name]
    raise ValueError(
        f"Unknown regime: {name!r}. Choose from canonical {list(REGIME_NAMES)} "
        f"or aliases {sorted(set(REGIME_ALIAS_MAP) - set(REGIME_NAMES))}."
    )


def get_regime(name: str) -> dict:
    """Get regime configuration by name (canonical or alias)."""
    return REGIMES[normalize_regime_name(name)]


def clamp_to_regime(param_name: str, value: float, regime_name: str) -> float:
    """Clamp a policy parameter to its regime-specific bounds."""
    regime = get_regime(regime_name)
    lo, hi = regime["bounds"][param_name]
    return float(np.clip(value, lo, hi))


def init_policy_state(regime_name: str) -> dict:
    """Initialize policy state from regime anchors."""
    regime = get_regime(regime_name)
    return dict(regime["anchors"])
