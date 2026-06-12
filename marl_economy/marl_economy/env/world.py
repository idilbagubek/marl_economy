"""Single-world simulation facade.

Refactored 2026-05-14 as part of the "better design" Stage 3 work: `World`
is now a thin wrapper around `BatchedWorld(K=1)`. There is ONE implementation
of the economic dynamics (in `batched_world.py` plus the vectorized helpers
in `dynamics_*.py` and `routing.py`). The legacy single-world `World` class
in this module used to duplicate that logic; that duplication is gone.

Why a wrapper:
* `BatchedWorld(K=1)` was proven bit-exact with the legacy `World` via
  `stage3_test1_k1_equivalence.py` (6259 fields, 0 diffs across 15
  regime x seed combos).
* Maintaining two implementations of the dynamics meant every change to
  e.g. health decay had to land in two places, and drift was caught only
  by the K=1 test. Folding `World` into `BatchedWorld(K=1)` makes drift
  structurally impossible.
* Per-world RNG streams (via `SeedSequence.spawn(K)`) means
  `BatchedWorld(K, seed)[world 0]` produces the same trajectory regardless
  of K, so the wrapper gives identical results to "running BatchedWorld at
  K=1" or "running BatchedWorld at K=16 and looking at world 0".

The wrapper preserves the legacy `World` public API: `world.h`, `world.f`,
`world.metrics`, `world.policy_state`, `world.step(...)`, etc. all behave
as the old code expected, except they read/write through row-0 views of
the underlying `BatchedHouseholdState`/`BatchedFirmState`.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from marl_economy.config.constants import (
    DEFAULT_NUM_HOUSEHOLDS, DEFAULT_NUM_FIRMS,
    DEFAULT_REGIME_NAME, DEFAULT_RANDOM_SEED,
    FIRM_UPDATE_PERIOD, GOV_UPDATE_PERIOD, CB_UPDATE_PERIOD,
    DEFAULT_EPISODE_WEEKS, WARMUP_WEEKS,
    GOV_A, GOV_B, GOV_C, GOV_D, GOV_E, GOV_U, GOV_ZETA, FIRM_CHI, FIRM_ETA,
    CB_RHO, CB_LAMBDA, CB_NU, CB_ALPHA_R, CB_BETA_R,
    HOUSEHOLD_ACTION_LABELS, ILLNESS_SEVERE,
    WEALTH_CLASS_POOR_MAX, WEALTH_CLASS_MIDDLE_MAX, WEALTH_CLASS_RICH_MAX,
    UNITY_HOUSEHOLD_GRID_COLUMNS, UNITY_HOUSEHOLD_GRID_SPACING, UNITY_HOUSEHOLD_DISTRICT_Z,
    UNITY_FIRM_RING_RADIUS, UNITY_FIRM_RING_STEP, UNITY_FIRM_RING_LAYERS,
    UNITY_GOV_POSITION, UNITY_CB_POSITION, UNITY_MARKET_POSITION,
    UNITY_CLINIC_POSITION, UNITY_BANK_POSITION, UNITY_CRIME_POSITION,
    UNITY_REST_POSITION,
)
from marl_economy.env.disasters import disaster_signal as _disaster_signal
from marl_economy.env.batched_world import BatchedWorld
from marl_economy.env.batched_state import BatchedHouseholdState, BatchedFirmState


# ---------------------------------------------------------------------------
# Row-0 view adapters: expose row 0 of a BatchedHouseholdState /
# BatchedFirmState as the legacy single-world HouseholdState / FirmState API.
# All field properties return 1D views (numpy basic indexing on a 2D array
# with an int index), so in-place mutations propagate back to the batched
# state automatically.
# ---------------------------------------------------------------------------


class _SingleHouseholdView:
    """Read/write view over row 0 of a `BatchedHouseholdState(K=1, N)`."""

    __slots__ = ("_hb",)

    def __init__(self, hb: BatchedHouseholdState):
        self._hb = hb

    @property
    def size(self) -> int:
        return self._hb.N

    @property
    def ids(self) -> np.ndarray:
        return np.arange(self._hb.N, dtype=np.int32)

    @property
    def wealth(self): return self._hb.wealth[0]
    @wealth.setter
    def wealth(self, v): self._hb.wealth[0, :] = v

    @property
    def health(self): return self._hb.health[0]
    @health.setter
    def health(self, v): self._hb.health[0, :] = v

    @property
    def hunger(self): return self._hb.hunger[0]
    @hunger.setter
    def hunger(self, v): self._hb.hunger[0, :] = v

    @property
    def fatigue(self): return self._hb.fatigue[0]
    @fatigue.setter
    def fatigue(self, v): self._hb.fatigue[0, :] = v

    @property
    def illness(self): return self._hb.illness[0]
    @illness.setter
    def illness(self, v): self._hb.illness[0, :] = v

    @property
    def experience(self): return self._hb.experience[0]
    @experience.setter
    def experience(self, v): self._hb.experience[0, :] = v

    @property
    def age(self): return self._hb.age[0]
    @age.setter
    def age(self, v): self._hb.age[0, :] = v

    @property
    def last_action(self): return self._hb.last_action[0]
    @last_action.setter
    def last_action(self, v): self._hb.last_action[0, :] = v

    @property
    def last_routed_firm_id(self): return self._hb.last_routed_firm_id[0]
    @last_routed_firm_id.setter
    def last_routed_firm_id(self, v): self._hb.last_routed_firm_id[0, :] = v

    @property
    def last_action_success(self): return self._hb.last_action_success[0]
    @last_action_success.setter
    def last_action_success(self, v): self._hb.last_action_success[0, :] = v

    @property
    def last_reward(self): return self._hb.last_reward[0]
    @last_reward.setter
    def last_reward(self, v): self._hb.last_reward[0, :] = v

    @property
    def last_wealth_change(self): return self._hb.last_wealth_change[0]
    @last_wealth_change.setter
    def last_wealth_change(self, v): self._hb.last_wealth_change[0, :] = v

    @property
    def last_health_change(self): return self._hb.last_health_change[0]
    @last_health_change.setter
    def last_health_change(self, v): self._hb.last_health_change[0, :] = v

    @property
    def last_crime_success(self): return self._hb.last_crime_success[0]
    @last_crime_success.setter
    def last_crime_success(self, v): self._hb.last_crime_success[0, :] = v

    @property
    def displaced(self): return self._hb.displaced[0]
    @displaced.setter
    def displaced(self, v): self._hb.displaced[0, :] = v

    def __len__(self):
        return self._hb.N


class _SingleFirmView:
    """Read/write view over row 0 of a `BatchedFirmState(K=1, F_max)`."""

    __slots__ = ("_fb",)

    def __init__(self, fb: BatchedFirmState):
        self._fb = fb

    @property
    def max_firms(self) -> int:
        return self._fb.max_firms

    @property
    def size(self) -> int:
        return int(self._fb.size[0])

    @property
    def num_alive(self) -> int:
        return int(self._fb.num_alive(0))

    @property
    def alive(self): return self._fb.alive[0]
    @property
    def id(self): return self._fb.id[0]
    @property
    def price(self): return self._fb.price[0]
    @price.setter
    def price(self, v): self._fb.price[0, :] = v
    @property
    def wage_offer(self): return self._fb.wage_offer[0]
    @wage_offer.setter
    def wage_offer(self, v): self._fb.wage_offer[0, :] = v
    @property
    def cash(self): return self._fb.cash[0]
    @cash.setter
    def cash(self, v): self._fb.cash[0, :] = v
    @property
    def capacity(self): return self._fb.capacity[0]
    @capacity.setter
    def capacity(self, v): self._fb.capacity[0, :] = v
    @property
    def target_capacity(self): return self._fb.target_capacity[0]
    @target_capacity.setter
    def target_capacity(self, v): self._fb.target_capacity[0, :] = v
    @property
    def stock(self): return self._fb.stock[0]
    @stock.setter
    def stock(self, v): self._fb.stock[0, :] = v
    @property
    def productivity(self): return self._fb.productivity[0]
    @productivity.setter
    def productivity(self, v): self._fb.productivity[0, :] = v
    @property
    def sales(self): return self._fb.sales[0]
    @property
    def hires(self): return self._fb.hires[0]
    @property
    def unmet_demand(self): return self._fb.unmet_demand[0]
    @property
    def profit(self): return self._fb.profit[0]
    @property
    def prev_price(self): return self._fb.prev_price[0]
    @property
    def prev_wage(self): return self._fb.prev_wage[0]
    @property
    def negative_cash_weeks(self): return self._fb.negative_cash_weeks[0]
    @property
    def window_sales(self): return self._fb.window_sales[0]
    @property
    def window_hires(self): return self._fb.window_hires[0]
    @property
    def window_unmet(self): return self._fb.window_unmet[0]
    @property
    def window_revenue(self): return self._fb.window_revenue[0]
    @property
    def window_wage_bill(self): return self._fb.window_wage_bill[0]
    @property
    def disaster_downtime(self): return self._fb.disaster_downtime[0]

    def alive_slots(self) -> np.ndarray:
        return self._fb.alive_slots_in_world(0)

    def slot_of(self, firm_id: int) -> int:
        return self._fb.slot_of(0, firm_id)

    def spawn(self, **kwargs) -> int:
        return self._fb.spawn(0, **kwargs)

    def kill(self, slot: int) -> None:
        self._fb.kill(0, slot)


# ---------------------------------------------------------------------------
# World - thin compatibility wrapper around BatchedWorld(K=1).
# ---------------------------------------------------------------------------


class World:
    """Single-world simulation facade backed by `BatchedWorld(K=1)`.

    Public API matches the legacy `World` class so existing callers (the
    RLlib env wrapper, smoke tests, baselines, visualization) keep working
    without changes. Internally there is no duplicated dynamics code -
    every step delegates to `BatchedWorld.step()` and unwraps the K=1
    result.
    """

    def __init__(self,
                 regime_name: str = DEFAULT_REGIME_NAME,
                 num_households: int = DEFAULT_NUM_HOUSEHOLDS,
                 num_firms: int = DEFAULT_NUM_FIRMS,
                 episode_weeks: int = DEFAULT_EPISODE_WEEKS,
                 warmup_weeks: int = WARMUP_WEEKS,
                 seed: int = DEFAULT_RANDOM_SEED,
                 curriculum_phase: int = 1,
                 compressed_quarterly: bool = False):
        self._bw = BatchedWorld(
            regime_name=regime_name, K=1,
            num_households=num_households,
            num_firms=num_firms,
            episode_weeks=episode_weeks,
            warmup_weeks=warmup_weeks,
            seed=seed,
            curriculum_phase=curriculum_phase,
            compressed_quarterly=compressed_quarterly,
        )
        # Row-0 view adapters. Note: `BatchedWorld.reset()` ALLOCATES new
        # state objects rather than reusing existing ones, so the adapters
        # must be rebuilt after each reset; see `_rebuild_views` below.
        self._rebuild_views()

    # ------------------------------------------------------------------
    # View rebuilding (called after BatchedWorld allocates new state).
    # ------------------------------------------------------------------

    def _rebuild_views(self) -> None:
        self.h = _SingleHouseholdView(self._bw.h)
        self.f = _SingleFirmView(self._bw.f)

    # ------------------------------------------------------------------
    # Properties: read/write proxies into BatchedWorld (K=1).
    # ------------------------------------------------------------------

    # Config (set at __init__, read-only after).
    @property
    def regime_name(self): return self._bw.regime_name
    @property
    def num_households(self): return self._bw.num_households
    @property
    def num_firms_init(self): return self._bw.num_firms_init
    @property
    def episode_weeks(self): return self._bw.episode_weeks
    @property
    def warmup_weeks(self): return self._bw.warmup_weeks
    @property
    def curriculum_phase(self): return self._bw.curriculum_phase
    @property
    def compressed_quarterly(self): return self._bw.compressed_quarterly
    @property
    def gov_update_period(self): return self._bw.gov_update_period
    @property
    def cb_update_period(self): return self._bw.cb_update_period
    @property
    def _max_firms(self): return self._bw._max_firms

    # Seed/rng pair. marl_env_rllib reseeds via `world.seed = X` then
    # `world.rng = default_rng(X)`. We honor both by writing into the
    # underlying BatchedWorld's rngs[0] slot.
    @property
    def seed(self): return self._bw.seed
    @seed.setter
    def seed(self, v): self._bw.seed = int(v)

    @property
    def rng(self): return self._bw.rngs[0]
    @rng.setter
    def rng(self, new_rng):
        self._bw.rngs[0] = new_rng

    # Lifecycle scalars (read row-0 of the K=1 batched arrays / lists).
    @property
    def week(self): return self._bw.week
    @week.setter
    def week(self, v): self._bw.week = int(v)

    @property
    def done(self): return bool(self._bw.done[0])

    @property
    def terminated_early(self): return bool(self._bw.terminated_early[0])

    @property
    def termination_reason(self) -> Optional[str]:
        return self._bw.termination_reason[0]

    # Mutable per-step state (lists of dicts in BatchedWorld -> row 0).
    @property
    def policy_state(self): return self._bw.policy_state[0]
    @policy_state.setter
    def policy_state(self, v): self._bw.policy_state[0] = v

    @property
    def prev_policy_state(self): return self._bw.prev_policy_state[0]
    @prev_policy_state.setter
    def prev_policy_state(self, v): self._bw.prev_policy_state[0] = v

    @property
    def interest_rate(self): return float(self._bw.interest_rate[0])
    @interest_rate.setter
    def interest_rate(self, v): self._bw.interest_rate[0] = float(v)

    @property
    def prev_interest_rate(self): return float(self._bw.prev_interest_rate[0])
    @prev_interest_rate.setter
    def prev_interest_rate(self, v): self._bw.prev_interest_rate[0] = float(v)

    # Trackers / ledgers (per-world objects in BatchedWorld).
    @property
    def metrics(self): return self._bw.metrics[0]
    @property
    def event_ledger(self): return self._bw.event_ledger[0]
    @property
    def collapse_detector(self): return self._bw.collapse_detector[0]
    @property
    def disaster_state(self): return self._bw.disaster_state[0]

    @property
    def visual_events(self) -> list[dict]:
        """Per-world visualization events with positions resolved on read.

        `BatchedWorld` stores raw events (firm/household IDs but no positions);
        we enrich each event here using the World-level visualization helpers
        so callers like `get_snapshot()` and `viewer_pygame` see the legacy
        shape (`{"type": ..., "week": ..., "position": {x,y,z}, ...}`).
        """
        out = []
        for ev in self._bw.visual_events[0]:
            ev_copy = dict(ev)
            ev_type = ev.get("type")
            if ev_type in ("firm_closed", "firm_created"):
                fid = ev.get("firm_id")
                if fid is not None:
                    ev_copy["position"] = self._firm_visual_position(int(fid))
            elif ev_type in (
                "household_hunger_crisis", "household_fatigue_crisis",
                "household_severe_illness", "household_death",
            ):
                hid = ev.get("household_id", ev.get("actor_id"))
                if hid is not None:
                    ev_copy["position"] = self._household_home_position(int(hid))
            out.append(ev_copy)
        return out

    # Reward storage.
    @property
    def household_rewards(self): return self._bw.household_rewards[0]
    @household_rewards.setter
    def household_rewards(self, v): self._bw.household_rewards[0] = v

    @property
    def firm_rewards(self): return self._bw.firm_rewards[0]
    @firm_rewards.setter
    def firm_rewards(self, v): self._bw.firm_rewards[0] = v

    @property
    def gov_reward(self): return float(self._bw.gov_reward[0])
    @gov_reward.setter
    def gov_reward(self, v): self._bw.gov_reward[0] = float(v)

    @property
    def cb_reward(self): return float(self._bw.cb_reward[0])
    @cb_reward.setter
    def cb_reward(self, v): self._bw.cb_reward[0] = float(v)

    # Action storage.
    @property
    def firm_actions(self): return self._bw.firm_actions[0]
    @firm_actions.setter
    def firm_actions(self, v): self._bw.firm_actions[0] = v

    # Histories.
    @property
    def unmet_demand_history(self): return self._bw.unmet_demand_history[0]

    @property
    def debug_log(self): return self._bw.debug_log

    # Weekly counters (broadcast row-0 of the (K,) arrays).
    @property
    def weekly_work_attempts(self): return int(self._bw.weekly_work_attempts[0])
    @property
    def weekly_work_matched(self): return int(self._bw.weekly_work_matched[0])
    @property
    def weekly_work_refused_solvency(self): return int(self._bw.weekly_work_refused_solvency[0])
    @property
    def weekly_shop_attempts(self): return int(self._bw.weekly_shop_attempts[0])
    @property
    def weekly_shop_matched(self): return int(self._bw.weekly_shop_matched[0])
    @property
    def weekly_clinic_attempts(self): return int(self._bw.weekly_clinic_attempts[0])
    @property
    def weekly_clinic_matched(self): return int(self._bw.weekly_clinic_matched[0])
    @property
    def weekly_crimes(self): return int(self._bw.weekly_crimes[0])
    @property
    def weekly_steal_success(self): return int(self._bw.weekly_steal_success[0])
    @property
    def weekly_unmet_demand(self): return int(self._bw.weekly_unmet_demand[0])
    @property
    def weekly_total_demand(self): return int(self._bw.weekly_total_demand[0])
    @property
    def weekly_hunger_crisis(self): return int(self._bw.weekly_hunger_crisis[0])
    @property
    def weekly_deaths(self): return int(self._bw.weekly_deaths[0])
    @property
    def weekly_displacements(self): return int(self._bw.weekly_displacements[0])

    # Private weekly stash (used by tests / visualization).
    @property
    def _weekly_clinic_capacity(self): return int(self._bw._weekly_clinic_capacity[0])
    @property
    def _weekly_clinic_used(self): return int(self._bw._weekly_clinic_used[0])
    @property
    def _weekly_place_stats(self): return self._bw._weekly_place_stats[0]

    # Lifecycle.
    def reset(self) -> dict:
        obs_dict = self._bw.reset()
        # BatchedWorld.reset() rebuilds h, f as fresh state objects, so the
        # adapter views need to point at the new ones.
        self._rebuild_views()
        return obs_dict["per_world"][0]

    def step(self, household_actions: dict,
             firm_actions: Optional[dict] = None,
             gov_action: Optional[tuple] = None,
             cb_action: Optional[int] = None) -> tuple:
        obs_dict, rewards_dict, done_arr, info_dict = self._bw.step(
            household_actions, firm_actions, gov_action, cb_action,
        )
        # Unwrap K=1.
        obs = obs_dict["per_world"][0]
        rewards = {
            "households": rewards_dict["households"][0],
            "firms": rewards_dict["firms"][0],
            "gov": float(rewards_dict["gov"][0]),
            "cb": float(rewards_dict["cb"][0]),
        }
        done = bool(done_arr[0])
        info = {
            "week": info_dict["week"],
            "metrics": info_dict["metrics"][0],
            "is_firm_update": info_dict["is_firm_update"],
            "is_gov_update": info_dict["is_gov_update"],
            "is_cb_update": info_dict["is_cb_update"],
            "is_warmup": info_dict["is_warmup"],
            "terminated_early": bool(info_dict["terminated_early"][0]),
            "termination_reason": info_dict["termination_reason"][0],
        }
        return obs, rewards, done, info

    def get_config(self) -> dict:
        return {
            "regime": self.regime_name,
            "seed": self.seed,
            "num_households": self.num_households,
            "num_firms_init": self.num_firms_init,
            "episode_weeks": self.episode_weeks,
            "warmup_weeks": self.warmup_weeks,
            "curriculum_phase": self.curriculum_phase,
            "compressed_quarterly": self.compressed_quarterly,
            "firm_update_period": FIRM_UPDATE_PERIOD,
            "gov_update_period": self.gov_update_period,
            "cb_update_period": self.cb_update_period,
            "coefficients": {
                "gov": {"a": GOV_A, "b": GOV_B, "c": GOV_C, "d": GOV_D,
                        "e": GOV_E, "u": GOV_U, "zeta": GOV_ZETA},
                "firm": {"chi": FIRM_CHI, "eta": FIRM_ETA},
                "cb": {"rho": CB_RHO, "lambda": CB_LAMBDA, "nu": CB_NU,
                       "alpha_r": CB_ALPHA_R, "beta_r": CB_BETA_R},
            },
        }

    # ------------------------------------------------------------------
    # Visualization snapshot (used by evaluate_checkpoint.py, export_json.py,
    # viewer_pygame.py, export_unity.py).
    # ------------------------------------------------------------------

    def get_snapshot(self) -> dict:
        """Produce a snapshot dict for visualization (Unity / pygame / web).

        Reads row 0 of the underlying BatchedWorld via the same h/f adapters
        the public API uses. `visual_events` is preserved but is always empty
        in the wrapper because BatchedWorld doesn't track per-event metadata
        (positions, etc.); training/playback paths that don't rely on event
        animations are unaffected.
        """
        alive_slots = self.f.alive_slots()
        summary_window = max(self.gov_update_period, self.cb_update_period)
        summary = self.metrics.get_quarterly_summary(window_weeks=summary_window)
        created_firm_ids = {
            event["firm_id"]
            for event in self.visual_events
            if event.get("type") == "firm_created"
        }
        downtime_count = int(np.sum(self.f.disaster_downtime[: self.f.size] > 0))

        households = []
        for i in range(self.h.size):
            hid = int(self.h.ids[i])
            last_action = int(self.h.last_action[i])
            action_label = HOUSEHOLD_ACTION_LABELS.get(last_action, "unknown")
            target_firm_id = int(self.h.last_routed_firm_id[i])
            if target_firm_id < 0:
                target_firm_id = None
            visual = self._visual_state_for(action_label)
            wealth = float(self.h.wealth[i])
            households.append({
                "id": hid,
                "wealth": wealth,
                "health": float(self.h.health[i]),
                "hunger": float(self.h.hunger[i]),
                "fatigue": float(self.h.fatigue[i]),
                "illness": int(self.h.illness[i]),
                "last_action": last_action,
                "action_label": action_label,
                "action_success": bool(self.h.last_action_success[i]),
                "target_firm_id": target_firm_id,
                "reward": float(self.h.last_reward[i]),
                "wealth_change": float(self.h.last_wealth_change[i]),
                "health_change": float(self.h.last_health_change[i]),
                "crime_success": bool(self.h.last_crime_success[i]),
                "wealth_class": self._wealth_class(wealth),
                "home_position": self._household_home_position(hid),
                "position": self._destination_for(action_label, target_firm_id, hid),
                "destination": self._destination_for(action_label, target_firm_id, hid),
                "visual_state": visual,
            })

        firms = []
        for slot in alive_slots.tolist():
            slot = int(slot)
            fid = int(self.f.id[slot])
            firms.append({
                "id": fid,
                "price": float(self.f.price[slot]),
                "wage_offer": float(self.f.wage_offer[slot]),
                "stock": float(self.f.stock[slot]),
                "cash": float(self.f.cash[slot]),
                "capacity": float(self.f.capacity[slot]),
                "position": self._firm_visual_position(fid),
                "spawned_this_step": fid in created_firm_ids,
                "visual_type": "firm",
            })

        n_h = self.h.size
        return {
            "week": self.week,
            "regime": self.regime_name,
            "households": households,
            "firms": firms,
            "institutions": {
                "government": {"position": self._vec3(UNITY_GOV_POSITION)},
                "central_bank": {"position": self._vec3(UNITY_CB_POSITION)},
                "market": {"position": self._vec3(UNITY_MARKET_POSITION)},
            },
            "macro": {
                "gdp": summary["gdp"],
                "gini": summary["gini"],
                "inflation": summary["inflation"],
                "avg_health": summary["avg_health"],
                "poverty": summary["poverty"],
                "crime_rate": summary["crime_rate"],
                "unemployment": summary["unemployment"],
                "num_firms": int(alive_slots.size),
                "mean_hunger": float(np.mean(self.h.hunger)) if n_h > 0 else 0.0,
                "frac_hungry": float(np.mean(self.h.hunger >= 0.5)) if n_h > 0 else 0.0,
                "mean_fatigue": float(np.mean(self.h.fatigue)) if n_h > 0 else 0.0,
                "frac_fatigued": float(np.mean(self.h.fatigue >= 0.55)) if n_h > 0 else 0.0,
                "frac_severe": (float(np.mean(self.h.illness == ILLNESS_SEVERE))
                                if n_h > 0 else 0.0),
                "frac_mild": (float(np.mean(self.h.illness == 2))  # ILLNESS_MILD
                              if n_h > 0 else 0.0),
                "mean_firm_cash": (
                    float(np.mean(self.f.cash[alive_slots])) if alive_slots.size > 0 else 0.0
                ),
                "clinic_capacity": self._weekly_clinic_capacity,
                "clinic_used": self._weekly_clinic_used,
                "disaster_signal": _disaster_signal(self.disaster_state),
                "firms_in_downtime": downtime_count,
            },
            "policy": dict(self.policy_state),
            "interest_rate": self.interest_rate,
            "events": list(self.visual_events),
        }

    # ------------------------------------------------------------------
    # Visualization helpers (verbatim from the legacy World class).
    # ------------------------------------------------------------------

    @staticmethod
    def _vec3(position: tuple) -> dict:
        return {
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2]),
        }

    def _household_home_position(self, household_id: int) -> dict:
        columns = max(1, UNITY_HOUSEHOLD_GRID_COLUMNS)
        rows = max(1, int(np.ceil(self.num_households / columns)))
        col = household_id % columns
        row = household_id // columns
        x = (col - (columns - 1) / 2.0) * UNITY_HOUSEHOLD_GRID_SPACING
        z = UNITY_HOUSEHOLD_DISTRICT_Z + (row - (rows - 1) / 2.0) * UNITY_HOUSEHOLD_GRID_SPACING
        return self._vec3((x, 0.0, z))

    def _firm_visual_position(self, firm_id: int) -> dict:
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        layer = firm_id % max(1, UNITY_FIRM_RING_LAYERS)
        radius = UNITY_FIRM_RING_RADIUS + layer * UNITY_FIRM_RING_STEP
        angle = firm_id * golden_angle
        return self._vec3((
            np.cos(angle) * radius,
            0.0,
            np.sin(angle) * radius,
        ))

    def _destination_for(self, label: str, target_firm_id: Optional[int],
                         household_id: int) -> dict:
        if target_firm_id is not None and label in ("work", "shop", "clinic"):
            return self._firm_visual_position(target_firm_id)
        if label == "clinic":
            return self._vec3(UNITY_CLINIC_POSITION)
        if label == "shop":
            return self._vec3(UNITY_MARKET_POSITION)
        if label == "invest":
            return self._vec3(UNITY_BANK_POSITION)
        if label == "steal":
            return self._vec3(UNITY_CRIME_POSITION)
        if label == "rest":
            return self._vec3(UNITY_REST_POSITION)
        return self._household_home_position(household_id)

    @staticmethod
    def _visual_state_for(label: str) -> str:
        if label == "displaced":
            return "displaced"
        if label in ("work", "shop", "clinic", "invest", "steal"):
            return "moving"
        return "idle"

    @staticmethod
    def _wealth_class(w: float) -> str:
        if w < WEALTH_CLASS_POOR_MAX:
            return "poor"
        elif w < WEALTH_CLASS_MIDDLE_MAX:
            return "middle"
        elif w < WEALTH_CLASS_RICH_MAX:
            return "rich"
        return "elite"
