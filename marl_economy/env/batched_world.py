"""BatchedWorld - K parallel MARL economy worlds stepped in lockstep.

Architecture: hybrid vectorization.

  * Hot path - the per-household weekly update - is fully vectorized across a
    leading K dimension. `step_households_batch_kn` operates on shape
    `(K, N)` arrays so the bulk of the work (wage/utility/clamp math over
    K*N agent-steps) runs as a single numpy call per inner block.

  * Cold path - disasters, firm entry/exit, routing, event recording,
    metrics, clinic-capacity triage, epidemics, illness transitions - is
    per-world Python loops. These ops are either rare per week or have
    sequential semantics (entry/exit) that don't vectorize cleanly across
    worlds. Looping over K=64 worlds with ~6 firms each is still much
    cheaper than the K*N household math.

Per-world state is kept as `K`-length lists of single-world objects:
  * policy_state, disaster_state, prev_policy_state            (dicts)
  * event ledger, collapse detector, metrics tracker            (objects)
  * interest_rate, prev_interest_rate, week                     (scalars in arrays)
  * firm_actions                                                (dict per world)
  * weekly accumulators                                         (arrays of length K)

Termination semantics: each world has its own `done[k]` flag. Once a world
terminates, subsequent calls to `step()` still advance the underlying state
arrays for that world (so the vectorized math stays uniform) but no new
metrics or events are recorded.

K=1 bit-equivalence target: running a `BatchedWorld(K=1)` should produce the
same metrics history (and same per-step RNG consumption) as a single
`World`. Verified via the validation script.
"""

from __future__ import annotations

import numpy as np
from copy import deepcopy
from typing import Optional

from marl_economy.config.constants import (
    EPSILON, DEFAULT_NUM_HOUSEHOLDS, DEFAULT_NUM_FIRMS, F_MIN, F_MAX,
    DEFAULT_REGIME_NAME, DEFAULT_RANDOM_SEED,
    FIRM_UPDATE_PERIOD, GOV_UPDATE_PERIOD, CB_UPDATE_PERIOD,
    COMPRESSED_QUARTERLY_UPDATE_PERIOD,
    DEFAULT_EPISODE_WEEKS, WARMUP_WEEKS,
    EARLY_TERM_PENALTY_SCALE, DEATH_THRESHOLD,
    HOUSEHOLD_EARLY_TERM_SCALE, FIRM_BORROW_RATE_SCALE, STEAL_REWARD_PENALTY,
    FIRM_EARLY_TERM_SCALE,
    TREASURY_DIVIDEND_FLOOR, TREASURY_DIVIDEND_KAPPA,
    TREASURY_DIVIDEND_CEILING, TREASURY_DIVIDEND_RESERVE,
    ACTION_WORK, ACTION_SHOP, ACTION_STEAL, ACTION_REST, ACTION_INVEST, ACTION_CLINIC,
    HOUSEHOLD_ACTION_LABELS,
    DISPLACEMENT_PENALTY,
    GOV_A, GOV_B, GOV_C, GOV_D, GOV_E, GOV_U, GOV_ZETA, FIRM_CHI, FIRM_ETA,
    GDP_COLLAPSE_FLOOR, GDP_COLLAPSE_PENALTY,
    GDP_COLLAPSE_RELATIVE_FRAC, GDP_PC_PEAK_DECAY,
    GOV_HEALTH_FLOOR, GOV_HEALTH_FLOOR_PEN,
    CLINIC_COST_MULTIPLIER, SHOP_COST_FRACTION,
    CB_RHO, CB_LAMBDA, CB_NU, CB_INFLATION_TARGET,
    CB_ALPHA_R, CB_BETA_R, CB_RATE_MIN, CB_RATE_MAX,
    HOUSEHOLD_OBS_DIM, FIRM_OBS_DIM, GOV_OBS_DIM, CB_OBS_DIM,
    FIRM_MARKET_NOISE_STD,
    WORLD_INIT_FIRM_CASH_JITTER, WORLD_INIT_FIRM_CAPACITY, WORLD_INIT_FIRM_STOCK,
    WORLD_INIT_HOUSEHOLD_WEALTH_MIN, WORLD_INIT_HOUSEHOLD_WEALTH_MAX,
    WORLD_INIT_HOUSEHOLD_HEALTH_MIN, WORLD_INIT_HOUSEHOLD_HEALTH_MAX,
    FIRM_NEUTRAL_ACTION,
    FIRM_DEFAULT_SMOOTHNESS_SCALE, FIRM_JOINT_FINE_TUNING_SMOOTHNESS_SCALE,
    MACRO_DEFAULT_SMOOTHNESS_SCALE,
    ILLNESS_HEALTHY, ILLNESS_SEVERE, ILLNESS_RECOVERING,
    NUM_ILLNESS_STATES, FATIGUE_HEALTH_PENALTY_THRESHOLD,
    RLLIB_MAX_FIRM_AGENT_SLOTS,
)
from marl_economy.config.presets import get_regime, init_policy_state, clamp_to_regime
from marl_economy.env.batched_state import (
    BatchedHouseholdState, BatchedFirmState,
)
from marl_economy.env.state import NO_ROUTED_FIRM_ID
from marl_economy.env.dynamics_household import (
    compute_household_reward, step_households_batch_kn,
)
from marl_economy.env.dynamics_firm import (
    firm_produce_stock_slot, firm_pay_wage_slot, firm_sell_slot, firm_pay_overhead_slot,
    firm_update_capacity_slot, firm_apply_action_slot,
    compute_firm_reward_slot, firm_reset_window_slot,
)
from marl_economy.env.routing import (
    route_work_vec, route_shop_vec, route_clinic_vec, route_steal_target,
    route_work_vec_kn, route_shop_vec_kn, route_clinic_vec_kn,
)
from marl_economy.env.events import (
    EventLedger,
    FIRM_ENTRY_NORMAL,
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
from marl_economy.env.collapse_detector import CollapseDetector
from marl_economy.env.dynamics_hunger import init_hunger, hunger_summary
from marl_economy.env.dynamics_fatigue import init_fatigue, fatigue_summary
from marl_economy.env.illness import (
    illness_summary, step_illness_transitions,
)
from marl_economy.env.epidemics import transmit_at_places
from marl_economy.env.clinic_capacity import weekly_capacity, allocate_treatment
from marl_economy.env.disasters import (
    init_disaster_state,
    maybe_trigger as maybe_trigger_disasters,
    disaster_signal,
    epidemic_wave_boost,
    supply_shock_capacity_mult,
)
from marl_economy.env.metrics import MetricsTracker
from marl_economy.env.firm_solvency import firm_solvency_diagnostics


# We piggyback on the single-world `FirmState` / `HouseholdState` adapter
# pattern: per-world cold-path functions (disasters, entry_exit, etc.) take a
# single-world view of the batched state. The adapter exposes one world's
# row of each (K, ...) array as a properly-typed single-world state object.

from marl_economy.env.state import HouseholdState, FirmState


class _WorldView:
    """Light adapter exposing one world's row of BatchedHouseholdState /
    BatchedFirmState as the single-world HouseholdState / FirmState classes.

    This lets us call into the existing single-world cold-path helpers
    (`disasters.maybe_trigger`, `entry_exit.process_mass_exit`, etc.) without
    duplicating their bodies. The view aliases the array slices, so mutations
    propagate back to the batched state automatically.
    """

    def __init__(self, h_batched: BatchedHouseholdState, f_batched: BatchedFirmState,
                 world_idx: int):
        self.k = int(world_idx)
        self._hb = h_batched
        self._fb = f_batched
        self.h = self._HouseholdView(h_batched, self.k)
        self.f = self._FirmView(f_batched, self.k)

    class _HouseholdView:
        """Acts like a `HouseholdState` for one world. Field reads/writes
        return / mutate the corresponding row of the batched array."""

        __slots__ = ("_hb", "k", "size")

        def __init__(self, hb: BatchedHouseholdState, k: int):
            self._hb = hb
            self.k = k
            self.size = hb.N

        @property
        def wealth(self): return self._hb.wealth[self.k]
        @property
        def health(self): return self._hb.health[self.k]
        @property
        def hunger(self): return self._hb.hunger[self.k]
        @property
        def fatigue(self): return self._hb.fatigue[self.k]
        @property
        def illness(self): return self._hb.illness[self.k]
        @illness.setter
        def illness(self, value):
            self._hb.illness[self.k] = value
        @property
        def experience(self): return self._hb.experience[self.k]
        @property
        def age(self): return self._hb.age[self.k]
        @property
        def last_action(self): return self._hb.last_action[self.k]
        @property
        def last_routed_firm_id(self): return self._hb.last_routed_firm_id[self.k]
        @property
        def last_action_success(self): return self._hb.last_action_success[self.k]
        @property
        def last_reward(self): return self._hb.last_reward[self.k]
        @property
        def last_wealth_change(self): return self._hb.last_wealth_change[self.k]
        @property
        def last_health_change(self): return self._hb.last_health_change[self.k]
        @property
        def last_crime_success(self): return self._hb.last_crime_success[self.k]
        @property
        def displaced(self): return self._hb.displaced[self.k]
        @property
        def ids(self): return np.arange(self.size, dtype=np.int32)

        def __len__(self):
            return self.size

    class _FirmView:
        """Acts like a single-world `FirmState` for one world.

        The view forwards `max_firms`, `size`, `spawn`, `kill` to the batched
        owner so that entry/exit semantics match `FirmState` exactly.
        """

        def __init__(self, fb: BatchedFirmState, k: int):
            self._fb = fb
            self.k = k
            self.max_firms = fb.max_firms

        @property
        def size(self):
            return int(self._fb.size[self.k])

        @property
        def alive(self): return self._fb.alive[self.k]
        @property
        def id(self): return self._fb.id[self.k]
        @property
        def price(self): return self._fb.price[self.k]
        @property
        def wage_offer(self): return self._fb.wage_offer[self.k]
        @property
        def cash(self): return self._fb.cash[self.k]
        @property
        def capacity(self): return self._fb.capacity[self.k]
        @property
        def target_capacity(self): return self._fb.target_capacity[self.k]
        @property
        def stock(self): return self._fb.stock[self.k]
        @property
        def productivity(self): return self._fb.productivity[self.k]
        @property
        def sales(self): return self._fb.sales[self.k]
        @property
        def hires(self): return self._fb.hires[self.k]
        @property
        def unmet_demand(self): return self._fb.unmet_demand[self.k]
        @property
        def profit(self): return self._fb.profit[self.k]
        @property
        def prev_price(self): return self._fb.prev_price[self.k]
        @property
        def prev_wage(self): return self._fb.prev_wage[self.k]
        @property
        def negative_cash_weeks(self): return self._fb.negative_cash_weeks[self.k]
        @property
        def window_sales(self): return self._fb.window_sales[self.k]
        @property
        def window_hires(self): return self._fb.window_hires[self.k]
        @property
        def window_unmet(self): return self._fb.window_unmet[self.k]
        @property
        def window_revenue(self): return self._fb.window_revenue[self.k]
        @property
        def window_wage_bill(self): return self._fb.window_wage_bill[self.k]
        @property
        def disaster_downtime(self): return self._fb.disaster_downtime[self.k]

        @property
        def num_alive(self):
            return int(self._fb.alive[self.k, : self.size].sum())

        def alive_slots(self):
            return np.where(self._fb.alive[self.k, : self.size])[0]

        def slot_of(self, firm_id):
            return self._fb.slot_of(self.k, firm_id)

        def spawn(self, *, price, wage, cash, capacity, stock, productivity=None):
            kwargs = {
                "price": price, "wage": wage, "cash": cash,
                "capacity": capacity, "stock": stock,
            }
            if productivity is not None:
                kwargs["productivity"] = productivity
            return self._fb.spawn(self.k, **kwargs)

        def kill(self, slot):
            self._fb.kill(self.k, slot)


def _fixed_firm_capacity(num_firms_init: int) -> int:
    return max(RLLIB_MAX_FIRM_AGENT_SLOTS, num_firms_init * 4 + 8)


class BatchedWorld:
    """K parallel MARL economy worlds stepped in lockstep.

    Mirrors the public surface of `World`: `reset()`, `step()`, per-week
    metrics history is available per world via `self.metrics[k].history`.
    """

    def __init__(self, regime_name: str = DEFAULT_REGIME_NAME,
                 K: int = 1,
                 num_households: int = DEFAULT_NUM_HOUSEHOLDS,
                 num_firms: int = DEFAULT_NUM_FIRMS,
                 episode_weeks: int = DEFAULT_EPISODE_WEEKS,
                 warmup_weeks: int = WARMUP_WEEKS,
                 seed: int = DEFAULT_RANDOM_SEED,
                 curriculum_phase: int = 1,
                 compressed_quarterly: bool = False):
        self.K = int(K)
        self.regime_name = regime_name
        self.num_households = int(num_households)
        self.num_firms_init = int(num_firms)
        self.episode_weeks = int(episode_weeks)
        self.warmup_weeks = int(warmup_weeks)
        self.curriculum_phase = int(curriculum_phase)
        self.compressed_quarterly = bool(compressed_quarterly)
        self.seed = int(seed)

        self.gov_update_period = GOV_UPDATE_PERIOD
        self.cb_update_period = CB_UPDATE_PERIOD
        if self.compressed_quarterly and self.curriculum_phase <= 2:
            self.gov_update_period = COMPRESSED_QUARTERLY_UPDATE_PERIOD
            self.cb_update_period = COMPRESSED_QUARTERLY_UPDATE_PERIOD

        self._max_firms = _fixed_firm_capacity(self.num_firms_init)
        # Per-world RNG streams via SeedSequence.spawn(K). Each world has its
        # own independent Generator so per-seed trajectories are reproducible
        # across different K values (BatchedWorld(K=K1, seed=s)[world i]
        # produces the same draws as BatchedWorld(K=K2, seed=s)[world i] for
        # any K1, K2 >= i+1). Cost 3 fix from the "better design" refactor.
        seq = np.random.SeedSequence(self.seed)
        sub_seqs = seq.spawn(self.K)
        self.rngs: list[np.random.Generator] = [
            np.random.default_rng(s) for s in sub_seqs
        ]
        self.debug_log: list[str] = []

        self.reset()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self):
        self.week = 0
        self.done = np.zeros(self.K, dtype=bool)
        self.terminated_early = np.zeros(self.K, dtype=bool)
        self.termination_reason: list[Optional[str]] = [None] * self.K

        # Per-world event ledgers / collapse detectors / metrics trackers.
        self.event_ledger = [EventLedger() for _ in range(self.K)]
        self.collapse_detector = [CollapseDetector() for _ in range(self.K)]
        self.metrics = [MetricsTracker() for _ in range(self.K)]
        self.disaster_state = [init_disaster_state() for _ in range(self.K)]
        # Government treasury (Option B): collects the wage tax wedge each week
        # and funds welfare. Conserves money (tax was previously destroyed).
        # Deficit allowed (can go negative = government borrowing) and tracked.
        self.treasury = np.zeros(self.K, dtype=np.float64)
        # Phase 3 (#4 relative collapse): slow-decaying per-world peak of
        # gdp_pc; the gov collapse penalty ramps as gdp_pc falls below
        # GDP_COLLAPSE_RELATIVE_FRAC * peak (robust to the unknown scale).
        self.gdp_pc_peak = np.zeros(self.K, dtype=np.float64)
        # C1 conservation ledger (pure diagnostic, no behavior change). Per-step
        # money MINTED / DESTROYED accumulators (destroyed stored as a positive
        # magnitude). Reset every step. residual = (Delta total_money) - (minted -
        # destroyed) must be ~0; a nonzero residual reveals a missed/mis-signed
        # site. Write-off is ledgered as DESTROYED (NOT routed to treasury) per
        # the review - it is genuine unrepayable bad debt and the signal must
        # stay visible.
        self._minted = np.zeros(self.K, dtype=np.float64)
        self._destroyed = np.zeros(self.K, dtype=np.float64)
        self._prev_total_money = np.full(self.K, np.nan, dtype=np.float64)
        # Welfare-valve instrumentation (per-week, per-world; pure diagnostic).
        self._wk_tax_collected = np.zeros(self.K, dtype=np.float64)
        self._wk_welfare_paid = np.zeros(self.K, dtype=np.float64)
        self._wk_dividend_paid = np.zeros(self.K, dtype=np.float64)

        # Per-world policy state, initial values (regime defaults).
        self.policy_state: list[dict] = []
        self.prev_policy_state: list[dict] = []
        self.interest_rate = np.zeros(self.K, dtype=np.float64)
        self.prev_interest_rate = np.zeros(self.K, dtype=np.float64)
        for k in range(self.K):
            ps = init_policy_state(self.regime_name)
            self.interest_rate[k] = ps.pop("interest_rate")
            self.prev_interest_rate[k] = self.interest_rate[k]
            self.policy_state.append(ps)
            self.prev_policy_state.append(deepcopy(ps))

        # Per-episode "first-time" event sets, per world.
        self._fatigue_crisis_emitted: list[set[int]] = [set() for _ in range(self.K)]
        self._severe_illness_emitted: list[set[int]] = [set() for _ in range(self.K)]
        self._hunger_crisis_emitted: list[set[int]] = [set() for _ in range(self.K)]
        self._death_emitted: list[set[int]] = [set() for _ in range(self.K)]

        # Per-world visualization event log. Positions are intentionally NOT
        # stored here - `World.visual_events` (the K=1 wrapper accessor)
        # enriches each event with positions on read via its visualization
        # helpers (`_firm_visual_position`, `_household_home_position`). This
        # keeps `BatchedWorld` free of viz concerns and lets future BatchedWorld
        # snapshot generators choose their own coordinate scheme.
        self.visual_events: list[list[dict]] = [[] for _ in range(self.K)]

        # Allocate batched state.
        self.h = BatchedHouseholdState(self.K, self.num_households)
        self.f = BatchedFirmState(self.K, self._max_firms)

        # Initialize households per world using per-world rng streams. Within
        # each world: uniform(wealth), uniform(health), init_hunger,
        # init_fatigue. Order matches single-world `World.reset` so K=1
        # bit-equiv with the new World-as-wrapper holds trivially.
        for k in range(self.K):
            rng_k = self.rngs[k]
            for i in range(self.num_households):
                self.h.wealth[k, i] = rng_k.uniform(
                    WORLD_INIT_HOUSEHOLD_WEALTH_MIN, WORLD_INIT_HOUSEHOLD_WEALTH_MAX,
                )
                self.h.health[k, i] = rng_k.uniform(
                    WORLD_INIT_HOUSEHOLD_HEALTH_MIN, WORLD_INIT_HOUSEHOLD_HEALTH_MAX,
                )
                self.h.hunger[k, i] = init_hunger(rng_k)
                self.h.fatigue[k, i] = init_fatigue(rng_k)

        # Initialize firms per world using per-world rng streams. Per-firm
        # init RNG order: normal(price), normal(wage), uniform(cash jitter).
        regime = get_regime(self.regime_name)
        anchor_price = regime["anchors"]["price_level"]
        anchor_wage = regime["anchors"]["wage_base"]
        self.firm_actions: list[dict] = [dict() for _ in range(self.K)]
        for k in range(self.K):
            rng_k = self.rngs[k]
            for j in range(self.num_firms_init):
                price = anchor_price + rng_k.normal(0, FIRM_MARKET_NOISE_STD)
                wage = anchor_wage + rng_k.normal(0, FIRM_MARKET_NOISE_STD)
                cash = 0.5 + rng_k.uniform(0, WORLD_INIT_FIRM_CASH_JITTER)
                slot = self.f.spawn(
                    k, price=price, wage=wage, cash=cash,
                    capacity=WORLD_INIT_FIRM_CAPACITY,
                    stock=WORLD_INIT_FIRM_STOCK,
                )
                self.firm_actions[k][int(self.f.id[k, slot])] = FIRM_NEUTRAL_ACTION

        # Unmet demand histories per world (used by firm-entry rule).
        self.unmet_demand_history: list[list[dict]] = [[] for _ in range(self.K)]

        # Reset weekly accumulators.
        self._init_weekly_accumulators()

        # Slow-agent reward storage per world.
        self.household_rewards: list[dict] = [dict() for _ in range(self.K)]
        self.firm_rewards: list[dict] = [dict() for _ in range(self.K)]
        self.gov_reward = np.zeros(self.K, dtype=np.float64)
        self.cb_reward = np.zeros(self.K, dtype=np.float64)

        # Per-world weekly stats stash for snapshot use.
        self._weekly_clinic_capacity = np.zeros(self.K, dtype=np.int32)
        self._weekly_clinic_used = np.zeros(self.K, dtype=np.int32)
        self._weekly_place_stats: list[dict] = [dict() for _ in range(self.K)]

        # C1: seed the conservation ledger with the ACTUAL initial total money
        # per world (NOT NaN) now that wealth/cash/treasury are all set, so the
        # FIRST stepped week's `money_residual` is a real conservation check
        # rather than silently force-zeroed by the NaN guard (which is kept in
        # step() only as a defensive fallback). Also (re)zero the per-step
        # accumulators so a reused instance can't carry stale ledger state.
        # MUST match step()'s `_tm` formula exactly (alive-firm cash only).
        self._minted[:] = 0.0
        self._destroyed[:] = 0.0
        self._prev_total_money = (
            self.h.wealth.sum(axis=1)
            + np.where(self.f.alive, self.f.cash, 0.0).sum(axis=1)
            + self.treasury
        ).astype(np.float64)

        return self._get_all_observations()

    def _init_weekly_accumulators(self):
        # Each weekly counter is a (K,) array.
        zero = lambda: np.zeros(self.K, dtype=np.int64)
        self.weekly_work_attempts = zero()
        self.weekly_work_matched = zero()
        self.weekly_work_refused_solvency = zero()
        self.weekly_shop_attempts = zero()
        self.weekly_shop_matched = zero()
        self.weekly_clinic_attempts = zero()
        self.weekly_clinic_matched = zero()
        self.weekly_crimes = zero()
        self.weekly_steal_success = zero()
        self.weekly_unmet_demand = zero()
        self.weekly_total_demand = zero()
        self.weekly_hunger_crisis = zero()
        self.weekly_deaths = zero()
        self.weekly_displacements = zero()

    def _reset_weekly_accumulators(self):
        for arr in (
            self.weekly_work_attempts, self.weekly_work_matched,
            self.weekly_work_refused_solvency,
            self.weekly_shop_attempts, self.weekly_shop_matched,
            self.weekly_clinic_attempts, self.weekly_clinic_matched,
            self.weekly_crimes, self.weekly_steal_success,
            self.weekly_unmet_demand, self.weekly_total_demand,
            self.weekly_hunger_crisis, self.weekly_deaths,
            self.weekly_displacements,
        ):
            arr[:] = 0

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self,
             household_actions: list[dict] | dict,
             firm_actions: Optional[list[Optional[dict]]] = None,
             gov_action: Optional[list[Optional[tuple]]] = None,
             cb_action: Optional[list[Optional[int]]] = None) -> tuple:
        """Step all K worlds one week in lockstep.

        Parameters
        ----------
        household_actions
            Either a single dict (applied to every world) or a list of length
            K of dicts (per-world `{hid: action}` mappings).
        firm_actions, gov_action, cb_action
            Optional per-world overrides. If `None`, no action that week.

        Returns
        -------
        obs, rewards, done, info
            obs: dict with per-agent observation arrays keyed by world.
            rewards: per-world reward dict.
            done: shape (K,) bool - per-world episode done state.
            info: per-world dict.
        """
        # Normalize action inputs to per-world lists.
        hh_actions = self._normalize_action_list(household_actions, dict_default={})
        firm_actions_list = self._normalize_action_list(firm_actions, default=None, allow_scalar_dict=False)
        gov_action_list = self._normalize_action_list(gov_action, default=None, allow_scalar_dict=False)
        cb_action_list = self._normalize_action_list(cb_action, default=None, allow_scalar_dict=False)

        # Don't step worlds that have already terminated.
        if np.all(self.done):
            raise RuntimeError("All worlds are done. Call reset().")

        self._reset_weekly_accumulators()
        for k in range(self.K):
            self.visual_events[k].clear()
        # C1: reset the per-step money ledger (all K; done worlds are skipped in
        # every phase so their deltas stay 0).
        self._minted[:] = 0.0
        self._destroyed[:] = 0.0

        # ---- 1) Disasters (per-world, sequential). ----
        triggered_disasters: list[list[dict]] = [[] for _ in range(self.K)]
        for k in range(self.K):
            if self.done[k]:
                continue
            view = _WorldView(self.h, self.f, k)
            illness_slice = self.h.illness[k]
            triggered = maybe_trigger_disasters(
                self.week, self.disaster_state[k],
                view.f, view.h, illness_slice, self.rngs[k],
            )
            # `illness_slice` is a view; epidemic_wave_seed mutated in place.
            triggered_disasters[k] = triggered
            # C1 G: earthquake flat firm cash loss -> destroyed (only the
            # earthquake info carries `cash_destroyed`; .get default 0 for the
            # other disaster types). This loop body is active-world only (the
            # `if self.done[k]: continue` guard above).
            for _di in triggered:
                self._destroyed[k] += float(_di.get("cash_destroyed", 0.0))
            for info in triggered:
                etype = {
                    "earthquake": DISASTER_EARTHQUAKE,
                    "fire": DISASTER_FIRE,
                    "flood": DISASTER_FLOOD,
                    "supply_shock": DISASTER_SUPPLY_SHOCK,
                    "epidemic_wave": DISASTER_EPIDEMIC_WAVE,
                }.get(info["type"])
                if etype is not None:
                    extras = {kk: vv for kk, vv in info.items() if kk not in ("type", "week")}
                    self.event_ledger[k].record(etype, week=self.week, **extras)
                    self.visual_events[k].append({
                        "type": etype,
                        "week": self.week,
                        **extras,
                    })

        # ---- 2) Slow-agent action application (per-world). ----
        is_firm_update = (self.week > 0 and self.week % FIRM_UPDATE_PERIOD == 0)
        is_gov_update = (self.week > 0 and self.week % self.gov_update_period == 0)
        is_cb_update = (self.week > 0 and self.week % self.cb_update_period == 0)

        for k in range(self.K):
            if self.done[k]:
                continue
            view = _WorldView(self.h, self.f, k)
            if is_firm_update and firm_actions_list[k] is not None:
                self.firm_actions[k].update(firm_actions_list[k])
                for fid, action in firm_actions_list[k].items():
                    slot = view.f.slot_of(int(fid))
                    if slot >= 0:
                        firm_apply_action_slot(
                            view.f, slot, action,
                            self.policy_state[k].get("min_wage_floor", 0.0),
                        )
            if is_gov_update and gov_action_list[k] is not None:
                self._apply_gov_action(k, gov_action_list[k])
                self.visual_events[k].append({
                    "type": "government_policy_changed",
                    "week": self.week,
                    "policy": dict(self.policy_state[k]),
                })
            if is_cb_update and cb_action_list[k] is not None:
                prev_rate = float(self.interest_rate[k])
                self._apply_cb_action(k, cb_action_list[k])
                self.visual_events[k].append({
                    "type": "central_bank_rate_changed",
                    "week": self.week,
                    "previous_rate": prev_rate,
                    "interest_rate": float(self.interest_rate[k]),
                })

        # ---- 3) Firm entry / exit (per-world). ----
        from marl_economy.env.entry_exit import (
            process_mass_exit, check_firm_entry, spawn_firm,
            recompute_market_averages,
        )
        for k in range(self.K):
            if self.done[k]:
                continue
            view = _WorldView(self.h, self.f, k)
            # Snapshot alive firm IDs BEFORE entry/exit so we can emit
            # firm_closed / firm_created visual events.
            alive_slots_before = view.f.alive_slots()
            alive_ids_before = {int(view.f.id[s]) for s in alive_slots_before.tolist()}

            # C1 sites A/B (firm spawn -> mint) + C (exit write-off/destroy).
            # This loop is active-world only (the `if self.done[k]: continue`
            # guard above), so accumulate straight into the per-world ledger.
            _mx = process_mass_exit(
                view.f, view.h, self.unmet_demand_history[k],
                self.rngs[k], self.debug_log,
                event_ledger=self.event_ledger[k], week=self.week,
            )
            self._minted[k] += float(_mx["money_minted"])
            self._destroyed[k] += float(_mx["money_destroyed"])
            if check_firm_entry(view.f, self.unmet_demand_history[k], self.rngs[k]):
                mean_p, mean_w = recompute_market_averages(view.f)
                new_slot = spawn_firm(view.f, mean_p, mean_w, self.rngs[k])
                # C1 A/B: a normal-entry firm joins the alive set with fresh
                # cash from nowhere -> MINT. Ledger the ACTUAL spawned cash
                # (ground truth, robust if spawn cash logic ever changes).
                self._minted[k] += float(view.f.cash[new_slot])
                new_id = int(view.f.id[new_slot])
                self.firm_actions[k][new_id] = FIRM_NEUTRAL_ACTION
                self.event_ledger[k].record(
                    FIRM_ENTRY_NORMAL, week=self.week, actor_id=new_id,
                    price=float(view.f.price[new_slot]),
                    wage=float(view.f.wage_offer[new_slot]),
                )
            mean_price, mean_wage = recompute_market_averages(view.f)
            self.policy_state[k]["price_level"] = mean_price
            self.policy_state[k]["wage_base"] = mean_wage

            # Emit firm_closed / firm_created visual events based on diff.
            alive_slots_after = view.f.alive_slots()
            alive_ids_after = {int(view.f.id[s]) for s in alive_slots_after.tolist()}
            for firm_id in sorted(alive_ids_before - alive_ids_after):
                self.visual_events[k].append({
                    "type": "firm_closed",
                    "week": self.week,
                    "firm_id": firm_id,
                })
            for firm_id in sorted(alive_ids_after - alive_ids_before):
                self.visual_events[k].append({
                    "type": "firm_created",
                    "week": self.week,
                    "firm_id": firm_id,
                })

        # Reset weekly firm counters for all alive firms across all worlds.
        # (Easy to vectorize; mask multiplication.)
        for k in range(self.K):
            if self.done[k]:
                continue
            view = _WorldView(self.h, self.f, k)
            slots = view.f.alive_slots()
            self.f.sales[k, slots] = 0.0
            self.f.hires[k, slots] = 0
            self.f.unmet_demand[k, slots] = 0.0

        # ---- 4) Routing (per-world Stage-2 calls; builds (K, N) routes). ----
        # Build per-world action arrays.
        actions_arr = np.zeros((self.K, self.num_households), dtype=np.int32)
        for k in range(self.K):
            if self.done[k]:
                actions_arr[k, :] = ACTION_REST
                continue
            ha = hh_actions[k]
            for i in range(self.num_households):
                actions_arr[k, i] = int(ha.get(i, ACTION_REST))

        # Consume + record displaced flags (per-world, per-HH event recording).
        for k in range(self.K):
            if self.done[k]:
                continue
            dmask = self.h.displaced[k].copy()
            if np.any(dmask):
                self.h.displaced[k, :] = False
                idxs = np.where(dmask)[0]
                self.weekly_displacements[k] += int(idxs.size)
                for i in idxs.tolist():
                    self.event_ledger[k].record(
                        HOUSEHOLD_DISPLACED, week=self.week, actor_id=i,
                    )

        # Mask helpers.
        work_mask = (actions_arr == ACTION_WORK)
        shop_mask = (actions_arr == ACTION_SHOP)
        clinic_mask = (actions_arr == ACTION_CLINIC)
        steal_mask = (actions_arr == ACTION_STEAL)
        self.weekly_work_attempts[:] = work_mask.sum(axis=1)
        self.weekly_shop_attempts[:] = shop_mask.sum(axis=1)
        self.weekly_clinic_attempts[:] = clinic_mask.sum(axis=1)
        self.weekly_crimes[:] = steal_mask.sum(axis=1)
        self.weekly_total_demand[:] = (shop_mask | clinic_mask).sum(axis=1)

        # K-vectorized routing across all (non-done) worlds. At K=1 the K-vec
        # functions internally delegate to the single-world `route_*_vec`
        # implementations, preserving bit-exact RNG consumption with the
        # post-Stage-2 single-world baseline. At K>1 they run a single
        # vectorized pass per action type (work, shop, clinic).
        routed_firm_slots = np.full(
            (self.K, self.num_households), NO_ROUTED_FIRM_ID, dtype=np.int32,
        )
        active_mask_kn = (~self.done)[:, None]  # (K, 1)
        wm_kn = work_mask & active_mask_kn
        sm_kn = shop_mask & active_mask_kn
        cm_kn = clinic_mask & active_mask_kn

        min_wage_floor_k = np.array(
            [float(self.policy_state[k].get("min_wage_floor", 0.0)) for k in range(self.K)],
            dtype=np.float64,
        )

        work_slots_kn = route_work_vec_kn(self.f, wm_kn, min_wage_floor_k, self.rngs)
        shop_slots_kn = route_shop_vec_kn(self.f, sm_kn, self.rngs)
        clinic_slots_kn = route_clinic_vec_kn(self.f, cm_kn, self.rngs)

        # Overwrite the -1 default only where the respective action mask is
        # True AND routing returned a non-negative slot. Refused workers /
        # no-eligible-firm shoppers stay at -1.
        routed_firm_slots = np.where(
            wm_kn & (work_slots_kn >= 0), work_slots_kn, routed_firm_slots,
        )
        routed_firm_slots = np.where(
            sm_kn & (shop_slots_kn >= 0), shop_slots_kn, routed_firm_slots,
        )
        routed_firm_slots = np.where(
            cm_kn & (clinic_slots_kn >= 0), clinic_slots_kn, routed_firm_slots,
        )

        # Per-world bookkeeping: refused-hire events + weekly counters.
        for k in range(self.K):
            if self.done[k]:
                continue
            alive_firms_count = int(self.f.alive[k].sum())
            wm_k = wm_kn[k]
            sm_k = sm_kn[k]
            cm_k = cm_kn[k]
            if alive_firms_count > 0:
                refused = wm_k & (routed_firm_slots[k] == NO_ROUTED_FIRM_ID)
                n_refused = int(refused.sum())
                self.weekly_work_refused_solvency[k] = n_refused
                if n_refused > 0:
                    for i in np.where(refused)[0].tolist():
                        self.event_ledger[k].record(
                            FIRM_HIRE_REFUSED, week=self.week,
                            actor_id=i, alive_firms=alive_firms_count,
                        )
            self.weekly_work_matched[k] = int(
                np.sum(wm_k & (routed_firm_slots[k] != NO_ROUTED_FIRM_ID))
            )
            unmet_routing = (sm_k | cm_k) & (routed_firm_slots[k] == NO_ROUTED_FIRM_ID)
            self.weekly_unmet_demand[k] += int(unmet_routing.sum())

        # ---- 4b) Firm production - STOCK ONLY (FIX A, 2026-05-18 v2). ----
        # Runs AFTER routing (so f.hires is final) and BEFORE the phase-5
        # household step, so the `_ration_stock` budget snapshot inside that
        # step sees THIS week's production (the real food-supply throttle was
        # that the ration/sell only saw last-week leftover stock -> ~25% fill).
        # Stock/goods only - NO money op -> C1-conservation-neutral; draws NO
        # rng -> household-step rng stream bit-identical (determinism intact).
        # The matching WAGE debit is phase 10 (needs the phase-5 Option-B
        # gross wage). Downtime firms produce nothing (skipped here AND in
        # phase 10, exactly as before). cap_mult (supply shock) wraps capacity
        # here since it scales production q=min(prod*sqrt(h), cap*cap_mult).
        # NOTE (known secondary, flagged): routing eligibility (phase 4) still
        # uses pre-production stock for shop/clinic ROUTING_MIN_STOCK; with
        # prod*3 leftover is ample after wk1 so most firms stay routable.
        for k in range(self.K):
            if self.done[k]:
                continue
            view = _WorldView(self.h, self.f, k)
            cap_mult = supply_shock_capacity_mult(self.disaster_state[k])
            for slot in view.f.alive_slots().tolist():
                if int(self.f.disaster_downtime[k, slot]) > 0:
                    continue
                if cap_mult < 1.0:
                    saved = float(self.f.capacity[k, slot])
                    self.f.capacity[k, slot] = saved * cap_mult
                    firm_produce_stock_slot(view.f, slot)
                    self.f.capacity[k, slot] = saved
                else:
                    firm_produce_stock_slot(view.f, slot)

        # ---- 5) VECTORIZED household batch step (hot path). ----
        # Build per-world policy arrays for the K-vectorized step.
        tax_rate_k = np.array([self.policy_state[k].get("tax_rate", 0.0) for k in range(self.K)], dtype=np.float64)
        welfare_k = np.array([self.policy_state[k].get("welfare", 0.0) for k in range(self.K)], dtype=np.float64)
        policing_k = np.array([self.policy_state[k].get("policing", 0.0) for k in range(self.K)], dtype=np.float64)
        healthcare_k = np.array([self.policy_state[k].get("healthcare_access", 0.0) for k in range(self.K)], dtype=np.float64)
        education_k = np.array([self.policy_state[k].get("education_quality", 0.0) for k in range(self.K)], dtype=np.float64)
        episode_progress_k = np.full(self.K, self.week / max(1, self.episode_weeks), dtype=np.float64)

        wealth_before = self.h.wealth.copy()
        health_before = self.h.health.copy()
        illness_pre_step = self.h.illness.copy()

        # Snapshot done worlds' household state so we can freeze it after the
        # vectorized step (the batch step mutates ALL K rows; we restore the
        # done rows to keep their terminal state stable).
        done_mask = self.done.copy()
        _snapshot = None
        if np.any(done_mask):
            _snapshot = {
                "wealth":               self.h.wealth[done_mask].copy(),
                "health":               self.h.health[done_mask].copy(),
                "hunger":               self.h.hunger[done_mask].copy(),
                "fatigue":              self.h.fatigue[done_mask].copy(),
                "illness":              self.h.illness[done_mask].copy(),
                "experience":           self.h.experience[done_mask].copy(),
                "age":                  self.h.age[done_mask].copy(),
                "last_action":          self.h.last_action[done_mask].copy(),
                "last_routed_firm_id":  self.h.last_routed_firm_id[done_mask].copy(),
                "last_action_success":  self.h.last_action_success[done_mask].copy(),
                "last_reward":          self.h.last_reward[done_mask].copy(),
                "last_wealth_change":   self.h.last_wealth_change[done_mask].copy(),
                "last_health_change":   self.h.last_health_change[done_mask].copy(),
                "last_crime_success":   self.h.last_crime_success[done_mask].copy(),
                "displaced":            self.h.displaced[done_mask].copy(),
                "tainted_wealth":       self.h.tainted_wealth[done_mask].copy(),
            }

        step_result = step_households_batch_kn(
            self.h, self.f, actions_arr, routed_firm_slots,
            tax_rate_k=tax_rate_k, welfare_k=welfare_k,
            policing_k=policing_k, healthcare_k=healthcare_k,
            education_k=education_k,
            interest_rate_k=self.interest_rate,
            rngs=self.rngs,
            episode_progress_k=episode_progress_k,
        )

        # Restore done worlds' state - they should remain frozen at terminal.
        if _snapshot is not None:
            self.h.wealth[done_mask] = _snapshot["wealth"]
            self.h.health[done_mask] = _snapshot["health"]
            self.h.hunger[done_mask] = _snapshot["hunger"]
            self.h.fatigue[done_mask] = _snapshot["fatigue"]
            self.h.illness[done_mask] = _snapshot["illness"]
            self.h.experience[done_mask] = _snapshot["experience"]
            self.h.age[done_mask] = _snapshot["age"]
            self.h.last_action[done_mask] = _snapshot["last_action"]
            self.h.last_routed_firm_id[done_mask] = _snapshot["last_routed_firm_id"]
            self.h.last_action_success[done_mask] = _snapshot["last_action_success"]
            self.h.last_reward[done_mask] = _snapshot["last_reward"]
            self.h.last_wealth_change[done_mask] = _snapshot["last_wealth_change"]
            self.h.last_health_change[done_mask] = _snapshot["last_health_change"]
            self.h.last_crime_success[done_mask] = _snapshot["last_crime_success"]
            self.h.displaced[done_mask] = _snapshot["displaced"]
            self.h.tainted_wealth[done_mask] = _snapshot["tainted_wealth"]
        shop_matched = step_result["shop_matched"]
        clinic_matched = step_result["clinic_matched"]
        crime_success = step_result["crime_success"]
        steal_amount = step_result["steal_amount"]
        hunger_crisis = step_result["hunger_crisis"]
        work_matched = step_result["work_matched"]
        # Wage conservation (Option B): firms will be charged the GROSS effective
        # wage (net_wage_by_firm) in the production phase; the tax wedge funds
        # the treasury, which pays welfare. Apply to active worlds only (done
        # worlds are frozen, same as rewards/metrics).
        net_wage_by_firm = step_result["net_wage_by_firm"]
        tainted_old_kn = step_result["tainted_old"]   # A3 pre-step taint baseline
        _active = ~self.done
        self.treasury[_active] += (
            step_result["tax_collected_k"][_active]
            - step_result["welfare_paid_k"][_active]
        )
        # Welfare-valve instrumentation: stash this week's treasury IN/OUT
        # (pure diagnostic; surfaced in metrics extras + CSV so the retrain
        # verdict can verify treasury bounded/mean-reverting directly).
        self._wk_tax_collected[_active] = step_result["tax_collected_k"][_active]
        self._wk_welfare_paid[_active] = step_result["welfare_paid_k"][_active]
        self._wk_dividend_paid[_active] = 0.0   # set in the TCD block (phase 11b)
        # C1 ledger sites D (wealth drag -> destroyed) and F (INVEST returns:
        # gains minted, losses destroyed). Active worlds only - done worlds are
        # frozen, exactly like treasury/rewards/metrics above.
        self._destroyed[_active] += step_result["drag_destroyed_k"][_active]
        self._minted[_active] += step_result["invest_minted_k"][_active]
        self._destroyed[_active] += step_result["invest_destroyed_k"][_active]
        # SHOP/CLINIC stock-rationed demand -> firm-entry unmet signal (money-
        # neutral; preserves the pre-fix entry behavior that used to ride on
        # the now-removed over-charge bug). Active worlds only.
        self.weekly_unmet_demand[_active] += step_result["shop_clinic_unmet_k"][_active]

        # Write per-world last_routed_firm_id (ID == slot in our scheme).
        # routed_firm_slots[k, n] is the slot; ID is the same.
        for k in range(self.K):
            self.h.last_routed_firm_id[k] = routed_firm_slots[k]

        # ---- 6) Clinic capacity + place-based transmission + illness (per-world). ----
        for k in range(self.K):
            if self.done[k]:
                continue
            view = _WorldView(self.h, self.f, k)
            healthcare_access = float(self.policy_state[k].get("healthcare_access", 0.3))
            capacity = weekly_capacity(healthcare_access)
            clinic_visitor_idxs = np.where(clinic_matched[k])[0].astype(np.int64)
            illness_post_k = self.h.illness[k].copy()
            treated_global = np.zeros(self.num_households, dtype=bool)
            if clinic_visitor_idxs.size > 0:
                sev = np.zeros(clinic_visitor_idxs.size, dtype=np.float64)
                for j, idx in enumerate(clinic_visitor_idxs):
                    state = int(illness_post_k[idx])
                    sev[j] = (
                        3.0 if state == ILLNESS_SEVERE else
                        1.0 if state == ILLNESS_RECOVERING else 1.0
                        if state != ILLNESS_HEALTHY else 0.0
                    )
                treated_mask, used = allocate_treatment(
                    clinic_visitor_idxs, capacity, sev, self.rngs[k],
                )
                treated_global[clinic_visitor_idxs[treated_mask]] = True
                denied = int((~treated_mask).sum())
                if denied > 0:
                    for idx in clinic_visitor_idxs[~treated_mask]:
                        self.event_ledger[k].record(
                            CLINIC_TREATMENT_DENIED, week=self.week, actor_id=int(idx),
                        )
                    if used >= capacity:
                        self.event_ledger[k].record(
                            CLINIC_OVERCROWDED, week=self.week,
                            used=used, capacity=capacity, denied=denied,
                        )
            # Place-based transmission.
            wave_boost = epidemic_wave_boost(self.disaster_state[k])
            # routed firm IDs for this world (-1 if no firm).
            routed_ids_k = np.where(
                routed_firm_slots[k] >= 0,
                self.f.id[k, np.maximum(routed_firm_slots[k], 0)],
                -1,
            ).astype(np.int32)
            newly_exposed, place_stats = transmit_at_places(
                actions_arr[k], routed_ids_k, illness_post_k,
                self.h.hunger[k], self.h.fatigue[k], self.h.health[k],
                self.rngs[k], healthcare_access,
                capacity, int(treated_global.sum()),
                infection_boost=wave_boost,
            )
            rested_k = (actions_arr[k] == ACTION_REST)
            illness_new, illness_health_loss = step_illness_transitions(
                illness_post_k, self.h.health[k], self.rngs[k],
                rested_k, treated_global, newly_exposed,
            )
            if np.any(illness_health_loss > 0):
                self.h.health[k] = np.maximum(0.0, self.h.health[k] - illness_health_loss)
            self.h.illness[k] = illness_new.astype(np.int32)
            self._weekly_clinic_capacity[k] = int(capacity)
            self._weekly_clinic_used[k] = int(treated_global.sum())
            self._weekly_place_stats[k] = place_stats

        # ---- 7) Per-world event recording + displacement penalty + counters. ----
        # Displaced flags were consumed at step 4's start (cleared + events
        # recorded). The DISPLACEMENT_PENALTY is applied below per world by
        # reconstructing the displaced set from HOUSEHOLD_DISPLACED ledger
        # entries for this week.

        self.household_rewards = [dict() for _ in range(self.K)]
        for k in range(self.K):
            if self.done[k]:
                continue
            for i in range(self.num_households):
                hid = i
                reward = float(self.h.last_reward[k, i])
                # Displacement penalty is NOT applied here. The displaced flags
                # were consumed (cleared + event-recorded) at step 4's start,
                # before the vectorized household step. The penalty is applied
                # below by reconstructing the displaced set from the ledger.
                pass
            # Recompute displaced set from ledger to apply penalty.
            displaced_ids = {
                ev.actor_id for ev in self.event_ledger[k].events
                if ev.type == HOUSEHOLD_DISPLACED and ev.week == self.week
            }
            for hid in displaced_ids:
                self.h.last_reward[k, hid] = self.h.last_reward[k, hid] + DISPLACEMENT_PENALTY

            # Now collect rewards + record per-step events.
            for i in range(self.num_households):
                hid = i
                reward = float(self.h.last_reward[k, i])
                self.household_rewards[k][hid] = reward
                if shop_matched[k, i]:
                    self.weekly_shop_matched[k] += 1
                if clinic_matched[k, i]:
                    self.weekly_clinic_matched[k] += 1
                if crime_success[k, i]:
                    self.weekly_steal_success[k] += 1
                if hunger_crisis[k, i]:
                    self.weekly_hunger_crisis[k] += 1
                    self.event_ledger[k].record(
                        HOUSEHOLD_HUNGER_CRISIS, week=self.week, actor_id=hid,
                        hunger=float(self.h.hunger[k, i]),
                    )
                    if hid not in self._hunger_crisis_emitted[k]:
                        self._hunger_crisis_emitted[k].add(hid)
                        self.visual_events[k].append({
                            "type": "household_hunger_crisis",
                            "week": self.week,
                            "household_id": hid,
                            "hunger": float(self.h.hunger[k, i]),
                        })
                if float(self.h.fatigue[k, i]) >= FATIGUE_HEALTH_PENALTY_THRESHOLD:
                    self.event_ledger[k].record(
                        HOUSEHOLD_FATIGUE_CRISIS, week=self.week, actor_id=hid,
                        fatigue=float(self.h.fatigue[k, i]),
                    )
                    if hid not in self._fatigue_crisis_emitted[k]:
                        self._fatigue_crisis_emitted[k].add(hid)
                        self.visual_events[k].append({
                            "type": "household_fatigue_crisis",
                            "week": self.week,
                            "household_id": hid,
                            "fatigue": float(self.h.fatigue[k, i]),
                        })

                prev_illness = int(illness_pre_step[k, i])
                new_illness = int(self.h.illness[k, i])
                if prev_illness == ILLNESS_HEALTHY and new_illness != ILLNESS_HEALTHY:
                    self.event_ledger[k].record(
                        HOUSEHOLD_INFECTED, week=self.week, actor_id=hid,
                        state=new_illness,
                    )
                if new_illness == ILLNESS_SEVERE and prev_illness != ILLNESS_SEVERE:
                    self.event_ledger[k].record(
                        HOUSEHOLD_SEVERE_ILLNESS, week=self.week, actor_id=hid,
                    )
                    if hid not in self._severe_illness_emitted[k]:
                        self._severe_illness_emitted[k].add(hid)
                        self.visual_events[k].append({
                            "type": "household_severe_illness",
                            "week": self.week,
                            "household_id": hid,
                        })
                if new_illness == ILLNESS_HEALTHY and prev_illness != ILLNESS_HEALTHY:
                    self.event_ledger[k].record(
                        HOUSEHOLD_RECOVERED, week=self.week, actor_id=hid,
                    )
                if float(self.h.health[k, i]) <= DEATH_THRESHOLD:
                    self.weekly_deaths[k] += 1
                    self.event_ledger[k].record(
                        HOUSEHOLD_DEATH, week=self.week, actor_id=hid,
                        hunger=float(self.h.hunger[k, i]),
                        wealth=float(self.h.wealth[k, i]),
                    )
                    if hid not in self._death_emitted[k]:
                        self._death_emitted[k].add(hid)
                        self.visual_events[k].append({
                            "type": "household_death",
                            "week": self.week,
                            "household_id": hid,
                            "hunger": float(self.h.hunger[k, i]),
                            "wealth": float(self.h.wealth[k, i]),
                        })

        # ---- 8) Firm-side sell effects (per-world). ----
        for k in range(self.K):
            if self.done[k]:
                continue
            view = _WorldView(self.h, self.f, k)
            for i in range(self.num_households):
                slot = int(routed_firm_slots[k, i])
                if slot < 0 or not bool(self.f.alive[k, slot]):
                    continue
                a = int(actions_arr[k, i])
                if a == ACTION_SHOP and bool(shop_matched[k, i]):
                    ok = firm_sell_slot(
                        view.f, slot,
                        unit_price=float(self.f.price[k, slot]) * SHOP_COST_FRACTION,
                    )
                    if not ok:
                        self.weekly_unmet_demand[k] += 1
                elif a == ACTION_CLINIC and bool(clinic_matched[k, i]):
                    ok = firm_sell_slot(
                        view.f, slot,
                        unit_price=float(self.f.price[k, slot]) * CLINIC_COST_MULTIPLIER,
                    )
                    if not ok:
                        self.weekly_unmet_demand[k] += 1

        # ---- 9) Theft (per-world, rng draws in slot order). ----
        for k in range(self.K):
            if self.done[k]:
                continue
            view = _WorldView(self.h, self.f, k)
            alive_slots = view.f.alive_slots()
            ep_progress = float(episode_progress_k[k])
            for i in range(self.num_households):
                if int(actions_arr[k, i]) != ACTION_STEAL or not bool(crime_success[k, i]):
                    continue
                steal_amt = float(steal_amount[k, i])
                target_slot = route_steal_target(view.f, alive_slots, self.rngs[k])
                net_steal_gain = 0.0
                if target_slot >= 0:
                    victim_cash = float(self.f.cash[k, target_slot])
                    actual_loss = min(steal_amt, max(0.0, victim_cash))
                    self.f.cash[k, target_slot] = victim_cash - actual_loss
                    net_steal_gain = actual_loss   # what the thief actually kept
                    if actual_loss < steal_amt:
                        excess = steal_amt - actual_loss
                        _w_pre_claw = float(self.h.wealth[k, i])
                        self.h.wealth[k, i] = max(0.0, _w_pre_claw - excess)
                        # C1 H: the household step provisionally minted the full
                        # steal_amt into wealth; the firm->hh transfer (actual_loss)
                        # + this excess clawback exactly cancel that mint -
                        # EXCEPT the clawback clamps at 0. Any un-removed
                        # remainder is money the thief retained that conservation
                        # says should be gone -> MINT.
                        self._minted[k] += max(0.0, excess - _w_pre_claw)
                else:
                    # No firm to steal from: the steal_amt added in the household
                    # step is fully removed -> net gain 0.
                    _w_pre_claw = float(self.h.wealth[k, i])
                    self.h.wealth[k, i] = max(0.0, _w_pre_claw - steal_amt)
                    # C1 H: same clamp-shortfall mint as the target case (here
                    # the full steal_amt is the clawback, no firm transfer).
                    self._minted[k] += max(0.0, steal_amt - _w_pre_claw)
                # A3: the kept steal becomes TAINTED wealth - spendable in-world
                # but permanently excluded from the reward's wealth channel
                # (cap = min(tainted, wealth) clears as it's spent; legit
                # earnings above the taint reward normally). Recompute the
                # thief's reward on EFFECTIVE (untainted) wealth, telescoped
                # against the same pre-step effective baseline the per-step
                # reward used (wealth_before - tainted_old). Then a flat
                # STEAL_REWARD_PENALTY prices the ACT itself - A3 cannot reach
                # the consumption/health channel (stolen food -> fed/health
                # reward), so this complementary penalty covers it.
                new_w = float(self.h.wealth[k, i])
                # A3 redesign: taint = pre-step taint + the kept steal. NO
                # wealth-level cap (the old min(.,new_w) was the laundering
                # ratchet + crash-recover spike). Taint only ever decreases via
                # genuine consumption outflow (handled in the household step).
                self.h.tainted_wealth[k, i] = (
                    float(tainted_old_kn[k, i]) + net_steal_gain
                )
                eff_w_new = max(0.0, new_w - float(self.h.tainted_wealth[k, i]))
                eff_w_old = max(
                    0.0, float(wealth_before[k, i]) - float(tainted_old_kn[k, i])
                )
                adjusted_reward = compute_household_reward(
                    eff_w_old, float(health_before[k, i]),
                    eff_w_new, float(self.h.health[k, i]),
                    float(self.h.hunger[k, i]),
                    ep_progress,
                    tainted_new=float(self.h.tainted_wealth[k, i]),
                ) - STEAL_REWARD_PENALTY
                self.household_rewards[k][i] = adjusted_reward
                self.h.last_reward[k, i] = adjusted_reward
                self.h.last_wealth_change[k, i] = float(self.h.wealth[k, i]) - float(wealth_before[k, i])

        # ---- 10) Firm WAGE + overhead + interest (per-world). ----
        # FIX A: production STOCK moved to phase 4b (before the household
        # ration). This phase now pays the Option-B GROSS wage (computed in
        # the phase-5 household step -> net_wage_by_firm), then overhead, then
        # Option-C interest. Downtime firms still produce nothing and pay NO
        # wage (only overhead+interest) - IDENTICAL to the pre-split behavior
        # (workers are never routed to a downtime firm, so net_wage_by_firm is
        # 0 there anyway). No capacity/cap_mult here (it only affected
        # production, now in 4b). window_wage_bill accrues here;
        # window_hires accrued in 4b - both before phase 11, totals unchanged.
        for k in range(self.K):
            if self.done[k]:
                continue
            view = _WorldView(self.h, self.f, k)
            rate_k = float(self.interest_rate[k])
            for slot in view.f.alive_slots().tolist():
                if int(self.f.disaster_downtime[k, slot]) > 0:
                    # C1 E: overhead is paid to NO ONE -> DESTROYED. Exact: the
                    # firm stays alive, so the full overhead leaves the
                    # alive-firm cash sum (no clamp; cash may go negative).
                    self._destroyed[k] += float(firm_pay_overhead_slot(view.f, slot))
                    # Option C: interest on negative cash -> treasury (conserved).
                    _c = float(self.f.cash[k, slot])
                    if _c < 0.0:
                        _int = rate_k * (-_c) * FIRM_BORROW_RATE_SCALE
                        self.f.cash[k, slot] = _c - _int
                        self.treasury[k] += _int
                    continue
                _wb = float(net_wage_by_firm[k, slot])
                # Pay the Option-B gross wage (stock already produced in 4b).
                firm_pay_wage_slot(view.f, slot, wage_bill_override=_wb)
                # C1 E: overhead paid to NO ONE -> DESTROYED (see downtime branch).
                self._destroyed[k] += float(firm_pay_overhead_slot(view.f, slot))
                # Option C: a firm running negative cash pays interest at the CB
                # rate into the treasury. Conserved (firm -interest, treasury
                # +interest). Makes the CB rate causally move firms -> inflation/
                # GDP/unemployment, so the CB macro reward is meaningful.
                _c = float(self.f.cash[k, slot])
                if _c < 0.0:
                    _int = rate_k * (-_c) * FIRM_BORROW_RATE_SCALE
                    self.f.cash[k, slot] = _c - _int
                    self.treasury[k] += _int

        # ---- 11) Monthly firm rewards (per-world). ----
        if is_firm_update:
            for k in range(self.K):
                if self.done[k]:
                    continue
                view = _WorldView(self.h, self.f, k)
                self.firm_rewards[k] = dict()
                ss = FIRM_DEFAULT_SMOOTHNESS_SCALE
                if self.curriculum_phase == 4:
                    ss = FIRM_JOINT_FINE_TUNING_SMOOTHNESS_SCALE
                for slot in view.f.alive_slots().tolist():
                    firm_update_capacity_slot(view.f, slot)
                    fid = int(self.f.id[k, slot])
                    self.firm_rewards[k][fid] = compute_firm_reward_slot(
                        view.f, slot, ss,
                    )
                    firm_reset_window_slot(view.f, slot)

        # ---- 11b) Treasury Citizen Dividend (welfare-valve recycle). ----
        # Treasury is a multi-inflow / single-poverty-gated-outflow sink; in a
        # healthy economy poverty->0 => welfare->0 => it hoards money unrecyclably
        # (3-Opus + 5-Opus reviewed; v2-adjudicated). Recycle to ALIVE
        # households a fixed per-hh floor D + kappa*(surplus above CEILING),
        # affordability-clamped, never draining below RESERVE. Steady state:
        # treasury* = CEILING + (tau - N*D)/kappa - finite & distribution-INDEPENDENT.
        # Pure treasury<->hh transfer: the EXACT float sum added to wealth is
        # subtracted from treasury => money_residual unchanged ~=1e-14; touches
        # NO C1 mint/destroy site; deterministic (no RNG), per-world; done
        # worlds frozen. Placed AFTER theft(9)/wage(10)/firm-rewards(11) and
        # BEFORE metrics(12) so it cannot perturb the C1-H steal-clamp or A3
        # taint (R1). Dead excluded => no die-and-farm (R2). D is the
        # tau-independent behavioral knob (kappa does NOT set the payout - R2/R3).
        for k in range(self.K):
            if self.done[k]:
                continue
            tre_k = float(self.treasury[k])
            avail = tre_k - TREASURY_DIVIDEND_RESERVE
            if avail <= 0.0:
                self._wk_dividend_paid[k] = 0.0
                continue
            alive = self.h.health[k] > DEATH_THRESHOLD
            n_alive = int(np.count_nonzero(alive))
            if n_alive == 0:
                self._wk_dividend_paid[k] = 0.0
                continue
            surplus = max(0.0, tre_k - TREASURY_DIVIDEND_CEILING)
            payout = (n_alive * TREASURY_DIVIDEND_FLOOR
                      + TREASURY_DIVIDEND_KAPPA * surplus)
            payout = min(payout, avail)   # keep treasury >= RESERVE
            if payout <= 0.0:
                self._wk_dividend_paid[k] = 0.0
                continue
            d = payout / n_alive
            inc = np.where(alive, d, 0.0)        # exact array added to wealth
            paid = float(inc.sum())              # exact sum injected
            self.h.wealth[k] += inc
            self.treasury[k] -= paid             # bit-exact: debit == credit sum
            self._wk_dividend_paid[k] = paid

        # ---- 12) Metrics update (per-world). ----
        week_metrics: list[dict] = [None] * self.K  # type: ignore
        for k in range(self.K):
            if self.done[k]:
                continue
            view = _WorldView(self.h, self.f, k)
            firm_diag = firm_solvency_diagnostics(view.f)
            hunger_diag = hunger_summary(self.h.hunger[k])
            fatigue_diag = fatigue_summary(self.h.fatigue[k])
            illness_diag = illness_summary(self.h.illness[k])
            total_inventory = firm_diag.pop("total_inventory")
            downtime_count = int(np.sum(self.f.disaster_downtime[k, : view.f.size] > 0))

            # C1: total system money + per-step conservation residual.
            _tm = float(
                np.sum(self.h.wealth[k])
                + np.sum(self.f.cash[k][self.f.alive[k]])
                + self.treasury[k]
            )
            _prev_tm = float(self._prev_total_money[k])
            if np.isnan(_prev_tm):
                _residual = 0.0   # first step post-reset: no prior to diff
            else:
                _residual = (_tm - _prev_tm) - (
                    float(self._minted[k]) - float(self._destroyed[k])
                )
            self._prev_total_money[k] = _tm

            extras = {
                "mean_firm_cash": firm_diag["mean_firm_cash"],
                "min_firm_cash": firm_diag["min_firm_cash"],
                "max_firm_cash": firm_diag["max_firm_cash"],
                "frac_insolvent_firms": firm_diag["frac_insolvent_firms"],
                "total_inventory": total_inventory,
                # C1 money-conservation: total_money should change each step by
                # exactly (minted - destroyed); money_residual ~= 0 certifies the
                # ledger is complete. A persistent nonzero residual reveals a
                # missed/mis-signed site (positive => missed destroy; negative =>
                # missed mint; ~2x a channel => that channel's sign flipped).
                "total_money": _tm,
                "money_minted": float(self._minted[k]),
                "money_destroyed": float(self._destroyed[k]),
                "money_residual": float(_residual),
                # Welfare-valve instrumentation (pure diagnostic): verify
                # treasury bounded/mean-reverting + flows directly post-retrain.
                "treasury": float(self.treasury[k]),
                "tax_collected_wk": float(self._wk_tax_collected[k]),
                "welfare_paid_wk": float(self._wk_welfare_paid[k]),
                "dividend_paid_wk": float(self._wk_dividend_paid[k]),
                "mean_hunger": hunger_diag["mean_hunger"],
                "max_hunger": hunger_diag["max_hunger"],
                "frac_hungry": hunger_diag["frac_hungry"],
                "frac_crisis": hunger_diag["frac_crisis"],
                "mean_fatigue": fatigue_diag["mean_fatigue"],
                "max_fatigue": fatigue_diag["max_fatigue"],
                "frac_fatigued": fatigue_diag["frac_fatigued"],
                "frac_exhausted": fatigue_diag["frac_exhausted"],
                "frac_healthy": illness_diag["frac_healthy"],
                "frac_exposed": illness_diag["frac_exposed"],
                "frac_mild": illness_diag["frac_mild"],
                "frac_severe": illness_diag["frac_severe"],
                "frac_recovering": illness_diag["frac_recovering"],
                "clinic_capacity": int(self._weekly_clinic_capacity[k]),
                "clinic_used": int(self._weekly_clinic_used[k]),
                "disaster_signal": disaster_signal(self.disaster_state[k]),
                "supply_shock_active": int(self.disaster_state[k].get("supply_shock_remaining", 0) > 0),
                "epidemic_wave_active": int(self.disaster_state[k].get("epidemic_wave_remaining", 0) > 0),
                "firms_in_downtime": downtime_count,
                "shop_attempts": int(self.weekly_shop_attempts[k]),
                "shop_matched": int(self.weekly_shop_matched[k]),
                "clinic_attempts": int(self.weekly_clinic_attempts[k]),
                "clinic_matched": int(self.weekly_clinic_matched[k]),
                "steal_success": int(self.weekly_steal_success[k]),
                "work_refused_solvency": int(self.weekly_work_refused_solvency[k]),
                "weekly_unmet_demand": int(self.weekly_unmet_demand[k]),
                "weekly_total_demand": int(self.weekly_total_demand[k]),
                "weekly_deaths": int(self.weekly_deaths[k]),
                "weekly_displacements": int(self.weekly_displacements[k]),
                "weekly_hunger_crisis": int(self.weekly_hunger_crisis[k]),
            }
            extras.update(self.event_ledger[k].to_csv_row(self.week))
            week_metrics[k] = self.metrics[k].update(
                view.h, view.f, self.week,
                int(self.weekly_work_attempts[k]), int(self.weekly_work_matched[k]),
                int(self.weekly_crimes[k]),
                steal_success=int(self.weekly_steal_success[k]),
                extra=extras,
            )

            self.unmet_demand_history[k].append({
                "unmet": int(self.weekly_unmet_demand[k]),
                "demand": int(self.weekly_total_demand[k]) + int(self.weekly_work_attempts[k]),
            })

        # ---- 13) Gov / CB rewards (per-world). ----
        for k in range(self.K):
            if self.done[k]:
                continue
            if is_gov_update:
                self.gov_reward[k] = self._compute_gov_reward(k)
            if is_cb_update:
                self.cb_reward[k] = self._compute_cb_reward(k)

        # ---- 14) Termination (per-world). ----
        self.week += 1
        for k in range(self.K):
            if self.done[k]:
                continue
            self._check_termination(k)

        # ---- 14b) Always-on invariant asserts ----
        # Mitigates Cost 2 (silent vectorized bugs). Each check is a single
        # numpy comparison over (K, N) or (K, F_max) - total cost ~tens of usec
        # per step. Catches NaN / out-of-range values the moment a bug
        # introduces them, instead of letting them propagate to a downstream
        # symptom.
        self._check_invariants()

        # ---- 15) Pack return value. ----
        is_warmup = self.week <= self.warmup_weeks
        # NOTE: we do NOT zero out rewards for worlds that terminated DURING
        # this step. Acting agents must receive their final reward (RLlib
        # MultiAgentEnv contract). Worlds that were already done BEFORE this
        # step have empty `household_rewards[k]` dicts (the per-world loop
        # above skipped them), so they emit no rewards naturally.
        rewards = {
            "households": [self.household_rewards[k] if not is_warmup else {} for k in range(self.K)],
            "firms": [self.firm_rewards[k] if ((is_firm_update or self.done[k]) and not is_warmup) else {} for k in range(self.K)],
            # `or self.done[k]`: same escape as firms - on a collapse week the
            # terminal early-term penalty must be EMITTED even if it is not a
            # gov/cb-update week, else it is masked to 0.0 here and never
            # reaches the wrapper accumulator (gov/cb would learn collapse is
            # free - the B1 bug, second drop point below the wrapper).
            "gov": [float(self.gov_reward[k]) if ((is_gov_update or self.done[k]) and not is_warmup) else 0.0 for k in range(self.K)],
            "cb": [float(self.cb_reward[k]) if ((is_cb_update or self.done[k]) and not is_warmup) else 0.0 for k in range(self.K)],
        }

        info = {
            "week": self.week,
            "metrics": week_metrics,
            "is_firm_update": is_firm_update,
            "is_gov_update": is_gov_update,
            "is_cb_update": is_cb_update,
            "is_warmup": is_warmup,
            "terminated_early": self.terminated_early.copy(),
            "termination_reason": list(self.termination_reason),
        }
        obs = self._get_all_observations()
        return obs, rewards, self.done.copy(), info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalize_action_list(self, actions, default=None, dict_default=None, allow_scalar_dict=True):
        """Convert action input to a length-K list."""
        if actions is None:
            return [default] * self.K
        if allow_scalar_dict and isinstance(actions, dict):
            return [dict(actions) for _ in range(self.K)]
        if isinstance(actions, list):
            assert len(actions) == self.K, f"expected list of length K={self.K}"
            return actions
        # Otherwise assume scalar that applies to every world.
        return [actions] * self.K

    def _apply_gov_action(self, k: int, action: tuple):
        from marl_economy.config.constants import (
            GOV_TAX_DELTAS, GOV_WELFARE_DELTAS, GOV_POLICING_DELTAS,
            GOV_HEALTHCARE_DELTAS, GOV_EDUCATION_DELTAS, GOV_MINWAGE_DELTAS,
        )
        delta_maps = [
            ("tax_rate", GOV_TAX_DELTAS),
            ("welfare", GOV_WELFARE_DELTAS),
            ("policing", GOV_POLICING_DELTAS),
            ("healthcare_access", GOV_HEALTHCARE_DELTAS),
            ("education_quality", GOV_EDUCATION_DELTAS),
            ("min_wage_floor", GOV_MINWAGE_DELTAS),
        ]
        self.prev_policy_state[k] = deepcopy(self.policy_state[k])
        for i, (key, deltas) in enumerate(delta_maps):
            if i < len(action):
                idx = action[i]
                idx = max(0, min(idx, len(deltas) - 1))
                self.policy_state[k][key] = clamp_to_regime(
                    key, self.policy_state[k][key] + deltas[idx], self.regime_name
                )

    def _apply_cb_action(self, k: int, action_idx: int):
        from marl_economy.config.constants import CB_RATE_DELTAS
        idx = max(0, min(action_idx, len(CB_RATE_DELTAS) - 1))
        self.prev_interest_rate[k] = self.interest_rate[k]
        rate_candidate = np.clip(
            self.interest_rate[k] + CB_RATE_DELTAS[idx], CB_RATE_MIN, CB_RATE_MAX
        )
        self.interest_rate[k] = clamp_to_regime(
            "interest_rate", rate_candidate, self.regime_name
        )

    def _compute_gov_reward(self, k: int) -> float:
        summary = self.metrics[k].get_quarterly_summary(window_weeks=self.gov_update_period)
        gdp_term = GOV_A * np.log(1 + summary["gdp_pc"])
        gini_term = GOV_B * summary["gini"]
        poverty_term = GOV_E * summary["poverty"]
        crime_term = GOV_D * summary["crime_rate"]
        health_term = GOV_C * summary["avg_health"]
        unemp_term = GOV_U * summary["unemployment"]

        delta_sq = 0.0
        for key in ["tax_rate", "welfare", "policing", "healthcare_access",
                    "education_quality", "min_wage_floor"]:
            d = self.policy_state[k].get(key, 0) - self.prev_policy_state[k].get(key, 0)
            delta_sq += d ** 2
        ss = MACRO_DEFAULT_SMOOTHNESS_SCALE
        if self.curriculum_phase == 4:
            ss = FIRM_JOINT_FINE_TUNING_SMOOTHNESS_SCALE
        smooth_term = GOV_ZETA * ss * delta_sq
        gdp_pc = float(summary["gdp_pc"])
        # Phase 3 (#4): RELATIVE collapse penalty vs a decaying peak.
        # Harsh-review fix: do NOT update the peak or apply the collapse penalty
        # until AFTER warmup. During warmup the economy is still spinning up
        # (gdp_pc~=0); with peak seeded at 0 the old code forced severity~=1 via
        # the absolute floor and hammered the very first gov reward with ~-8.
        if self.week > self.warmup_weeks:
            self.gdp_pc_peak[k] = max(self.gdp_pc_peak[k] * GDP_PC_PEAK_DECAY, gdp_pc)
            # Only penalize once a real post-warmup peak exists (peak>floor),
            # so a never-yet-grown economy isn't flagged as "collapsed".
            if self.gdp_pc_peak[k] > GDP_COLLAPSE_FLOOR:
                ref = GDP_COLLAPSE_RELATIVE_FRAC * float(self.gdp_pc_peak[k])
                collapse_severity = float(np.clip((ref - gdp_pc) / max(EPSILON, ref), 0.0, 1.0))
            else:
                collapse_severity = 0.0
        else:
            collapse_severity = 0.0
        collapse_penalty = GDP_COLLAPSE_PENALTY * collapse_severity
        # Continuous health-floor penalty: a sick economy is punished every gov
        # decision, not only at the (formerly inert) collapse floor.
        health_floor_pen = GOV_HEALTH_FLOOR_PEN * max(
            0.0, GOV_HEALTH_FLOOR - float(summary["avg_health"])
        )
        return float(gdp_term - gini_term - poverty_term - crime_term
                     + health_term - unemp_term - smooth_term
                     - collapse_penalty - health_floor_pen)

    def _compute_cb_reward(self, k: int) -> float:
        summary = self.metrics[k].get_quarterly_summary(window_weeks=self.cb_update_period)
        inflation_dev = (summary["inflation"] - CB_INFLATION_TARGET) ** 2
        gdp_growth_term = CB_LAMBDA * summary["gdp_growth"]
        dr = self.interest_rate[k] - self.prev_interest_rate[k]
        smooth_term = CB_RHO * dr ** 2
        unemp_term = CB_NU * summary["unemployment"]
        return float(-inflation_dev + gdp_growth_term - smooth_term - unemp_term)

    def _apply_household_collapse_penalty(self, k: int, remaining: int) -> None:
        """Phase 3 (#C symmetric): on early collapse/mass-death, every household
        in world k takes a terminal penalty (gov/cb already do) - closes the
        'drive collapse to end the episode early' exploit. Applied to this
        terminal step's reward, which is the last one RLlib sees for them.
        """
        hh_pen = HOUSEHOLD_EARLY_TERM_SCALE * (remaining / self.episode_weeks)
        if self.h.N > 0:
            self.h.last_reward[k, :] += hh_pen
        for _hid in self.household_rewards[k]:
            self.household_rewards[k][_hid] += hh_pen

    def _apply_firm_collapse_penalty(self, k: int, remaining: int) -> None:
        """B1: symmetric terminal collapse penalty for FIRMS (was deferred).
        Adds FIRM_EARLY_TERM_SCALE*(remaining/episode) to every ALIVE firm's
        reward for the terminal step. On a non-firm-update week firm_rewards[k]
        is empty; we populate it here, and the reward-dict assembly delivers
        firm rewards for a DONE world regardless of the firm-update cadence
        (see the `is_firm_update or self.done[k]` gate). Closes the
        firm-driven-collapse half of the early-termination exploit.
        """
        # Assign a CLEAN penalty-only dict (do NOT += onto firm_rewards[k]):
        # between firm-update weeks firm_rewards[k] holds STALE values already
        # delivered at the last firm-tick; += would double-pay them on a
        # non-firm-update collapse week (harsh-review S6). On a collapse the
        # firm's terminal signal is the collapse penalty; forfeiting a partial
        # in-progress month is acceptable and guarantees no stale double-pay.
        f_pen = FIRM_EARLY_TERM_SCALE * (remaining / self.episode_weeks)
        alive_slots = np.nonzero(self.f.alive[k])[0]
        self.firm_rewards[k] = {
            int(self.f.id[k, _s]): f_pen for _s in alive_slots.tolist()
        }

    def _check_invariants(self) -> None:
        """Cheap always-on invariant asserts. Mitigates Cost 2 from the
        better-design refactor: vectorized bugs are silent (wrong numbers,
        no crash) without these checks. Each check is one numpy comparison
        across (K, N) or (K, F_max) - total cost a small fraction of step().

        Skips checks for done worlds (their state is intentionally frozen at
        terminal values and may be at threshold boundaries).
        """
        active = ~self.done  # (K,) bool
        if not active.any():
            return
        h = self.h
        f = self.f
        # Household state ranges.
        assert np.all(h.wealth[active] >= 0.0), \
            f"BatchedWorld invariant: negative wealth at week {self.week}"
        assert np.all((0.0 <= h.health[active]) & (h.health[active] <= 1.0)), \
            f"BatchedWorld invariant: health out of [0,1] at week {self.week}"
        assert np.all((0.0 <= h.hunger[active]) & (h.hunger[active] <= 1.0)), \
            f"BatchedWorld invariant: hunger out of [0,1] at week {self.week}"
        assert np.all((0.0 <= h.fatigue[active]) & (h.fatigue[active] <= 1.0)), \
            f"BatchedWorld invariant: fatigue out of [0,1] at week {self.week}"
        assert np.all((0 <= h.illness[active]) & (h.illness[active] < NUM_ILLNESS_STATES)), \
            f"BatchedWorld invariant: illness out of range at week {self.week}"
        # No NaN anywhere in active worlds.
        assert not np.any(np.isnan(h.wealth[active])), \
            f"BatchedWorld invariant: NaN wealth at week {self.week}"
        assert not np.any(np.isnan(h.health[active])), \
            f"BatchedWorld invariant: NaN health at week {self.week}"
        # Reward finiteness: the shaping terms (level/fed, firm activity) have no
        # other invariant coverage; a NaN/inf reward would otherwise be silent.
        assert np.all(np.isfinite(h.last_reward[active])), \
            f"BatchedWorld invariant: non-finite household reward at week {self.week}"
        # Firm cash finite where alive (cash can go negative under credit; only
        # NaN/inf is a bug).
        alive_mask_active = f.alive & active[:, None]
        if np.any(alive_mask_active):
            assert np.all(np.isfinite(f.cash[alive_mask_active])), \
                f"BatchedWorld invariant: non-finite firm cash at week {self.week}"
        # Firm count per active world stays in [F_MIN, F_MAX].
        num_alive_per_world = f.alive.sum(axis=1)
        active_num_alive = num_alive_per_world[active]
        assert np.all((F_MIN <= active_num_alive) & (active_num_alive <= F_MAX)), \
            (f"BatchedWorld invariant: firm count out of [F_MIN={F_MIN}, "
             f"F_MAX={F_MAX}] at week {self.week}; got {active_num_alive.tolist()}")

    def _check_termination(self, k: int):
        if self.week >= self.episode_weeks:
            self.done[k] = True
            # Graceful episode-end is NOT a collapse: no terminal penalty, and
            # clear any STALE firm_rewards[k] (last firm-tick values, already
            # delivered) so the `or self.done[k]` firms gate can't re-emit them
            # (harsh-review S6).
            self.firm_rewards[k] = {}
            return
        if self.metrics[k].check_mass_death(_WorldView(self.h, self.f, k).h):
            self.done[k] = True
            self.terminated_early[k] = True
            self.termination_reason[k] = "mass_death"
            remaining = self.episode_weeks - self.week
            penalty = EARLY_TERM_PENALTY_SCALE * (remaining / self.episode_weeks)
            # CLEAN assignment (NOT +=), mirroring _apply_firm_collapse_penalty:
            # between gov/cb-update weeks self.gov_reward[k]/cb_reward[k] hold
            # STALE last-tick values already accumulated+delivered; += would
            # re-emit them stacked with the penalty on a non-gov/cb-update
            # collapse week (double-pay). The terminal signal IS the collapse
            # penalty; forfeiting the partial in-progress period is acceptable
            # and consistent with the firm treatment. (All 3 collapse branches
            # use `= penalty` for the same reason.)
            self.gov_reward[k] = penalty
            self.cb_reward[k] = penalty
            self._apply_household_collapse_penalty(k, remaining)
            self._apply_firm_collapse_penalty(k, remaining)
            return

        # Collapse detector.
        view = _WorldView(self.h, self.f, k)
        alive_slots = view.f.alive_slots()
        alive_firm_views = [
            {"cash": float(self.f.cash[k, s])} for s in alive_slots.tolist()
        ]
        latest_metrics = self.metrics[k].history[-1] if self.metrics[k].history else {}
        weekly_gdp = float(latest_metrics.get("gdp", 0.0))
        mean_health_k = float(np.mean(self.h.health[k])) if self.h.N > 0 else 1.0
        if self.collapse_detector[k].update(
            self.week, weekly_gdp, alive_firm_views, self.event_ledger[k],
            mean_health=mean_health_k,
        ):
            self.done[k] = True
            self.terminated_early[k] = True
            reason = self.collapse_detector[k].trigger_reason or "economic_collapse"
            self.termination_reason[k] = (
                "economic_collapse" if reason == "no_firms"
                else f"economic_collapse:{reason}"
            )
            remaining = self.episode_weeks - self.week
            penalty = EARLY_TERM_PENALTY_SCALE * (remaining / self.episode_weeks)
            self.gov_reward[k] = penalty
            self.cb_reward[k] = penalty
            self._apply_household_collapse_penalty(k, remaining)
            self._apply_firm_collapse_penalty(k, remaining)
            return
        if self.metrics[k].check_hyperinflation():
            self.done[k] = True
            self.terminated_early[k] = True
            self.termination_reason[k] = "hyperinflation"
            remaining = self.episode_weeks - self.week
            penalty = EARLY_TERM_PENALTY_SCALE * (remaining / self.episode_weeks)
            self.gov_reward[k] = penalty
            self.cb_reward[k] = penalty
            self._apply_household_collapse_penalty(k, remaining)
            self._apply_firm_collapse_penalty(k, remaining)
            return
        if self.metrics[k].history:
            latest = self.metrics[k].history[-1]
            for kk, vv in latest.items():
                if isinstance(vv, float) and np.isnan(vv):
                    self.done[k] = True
                    self.terminated_early[k] = True
                    self.termination_reason[k] = f"nan_detected:{kk}"
                    return

    def _get_all_observations(self) -> dict:
        """Return per-world observations as a list of dicts."""
        obs_list = []
        for k in range(self.K):
            view = _WorldView(self.h, self.f, k)
            obs_list.append(self._single_world_obs(k, view))
        return {"per_world": obs_list}

    def _single_world_obs(self, k: int, view: _WorldView) -> dict:
        from marl_economy.config.constants import (
            OBS_FALLBACK_POLICING, OBS_FALLBACK_TAX_RATE,
            OBS_FALLBACK_HEALTHCARE_ACCESS, OBS_FALLBACK_EDUCATION_QUALITY,
            AGE_NORMALIZER, HOUSEHOLD_ACTION_NORMALIZER,
        )
        from marl_economy.env.entry_exit import recompute_market_averages
        from marl_economy.env.dynamics_firm import firm_get_observation_slot
        alive_slots = view.f.alive_slots()
        mean_price, mean_wage = recompute_market_averages(view.f)
        obs = {"households": {}, "firms": {}, "gov": None, "cb": None}
        policing = self.policy_state[k].get("policing", OBS_FALLBACK_POLICING)
        tax_rate = self.policy_state[k].get("tax_rate", OBS_FALLBACK_TAX_RATE)
        healthcare = self.policy_state[k].get("healthcare_access", OBS_FALLBACK_HEALTHCARE_ACCESS)
        education = self.policy_state[k].get("education_quality", OBS_FALLBACK_EDUCATION_QUALITY)
        for i in range(self.num_households):
            obs["households"][i] = np.array([
                float(self.h.wealth[k, i]),
                float(self.h.health[k, i]),
                float(policing), float(tax_rate),
                float(mean_price), float(mean_wage),
                float(healthcare), float(education),
                float(self.h.age[k, i]) / AGE_NORMALIZER,
                float(self.h.last_action[k, i]) / HOUSEHOLD_ACTION_NORMALIZER,
                float(self.h.hunger[k, i]),
                float(self.h.fatigue[k, i]),
                float(self.h.illness[k, i]) / float(NUM_ILLNESS_STATES),
            ], dtype=np.float32)
        min_wage = self.policy_state[k].get("min_wage_floor", 0.0)
        for slot in alive_slots.tolist():
            fid = int(view.f.id[slot])
            obs["firms"][fid] = firm_get_observation_slot(
                view.f, slot, mean_price, mean_wage, tax_rate, min_wage,
            )
        gov_summary = self.metrics[k].get_quarterly_summary(window_weeks=self.gov_update_period)
        gov_disaster_count = float(disaster_signal(self.disaster_state[k]))
        obs["gov"] = np.array([
            gov_summary["gdp_pc"], gov_summary["gini"], gov_summary["avg_health"],
            gov_summary["poverty"], gov_summary["crime_rate"], gov_summary["unemployment"],
            gov_summary["inflation"], tax_rate, gov_disaster_count,
        ], dtype=np.float32)
        cb_summary = self.metrics[k].get_quarterly_summary(window_weeks=self.cb_update_period)
        obs["cb"] = np.array([
            cb_summary["inflation"], cb_summary["gdp_growth"],
            cb_summary["unemployment"], self.interest_rate[k],
        ], dtype=np.float32)
        return obs
