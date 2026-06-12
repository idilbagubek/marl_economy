"""Batched MultiAgentEnv backed by `BatchedWorld(K)`.

Created 2026-05-14 as part of the "wire RLlib" Stage 3 task. Exposes K
parallel worlds as a SINGLE RLlib `MultiAgentEnv` whose agent IDs encode
the world index:

  households: ``h_w{k}_{i}``     (e.g. ``h_w0_5``, ``h_w7_42``)
  firms:      ``f_w{k}_{fid}``   (e.g. ``f_w3_8``)
  government: ``gov_w{k}``
  central bank: ``cb_w{k}``

`policy_mapping_fn_batched` strips the ``_w{k}_`` infix and maps the agent
type prefix to its policy (e.g. all ``h_*`` agents share `pi_household`),
so the K worlds train shared policies in parallel.

Multi-timescale filtering: households appear in obs every step (weekly);
firms appear only on `week % FIRM_UPDATE_PERIOD == 0`; gov/cb on their own
update cadences. All K worlds tick together (same `week` across worlds),
so a tick test against `self.bw.week` applies uniformly.

Done semantics: the env episode ends when **all** K worlds are done. Worlds
that terminated early stop contributing obs and accumulated rewards from
that point. A future enhancement could reset done worlds individually; for
now we keep RLlib's standard "one env, one episode boundary" contract.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, Set, Optional

from ..config.constants import (
    DEFAULT_NUM_HOUSEHOLDS, DEFAULT_NUM_FIRMS, DEFAULT_EPISODE_WEEKS, WARMUP_WEEKS,
    DEFAULT_RANDOM_SEED, DEFAULT_REGIME_NAME,
    CURRICULUM,
    F_MAX, FIRM_UPDATE_PERIOD, GOV_UPDATE_PERIOD, CB_UPDATE_PERIOD,
    RLLIB_MAX_FIRM_AGENT_SLOTS,
    HOUSEHOLD_OBS_DIM, FIRM_OBS_DIM, GOV_OBS_DIM, CB_OBS_DIM,
    NUM_HOUSEHOLD_ACTIONS, ACTION_REST,
    FIRM_PRICE_DELTAS, FIRM_WAGE_DELTAS, FIRM_CAPACITY_DELTAS, FIRM_ACTION_DIMS,
    GOV_TAX_DELTAS, GOV_WELFARE_DELTAS, GOV_POLICING_DELTAS, GOV_ACTION_DIMS,
    GOV_HEALTHCARE_DELTAS, GOV_EDUCATION_DELTAS, GOV_MINWAGE_DELTAS,
    CB_RATE_DELTAS,
)
from ..env.batched_world import BatchedWorld

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv as _BaseEnv
    HAS_RLLIB = True
except ImportError:
    import gymnasium as gym
    _BaseEnv = gym.Env
    HAS_RLLIB = False


def policy_mapping_fn_batched(agent_id: str, episode=None, worker=None, **kwargs) -> str:
    """Map world-prefixed agent IDs to shared policies."""
    if agent_id.startswith("h_"):
        return "pi_household"
    elif agent_id.startswith("f_"):
        return "pi_firm"
    elif agent_id.startswith("gov_"):
        return "pi_gov"
    elif agent_id.startswith("cb_"):
        return "pi_cb"
    else:
        raise ValueError(f"Unknown agent_id: {agent_id}")


def _parse_agent_id(agent_id: str) -> Tuple[str, int, Optional[int]]:
    """Decompose a world-prefixed agent id into (kind, world_idx, instance_id).

    Returns (kind, k, inst) where:
      * kind: "h", "f", "gov", or "cb"
      * k: world index (int)
      * inst: household/firm id (int), or None for gov/cb

    Examples:
      "h_w0_5"   -> ("h", 0, 5)
      "f_w3_12"  -> ("f", 3, 12)
      "gov_w7"   -> ("gov", 7, None)
      "cb_w2"    -> ("cb", 2, None)
    """
    if agent_id.startswith("h_w"):
        rest = agent_id[3:]  # e.g. "0_5"
        k_str, i_str = rest.split("_", 1)
        return ("h", int(k_str), int(i_str))
    if agent_id.startswith("f_w"):
        rest = agent_id[3:]
        k_str, i_str = rest.split("_", 1)
        return ("f", int(k_str), int(i_str))
    if agent_id.startswith("gov_w"):
        return ("gov", int(agent_id[5:]), None)
    if agent_id.startswith("cb_w"):
        return ("cb", int(agent_id[4:]), None)
    raise ValueError(f"Unknown agent id format: {agent_id}")


class BatchedMARLEconomyEnv(_BaseEnv):
    """RLlib MultiAgentEnv that wraps a single `BatchedWorld(K)` and exposes
    K worlds as one env with world-prefixed agent IDs.

    Use via:
        env = BatchedMARLEconomyEnv(config={"batched_k": 16, "regime": "LME", ...})

    Compared to running K independent `MARLEconomyEnv` instances:
      * One Python object (shared BatchedWorld) instead of K
      * Vectorized numpy in `step_households_batch_kn`
      * One disaster / clinic / illness pass per BatchedWorld step (still
        per-world internally, but no inter-process overhead)
    """

    def __init__(self, config: dict = None):
        config = config or {}

        self.K = int(config.get("batched_k", 1))
        self.regime_name = config.get("regime_name", config.get("regime", DEFAULT_REGIME_NAME))
        self.num_households = int(config.get("num_households", DEFAULT_NUM_HOUSEHOLDS))
        self.num_firms = int(config.get("num_firms", DEFAULT_NUM_FIRMS))
        self.episode_weeks = int(config.get("episode_weeks", config.get("episode_length", DEFAULT_EPISODE_WEEKS)))
        self.warmup_weeks = int(config.get("warmup_weeks", WARMUP_WEEKS))
        self.seed = int(config.get("seed", DEFAULT_RANDOM_SEED))
        self.curriculum_phase = int(config.get("curriculum_phase", 1))
        self.compressed_quarterly = bool(config.get("compressed_quarterly", False))

        phase_config = CURRICULUM.get(self.curriculum_phase, CURRICULUM[1])
        self.firms_fixed = bool(config.get("firms_fixed", phase_config.get("firms_fixed", False)))
        self.gov_scripted = bool(config.get("gov_scripted", phase_config.get("gov_scripted", False)))
        self.cb_scripted = bool(config.get("cb_scripted", phase_config.get("cb_scripted", False)))
        self.households_learning = bool(config.get("households_learning", phase_config.get("households_learning", True)))
        self.firms_use_rl = not self.firms_fixed
        self.gov_use_rl = not self.gov_scripted
        self.cb_use_rl = not self.cb_scripted

        # Underlying BatchedWorld.
        self.bw = BatchedWorld(
            regime_name=self.regime_name, K=self.K,
            num_households=self.num_households,
            num_firms=self.num_firms,
            episode_weeks=self.episode_weeks,
            warmup_weeks=self.warmup_weeks,
            seed=self.seed,
            curriculum_phase=self.curriculum_phase,
            compressed_quarterly=self.compressed_quarterly,
        )

        # Spaces: prefixed agent IDs across K worlds.
        hh_obs = spaces.Box(low=-np.inf, high=np.inf, shape=(HOUSEHOLD_OBS_DIM,), dtype=np.float32)
        firm_obs = spaces.Box(low=-np.inf, high=np.inf, shape=(FIRM_OBS_DIM,), dtype=np.float32)
        gov_obs = spaces.Box(low=-np.inf, high=np.inf, shape=(GOV_OBS_DIM,), dtype=np.float32)
        cb_obs = spaces.Box(low=-np.inf, high=np.inf, shape=(CB_OBS_DIM,), dtype=np.float32)

        hh_act = spaces.Discrete(NUM_HOUSEHOLD_ACTIONS)
        firm_act = spaces.MultiDiscrete(FIRM_ACTION_DIMS)
        gov_act = spaces.MultiDiscrete(GOV_ACTION_DIMS)
        cb_act = spaces.Discrete(len(CB_RATE_DELTAS))

        self.observation_spaces = {}
        self.action_spaces = {}
        for k in range(self.K):
            for i in range(self.num_households):
                self.observation_spaces[f"h_w{k}_{i}"] = hh_obs
                self.action_spaces[f"h_w{k}_{i}"] = hh_act
            for fid in range(RLLIB_MAX_FIRM_AGENT_SLOTS):
                self.observation_spaces[f"f_w{k}_{fid}"] = firm_obs
                self.action_spaces[f"f_w{k}_{fid}"] = firm_act
            self.observation_spaces[f"gov_w{k}"] = gov_obs
            self.action_spaces[f"gov_w{k}"] = gov_act
            self.observation_spaces[f"cb_w{k}"] = cb_obs
            self.action_spaces[f"cb_w{k}"] = cb_act

        self.possible_agents = list(self.observation_spaces.keys())
        self.agents = []
        self.observation_space = spaces.Dict(self.observation_spaces)
        self.action_space = spaces.Dict(self.action_spaces)

        # Per-world reward accumulators (slow agents).
        self._firm_reward_accum: list[dict] = [{} for _ in range(self.K)]
        self._gov_reward_accum: list[float] = [0.0] * self.K
        self._cb_reward_accum: list[float] = [0.0] * self.K
        # B1 open-episode set: slow agents (firm/gov/cb) that have received
        # >=1 obs and are NOT yet terminated - i.e., the episodes RLlib has
        # OPEN for them. A slow agent ticks every FIRM_UPDATE_PERIOD (firm) /
        # gov-cb period, so `_prev_obs_agents` (last step only) misses it on
        # most collapse weeks; the terminal collapse/early-term penalty must
        # be delivered to any OPEN slow agent of a done world, not just one
        # that happened to get an obs on the immediately preceding step.
        self._open_slow_agents: set[str] = set()

        super().__init__()

    # ------------------------------------------------------------------
    # Reset / step
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            # Re-seed the underlying BatchedWorld.
            self.bw.seed = int(seed)
            seq = np.random.SeedSequence(int(seed))
            sub_seqs = seq.spawn(self.K)
            self.bw.rngs = [np.random.default_rng(s) for s in sub_seqs]

        obs_dict = self.bw.reset()

        self._firm_reward_accum = [{} for _ in range(self.K)]
        self._gov_reward_accum = [0.0] * self.K
        self._cb_reward_accum = [0.0] * self.K
        # Track agents that received obs at end of last step - i.e., the
        # agents that will act in the NEXT step. RLlib's MultiAgentEnv
        # contract requires every acting agent to receive a reward AND a
        # terminated flag at the end of the step they acted in.
        self._prev_obs_agents: set[str] = set()
        self._open_slow_agents = set()
        # Track which worlds were done at start of last step so we know
        # when a world transitions alive -> done.
        self._prev_done: np.ndarray = np.zeros(self.K, dtype=bool)

        flat_obs = self._flatten_obs(
            obs_dict["per_world"],
            include_firms=True, include_gov=True, include_cb=True,
            done_mask=np.zeros(self.K, dtype=bool),
        )
        self.agents = list(flat_obs.keys())
        self._prev_obs_agents = set(flat_obs.keys())
        # Register slow agents (firm/gov/cb) with an OPEN episode. Households
        # ("h_w...") are fast (every step) and excluded - they never need the
        # terminal-flush path.
        self._open_slow_agents |= {
            a for a in flat_obs if a.startswith(("f_w", "gov_w", "cb_w"))
        }
        return flat_obs, {a: {} for a in flat_obs}

    def step(self, action_dict: Dict):
        # Group actions by world.
        household_actions_list, firm_actions_list, gov_action_list, cb_action_list = \
            self._unflatten_actions(action_dict)

        obs_dict, rewards_dict, done_arr, info = self.bw.step(
            household_actions_list,
            firm_actions=firm_actions_list,
            gov_action=gov_action_list,
            cb_action=cb_action_list,
        )

        is_warmup = info.get("is_warmup", False)

        # Rewards: households deliver every step, slow agents accumulate.
        # NOTE: we no longer skip done worlds - for a world that just
        # transitioned alive -> done, its acting agents must receive a final
        # reward (RLlib contract). Worlds that were already done at start of
        # step have empty rewards_dict entries naturally.
        flat_rewards: Dict[str, float] = {}
        for k in range(self.K):
            for hid, rew in rewards_dict["households"][k].items():
                flat_rewards[f"h_w{k}_{hid}"] = float(rew if not is_warmup else 0.0)
            if self.firms_use_rl:
                for fid, rew in rewards_dict["firms"][k].items():
                    self._firm_reward_accum[k][fid] = (
                        self._firm_reward_accum[k].get(fid, 0.0)
                        + float(rew if not is_warmup else 0.0)
                    )
            if self.gov_use_rl:
                self._gov_reward_accum[k] += float(
                    rewards_dict["gov"][k] if not is_warmup else 0.0
                )
            if self.cb_use_rl:
                self._cb_reward_accum[k] += float(
                    rewards_dict["cb"][k] if not is_warmup else 0.0
                )

        # Next-step slow-agent tick test (all worlds tick together).
        next_week = info["week"]
        next_is_firm = (next_week > 0 and next_week % FIRM_UPDATE_PERIOD == 0)
        next_is_gov = (next_week > 0 and next_week % self.bw.gov_update_period == 0)
        next_is_cb = (next_week > 0 and next_week % self.bw.cb_update_period == 0)

        all_done = bool(np.all(done_arr))
        if all_done:
            next_is_firm = self.firms_use_rl
            next_is_gov = self.gov_use_rl
            next_is_cb = self.cb_use_rl
        next_is_firm = next_is_firm and self.firms_use_rl
        next_is_gov = next_is_gov and self.gov_use_rl
        next_is_cb = next_is_cb and self.cb_use_rl

        # Next-step obs (only for non-done worlds).
        flat_obs = self._flatten_obs(
            obs_dict["per_world"],
            include_firms=next_is_firm, include_gov=next_is_gov, include_cb=next_is_cb,
            done_mask=done_arr,
        )

        # Deliver accumulated slow-agent rewards when those agents next appear.
        if next_is_firm:
            for k in range(self.K):
                if done_arr[k]:
                    continue
                world_obs = obs_dict["per_world"][k]
                for fid in world_obs["firms"].keys():
                    if fid in self._firm_reward_accum[k]:
                        flat_rewards[f"f_w{k}_{fid}"] = self._firm_reward_accum[k].pop(fid)
        if next_is_gov:
            for k in range(self.K):
                if done_arr[k]:
                    continue
                flat_rewards[f"gov_w{k}"] = self._gov_reward_accum[k]
                self._gov_reward_accum[k] = 0.0
        if next_is_cb:
            for k in range(self.K):
                if done_arr[k]:
                    continue
                flat_rewards[f"cb_w{k}"] = self._cb_reward_accum[k]
                self._cb_reward_accum[k] = 0.0

        # B1 + correctness fix: the next-tick slow-agent delivery above does
        # `if done_arr[k]: continue`, so a just-terminated world's ACCUMULATED
        # firm/gov/cb rewards - INCLUDING the terminal collapse / early-term
        # penalties (B1 firm, and the pre-existing gov/cb EARLY_TERM_PENALTY,
        # which had the SAME silent-drop bug) - were never delivered. Flush
        # them now to every slow agent of the done world with an OPEN episode.
        # The earlier `_prev_obs_agents` (last-step-only) gate dropped the
        # penalty on ~75%+ of collapse weeks (slow agents tick every
        # FIRM_UPDATE_PERIOD / gov-cb period, rarely the exact pre-collapse
        # step); `_open_slow_agents` (any un-terminated slow agent that has had
        # >=1 obs) closes that gap. Contract-safe: RLlib still has these
        # episodes open; final_rewards/terminated re-add below closes them
        # with terminated=True. (Needs a training-time smoke to confirm the
        # RLlib reward+terminated-without-current-obs path - eval cannot.)
        # B1 open-episode delivery: gate on `_open_slow_agents` (slow agents
        # with an OPEN RLlib episode - got >=1 obs, not yet terminated), NOT
        # `_prev_obs_agents` (only the immediately-preceding step). Slow agents
        # tick every FIRM_UPDATE_PERIOD (firm) / gov-cb period, so the old gate
        # dropped the terminal collapse/early-term penalty on ~75%+ of collapse
        # weeks. At the collapse step the accumulator is non-empty (the
        # collapse/early-term penalty is added that very step), so iterating it
        # delivers the penalty; the agent is then terminated and its episode
        # closed (removed from the open set) so it is never delivered twice.
        # Iterate OPEN firm agents of the done world (not just non-empty accum
        # keys): on a GRACEFUL episode-end the firm accum can be empty (no
        # collapse penalty written), but the open episode MUST still be closed
        # with terminated=True - else, under staggered per-world termination,
        # that firm's RLlib episode hangs. Deliver pop(fid, 0.0) so an
        # empty-accum firm gets a clean 0.0 final reward + terminated.
        _terminated_slow: set[str] = set()
        for k in range(self.K):
            if not done_arr[k]:
                continue
            if self.firms_use_rl:
                _fprefix = f"f_w{k}_"
                for aid in [a for a in self._open_slow_agents
                            if a.startswith(_fprefix)]:
                    fid = int(aid[len(_fprefix):])
                    flat_rewards[aid] = self._firm_reward_accum[k].pop(fid, 0.0)
                    _terminated_slow.add(aid)
            if self.gov_use_rl:
                aid = f"gov_w{k}"
                if aid in self._open_slow_agents:
                    flat_rewards[aid] = self._gov_reward_accum[k]
                    self._gov_reward_accum[k] = 0.0
                    _terminated_slow.add(aid)
            if self.cb_use_rl:
                aid = f"cb_w{k}"
                if aid in self._open_slow_agents:
                    flat_rewards[aid] = self._cb_reward_accum[k]
                    self._cb_reward_accum[k] = 0.0
                    _terminated_slow.add(aid)

        # Enforce obs <-> reward parity: every agent in next obs gets a
        # reward (default 0.0). Mirrors the single-world MARLEconomyEnv
        # behavior; required during warmup weeks when BatchedWorld returns
        # empty reward dicts.
        for agent_id in flat_obs:
            if agent_id not in flat_rewards:
                flat_rewards[agent_id] = 0.0

        # Also emit final 0.0 for agents that acted last step but won't
        # appear in next obs (slow agents in worlds that terminated before
        # their next tick).
        for agent_id in self._prev_obs_agents:
            if agent_id not in flat_rewards and agent_id not in flat_obs:
                flat_rewards[agent_id] = 0.0

        # Filter rewards to only agents in obs (per single-world contract).
        # Agents that just terminated keep their reward via _prev_obs_agents
        # branch above, plus get terminated=True below.
        final_rewards = {a: flat_rewards.get(a, 0.0) for a in flat_obs}
        # Re-add terminated-but-not-in-obs agents - slow agents in just-
        # terminated worlds. INCLUDE `_terminated_slow` (open slow agents
        # flushed above whose last obs may be many ticks back, so they are NOT
        # in `_prev_obs_agents`): they need their flushed reward AND
        # terminated=True so RLlib closes the episode with the penalty.
        for agent_id in (self._prev_obs_agents | _terminated_slow):
            if agent_id not in flat_obs and agent_id in flat_rewards:
                final_rewards[agent_id] = flat_rewards[agent_id]

        # Terminated / truncated.
        flat_terminated: Dict[str, bool] = {a: all_done for a in flat_obs.keys()}
        for agent_id in (self._prev_obs_agents | _terminated_slow):
            if agent_id not in flat_obs:
                flat_terminated[agent_id] = True
        flat_terminated["__all__"] = all_done
        flat_truncated = {a: False for a in flat_obs.keys()}
        flat_truncated["__all__"] = False
        flat_infos = {a: {} for a in flat_obs.keys()}

        # Snapshot agent set for next step's parity check.
        self._prev_obs_agents = set(flat_obs.keys())
        # Maintain the open-episode set: a slow agent opens when it gets an obs
        # and closes (is removed) when terminated/flushed this step. Doing the
        # add BEFORE the discard is safe - a flushed agent (done world) is never
        # in this step's flat_obs (done worlds emit no obs), so it cannot be
        # re-added in the same step it is closed.
        self._open_slow_agents |= {
            a for a in flat_obs if a.startswith(("f_w", "gov_w", "cb_w"))
        }
        self._open_slow_agents -= _terminated_slow
        self._prev_done = done_arr.copy()
        self.agents = list(flat_obs.keys())
        return flat_obs, final_rewards, flat_terminated, flat_truncated, flat_infos

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flatten_obs(self, per_world: list, include_firms: bool, include_gov: bool,
                     include_cb: bool, done_mask: np.ndarray) -> Dict[str, np.ndarray]:
        flat = {}
        for k in range(self.K):
            if done_mask[k]:
                continue
            world_obs = per_world[k]
            for hid, obs in world_obs["households"].items():
                flat[f"h_w{k}_{hid}"] = obs
            if include_firms and self.firms_use_rl:
                for fid, obs in world_obs["firms"].items():
                    flat[f"f_w{k}_{fid}"] = obs
            if include_gov and self.gov_use_rl and world_obs["gov"] is not None:
                flat[f"gov_w{k}"] = world_obs["gov"]
            if include_cb and self.cb_use_rl and world_obs["cb"] is not None:
                flat[f"cb_w{k}"] = world_obs["cb"]
        return flat

    def _unflatten_actions(self, action_dict: Dict) -> Tuple[list, list, list, list]:
        household_actions_list: list[dict] = [{} for _ in range(self.K)]
        firm_actions_list: list[Optional[dict]] = [None] * self.K
        gov_action_list: list[Optional[tuple]] = [None] * self.K
        cb_action_list: list[Optional[int]] = [None] * self.K

        for key, action in action_dict.items():
            kind, k, inst = _parse_agent_id(key)
            if kind == "h":
                household_actions_list[k][inst] = int(action)
            elif kind == "f":
                if not self.firms_use_rl:
                    continue
                if firm_actions_list[k] is None:
                    firm_actions_list[k] = {}
                if hasattr(action, "__len__"):
                    firm_actions_list[k][inst] = tuple(int(a) for a in action)
                else:
                    firm_actions_list[k][inst] = self._decode_firm_action(int(action))
            elif kind == "gov":
                if not self.gov_use_rl:
                    continue
                if hasattr(action, "__len__"):
                    gov_action_list[k] = tuple(int(a) for a in action)
                else:
                    gov_action_list[k] = self._decode_gov_action(int(action))
            elif kind == "cb":
                if not self.cb_use_rl:
                    continue
                cb_action_list[k] = int(action)

        # Fill in REST for any household slot not in the action dict (mirrors
        # MARLEconomyEnv's behavior of defaulting locked / silent households).
        for k in range(self.K):
            for i in range(self.num_households):
                if i not in household_actions_list[k]:
                    household_actions_list[k][i] = ACTION_REST

        return household_actions_list, firm_actions_list, gov_action_list, cb_action_list

    @staticmethod
    def _decode_firm_action(flat_action: int) -> tuple:
        n_p, n_w, n_c = FIRM_ACTION_DIMS
        cap_idx = flat_action % n_c
        remaining = flat_action // n_c
        wage_idx = remaining % n_w
        price_idx = remaining // n_w
        return (min(price_idx, n_p - 1), min(wage_idx, n_w - 1), min(cap_idx, n_c - 1))

    @staticmethod
    def _decode_gov_action(flat_action: int) -> tuple:
        dims = GOV_ACTION_DIMS
        indices = []
        remaining = flat_action
        for d in reversed(dims):
            indices.append(remaining % d)
            remaining //= d
        indices.reverse()
        return tuple(indices)

    def get_agent_ids(self) -> Set[str]:
        return set(self.agents)
