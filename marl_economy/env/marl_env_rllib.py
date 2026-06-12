"""
marl_env_rllib.py - RLlib MultiAgentEnv wrapper for World simulation.

Correctly translates between:
  - World's nested dict interface: {"households": {id: obs}, "firms": {id: obs}, "gov": obs, "cb": obs}
  - RLlib's flat dict interface: {"h_0": obs, ..., "f_0": obs, ..., "gov": obs, "cb": obs}

Multi-timescale obs filtering (per research):
  - Households appear in obs every step (weekly)
  - Firms appear only on firm update weeks (monthly, every 4 weeks)
  - Gov/CB appear only on their update weeks (quarterly, every 12 weeks)
  - Non-acting agents are OMITTED from obs dict (not padded with dummies)
  - Rewards are accumulated between action steps and delivered as a lump sum

Compatible with Ray RLlib 2.54.0 new API stack (PettingZoo-style contract).
"""

import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, Set

from ..config.constants import (
    DEFAULT_NUM_HOUSEHOLDS, DEFAULT_NUM_FIRMS, DEFAULT_EPISODE_WEEKS, WARMUP_WEEKS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_REGIME_NAME,
    CURRICULUM,
    F_MAX, FIRM_UPDATE_PERIOD, GOV_UPDATE_PERIOD, CB_UPDATE_PERIOD, RLLIB_MAX_FIRM_AGENT_SLOTS,
    HOUSEHOLD_OBS_DIM, FIRM_OBS_DIM, GOV_OBS_DIM, CB_OBS_DIM,
    NUM_HOUSEHOLD_ACTIONS, ACTION_REST,
    FIRM_PRICE_DELTAS, FIRM_WAGE_DELTAS, FIRM_CAPACITY_DELTAS, FIRM_ACTION_DIMS,
    GOV_TAX_DELTAS, GOV_WELFARE_DELTAS, GOV_POLICING_DELTAS, GOV_ACTION_DIMS,
    GOV_HEALTHCARE_DELTAS, GOV_EDUCATION_DELTAS, GOV_MINWAGE_DELTAS,
    CB_RATE_DELTAS,
)
from ..env.world import World

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv as _BaseEnv
    HAS_RLLIB = True
except ImportError:
    import gymnasium as gym
    _BaseEnv = gym.Env
    HAS_RLLIB = False


def policy_mapping_fn(agent_id: str, episode=None, worker=None, **kwargs) -> str:
    """Map agent IDs to shared policies (Section 11)."""
    if agent_id.startswith("h_"):
        return "pi_household"
    elif agent_id.startswith("f_"):
        return "pi_firm"
    elif agent_id == "gov":
        return "pi_gov"
    elif agent_id == "cb":
        return "pi_cb"
    else:
        raise ValueError(f"Unknown agent_id: {agent_id}")


class MARLEconomyEnv(_BaseEnv):
    """
    Multi-agent RLlib wrapper around the World simulation engine.

    Action spaces (MultiDiscrete for factored decisions per research):
      - Household: Discrete(6) - Work/Shop/Steal/Rest/Invest/Clinic
      - Firm: MultiDiscrete([5,5,3]) - price_delta x wage_delta x capacity_delta
      - Government: MultiDiscrete([5,5,5,5,5,3]) - 6 policy dimensions (28 logits vs 9375)
      - Central Bank: Discrete(5) - interest rate delta

    Multi-timescale handling:
      - Only agents about to act appear in the returned obs dict
      - Rewards are accumulated and delivered when the agent next acts
    """

    def __init__(self, config: dict = None):
        config = config or {}

        self.regime_name = config.get("regime_name", config.get("regime", DEFAULT_REGIME_NAME))
        self.num_households = config.get("num_households", DEFAULT_NUM_HOUSEHOLDS)
        self.num_firms = config.get("num_firms", DEFAULT_NUM_FIRMS)
        self.episode_weeks = config.get("episode_weeks", config.get("episode_length", DEFAULT_EPISODE_WEEKS))
        self.warmup_weeks = config.get("warmup_weeks", WARMUP_WEEKS)
        self.seed = config.get("seed", DEFAULT_RANDOM_SEED)
        self.curriculum_phase = config.get("curriculum_phase", 1)
        self.compressed_quarterly = config.get("compressed_quarterly", False)
        phase_config = CURRICULUM.get(self.curriculum_phase, CURRICULUM[1])
        self.firms_fixed = bool(config.get("firms_fixed", phase_config.get("firms_fixed", False)))
        self.gov_scripted = bool(config.get("gov_scripted", phase_config.get("gov_scripted", False)))
        self.cb_scripted = bool(config.get("cb_scripted", phase_config.get("cb_scripted", False)))
        self.households_learning = bool(config.get("households_learning", phase_config.get("households_learning", True)))
        self.firms_use_rl = not self.firms_fixed
        self.gov_use_rl = not self.gov_scripted
        self.cb_use_rl = not self.cb_scripted

        self.world = World(
            regime_name=self.regime_name,
            num_households=self.num_households,
            num_firms=self.num_firms,
            episode_weeks=self.episode_weeks,
            warmup_weeks=self.warmup_weeks,
            seed=self.seed,
            curriculum_phase=self.curriculum_phase,
            compressed_quarterly=self.compressed_quarterly,
        )

        # Define action / observation spaces.
        hh_obs = spaces.Box(low=-np.inf, high=np.inf, shape=(HOUSEHOLD_OBS_DIM,), dtype=np.float32)
        firm_obs = spaces.Box(low=-np.inf, high=np.inf, shape=(FIRM_OBS_DIM,), dtype=np.float32)
        gov_obs = spaces.Box(low=-np.inf, high=np.inf, shape=(GOV_OBS_DIM,), dtype=np.float32)
        cb_obs = spaces.Box(low=-np.inf, high=np.inf, shape=(CB_OBS_DIM,), dtype=np.float32)

        hh_act = spaces.Discrete(NUM_HOUSEHOLD_ACTIONS)
        firm_act = spaces.MultiDiscrete(FIRM_ACTION_DIMS)
        gov_act = spaces.MultiDiscrete(GOV_ACTION_DIMS)
        cb_act = spaces.Discrete(len(CB_RATE_DELTAS))

        # Per-agent-id space dicts. Pre-allocate firm slots so dynamic firm
        # entry can keep allocating ids without bumping the dict.
        self.observation_spaces = {}
        self.action_spaces = {}

        for i in range(self.num_households):
            self.observation_spaces[f"h_{i}"] = hh_obs
            self.action_spaces[f"h_{i}"] = hh_act

        # Pre-allocate firm slots - needs to be large because next_firm_id
        # increments on every entry (IDs are never reused within an episode).
        # Over a 156-week episode with frequent entry/exit, IDs can reach 50+.
        for fid in range(RLLIB_MAX_FIRM_AGENT_SLOTS):
            self.observation_spaces[f"f_{fid}"] = firm_obs
            self.action_spaces[f"f_{fid}"] = firm_act

        self.observation_spaces["gov"] = gov_obs
        self.action_spaces["gov"] = gov_act
        self.observation_spaces["cb"] = cb_obs
        self.action_spaces["cb"] = cb_act

        # PettingZoo-style agent lists (required by Ray 2.54.0)
        self.possible_agents = list(self.observation_spaces.keys())
        self.agents = []

        # gymnasium.Env requirements
        self.observation_space = spaces.Dict(self.observation_spaces)
        self.action_space = spaces.Dict(self.action_spaces)

        # Slow agents accumulate reward between their own action ticks and
        # have it delivered as a lump sum when they next appear in obs.
        self._firm_reward_accum = {}
        self._gov_reward_accum = 0.0
        self._cb_reward_accum = 0.0

        # Call super AFTER setting attributes
        super().__init__()

    def _flatten_obs(self, world_obs: dict, include_firms: bool,
                     include_gov: bool, include_cb: bool) -> Dict[str, np.ndarray]:
        """Convert World's nested obs to flat dict, filtering by timescale."""
        flat = {}
        for hid, obs in world_obs["households"].items():
            flat[f"h_{hid}"] = obs
        if include_firms and self.firms_use_rl:
            for fid, obs in world_obs["firms"].items():
                flat[f"f_{fid}"] = obs
        if include_gov and self.gov_use_rl and world_obs["gov"] is not None:
            flat["gov"] = world_obs["gov"]
        if include_cb and self.cb_use_rl and world_obs["cb"] is not None:
            flat["cb"] = world_obs["cb"]
        return flat

    def _unflatten_actions(self, action_dict: dict) -> tuple:
        """Convert RLlib's flat action_dict to World's expected formats."""
        household_actions = {}
        for i in range(self.num_households):
            key = f"h_{i}"
            if key in action_dict:
                household_actions[i] = int(action_dict[key])
            else:
                household_actions[i] = ACTION_REST

        firm_actions = {}
        if self.firms_use_rl:
            for key, action in action_dict.items():
                if key.startswith("f_"):
                    fid = int(key[2:])
                    if hasattr(action, '__len__'):
                        firm_actions[fid] = tuple(int(a) for a in action)
                    else:
                        firm_actions[fid] = self._decode_firm_action(int(action))

        gov_action = None
        if self.gov_use_rl and "gov" in action_dict:
            action = action_dict["gov"]
            if hasattr(action, '__len__'):
                gov_action = tuple(int(a) for a in action)
            else:
                gov_action = self._decode_gov_action(int(action))

        cb_action = None
        if self.cb_use_rl and "cb" in action_dict:
            cb_action = int(action_dict["cb"])

        return household_actions, firm_actions or None, gov_action, cb_action

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

    def reset(self, *, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset world and return initial observations for all agents."""
        if seed is not None:
            self.world.seed = seed
            self.world.rng = np.random.default_rng(seed)

        world_obs = self.world.reset()

        self._firm_reward_accum = {}
        self._gov_reward_accum = 0.0
        self._cb_reward_accum = 0.0

        # On reset, include the agents unlocked for this curriculum stage.
        # Policy modules are still declared in the RLlib config, but locked
        # agents should not act or contribute random rewards.
        flat_obs = self._flatten_obs(world_obs,
                                     include_firms=True,
                                     include_gov=True,
                                     include_cb=True)
        self.agents = list(flat_obs.keys())
        return flat_obs, {aid: {} for aid in flat_obs}

    def step(self, action_dict: Dict) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Execute one weekly step with multi-timescale obs filtering.

        Returns obs only for agents that should act on the NEXT step.
        Accumulated rewards are delivered when agents next appear in obs.
        """
        hh_actions, firm_actions, gov_action, cb_action = self._unflatten_actions(action_dict)

        world_obs, world_rewards, done, info = self.world.step(
            hh_actions, firm_actions, gov_action, cb_action
        )

        is_warmup = info.get("is_warmup", False)

        # Accumulate rewards.
        # Household rewards delivered immediately
        flat_rewards = {}
        for hid, rew in world_rewards.get("households", {}).items():
            flat_rewards[f"h_{hid}"] = rew if not is_warmup else 0.0

        # Firm/Gov/CB rewards are emitted on their own decision cadence
        # (monthly/quarterly summaries). Curriculum-locked agents are not
        # exposed to RLlib, so their rewards are intentionally not reported.
        if self.firms_use_rl:
            for fid, rew in world_rewards.get("firms", {}).items():
                self._firm_reward_accum[fid] = self._firm_reward_accum.get(fid, 0.0) + (rew if not is_warmup else 0.0)

        if self.gov_use_rl:
            self._gov_reward_accum += (world_rewards.get("gov", 0.0) if not is_warmup else 0.0)
        if self.cb_use_rl:
            self._cb_reward_accum += (world_rewards.get("cb", 0.0) if not is_warmup else 0.0)

        # Decide which slow agents will appear in next step's observation.
        # world.week was incremented inside world.step(), so it equals the NEXT week number
        next_week = self.world.week
        next_is_firm = (next_week > 0 and next_week % FIRM_UPDATE_PERIOD == 0)
        next_is_gov = (next_week > 0 and next_week % self.world.gov_update_period == 0)
        next_is_cb = (next_week > 0 and next_week % self.world.cb_update_period == 0)

        # On done, include everyone for final obs
        if done:
            next_is_firm = self.firms_use_rl
            next_is_gov = self.gov_use_rl
            next_is_cb = self.cb_use_rl

        next_is_firm = next_is_firm and self.firms_use_rl
        next_is_gov = next_is_gov and self.gov_use_rl
        next_is_cb = next_is_cb and self.cb_use_rl

        # Deliver accumulated rewards to slow agents that are about to act.
        if next_is_firm:
            for fid in list(self._firm_reward_accum.keys()):
                flat_rewards[f"f_{fid}"] = self._firm_reward_accum.pop(fid, 0.0)

        if next_is_gov:
            flat_rewards["gov"] = self._gov_reward_accum
            self._gov_reward_accum = 0.0

        if next_is_cb:
            flat_rewards["cb"] = self._cb_reward_accum
            self._cb_reward_accum = 0.0

        # Build the filtered obs dict.
        flat_obs = self._flatten_obs(world_obs,
                                     include_firms=next_is_firm,
                                     include_gov=next_is_gov,
                                     include_cb=next_is_cb)

        # Enforce obs <-> reward parity.
        for agent_id in flat_obs:
            if agent_id not in flat_rewards:
                flat_rewards[agent_id] = 0.0

        # Only return rewards for agents in obs
        final_rewards = {aid: flat_rewards.get(aid, 0.0) for aid in flat_obs}

        self.agents = list(flat_obs.keys())

        terminateds = {aid: done for aid in flat_obs}
        terminateds["__all__"] = done
        truncateds = {aid: False for aid in flat_obs}
        truncateds["__all__"] = False
        infos = {aid: {} for aid in flat_obs}

        return flat_obs, final_rewards, terminateds, truncateds, infos

    def random_actions(self) -> Dict:
        """Generate random actions for all currently active agents."""
        rng = self.world.rng
        actions = {}
        for aid in self.agents:
            if aid.startswith("h_"):
                actions[aid] = int(rng.integers(0, NUM_HOUSEHOLD_ACTIONS))
            elif aid.startswith("f_"):
                actions[aid] = np.array([
                    rng.integers(0, FIRM_ACTION_DIMS[0]),
                    rng.integers(0, FIRM_ACTION_DIMS[1]),
                    rng.integers(0, FIRM_ACTION_DIMS[2]),
                ], dtype=np.int64)
            elif aid == "gov":
                actions[aid] = np.array([
                    rng.integers(0, GOV_ACTION_DIMS[0]),
                    rng.integers(0, GOV_ACTION_DIMS[1]),
                    rng.integers(0, GOV_ACTION_DIMS[2]),
                    rng.integers(0, GOV_ACTION_DIMS[3]),
                    rng.integers(0, GOV_ACTION_DIMS[4]),
                    rng.integers(0, GOV_ACTION_DIMS[5]),
                ], dtype=np.int64)
            elif aid == "cb":
                actions[aid] = int(rng.integers(0, len(CB_RATE_DELTAS)))
        return actions
