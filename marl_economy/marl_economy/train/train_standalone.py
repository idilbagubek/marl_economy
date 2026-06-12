"""
Standalone PPO training loop for MARL Economy.
This provides a simple training framework that works without Ray/RLlib,
using only numpy. For full MARL training, use the RLlib wrapper.

Supports staged training (Section 10.2):
- Stage 1: Household policy with scripted firms/gov/cb
- Stage 2: Firm policy with a bootstrapped household policy
- Later stages are handled by the RLlib trainer
"""

import sys
import os
import json
import csv
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from marl_economy.config.constants import (
    CURRICULUM,
    DEFAULT_REGIME_NAME, REGIME_NAMES,
    DEFAULT_RANDOM_SEED,
    NUM_HOUSEHOLD_ACTIONS, HOUSEHOLD_OBS_DIM, FIRM_OBS_DIM,
    GOV_OBS_DIM, CB_OBS_DIM, GAMMA, HOUSEHOLD_GAMMA, FIRM_GAMMA, ACTION_REST,
    FIRM_PRICE_DELTAS, FIRM_WAGE_DELTAS, FIRM_CAPACITY_DELTAS, FIRM_ACTION_DIMS,
    GOV_TAX_DELTAS, GOV_WELFARE_DELTAS, GOV_POLICING_DELTAS,
    GOV_HEALTHCARE_DELTAS, GOV_EDUCATION_DELTAS, GOV_MINWAGE_DELTAS,
    CB_RATE_DELTAS, FIRM_UPDATE_PERIOD, GOV_UPDATE_PERIOD, FIRM_NEUTRAL_ACTION,
    TRAIN_STAGE1_LOG_INTERVAL_EPISODES, TRAIN_STAGE2_LOG_INTERVAL_EPISODES,
)
from marl_economy.env.world import World


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically safe categorical probabilities for the legacy trainer."""
    logits = np.asarray(logits, dtype=np.float64)
    if not np.all(np.isfinite(logits)):
        logits = np.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)

    shifted = logits - np.max(logits)
    exp_vals = np.exp(shifted)
    total = np.sum(exp_vals)
    if not np.isfinite(total) or total <= 0.0:
        return np.full_like(logits, 1.0 / len(logits), dtype=np.float64)

    probs = exp_vals / total
    prob_sum = np.sum(probs)
    if not np.isfinite(prob_sum) or prob_sum <= 0.0:
        return np.full_like(logits, 1.0 / len(logits), dtype=np.float64)
    return probs / prob_sum


class SimplePolicy:
    """
    Simple tabular softmax policy for prototyping.
    For production, replace with neural net via RLlib.
    """
    def __init__(self, obs_dim: int, n_actions: int, lr: float = 0.01):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.lr = lr
        # Simple linear policy: logits = W @ obs + b
        self.W = np.random.randn(n_actions, obs_dim) * 0.01
        self.b = np.zeros(n_actions)

    def get_action(self, obs: np.ndarray, rng: np.random.Generator) -> tuple:
        """Sample action from policy. Returns (action, log_prob)."""
        logits = self.W @ obs + self.b
        probs = _stable_softmax(logits)
        action = rng.choice(self.n_actions, p=probs)
        log_prob = np.log(probs[action] + 1e-10)
        return int(action), float(log_prob)

    def get_probs(self, obs: np.ndarray) -> np.ndarray:
        logits = self.W @ obs + self.b
        return _stable_softmax(logits)

    def update(self, trajectories: list, gamma: float = GAMMA):
        """
        Simple REINFORCE update.
        trajectories: list of (obs, action, reward, log_prob)
        """
        if not trajectories:
            return 0.0

        # Compute returns
        returns = []
        G = 0.0
        for _, _, r, _ in reversed(trajectories):
            G = r + gamma * G
            returns.insert(0, G)

        returns = np.array(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient
        total_loss = 0.0
        for (obs, action, _, log_prob), G in zip(trajectories, returns):
            probs = self.get_probs(obs)
            # Gradient of log pi(a|s) w.r.t. W
            one_hot = np.zeros(self.n_actions)
            one_hot[action] = 1.0
            grad = np.outer(one_hot - probs, obs)  # (n_actions, obs_dim)

            self.W += self.lr * G * grad
            self.b += self.lr * G * (one_hot - probs)
            total_loss += -log_prob * G

        return total_loss / len(trajectories)


class FactorizedPolicy:
    """Linear multi-head softmax policy for MultiDiscrete action spaces."""

    def __init__(self, obs_dim: int, action_dims: tuple[int, ...], lr: float = 0.01):
        self.obs_dim = obs_dim
        self.action_dims = tuple(action_dims)
        self.lr = lr
        self.W = [np.random.randn(dim, obs_dim) * 0.01 for dim in self.action_dims]
        self.b = [np.zeros(dim) for dim in self.action_dims]

    def get_probs(self, obs: np.ndarray) -> list[np.ndarray]:
        probs = []
        for W, b in zip(self.W, self.b):
            logits = W @ obs + b
            probs.append(_stable_softmax(logits))
        return probs

    def get_action(self, obs: np.ndarray, rng: np.random.Generator) -> tuple:
        """Sample one categorical action per head. Returns ((a1, ...), log_prob)."""
        probs = self.get_probs(obs)
        action = []
        log_prob = 0.0
        for head_probs in probs:
            a = int(rng.choice(len(head_probs), p=head_probs))
            action.append(a)
            log_prob += np.log(head_probs[a] + 1e-10)
        return tuple(action), float(log_prob)

    def update(self, trajectories: list, gamma: float = GAMMA):
        """
        REINFORCE update for factorized action heads.
        trajectories: list of (obs, action_tuple, reward, log_prob)
        """
        if not trajectories:
            return 0.0

        returns = []
        G = 0.0
        for _, _, r, _ in reversed(trajectories):
            G = r + gamma * G
            returns.insert(0, G)

        returns = np.array(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        total_loss = 0.0
        for (obs, action, _, log_prob), G in zip(trajectories, returns):
            probs = self.get_probs(obs)
            for head_idx, head_probs in enumerate(probs):
                one_hot = np.zeros(self.action_dims[head_idx])
                one_hot[action[head_idx]] = 1.0
                grad = np.outer(one_hot - head_probs, obs)
                self.W[head_idx] += self.lr * G * grad
                self.b[head_idx] += self.lr * G * (one_hot - head_probs)
            total_loss += -log_prob * G

        return total_loss / len(trajectories)


class ScriptedPolicy:
    """Returns fixed/neutral actions (for scripted agents)."""
    def __init__(self, action):
        self.action = action

    def get_action(self, obs, rng):
        return self.action, 0.0


def train_households(regime: str = DEFAULT_REGIME_NAME, num_episodes: int = 500,
                     episode_weeks: int | None = None, seed: int = DEFAULT_RANDOM_SEED,
                     lr: float = 0.01, output_dir: str = None):
    """
    Stage 1: Train household policy with scripted firms/gov/cb.
    """
    print(f"{'='*60}")
    print(f"STAGE 1: Training Households ({regime})")
    print(f"Episodes: {num_episodes}, Weeks/ep: {episode_weeks}, LR: {lr}")
    print(f"{'='*60}")

    rng = np.random.default_rng(seed)
    episode_weeks = episode_weeks or CURRICULUM[1]["episode_weeks"]

    if output_dir is None:
        output_dir = f"runs/train_households_{regime}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # Household policy (shared across all households)
    policy = SimplePolicy(HOUSEHOLD_OBS_DIM, NUM_HOUSEHOLD_ACTIONS, lr=lr)

    episode_rewards = []
    episode_metrics = []

    for ep in range(num_episodes):
        world = World(
            regime_name=regime,
            episode_weeks=episode_weeks,
            seed=seed + ep,
            curriculum_phase=1,
        )
        obs = world.reset()
        trajectories = {int(world.h.ids[i]): [] for i in range(world.h.size)}
        total_reward = 0.0

        for step in range(episode_weeks):
            # Households use learned policy
            household_actions = {}
            household_logprobs = {}
            household_step_obs = {}
            for i in range(world.h.size):
                hid = int(world.h.ids[i])
                if hid in obs["households"]:
                    household_step_obs[hid] = obs["households"][hid].copy()
                    action, logp = policy.get_action(obs["households"][hid], rng)
                    household_actions[hid] = action
                    household_logprobs[hid] = logp

            # Scripted firm actions (neutral)
            firm_actions = None
            if step > 0 and step % 4 == 0:
                alive_slots = world.f.alive_slots()
                firm_actions = {int(world.f.id[slot]): FIRM_NEUTRAL_ACTION
                                for slot in alive_slots.tolist()}

            obs, rewards, done, info = world.step(
                household_actions, firm_actions, None, None
            )

            # Collect trajectories (skip warmup)
            if not info.get("is_warmup", False):
                for hid in household_actions:
                    if hid in rewards.get("households", {}):
                        r = rewards["households"][hid]
                        total_reward += r
                        if hid in household_step_obs:
                            trajectories[hid].append((
                                household_step_obs[hid],
                                household_actions[hid],
                                r,
                                household_logprobs.get(hid, 0),
                            ))

            if done:
                break

        # Update policy with all household trajectories
        all_trajs = []
        for hid_trajs in trajectories.values():
            all_trajs.extend(hid_trajs)

        loss = policy.update(all_trajs, gamma=HOUSEHOLD_GAMMA)

        avg_reward = total_reward / max(1, world.h.size)
        episode_rewards.append(avg_reward)

        # Get final metrics
        if world.metrics.history:
            last_m = world.metrics.history[-1]
            episode_metrics.append(last_m)

        if ep % TRAIN_STAGE1_LOG_INTERVAL_EPISODES == 0 or ep == num_episodes - 1:
            m = world.metrics.history[-1] if world.metrics.history else {}
            print(f"  Ep {ep:4d} | AvgReward={avg_reward:8.4f} | "
                  f"GDP={m.get('gdp', 0):.3f} Gini={m.get('gini', 0):.3f} "
                  f"Health={m.get('avg_health', 0):.3f}")

    np.save(f"{output_dir}/policy_W.npy", policy.W)
    np.save(f"{output_dir}/policy_b.npy", policy.b)
    np.save(f"{output_dir}/episode_rewards.npy", np.array(episode_rewards))

    with open(f"{output_dir}/train_config.json", "w") as f:
        json.dump({
            "stage": 1,
            "regime": regime,
            "num_episodes": num_episodes,
            "episode_weeks": episode_weeks,
            "lr": lr,
            "seed": seed,
        }, f, indent=2)

    print(f"\nTraining complete. Saved to {output_dir}/")
    return policy, episode_rewards


def train_firms(regime: str = DEFAULT_REGIME_NAME, num_episodes: int = 1000,
                episode_weeks: int | None = None, seed: int = DEFAULT_RANDOM_SEED,
                household_policy_dir: str = None,
                lr: float = 0.005, output_dir: str = None):
    """
    Stage 2: Train firm policy with frozen household policy.
    """
    print(f"{'='*60}")
    print(f"STAGE 2: Training Firms ({regime})")
    print(f"{'='*60}")

    rng = np.random.default_rng(seed)
    episode_weeks = episode_weeks or CURRICULUM[2]["episode_weeks"]

    if output_dir is None:
        output_dir = f"runs/train_firms_{regime}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # Load frozen household policy
    h_policy = SimplePolicy(HOUSEHOLD_OBS_DIM, NUM_HOUSEHOLD_ACTIONS)
    if household_policy_dir:
        h_policy.W = np.load(f"{household_policy_dir}/policy_W.npy")
        h_policy.b = np.load(f"{household_policy_dir}/policy_b.npy")
        print(f"  Loaded household policy from {household_policy_dir}")

    # Firm policy (shared, factorized multi-action)
    firm_policy = FactorizedPolicy(FIRM_OBS_DIM, FIRM_ACTION_DIMS, lr=lr)

    episode_rewards = []

    for ep in range(num_episodes):
        world = World(
            regime_name=regime,
            episode_weeks=episode_weeks,
            seed=seed + ep,
            curriculum_phase=2,
        )
        obs = world.reset()
        firm_trajectories = {int(world.f.id[s]): []
                             for s in range(world.f.size)}
        total_firm_reward = 0.0

        for step in range(episode_weeks):
            # Frozen household policy
            household_actions = {}
            for i in range(world.h.size):
                hid = int(world.h.ids[i])
                if hid in obs["households"]:
                    action, _ = h_policy.get_action(obs["households"][hid], rng)
                    household_actions[hid] = action

            # Firm actions on update ticks
            firm_actions = None
            firm_logprobs = {}
            firm_step_obs = {}
            if step > 0 and step % FIRM_UPDATE_PERIOD == 0:
                firm_actions = {}
                for slot in world.f.alive_slots().tolist():
                    fid = int(world.f.id[slot])
                    if fid in obs.get("firms", {}):
                        firm_step_obs[fid] = obs["firms"][fid].copy()
                        action, logp = firm_policy.get_action(obs["firms"][fid], rng)
                        firm_actions[fid] = action
                        firm_logprobs[fid] = logp

            obs, rewards, done, info = world.step(
                household_actions, firm_actions, None, None
            )

            # Collect firm trajectories
            if info.get("is_firm_update", False) and not info.get("is_warmup", False):
                for fid, r in rewards.get("firms", {}).items():
                    total_firm_reward += r
                    if fid in firm_trajectories and fid in firm_step_obs and fid in firm_actions:
                        firm_trajectories[fid].append((
                            firm_step_obs[fid],
                            firm_actions[fid],
                            r,
                            firm_logprobs.get(fid, 0),
                        ))

            if done:
                break

        all_trajs = []
        for fid_trajs in firm_trajectories.values():
            all_trajs.extend(fid_trajs)
        firm_policy.update(all_trajs, gamma=FIRM_GAMMA)

        alive_count = world.f.num_alive
        avg_reward = total_firm_reward / max(1, alive_count)
        episode_rewards.append(avg_reward)

        if ep % TRAIN_STAGE2_LOG_INTERVAL_EPISODES == 0 or ep == num_episodes - 1:
            print(f"  Ep {ep:4d} | FirmReward={avg_reward:8.4f} | Firms={alive_count}")

    np.save(f"{output_dir}/firm_policy_W.npy",
            np.array(firm_policy.W, dtype=object), allow_pickle=True)
    np.save(f"{output_dir}/firm_policy_b.npy",
            np.array(firm_policy.b, dtype=object), allow_pickle=True)
    np.save(f"{output_dir}/episode_rewards.npy", np.array(episode_rewards))
    print(f"\nStage 2 complete. Saved to {output_dir}/")
    return firm_policy, episode_rewards


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train MARL Economy agents")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--regime", default=DEFAULT_REGIME_NAME, choices=REGIME_NAMES)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--household-policy", default=None, help="Path to stage 1 output")
    args = parser.parse_args()

    if args.stage == 1:
        train_households(args.regime, args.episodes, seed=args.seed, lr=args.lr)
    elif args.stage == 2:
        train_firms(args.regime, args.episodes, seed=args.seed,
                    household_policy_dir=args.household_policy, lr=args.lr)
    else:
        print(f"Stage {args.stage} requires RLlib. Use train_rllib.py for stages 3-4.")


if __name__ == "__main__":
    main()
