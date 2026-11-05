"""Flat Monte Carlo action evaluation from the controlled player's perspective."""

import pickle
from pathlib import Path
import numpy as np
from .base_agent import BaseAgent
from .search import rollout


class RandomRolloutAgent(BaseAgent):
    def __init__(
        self,
        action_space_size,
        num_rollouts=50,
        max_rollout_depth=100,
        discount_factor=0.99,
        is_two_player=False,
        seed=None,
        **kwargs,
    ):
        super().__init__("RandomRollout", action_space_size)
        if num_rollouts < 1 or max_rollout_depth < 1:
            raise ValueError("Rollout budgets must be positive")
        self.rng = np.random.default_rng(seed)
        self.num_rollouts, self.max_depth, self.gamma = num_rollouts, max_rollout_depth, discount_factor
        self.is_two_player = is_two_player
        self._env = None

    def set_env(self, env):
        self._env = env

    def select_action(self, state, valid_actions=None):
        if self._env is None:
            raise RuntimeError("RandomRollout requires set_env(env)")
        valid = self._env.get_valid_actions(state) if valid_actions is None else valid_actions
        if not len(valid):
            raise ValueError("No legal actions")
        values = []
        for action in valid:
            samples = []
            for _ in range(self.num_rollouts):
                sim = self._env.clone(seed=int(self.rng.integers(2**32)))
                _, r, done, _ = sim.step(int(action))
                samples.append(
                    r if done else r + self.gamma * rollout(sim, self.rng, self.max_depth - 1, self.gamma)
                )
            values.append(np.mean(samples))
        return int(valid[np.argmax(values)])

    def learn(self, *args):
        pass

    def save(self, filepath):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath + "_config.pkl", "wb") as stream:
            pickle.dump(
                {
                    "num_rollouts": self.num_rollouts,
                    "max_depth": self.max_depth,
                    "gamma": self.gamma,
                    "is_two_player": False,
                },
                stream,
            )

    def load(self, filepath):
        with open(filepath + "_config.pkl", "rb") as stream:
            config = pickle.load(stream)
        self.num_rollouts, self.max_depth = config["num_rollouts"], config["max_depth"]
        self.gamma = config.get("gamma", 0.99)
