"""UCT planning over sampled fixed-opponent transitions."""

import pickle
from pathlib import Path
import numpy as np
from .base_agent import BaseAgent
from .search import tree_search


class MCTSAgent(BaseAgent):
    def __init__(
        self,
        action_space_size,
        num_simulations=200,
        exploration_constant=1.41,
        max_rollout_depth=100,
        discount_factor=0.99,
        is_two_player=False,
        seed=None,
        **kwargs,
    ):
        super().__init__("MCTS_UCT", action_space_size)
        self.rng = np.random.default_rng(seed)
        self.num_simulations, self.c, self.max_depth = (
            num_simulations,
            exploration_constant,
            max_rollout_depth,
        )
        self.gamma = discount_factor
        self.is_two_player = is_two_player
        self._env = None

    def set_env(self, env):
        self._env = env

    def select_action(self, state, valid_actions=None):
        if self._env is None:
            raise RuntimeError("MCTS requires set_env(env)")
        visits, totals = tree_search(
            self._env, self.rng, self.num_simulations, self.max_depth, self.gamma, self.c
        )
        valid = self._env.get_valid_actions(state)
        most = valid[visits[valid] == visits[valid].max()]
        means = totals[most] / np.maximum(1, visits[most])
        return int(most[np.argmax(means)])

    def learn(self, *args):
        pass

    def save(self, filepath):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath + "_config.pkl", "wb") as stream:
            pickle.dump(
                {
                    "num_simulations": self.num_simulations,
                    "c": self.c,
                    "max_depth": self.max_depth,
                    "gamma": self.gamma,
                    "is_two_player": False,
                },
                stream,
            )

    def load(self, filepath):
        with open(filepath + "_config.pkl", "rb") as stream:
            config = pickle.load(stream)
        self.num_simulations, self.c, self.max_depth = (
            config["num_simulations"],
            config["c"],
            config["max_depth"],
        )
        self.gamma = config.get("gamma", 0.99)
