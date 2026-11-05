"""Tabular Q-Learning agent."""

import numpy as np
import pickle
from typing import Optional, Any
from .base_agent import BaseAgent


class TabularQLearningAgent(BaseAgent):
    def __init__(
        self,
        state_space_size: int,
        action_space_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.01,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__("TabularQLearning", action_space_size)
        self.uses_tabular = True
        self.is_value_based = True
        self.state_space_size = state_space_size
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.rng = np.random.default_rng(seed)
        self.q_table = np.zeros((state_space_size, action_space_size), dtype=np.float32)

    def select_action(self, state: Any, valid_actions: Optional[np.ndarray] = None) -> int:
        s = int(state) if not isinstance(state, (int, np.integer)) else state
        if self.training and self.rng.random() < self.epsilon:
            if valid_actions is not None and len(valid_actions) > 0:
                return int(self.rng.choice(valid_actions))
            return self.rng.integers(0, self.action_space_size)
        q = self.q_table[s].copy()
        if valid_actions is not None:
            mask = np.full(self.action_space_size, -1e9)
            mask[valid_actions] = 0
            q = q + mask
        if valid_actions is not None and len(valid_actions) == 0:
            raise ValueError("No legal actions")
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best)) if self.training else int(best[0])

    def learn(self, state, action, reward, next_state, done):
        if not self.training:
            return
        s = int(state)
        ns = int(next_state)
        mask = self.next_action_mask()
        bootstrap = np.max(self.q_table[ns, mask]) if not done and mask.any() else 0.0
        target = reward + self.gamma * bootstrap
        self.q_table[s, action] += self.lr * (target - self.q_table[s, action])
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        import os

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath + ".pkl", "wb") as f:
            pickle.dump({"q_table": self.q_table, "epsilon": self.epsilon}, f)

    def load(self, filepath: str):
        with open(filepath + ".pkl", "rb") as f:
            data = pickle.load(f)
        self.q_table = data["q_table"]
        self.epsilon = data["epsilon"]
