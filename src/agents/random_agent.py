"""Random agent"""

import numpy as np
from typing import Optional, Any
from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, action_space_size: int, seed: Optional[int] = None, **kwargs):
        super().__init__("Random", action_space_size)
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: Any, valid_actions: Optional[np.ndarray] = None) -> int:
        if valid_actions is not None and len(valid_actions) == 0:
            raise ValueError("No legal actions")
        if valid_actions is not None:
            return int(self.rng.choice(valid_actions))
        return self.rng.integers(0, self.action_space_size)

    def learn(self, state, action, reward, next_state, done):
        pass

    def save(self, filepath: str):
        pass

    def load(self, filepath: str):
        pass
