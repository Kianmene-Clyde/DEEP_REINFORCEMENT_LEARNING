"""Interface de l'agent basique."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Any


class BaseAgent(ABC):
    """classe abstraite pour tous nos agents"""

    def __init__(self, name: str, action_space_size: int, input_size: Any = None):
        self.name = name
        self.action_space_size = action_space_size
        self.input_size = input_size
        self.training = True
        self.uses_tabular = False
        self.rng = np.random.default_rng()
        self._next_valid_actions = None
        self._transition_terminated = False
        self._transition_truncated = False

    def observe_transition(
        self, state, action, reward, next_state, *, terminated, truncated=False, next_valid_actions=None
    ):
        """Preserve time-limit and legal-action information across the legacy API."""
        self._transition_terminated = bool(terminated)
        self._transition_truncated = bool(truncated)
        self._next_valid_actions = next_valid_actions
        boundary = bool(terminated or truncated)
        self.learn(
            state,
            action,
            reward,
            next_state,
            bool(terminated) if getattr(self, "is_value_based", False) else boundary,
        )

    def next_action_mask(self):
        mask = np.ones(self.action_space_size, dtype=bool)
        if self._next_valid_actions is not None:
            mask[:] = False
            mask[self._next_valid_actions] = True
        return mask

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != "_env"}

    @abstractmethod
    def select_action(self, state: Any, valid_actions: Optional[np.ndarray] = None) -> int:
        pass

    @abstractmethod
    def learn(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        pass

    @abstractmethod
    def load(self, filepath: str) -> None:
        pass

    def set_training_mode(self, training: bool):
        self.training = training

    def __repr__(self):
        return f"{self.name}(actions={self.action_space_size})"
