"""Interface de l'aegnt basique."""
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

    @abstractmethod
    def select_action(self, state: Any, valid_actions: Optional[np.ndarray] = None) -> int:
        pass

    @abstractmethod
    def learn(self, state: Any, action: int, reward: float,
              next_state: Any, done: bool) -> None:
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
