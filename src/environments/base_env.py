"""Base Environment class for all environments."""
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any, Optional


class BaseEnvironment(ABC):
    """Abstract base class for RL environments."""
    
    # Override in subclasses for adversarial environments
    is_two_player: bool = False

    def __init__(self, seed: Optional[int] = None):
        self.seed_value = seed
        if seed is not None:
            np.random.seed(seed)
        self.reset()
    
    @abstractmethod
    def reset(self) -> Any:
        pass
    
    @abstractmethod
    def step(self, action: int) -> Tuple[Any, float, bool, Dict]:
        pass
    
    @abstractmethod
    def render(self) -> None:
        pass
    
    @abstractmethod
    def _get_state(self) -> Any:
        """Return current state (needed by MCTS/Rollout for deep copy)."""
        pass

    @property
    @abstractmethod
    def observation_space(self) -> Any:
        pass
    
    @property
    @abstractmethod
    def action_space(self) -> int:
        pass
    
    @abstractmethod
    def get_valid_actions(self, state: Any) -> np.ndarray:
        pass
