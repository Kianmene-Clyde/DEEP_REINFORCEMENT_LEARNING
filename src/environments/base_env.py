from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any, Optional


class BaseEnvironment(ABC):
    """Classe abstraite pour les autres environnements"""

    # on le remplace dans les sous-classes pour les environnements adversaires
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
