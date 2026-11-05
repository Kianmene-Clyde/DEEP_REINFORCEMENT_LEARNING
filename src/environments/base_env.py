from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any, Optional


class BaseEnvironment(ABC):
    """Classe abstraite pour les autres environnements"""

    # on le remplace dans les sous-classes pour les environnements adversaires
    is_two_player: bool = False

    def __init__(self, seed: Optional[int] = None):
        self.seed_value = seed
        self.rng = np.random.default_rng(seed)
        self._done = False
        self.reset()

    def reseed(self, seed=None):
        if seed is not None:
            self.seed_value = int(seed)
            self.rng = np.random.default_rng(seed)

    def clone(self, seed=None):
        """Clone observable state with an independent simulation random stream."""
        import copy

        env = copy.deepcopy(self)
        env.reseed(seed)
        return env

    def _validate_action(self, action):
        if self._done:
            raise RuntimeError("Episode finished: call reset before step")
        if not isinstance(action, (int, np.integer)) or action not in self.get_valid_actions(
            self._get_state()
        ):
            raise ValueError(f"Invalid action: {action}")

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
