import numpy as np
from typing import Tuple, Dict, Any, Optional
from .base_env import BaseEnvironment


class LineWorld(BaseEnvironment):
    def __init__(self, length: int = 10, seed: Optional[int] = None):
        self.length = length
        if length < 2:
            raise ValueError("LineWorld requires length >= 2")
        self.position = 0
        self._action_space_size = 2
        super().__init__(seed)

    def _get_state(self) -> np.ndarray:
        state = np.zeros(self.length, dtype=np.float32)
        state[self.position] = 1.0
        return state

    def state_to_index(self, state: Any) -> int:
        if isinstance(state, (int, np.integer)):
            return int(state)
        state_arr = np.asarray(state)
        return int(np.argmax(state_arr))

    def reset(self, seed=None) -> np.ndarray:
        self.reseed(seed)
        self._done = False
        self.position = 0
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        self._validate_action(action)
        if action == 0:
            self.position = max(0, self.position - 1)
        elif action == 1:
            self.position = min(self.length - 1, self.position + 1)

        reward = 1.0 if self.position == self.length - 1 else -0.1
        done = self.position == self.length - 1
        self._done = done
        return self._get_state(), reward, done, {"outcome": "success", "terminal": "goal"} if done else {}

    def render(self) -> None:
        line = ["_"] * self.length
        line[self.position] = "G"
        print("".join(line))

    @property
    def observation_space(self) -> int:
        return self.length

    @property
    def action_space(self) -> int:
        return self._action_space_size

    def get_valid_actions(self, state: Any) -> np.ndarray:
        return np.array([] if self._done else [0, 1], dtype=np.int32)
