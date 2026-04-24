"""Line World environment - agent moves on a 1D line.

State encoding (vector):
- One-hot vector of length `length`, with 1 at the current position.

Action encoding:
- 0: Left
- 1: Right
"""
import numpy as np
from typing import Tuple, Dict, Any, Optional
from .base_env import BaseEnvironment


class LineWorld(BaseEnvironment):
    """
    Simple 1D environment where agent moves left/right on a line.
    Goal is to reach the right end of the line.
    """

    def __init__(self, length: int = 10, seed: Optional[int] = None):
        self.length = length
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

    def reset(self) -> np.ndarray:
        self.position = 0
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if action == 0:
            self.position = max(0, self.position - 1)
        elif action == 1:
            self.position = min(self.length - 1, self.position + 1)

        reward = 1.0 if self.position == self.length - 1 else -0.1
        done = self.position == self.length - 1
        return self._get_state(), reward, done, {}

    def render(self) -> None:
        line = ['_'] * self.length
        line[self.position] = 'G'
        print(''.join(line))

    @property
    def observation_space(self) -> int:
        return self.length

    @property
    def action_space(self) -> int:
        return self._action_space_size

    def get_valid_actions(self, state: Any) -> np.ndarray:
        return np.array([0, 1], dtype=np.int32)
