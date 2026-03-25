"""Grid World environment - agent navigates a 2D grid.

State encoding (vector):
- One-hot vector of length `width * height`.
- Index = x + y * width

Action encoding:
- 0: Up
- 1: Down
- 2: Left
- 3: Right
"""
import numpy as np
from typing import Tuple, Dict, Any, Optional
from .base_env import BaseEnvironment


class GridWorld(BaseEnvironment):
    def __init__(self, width: int = 5, height: int = 5, seed: Optional[int] = None):
        self.width = width
        self.height = height
        self.start_pos = np.array([0, 0], dtype=np.int32)
        self.goal_pos = np.array([width - 1, height - 1], dtype=np.int32)
        self.agent_pos = self.start_pos.copy()
        self._action_space_size = 4
        super().__init__(seed)

    def _pos_to_index(self, pos: np.ndarray) -> int:
        return int(pos[0] + pos[1] * self.width)

    def _get_state(self) -> np.ndarray:
        idx = self._pos_to_index(self.agent_pos)
        state = np.zeros(self.width * self.height, dtype=np.float32)
        state[idx] = 1.0
        return state

    def state_to_index(self, state: Any) -> int:
        state_arr = np.asarray(state)
        if state_arr.ndim == 0:
            return int(state_arr)
        return int(np.argmax(state_arr))

    def reset(self) -> np.ndarray:
        self.agent_pos = self.start_pos.copy()
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        old_pos = self.agent_pos.copy()

        if action == 0:  # Up
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:  # Down
            self.agent_pos[1] = min(self.height - 1, self.agent_pos[1] + 1)
        elif action == 2:  # Left
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3:  # Right
            self.agent_pos[0] = min(self.width - 1, self.agent_pos[0] + 1)

        distance = np.sum(np.abs(self.agent_pos - self.goal_pos))
        old_distance = np.sum(np.abs(old_pos - self.goal_pos))

        if np.array_equal(self.agent_pos, self.goal_pos):
            reward = 10.0
            done = True
        else:
            reward = (old_distance - distance) * 0.1 - 0.01
            done = False

        return self._get_state(), reward, done, {}

    def render(self) -> None:
        grid = np.zeros((self.height, self.width), dtype=str)
        grid[:] = '.'
        grid[self.goal_pos[1], self.goal_pos[0]] = 'G'
        grid[self.agent_pos[1], self.agent_pos[0]] = 'A'
        for row in grid:
            print(' '.join(row))
        print()

    @property
    def observation_space(self) -> int:
        return self.width * self.height

    @property
    def action_space(self) -> int:
        return self._action_space_size

    def get_valid_actions(self, state: Any) -> np.ndarray:
        return np.arange(self._action_space_size, dtype=np.int32)
