import numpy as np
from typing import Tuple, Dict, Any, Optional
from .base_env import BaseEnvironment


class GridWorld(BaseEnvironment):
    def __init__(self, width: int = 5, height: int = 5, seed: Optional[int] = None):
        if width < 2 or height < 2:
            raise ValueError("GridWorld requires width>=2 and height>=2 (because start is (1,0) and pit is (0,0)).")

        self.width = width
        self.height = height

        # Case (0,0) est éliminatoire
        self.pit_pos = np.array([0, 0], dtype=np.int32)

        # Point de départ c'est la case (1,0)
        self.start_pos = np.array([1, 0], dtype=np.int32)

        # La case gagnante reste le coin inférieur droit
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

        # Move
        if action == 0:  # Up
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 1:  # Down
            self.agent_pos[1] = min(self.height - 1, self.agent_pos[1] + 1)
        elif action == 2:  # Left
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 3:  # Right
            self.agent_pos[0] = min(self.width - 1, self.agent_pos[0] + 1)

        # La case éliminatoire a la priorité sur tout le reste
        if np.array_equal(self.agent_pos, self.pit_pos):
            return self._get_state(), -10.0, True, {"terminal": "pit"}

        # Goal
        if np.array_equal(self.agent_pos, self.goal_pos):
            return self._get_state(), 10.0, True, {"terminal": "goal"}

        distance = np.sum(np.abs(self.agent_pos - self.goal_pos))
        old_distance = np.sum(np.abs(old_pos - self.goal_pos))
        reward = (old_distance - distance) * 0.1 - 0.01
        done = False

        return self._get_state(), float(reward), done, {}

    def render(self) -> None:
        grid = np.zeros((self.height, self.width), dtype=str)
        grid[:] = '.'

        grid[self.pit_pos[1], self.pit_pos[0]] = 'X'
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
