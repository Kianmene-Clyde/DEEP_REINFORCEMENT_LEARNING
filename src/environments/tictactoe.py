"""TicTacToe environment - agent plays X against configurable opponent.

State encoding: flat array of 9 values: 0=empty, 1=agent(X), -1=opponent(O)
Action encoding: board position 0-8 (row*3 + col)
"""
import numpy as np
from typing import Tuple, Dict, Any, Optional
from .base_env import BaseEnvironment


class TicTacToe(BaseEnvironment):
    is_two_player = True

    def __init__(self, opponent_type: str = "random", seed: Optional[int] = None):
        self.opponent_type = opponent_type
        self.board = np.zeros(9, dtype=np.float32)
        self._action_space_size = 9
        super().__init__(seed)

    def _get_state(self) -> np.ndarray:
        return self.board.copy()

    def state_to_index(self, state: Any) -> int:
        """Convert board to unique index (for tabular agents). 3^9 = 19683 states."""
        s = np.asarray(state, dtype=np.int32) + 1  # map -1,0,1 -> 0,1,2
        idx = 0
        for i in range(9):
            idx += int(s[i]) * (3 ** i)
        return idx

    def reset(self) -> np.ndarray:
        self.board = np.zeros(9, dtype=np.float32)
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.board[action] != 0:
            return self._get_state(), -1.0, True, {"invalid": True}

        self.board[action] = 1.0
        w = self._check_winner()
        if w == 1:
            return self._get_state(), 1.0, True, {}
        if self._is_full():
            return self._get_state(), 0.0, True, {}

        # Opponent move
        opp_action = self._opponent_move()
        self.board[opp_action] = -1.0
        w = self._check_winner()
        if w == -1:
            return self._get_state(), -1.0, True, {}
        if self._is_full():
            return self._get_state(), 0.0, True, {}

        return self._get_state(), 0.0, False, {}

    def _check_winner(self) -> int:
        b = self.board.reshape(3, 3)
        for i in range(3):
            if abs(b[i].sum()) == 3:
                return int(np.sign(b[i].sum()))
            if abs(b[:, i].sum()) == 3:
                return int(np.sign(b[:, i].sum()))
        if abs(np.diag(b).sum()) == 3:
            return int(np.sign(np.diag(b).sum()))
        if abs(np.diag(b[::-1]).sum()) == 3:
            return int(np.sign(np.diag(b[::-1]).sum()))
        return 0

    def _is_full(self) -> bool:
        return np.all(self.board != 0)

    def _opponent_move(self) -> int:
        valid = np.where(self.board == 0)[0]
        if self.opponent_type == "heuristic":
            return self._heuristic_move(valid)
        return int(np.random.choice(valid))

    def _heuristic_move(self, valid: np.ndarray) -> int:
        # Win if possible
        for a in valid:
            self.board[a] = -1.0
            if self._check_winner() == -1:
                self.board[a] = 0
                return int(a)
            self.board[a] = 0
        # Block agent
        for a in valid:
            self.board[a] = 1.0
            if self._check_winner() == 1:
                self.board[a] = 0
                return int(a)
            self.board[a] = 0
        # Center
        if 4 in valid:
            return 4
        # Corner
        corners = np.array([0, 2, 6, 8])
        avail = valid[np.isin(valid, corners)]
        if len(avail) > 0:
            return int(np.random.choice(avail))
        return int(np.random.choice(valid))

    def render(self) -> None:
        sym = {0: ' ', 1: 'X', -1: 'O'}
        b = self.board.reshape(3, 3)
        for i in range(3):
            print(f" {sym[int(b[i,0])]} | {sym[int(b[i,1])]} | {sym[int(b[i,2])]}")
            if i < 2:
                print("-----------")
        print()

    @property
    def observation_space(self) -> int:
        return 9

    @property
    def action_space(self) -> int:
        return self._action_space_size

    def get_valid_actions(self, state: Any) -> np.ndarray:
        s = np.asarray(state) if state is not None else self.board
        return np.where(s == 0)[0].astype(np.int32)
