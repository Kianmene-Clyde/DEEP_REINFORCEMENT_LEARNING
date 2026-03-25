"""Quarto environment - strategic 4x4 board game.

Rules:
- 16 unique pieces, each with 4 binary attributes (tall/short, dark/light, round/square, hollow/solid)
- Each turn: player places the piece given to them, then picks a piece for opponent
- Win: 4 pieces in a row/column/diagonal sharing at least 1 common attribute

Simplification for RL:
- Agent action = board position (0-15) to place the current piece
- After agent places, agent picks a random piece for opponent
- Opponent places and picks a piece for agent (random or heuristic)

State encoding: 33-dimensional vector:
  [board (16 slots: -1=empty, 0-15=piece), current_piece, available_pieces_mask (16 bits)]
"""
import numpy as np
from typing import Tuple, Dict, Any, Optional
from .base_env import BaseEnvironment


class Quarto(BaseEnvironment):
    # Piece attributes: each piece 0-15 has 4 binary attributes
    ATTRS = np.array([[(p >> b) & 1 for b in range(4)] for p in range(16)], dtype=np.int32)

    def __init__(self, opponent_type: str = "random", seed: Optional[int] = None):
        self.opponent_type = opponent_type
        self.board = np.full(16, -1, dtype=np.int32)  # 4x4 flattened, -1=empty
        self.available = np.ones(16, dtype=bool)  # which pieces are still available
        self.current_piece = -1  # piece that must be placed
        self._action_space_size = 16  # board positions
        self._done = False
        super().__init__(seed)

    def _get_state(self) -> np.ndarray:
        """State: board(16) + current_piece(1) + available_mask(16) = 33 dims."""
        return np.concatenate([
            self.board.astype(np.float32) / 15.0,  # normalize to [0,1], -1/15 for empty
            np.array([self.current_piece / 15.0], dtype=np.float32),
            self.available.astype(np.float32),
        ])

    def reset(self) -> np.ndarray:
        self.board = np.full(16, -1, dtype=np.int32)
        self.available = np.ones(16, dtype=bool)
        self._done = False
        # Opponent picks a first piece for the agent
        piece = self._pick_piece_for("agent")
        self.current_piece = piece
        self.available[piece] = False
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Agent places current_piece at position `action`, then picks piece for opponent."""
        if self._done:
            return self._get_state(), 0.0, True, {}

        pos = action
        # Validate placement
        if pos < 0 or pos >= 16 or self.board[pos] != -1:
            return self._get_state(), -1.0, True, {"error": "invalid_placement"}
        if self.current_piece < 0:
            return self._get_state(), -1.0, True, {"error": "no_piece"}

        # Agent places piece
        self.board[pos] = self.current_piece
        if self._check_win():
            self._done = True
            return self._get_state(), 1.0, True, {"winner": "agent"}

        if not np.any(self.board == -1) or not np.any(self.available):
            self._done = True
            return self._get_state(), 0.0, True, {"result": "draw"}

        # Agent picks piece for opponent
        opp_piece = self._pick_piece_for("opponent")
        if opp_piece < 0:
            self._done = True
            return self._get_state(), 0.0, True, {"result": "draw"}
        self.available[opp_piece] = False

        # Opponent places the piece
        opp_pos = self._opponent_place(opp_piece)
        if opp_pos is not None:
            self.board[opp_pos] = opp_piece
            if self._check_win():
                self._done = True
                return self._get_state(), -1.0, True, {"winner": "opponent"}

        if not np.any(self.board == -1) or not np.any(self.available):
            self._done = True
            return self._get_state(), 0.0, True, {"result": "draw"}

        # Opponent picks piece for agent
        agent_piece = self._opponent_pick_piece()
        if agent_piece < 0:
            self._done = True
            return self._get_state(), 0.0, True, {"result": "draw"}
        self.current_piece = agent_piece
        self.available[agent_piece] = False

        return self._get_state(), 0.0, False, {}

    def _check_win(self) -> bool:
        """Check all lines of 4 for shared attribute."""
        b = self.board.reshape(4, 4)
        lines = []
        # Rows & columns
        for i in range(4):
            lines.append(b[i, :])
            lines.append(b[:, i])
        # Diagonals
        lines.append(np.array([b[i, i] for i in range(4)]))
        lines.append(np.array([b[i, 3 - i] for i in range(4)]))

        for line in lines:
            if np.any(line == -1):
                continue
            pieces_attrs = self.ATTRS[line]  # (4, 4)
            # Check if all 4 share at least one attribute value
            for attr in range(4):
                if len(set(pieces_attrs[:, attr])) == 1:
                    return True
        return False

    def _pick_piece_for(self, target: str) -> int:
        """Agent picks a piece (random). Returns piece index or -1."""
        avail = np.where(self.available)[0]
        if len(avail) == 0:
            return -1
        return int(np.random.choice(avail))

    def _opponent_place(self, piece: int) -> Optional[int]:
        """Opponent places piece on the board."""
        empty = np.where(self.board == -1)[0]
        if len(empty) == 0:
            return None
        if self.opponent_type == "heuristic":
            return self._heuristic_place(piece, empty)
        return int(np.random.choice(empty))

    def _opponent_pick_piece(self) -> int:
        """Opponent picks a piece for the agent."""
        avail = np.where(self.available)[0]
        if len(avail) == 0:
            return -1
        if self.opponent_type == "heuristic":
            return self._heuristic_pick(avail)
        return int(np.random.choice(avail))

    def _heuristic_place(self, piece: int, empty: np.ndarray) -> int:
        """Try to win, else random."""
        for pos in empty:
            self.board[pos] = piece
            if self._check_win():
                self.board[pos] = -1
                return int(pos)
            self.board[pos] = -1
        return int(np.random.choice(empty))

    def _heuristic_pick(self, avail: np.ndarray) -> int:
        """Avoid giving a piece that lets opponent win. Simplified."""
        # Just pick random for now (full heuristic is complex)
        return int(np.random.choice(avail))

    def render(self) -> None:
        b = self.board.reshape(4, 4)
        print("Quarto Board:")
        for row in b:
            print([f"{p:2d}" if p != -1 else " ." for p in row])
        print(f"Current piece: {self.current_piece}, Available: {list(np.where(self.available)[0])}")

    @property
    def observation_space(self) -> int:
        return 33

    @property
    def action_space(self) -> int:
        return self._action_space_size

    def get_valid_actions(self, state: Any = None) -> np.ndarray:
        """Valid actions = empty board positions."""
        # Use internal board (always in sync with state after step/reset)
        return np.where(self.board == -1)[0].astype(np.int32)
