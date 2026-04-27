"""Environnement Quarto - jeu de strategie sur plateau 4x4.

Encodage de l'etat (101 dimensions) :
  Pour chaque case du plateau (16 cases x 5 valeurs = 80) :
    [occupee, taille, couleur, forme, remplissage]
    occupee=0 -> case vide (les 4 attrs sont a 0)
    occupee=1 -> piece presente, attrs en binaire (0 ou 1)
  Piece courante (5 valeurs) :
    [a_piece, taille, couleur, forme, remplissage]
  Disponibilite des pieces (16 valeurs) :
    1.0 = disponible, 0.0 = utilisee

Attributs binaires des pieces (chaque piece 0-15 = 4 bits) :
  bit 0 : Taille      (0=Petite, 1=Grande)
  bit 1 : Couleur     (0=Claire, 1=Foncee)
  bit 2 : Forme       (0=Carree, 1=Ronde)
  bit 3 : Remplissage (0=Pleine, 1=Creuse)
"""
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from .base_env import BaseEnvironment

# Symboles visuels uniques pour chaque piece
# Lettre : majuscule=Grande, minuscule=Petite, R/r=Foncee, B/b=Claire
# Forme  : carre/rond x plein/creux
PIECE_SYMBOLS = {
    0: "b\u25a0",  # Petite Claire Carree Pleine
    1: "B\u25a0",  # Grande Claire Carree Pleine
    2: "r\u25a0",  # Petite Foncee Carree Pleine
    3: "R\u25a0",  # Grande Foncee Carree Pleine
    4: "b\u25cf",  # Petite Claire Ronde  Pleine
    5: "B\u25cf",  # Grande Claire Ronde  Pleine
    6: "r\u25cf",  # Petite Foncee Ronde  Pleine
    7: "R\u25cf",  # Grande Foncee Ronde  Pleine
    8: "b\u25a1",  # Petite Claire Carree Creuse
    9: "B\u25a1",  # Grande Claire Carree Creuse
    10: "r\u25a1",  # Petite Foncee Carree Creuse
    11: "R\u25a1",  # Grande Foncee Carree Creuse
    12: "b\u25cb",  # Petite Claire Ronde  Creuse
    13: "B\u25cb",  # Grande Claire Ronde  Creuse
    14: "r\u25cb",  # Petite Foncee Ronde  Creuse
    15: "R\u25cb",  # Grande Foncee Ronde  Creuse
}

PIECE_ATTRS_NAMES = [
    ("Petite", "Grande"),
    ("Claire", "Foncee"),
    ("Carree", "Ronde"),
    ("Pleine", "Creuse"),
]


class Quarto(BaseEnvironment):
    is_two_player = True
    ATTRS = np.array([[(p >> b) & 1 for b in range(4)] for p in range(16)], dtype=np.int32)

    def __init__(self, opponent_type: str = "random", seed: Optional[int] = None):
        self.opponent_type = opponent_type
        self.board = np.full(16, -1, dtype=np.int32)
        self.available = np.ones(16, dtype=bool)
        self.current_piece = -1
        self._action_space_size = 16
        self._done = False
        super().__init__(seed)

    # ──── Utilitaires pieces ────

    @staticmethod
    def piece_symbol(piece_id: int) -> str:
        if piece_id < 0 or piece_id > 15:
            return "?"
        return PIECE_SYMBOLS[piece_id]

    @staticmethod
    def piece_description(piece_id: int) -> str:
        if piece_id < 0:
            return "aucune"
        attrs = [(piece_id >> b) & 1 for b in range(4)]
        return ", ".join(PIECE_ATTRS_NAMES[i][attrs[i]] for i in range(4))

    # ──── Encodage de l'etat (101 dimensions) ────

    def _encode_piece_attrs(self, piece_id: int) -> np.ndarray:
        """Encode une piece en 5 valeurs : [occupee, taille, couleur, forme, remplissage]."""
        if piece_id < 0:
            return np.zeros(5, dtype=np.float32)
        enc = np.zeros(5, dtype=np.float32)
        enc[0] = 1.0  # case occupee / piece presente
        enc[1] = float((piece_id >> 0) & 1)  # taille
        enc[2] = float((piece_id >> 1) & 1)  # couleur
        enc[3] = float((piece_id >> 2) & 1)  # forme
        enc[4] = float((piece_id >> 3) & 1)  # remplissage
        return enc

    def _get_state(self) -> np.ndarray:
        # Plateau : 16 cases x 5 valeurs = 80 dimensions
        board_enc = np.zeros(80, dtype=np.float32)
        for i in range(16):
            board_enc[i * 5:(i + 1) * 5] = self._encode_piece_attrs(self.board[i])

        # Piece courante : 5 dimensions
        piece_enc = self._encode_piece_attrs(self.current_piece)

        # Disponibilite : 16 dimensions
        avail_enc = self.available.astype(np.float32)

        return np.concatenate([board_enc, piece_enc, avail_enc])
        # Total : 80 + 5 + 16 = 101

    def reset(self) -> np.ndarray:
        self.board = np.full(16, -1, dtype=np.int32)
        self.available = np.ones(16, dtype=bool)
        self._done = False
        # L'adversaire choisit une premiere piece pour l'agent
        piece = self._opponent_pick_piece_for_agent()
        self.current_piece = piece
        self.available[piece] = False
        return self._get_state()

    # ──── Step ────

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self._done:
            return self._get_state(), 0.0, True, {}

        pos = action
        if pos < 0 or pos >= 16 or self.board[pos] != -1:
            return self._get_state(), -1.0, True, {"error": "placement invalide"}
        if self.current_piece < 0:
            return self._get_state(), -1.0, True, {"error": "aucune piece"}

        # 1. L'agent place sa piece
        self.board[pos] = self.current_piece
        if self._check_win():
            self._done = True
            return self._get_state(), 1.0, True, {"winner": "agent"}

        if not np.any(self.board == -1) or not np.any(self.available):
            self._done = True
            return self._get_state(), 0.0, True, {"result": "draw"}

        # 2. L'agent choisit une piece pour l'adversaire (TOUJOURS heuristique)
        opp_piece = self._agent_pick_piece_for_opponent()
        if opp_piece < 0:
            self._done = True
            return self._get_state(), 0.0, True, {"result": "draw"}
        self.available[opp_piece] = False

        # 3. L'adversaire place sa piece
        opp_pos = self._opponent_place(opp_piece)
        if opp_pos is not None:
            self.board[opp_pos] = opp_piece
            if self._check_win():
                self._done = True
                return self._get_state(), -1.0, True, {"winner": "opponent"}

        if not np.any(self.board == -1) or not np.any(self.available):
            self._done = True
            return self._get_state(), 0.0, True, {"result": "draw"}

        # 4. L'adversaire choisit une piece pour l'agent (selon opponent_type)
        agent_piece = self._opponent_pick_piece_for_agent()
        if agent_piece < 0:
            self._done = True
            return self._get_state(), 0.0, True, {"result": "draw"}
        self.current_piece = agent_piece
        self.available[agent_piece] = False

        return self._get_state(), 0.0, False, {}

    # ──── Verification de victoire ────

    def _get_lines(self) -> List[np.ndarray]:
        b = self.board.reshape(4, 4)
        lines = []
        for i in range(4):
            lines.append(b[i, :])
            lines.append(b[:, i])
        lines.append(np.array([b[i, i] for i in range(4)]))
        lines.append(np.array([b[i, 3 - i] for i in range(4)]))
        return lines

    def _check_win(self) -> bool:
        for line in self._get_lines():
            if np.any(line == -1):
                continue
            pieces_attrs = self.ATTRS[line]
            for attr in range(4):
                if len(set(pieces_attrs[:, attr])) == 1:
                    return True
        return False

    def _would_piece_win_somewhere(self, piece: int) -> bool:
        """Verifie si cette piece peut gagner en etant placee quelque part."""
        empty = np.where(self.board == -1)[0]
        for pos in empty:
            self.board[pos] = piece
            won = self._check_win()
            self.board[pos] = -1
            if won:
                return True
        return False

    def _count_dangerous_lines(self, piece: int) -> int:
        """Score de dangerosite : combien de lignes cette piece pourrait aider a completer."""
        danger = 0
        piece_attrs = self.ATTRS[piece]
        for line in self._get_lines():
            filled = line[line != -1]
            empty_count = np.sum(line == -1)
            if len(filled) < 2 or empty_count == 0:
                continue
            filled_attrs = self.ATTRS[filled]
            for attr in range(4):
                vals = set(filled_attrs[:, attr])
                if len(vals) == 1 and piece_attrs[attr] == list(vals)[0]:
                    danger += len(filled) ** 2  # 3 pieces = 9, 2 pieces = 4
        return danger

    # ──── Choix de piece : AGENT pour l'adversaire ────

    def _agent_pick_piece_for_opponent(self) -> int:
        """L'agent choisit une piece pour l'adversaire.

        TOUJOURS heuristique : donner la piece la MOINS dangereuse
        pour minimiser les chances de victoire de l'adversaire.

        Strategie :
        1. Eliminer les pieces qui permettent a l'adversaire de gagner immediatement
        2. Parmi les restantes, choisir celle avec le moins de lignes dangereuses
        """
        avail = np.where(self.available)[0]
        if len(avail) == 0:
            return -1

        # Niveau 1 : eliminer les pieces qui donnent la victoire
        safe = [int(p) for p in avail if not self._would_piece_win_somewhere(int(p))]
        candidates = safe if safe else [int(p) for p in avail]

        if len(candidates) == 1:
            return candidates[0]

        # Niveau 2 : choisir la moins dangereuse
        scored = [(p, self._count_dangerous_lines(p)) for p in candidates]
        scored.sort(key=lambda x: x[1])
        best_score = scored[0][1]
        best = [p for p, s in scored if s == best_score]
        return int(np.random.choice(best))

    # ──── Choix de piece : ADVERSAIRE pour l'agent ────

    def _opponent_pick_piece_for_agent(self) -> int:
        """L'adversaire choisit une piece pour l'agent.

        En mode 'random' : choix aleatoire.
        En mode 'heuristic' : donner la piece la MOINS dangereuse
        pour minimiser les chances de victoire de l'agent.
        """
        avail = np.where(self.available)[0]
        if len(avail) == 0:
            return -1

        if self.opponent_type == "random":
            return int(np.random.choice(avail))

        # Mode heuristic : meme logique que l'agent
        safe = [int(p) for p in avail if not self._would_piece_win_somewhere(int(p))]
        candidates = safe if safe else [int(p) for p in avail]

        if len(candidates) == 1:
            return candidates[0]

        scored = [(p, self._count_dangerous_lines(p)) for p in candidates]
        scored.sort(key=lambda x: x[1])
        best_score = scored[0][1]
        best = [p for p, s in scored if s == best_score]
        return int(np.random.choice(best))

    # ──── Placement de l'adversaire ────

    def _opponent_place(self, piece: int) -> Optional[int]:
        empty = np.where(self.board == -1)[0]
        if len(empty) == 0:
            return None
        if self.opponent_type == "heuristic":
            return self._heuristic_place(piece, empty)
        return int(np.random.choice(empty))

    def _heuristic_place(self, piece: int, empty: np.ndarray) -> int:
        # Gagner si possible
        for pos in empty:
            self.board[pos] = piece
            if self._check_win():
                self.board[pos] = -1
                return int(pos)
            self.board[pos] = -1
        return int(np.random.choice(empty))

    # ──── Affichage ────

    def render(self) -> None:
        b = self.board.reshape(4, 4)
        print("  Quarto Board:")
        print("       0     1     2     3")
        print("    +-----+-----+-----+-----+")
        for i, row in enumerate(b):
            cells = []
            for p in row:
                if p == -1:
                    cells.append("  .  ")
                else:
                    sym = PIECE_SYMBOLS[p]
                    cells.append(f"{p:2d} {sym}")
            print(f"  {i} |{'|'.join(cells)}|")
            print("    +-----+-----+-----+-----+")
        if self.current_piece >= 0:
            sym = PIECE_SYMBOLS[self.current_piece]
            desc = self.piece_description(self.current_piece)
            print(f"  Piece a placer: {self.current_piece} ({sym}) - {desc}")
        avail = np.where(self.available)[0]
        if len(avail) > 0:
            print(f"  Disponibles: {', '.join(f'{p}({PIECE_SYMBOLS[p]})' for p in avail)}")

    @property
    def observation_space(self) -> int:
        return 101  # 16*5 + 5 + 16

    @property
    def action_space(self) -> int:
        return self._action_space_size

    def get_valid_actions(self, state: Any = None) -> np.ndarray:
        return np.where(self.board == -1)[0].astype(np.int32)
