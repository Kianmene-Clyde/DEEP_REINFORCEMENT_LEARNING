import numpy as np
import copy
import math
from typing import Optional, Any, List
from .base_agent import BaseAgent
import pickle
import os


class MCTSNode:
    """Nœud de l'arbre MCTS."""
    __slots__ = ['state', 'parent', 'action', 'children', 'visits', 'value',
                 'untried_actions', 'is_terminal']

    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions: Optional[np.ndarray] = None
        self.is_terminal = False

    def ucb1(self, c: float = 1.41) -> float:
        if self.visits == 0:
            return float('inf')
        exploit = self.value / self.visits
        explore = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore

    def best_child(self, c: float = 1.41) -> 'MCTSNode':
        return max(self.children, key=lambda ch: ch.ucb1(c))

    def is_fully_expanded(self) -> bool:
        return (self.untried_actions is not None
                and len(self.untried_actions) == 0)


class MCTSAgent(BaseAgent):

    def __init__(self, action_space_size: int, num_simulations: int = 200, exploration_constant: float = 1.41,
                 max_rollout_depth: int = 100, is_two_player: bool = False, seed: Optional[int] = None, **kwargs):

        super().__init__("MCTS_UCT", action_space_size)
        if seed is not None:
            np.random.seed(seed)
        self.num_simulations = num_simulations
        self.c = exploration_constant
        self.max_depth = max_rollout_depth
        self.is_two_player = is_two_player
        self._env = None

    def set_env(self, env):
        self._env = env
        # Détection automatique si l'env expose l'attribut
        if hasattr(env, 'is_two_player'):
            self.is_two_player = env.is_two_player

    # Sélection d'action
    def select_action(self, state: Any, valid_actions: Optional[np.ndarray] = None) -> int:
        if self._env is None:
            return (int(np.random.choice(valid_actions))
                    if valid_actions is not None else 0)

        root = MCTSNode(state=state)
        root.untried_actions = (
            valid_actions.copy() if valid_actions is not None
            else np.arange(self.action_space_size)
        )

        for _ in range(self.num_simulations):
            node = root
            sim_env = copy.deepcopy(self._env)
            reward_acc = 0.0

            # Descendre tant que le nœud est entièrement développé et
            # non-terminal. On RÉCUPÈRE la récompense de chaque step.
            while (node.is_fully_expanded()
                   and len(node.children) > 0
                   and not node.is_terminal):
                node = node.best_child(self.c)
                _, r, done, _ = sim_env.step(int(node.action))
                reward_acc += r
                if done:
                    node.is_terminal = True
                    break  # nœud terminal atteint : arrêter la descente

            # Expansion
            if (not node.is_terminal
                    and node.untried_actions is not None
                    and len(node.untried_actions) > 0):
                action = int(np.random.choice(node.untried_actions))
                node.untried_actions = node.untried_actions[
                    node.untried_actions != action
                    ]
                next_state, r, done, _ = sim_env.step(action)
                reward_acc += r
                child = MCTSNode(state=next_state, parent=node, action=action)
                child.is_terminal = done
                if not done:
                    try:
                        child.untried_actions = sim_env.get_valid_actions(next_state)
                    except Exception:
                        child.untried_actions = np.arange(self.action_space_size)
                else:
                    child.untried_actions = np.array([])
                node.children.append(child)
                node = child

            # Simulation rollout aléatoire
            if not node.is_terminal:
                reward_acc += self._rollout(sim_env)

            # ── 4. Rétropropagation ───────────────────────────────────
            current = node
            val = reward_acc
            while current is not None:
                current.visits += 1
                current.value += val
                if self.is_two_player:
                    val = -val  # inversion de perspective
                current = current.parent

        # Choisir l'action la plus visitée
        if len(root.children) == 0:
            return (int(np.random.choice(valid_actions))
                    if valid_actions is not None else 0)
        best = max(root.children, key=lambda ch: ch.visits)
        return int(best.action)

    # Rollout aléatoire
    def _rollout(self, env) -> float:
        total = 0.0
        for _ in range(self.max_depth):
            try:
                st = env._get_state() if hasattr(env, '_get_state') else None
                valid = env.get_valid_actions(st)
                a = int(np.random.choice(valid)) if len(valid) > 0 else 0
                _, r, done, _ = env.step(a)
                total += r
                if done:
                    break
            except Exception:
                break
        return total

    # Interface BaseAgent
    def learn(self, state, action, reward, next_state, done):
        pass  # Agent de planification pure, pas d'apprentissage

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath + '_config.pkl', 'wb') as f:
            pickle.dump({
                'num_simulations': self.num_simulations,
                'c': self.c,
                'max_depth': self.max_depth,
                'is_two_player': self.is_two_player,
            }, f)

    def load(self, filepath: str):
        with open(filepath + '_config.pkl', 'rb') as f:
            cfg = pickle.load(f)
        self.num_simulations = cfg['num_simulations']
        self.c = cfg['c']
        self.max_depth = cfg['max_depth']
        self.is_two_player = cfg.get('is_two_player', False)
