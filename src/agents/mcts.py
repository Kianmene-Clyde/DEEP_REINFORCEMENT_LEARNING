"""Monte Carlo Tree Search (UCT) agent."""
import numpy as np
import copy
import math
from typing import Optional, Any, Dict, List
from .base_agent import BaseAgent


class MCTSNode:
    """Node in the MCTS tree."""
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
        return self.untried_actions is not None and len(self.untried_actions) == 0


class MCTSAgent(BaseAgent):
    """Monte Carlo Tree Search with Upper Confidence Bound (UCT).
    
    For each decision:
    1. Selection: traverse tree using UCB1
    2. Expansion: add new child node
    3. Simulation: random rollout from new node
    4. Backpropagation: update values up the tree
    
    Requires environment to support deep copy.
    """

    def __init__(self, action_space_size: int, num_simulations: int = 200,
                 exploration_constant: float = 1.41,
                 max_rollout_depth: int = 100,
                 seed: Optional[int] = None, **kwargs):
        super().__init__("MCTS_UCT", action_space_size)
        if seed is not None:
            np.random.seed(seed)
        self.num_simulations = num_simulations
        self.c = exploration_constant
        self.max_depth = max_rollout_depth
        self._env = None

    def set_env(self, env):
        self._env = env

    def select_action(self, state: Any, valid_actions: Optional[np.ndarray] = None) -> int:
        if self._env is None:
            return int(np.random.choice(valid_actions)) if valid_actions is not None else 0

        root = MCTSNode(state=state)
        root.untried_actions = valid_actions.copy() if valid_actions is not None \
            else np.arange(self.action_space_size)

        for _ in range(self.num_simulations):
            node = root
            sim_env = copy.deepcopy(self._env)

            # Selection
            while node.is_fully_expanded() and len(node.children) > 0 and not node.is_terminal:
                node = node.best_child(self.c)
                _, _, done, _ = sim_env.step(int(node.action))
                if done:
                    node.is_terminal = True

            # Expansion
            if not node.is_terminal and node.untried_actions is not None and len(node.untried_actions) > 0:
                action = int(np.random.choice(node.untried_actions))
                node.untried_actions = node.untried_actions[node.untried_actions != action]
                next_state, reward, done, _ = sim_env.step(action)
                child = MCTSNode(state=next_state, parent=node, action=action)
                child.is_terminal = done
                try:
                    child.untried_actions = sim_env.get_valid_actions(next_state)
                except Exception:
                    child.untried_actions = np.arange(self.action_space_size)
                node.children.append(child)
                node = child
                reward_acc = reward
            else:
                reward_acc = 0.0

            # Simulation (random rollout)
            if not node.is_terminal:
                reward_acc += self._rollout(sim_env)

            # Backpropagation
            while node is not None:
                node.visits += 1
                node.value += reward_acc
                node = node.parent

        # Pick best action (most visited)
        if len(root.children) == 0:
            return int(np.random.choice(valid_actions)) if valid_actions is not None else 0
        best = max(root.children, key=lambda c: c.visits)
        return int(best.action)

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

    def learn(self, state, action, reward, next_state, done):
        pass  # Pure planning agent

    def save(self, filepath: str):
        import pickle, os
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath + '_config.pkl', 'wb') as f:
            pickle.dump({'num_simulations': self.num_simulations,
                         'c': self.c, 'max_depth': self.max_depth}, f)

    def load(self, filepath: str):
        import pickle
        with open(filepath + '_config.pkl', 'rb') as f:
            cfg = pickle.load(f)
        self.num_simulations = cfg['num_simulations']
        self.c = cfg['c']
        self.max_depth = cfg['max_depth']
