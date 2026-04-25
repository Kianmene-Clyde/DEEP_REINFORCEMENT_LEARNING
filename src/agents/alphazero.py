import numpy as np
import os
import copy
import math
from typing import Optional, Any, List, Tuple

from .base_agent import BaseAgent
from .utils import mask_and_normalize as _safe_probs
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from neural_network.model import NeuralNetwork


# Nœud MCTS
class AlphaZeroNode:
    __slots__ = ['state', 'parent', 'action', 'children', 'visits',
                 'value_sum', 'prior', 'is_terminal', 'valid_actions', 'expanded']

    def __init__(self, state=None, parent=None, action=None, prior: float = 0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior
        self.is_terminal = False
        self.valid_actions = None
        self.expanded = False

    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def ucb_score(self, c_puct: float = 1.5) -> float:
        parent_visits = self.parent.visits if self.parent else 1
        return (self.q_value()
                + c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visits))


# Agent AlphaZero
class AlphaZeroAgent(BaseAgent):

    def __init__(self, input_size: int, action_space_size: int, learning_rate: float = 0.001,
                 num_simulations: int = 800, c_puct: float = 1.5, dirichlet_alpha: float = 0.3,
                 dirichlet_epsilon: float = 0.25, temperature: float = 1.0, buffer_size: int = 500_000,
                 train_batch_size: int = 4096, hidden_layers=None, seed: Optional[int] = None, **kwargs):

        super().__init__("AlphaZero", action_space_size, input_size)
        if seed is not None:
            np.random.seed(seed)

        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temperature = temperature
        self.train_batch_size = train_batch_size

        h = hidden_layers or [128, 128]
        self.net = NeuralNetwork.build_dual_head(
            input_size, h, action_space_size, lr=learning_rate
        )

        self.buffer: List[Tuple] = []
        self.buffer_size = buffer_size

        # Données de l'épisode courant
        self.current_episode_data: List[Tuple] = []
        self._env = None
        self._is_two_player = False

    def set_env(self, env):
        self._env = env
        self._is_two_player = getattr(env, 'is_two_player', False)

    # Prédiction du réseau de neurone
    def neural_network_predict(self, state_flat: np.ndarray):
        s = np.atleast_2d(state_flat.astype(np.float32))
        policy, value = self.net.predict(s)
        return policy[0], float(value[0, 0])

    # Sélection d'action via MCTS
    def select_action(self, state: Any, valid_actions: Optional[np.ndarray] = None) -> int:
        state_flat = np.asarray(state, dtype=np.float32).flatten()

        # Mode évaluation : greedy sur la politique du réseau
        if self._env is None or not self.training:
            p = _safe_probs(
                self.neural_network_predict(state_flat)[0], valid_actions, self.action_space_size
            )
            return int(np.argmax(p))

        if valid_actions is None:
            valid_actions = np.arange(self.action_space_size)

        # Initialisation de la racine
        root = AlphaZeroNode(state=state_flat)
        root.valid_actions = valid_actions

        policy, root_value = self.neural_network_predict(state_flat)
        policy = _safe_probs(policy, valid_actions, self.action_space_size)

        noise = np.random.dirichlet(
            [self.dirichlet_alpha] * len(valid_actions)
        )
        for idx, a in enumerate(valid_actions):
            noisy_prior = (
                    (1 - self.dirichlet_epsilon) * policy[a]
                    + self.dirichlet_epsilon * noise[idx]
            )
            root.children[int(a)] = AlphaZeroNode(
                parent=root, action=int(a), prior=noisy_prior
            )
        root.expanded = True
        root.visits = 1
        root.value_sum = root_value

        # num_simulations pour les simulations MCTS
        for _ in range(self.num_simulations):
            node = root
            sim_env = copy.deepcopy(self._env)

            # Sélection : descendre selon UCB
            while node.expanded and len(node.children) > 0 and not node.is_terminal:
                node = max(
                    node.children.values(),
                    key=lambda c: c.ucb_score(self.c_puct)
                )
                next_state, _, done, _ = sim_env.step(int(node.action))
                if done:
                    node.is_terminal = True
                if node.state is None:
                    node.state = np.asarray(next_state, dtype=np.float32).flatten()

            # Évaluation de la feuille
            if node.is_terminal:
                value = 0.0
            elif not node.expanded:
                policy_child, value = self.neural_network_predict(node.state)
                try:
                    st = sim_env._get_state()
                    va = sim_env.get_valid_actions(st)
                except Exception:
                    va = np.arange(self.action_space_size)
                node.valid_actions = va
                policy_child = _safe_probs(policy_child, va, self.action_space_size)
                for a in va:
                    if int(a) not in node.children:
                        node.children[int(a)] = AlphaZeroNode(
                            parent=node, action=int(a), prior=policy_child[a]
                        )
                node.expanded = True
            else:
                value = 0.0

            # Rétropropagation
            current = node
            while current is not None:
                current.visits += 1
                current.value_sum += value
                if self._is_two_player:
                    value = -value
                current = current.parent

        # Politique MCTS π à partir des visites
        visits = np.zeros(self.action_space_size, dtype=np.float32)
        for a, child in root.children.items():
            visits[a] = child.visits

        if self.temperature > 0:
            visits_temp = visits ** (1.0 / max(self.temperature, 0.01))
            pi = visits_temp / (visits_temp.sum() + 1e-8)
        else:
            pi = np.zeros(self.action_space_size)
            pi[np.argmax(visits)] = 1.0

        # On stocke l'état, π_MCTS, index_joueur pour les cibles de valeur
        player_idx = getattr(self._env, 'current_player', 0)
        self.current_episode_data.append((state_flat.copy(), pi.copy(), player_idx))

        pi_safe = _safe_probs(pi, valid_actions, self.action_space_size)
        action = int(np.random.choice(self.action_space_size, p=pi_safe))
        return action

    def learn(self, state, action, reward, next_state, done):
        if not self.training:
            return
        if not done:
            return

        final_outcome = float(reward)

        for s, pi, player_idx in self.current_episode_data:
            if self._is_two_player:
                # Le joueur qui a fait le dernier coup a reçu `final_outcome`.
                # Les positions du même joueur ont z = final_outcome,
                # celles de l'adversaire ont z = -final_outcome.
                last_player = getattr(self._env, 'last_player', player_idx)
                if player_idx == last_player:
                    z = final_outcome
                else:
                    z = -final_outcome
            else:
                # Jeu solo : z = résultat final, pas de discount
                z = final_outcome

            self.buffer.append((s, pi, z))
            if len(self.buffer) > self.buffer_size:
                self.buffer.pop(0)

        self.current_episode_data = []

        if len(self.buffer) >= self.train_batch_size:
            self._train_network()

    # Entraînement du réseau
    def _train_network(self):
        n = len(self.buffer)
        indices = np.random.choice(n, min(n, self.train_batch_size), replace=False)
        batch = [self.buffer[i] for i in indices]

        states = np.array([b[0] for b in batch], dtype=np.float32)
        target_pis = np.array([b[1] for b in batch], dtype=np.float32)
        target_vs = np.array([b[2] for b in batch], dtype=np.float32)

        policy_pred, value_pred = self.net.forward(states, training=True)

        # Gradient de la cross-entropie
        policy_grad = policy_pred - target_pis

        # Gradient de la MSE
        # (facteur 2 absorbé dans le learning rate)
        value_grad = np.zeros_like(value_pred)
        value_grad[:, 0] = value_pred[:, 0] - target_vs

        self.net.backward_dual(policy_grad, value_grad)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        self.net.save(filepath + '_net.pkl')

    def load(self, filepath: str):
        self.net = NeuralNetwork.load(filepath + '_net.pkl')
