"""AlphaZero: self-play MCTS guided by a dual-head neural network."""
import numpy as np
import os, copy, math
from typing import Optional, Any, List, Tuple
from .base_agent import BaseAgent
from .utils import mask_and_normalize as _safe_probs
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from nn.model import NeuralNetwork


class AlphaZeroNode:
    __slots__ = ['state', 'parent', 'action', 'children', 'visits',
                 'value_sum', 'prior', 'is_terminal', 'valid_actions', 'expanded']

    def __init__(self, state=None, parent=None, action=None, prior=0.0):
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

    def q_value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def ucb_score(self, c_puct=1.5):
        parent_visits = self.parent.visits if self.parent else 1
        return self.q_value() + c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visits)


class AlphaZeroAgent(BaseAgent):
    """AlphaZero: MCTS + dual-head NN + self-play."""

    def __init__(self, input_size: int, action_space_size: int,
                 learning_rate: float = 0.001, num_simulations: int = 100,
                 c_puct: float = 1.5, temperature: float = 1.0,
                 buffer_size: int = 10000,
                 hidden_layers=None, seed: Optional[int] = None, **kwargs):
        super().__init__("AlphaZero", action_space_size, input_size)
        if seed is not None:
            np.random.seed(seed)
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature

        h = hidden_layers or [128, 128]
        self.net = NeuralNetwork.build_dual_head(input_size, h, action_space_size, lr=learning_rate)

        self.buffer: List[Tuple] = []
        self.buffer_size = buffer_size
        self.current_episode_data: List[Tuple] = []
        self.current_episode_rewards: List[float] = []
        self._env = None
        self._is_two_player = False
        self.train_batch_size = 64
        self.gamma = 0.99

    def set_env(self, env):
        self._env = env
        self._is_two_player = getattr(env, 'is_two_player', False)

    def _nn_predict(self, state_flat: np.ndarray):
        """Get policy and value from network."""
        s = np.atleast_2d(state_flat.astype(np.float32))
        policy, value = self.net.predict(s)
        return policy[0], float(value[0, 0])

    def select_action(self, state: Any, valid_actions: Optional[np.ndarray] = None) -> int:
        state_flat = np.asarray(state, dtype=np.float32).flatten()

        if self._env is None or not self.training:
            p = _safe_probs(self._nn_predict(state_flat)[0], valid_actions, self.action_space_size)
            return int(np.argmax(p))

        if valid_actions is None:
            valid_actions = np.arange(self.action_space_size)

        root = AlphaZeroNode(state=state_flat)
        root.valid_actions = valid_actions

        # Expand root
        policy, root_value = self._nn_predict(state_flat)
        policy = _safe_probs(policy, valid_actions, self.action_space_size)
        for a in valid_actions:
            root.children[int(a)] = AlphaZeroNode(parent=root, action=int(a), prior=policy[a])
        root.expanded = True
        root.visits = 1
        root.value_sum = root_value

        for _ in range(self.num_simulations):
            node = root
            sim_env = copy.deepcopy(self._env)

            # Selection: descend the tree
            while node.expanded and len(node.children) > 0 and not node.is_terminal:
                node = max(node.children.values(), key=lambda c: c.ucb_score(self.c_puct))
                next_state, _, done, _ = sim_env.step(int(node.action))
                if done:
                    node.is_terminal = True
                if node.state is None:
                    node.state = np.asarray(next_state, dtype=np.float32).flatten()

            # Evaluate leaf
            if node.is_terminal:
                value = 0.0
            elif not node.expanded:
                # Expand this node
                policy_child, value = self._nn_predict(node.state)
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
                            parent=node, action=int(a), prior=policy_child[a])
                node.expanded = True
            else:
                value = 0.0

            # Backpropagation
            while node is not None:
                node.visits += 1
                node.value_sum += value
                if self._is_two_player:
                    value = -value  # flip for 2-player only
                node = node.parent

        # Build MCTS policy from visit counts
        visits = np.zeros(self.action_space_size, dtype=np.float32)
        for a, child in root.children.items():
            visits[a] = child.visits

        if self.temperature > 0:
            visits_temp = visits ** (1.0 / max(self.temperature, 0.01))
            pi = visits_temp / (visits_temp.sum() + 1e-8)
        else:
            pi = np.zeros(self.action_space_size)
            pi[np.argmax(visits)] = 1.0

        self.current_episode_data.append((state_flat.copy(), pi.copy()))

        if self.training:
            pi_safe = _safe_probs(pi, valid_actions, self.action_space_size)
            action = int(np.random.choice(self.action_space_size, p=pi_safe))
        else:
            action = int(np.argmax(pi))
        return action

    def learn(self, state, action, reward, next_state, done):
        if not self.training:
            return
        self.current_episode_rewards.append(reward)
        if done:
            # Compute discounted returns for value targets
            rewards = self.current_episode_rewards
            if self._is_two_player:
                # For games: all positions get the final outcome
                value_target = reward
                for s, pi in self.current_episode_data:
                    self.buffer.append((s, pi, value_target))
                    if len(self.buffer) > self.buffer_size:
                        self.buffer.pop(0)
            else:
                # For single-player: compute discounted returns per step
                T = len(rewards)
                returns = np.zeros(T)
                G = 0.0
                for t in reversed(range(T)):
                    G = rewards[t] + self.gamma * G
                    returns[t] = G
                # Normalize returns to [-1, 1] range for tanh value head
                max_abs = max(abs(returns.max()), abs(returns.min()), 1e-8)
                returns_norm = np.clip(returns / max_abs, -1, 1)
                for i, (s, pi) in enumerate(self.current_episode_data):
                    if i < T:
                        self.buffer.append((s, pi, float(returns_norm[i])))
                        if len(self.buffer) > self.buffer_size:
                            self.buffer.pop(0)
            self.current_episode_data = []
            self.current_episode_rewards = []
            if len(self.buffer) >= self.train_batch_size:
                self._train_network()

    def _train_network(self):
        n = len(self.buffer)
        indices = np.random.choice(n, min(n, self.train_batch_size), replace=False)
        batch = [self.buffer[i] for i in indices]
        states = np.array([b[0] for b in batch], dtype=np.float32)
        target_pis = np.array([b[1] for b in batch], dtype=np.float32)
        target_vs = np.array([b[2] for b in batch], dtype=np.float32)

        policy_pred, value_pred = self.net.forward(states, training=True)
        policy_grad = policy_pred - target_pis
        value_grad = np.zeros_like(value_pred)
        value_grad[:, 0] = value_pred[:, 0] - target_vs
        self.net.backward_dual(policy_grad, value_grad)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        self.net.save(filepath + '_net.pkl')

    def load(self, filepath: str):
        self.net = NeuralNetwork.load(filepath + '_net.pkl')
