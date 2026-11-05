"""AlphaZero-inspired policy/value search against a fixed opponent, without self-play."""

from collections import deque
from pathlib import Path
import numpy as np
from .base_agent import BaseAgent
from .search import tree_search, visit_policy
from .utils import action_mask, masked_batch
from ..neural_network.model import NeuralNetwork


class AlphaZeroAgent(BaseAgent):
    def __init__(
        self,
        input_size,
        action_space_size,
        learning_rate=0.001,
        num_simulations=30,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        temperature=1.0,
        buffer_size=50000,
        train_batch_size=64,
        hidden_layers=None,
        seed=None,
        discount_factor=0.99,
        max_rollout_depth=50,
        **kwargs,
    ):
        super().__init__("AlphaZero-inspired (fixed opponent)", action_space_size, input_size)
        self.rng = np.random.default_rng(seed)
        self.num_simulations, self.c_puct, self.max_depth = num_simulations, c_puct, max_rollout_depth
        self.dirichlet_alpha, self.dirichlet_epsilon = dirichlet_alpha, dirichlet_epsilon
        self.temperature, self.train_batch_size, self.gamma = temperature, train_batch_size, discount_factor
        h = [128, 128] if hidden_layers is None else hidden_layers
        self.net = NeuralNetwork.build_dual_head(
            input_size, h, action_space_size, lr=learning_rate, rng=self.rng
        )
        self.buffer = deque(maxlen=buffer_size)
        self.current_episode_data = []
        self._env = None

    def set_env(self, env):
        self._env = env

    def neural_network_predict(self, state):
        policy, value = self.net.predict(np.atleast_2d(state))
        return policy[0], float(value[0, 0])

    def select_action(self, state, valid_actions=None):
        if self._env is None:
            raise RuntimeError("AlphaZero-inspired search requires set_env(env)")
        valid = self._env.get_valid_actions(state) if valid_actions is None else valid_actions
        noise = (self.dirichlet_alpha, self.dirichlet_epsilon) if self.training else None
        visits, _ = tree_search(
            self._env,
            self.rng,
            self.num_simulations,
            self.max_depth,
            self.gamma,
            self.c_puct,
            self.neural_network_predict,
            noise,
        )
        target = visit_policy(visits, valid, 1.0)
        if self.training:
            self.current_episode_data.append(
                [np.asarray(state).copy(), target, action_mask(valid, self.action_space_size), 0.0]
            )
        policy = visit_policy(visits, valid, self.temperature if self.training else 0.0)
        return int(self.rng.choice(self.action_space_size, p=policy))

    def learn(self, state, action, reward, next_state, done):
        if not self.training:
            return
        if not self.current_episode_data:
            raise RuntimeError("select_action must precede learn")
        self.current_episode_data[-1][3] = float(reward)
        if done:
            value = self.neural_network_predict(next_state)[1] if self._transition_truncated else 0.0
            for s, pi, mask, r in reversed(self.current_episode_data):
                value = r + self.gamma * value
                self.buffer.append((s, pi, mask, value))
            self.current_episode_data = []
            if len(self.buffer) >= self.train_batch_size:
                self._train_network()

    def _train_network(self):
        indices = self.rng.choice(
            len(self.buffer), min(len(self.buffer), self.train_batch_size), replace=False
        )
        data = [self.buffer[i] for i in indices]
        states = np.array([r[0] for r in data], dtype=np.float32)
        targets = np.array([r[1] for r in data])
        masks = np.array([r[2] for r in data])
        values = np.array([r[3] for r in data])
        policy, value = self.net.forward(states, training=True)
        policy = masked_batch(policy, masks)
        self.net.backward_dual(policy - targets, (value[:, 0] - values)[:, None])

    def save(self, filepath):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.net.save(filepath + "_net.pkl")

    def load(self, filepath):
        self.net = NeuralNetwork.load(filepath + "_net.pkl")
