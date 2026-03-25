"""REINFORCE variants: vanilla, mean baseline, critic baseline."""
import numpy as np
import os, pickle
from typing import Optional, Any, List
from .base_agent import BaseAgent
from .utils import mask_and_normalize as _safe_probs
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from nn.model import NeuralNetwork


class REINFORCEAgent(BaseAgent):
    """Vanilla REINFORCE (Williams 1992). Policy gradient with Monte Carlo returns."""

    def __init__(self, input_size: int, action_space_size: int,
                 learning_rate: float = 0.001, discount_factor: float = 0.99,
                 hidden_layers=None, seed: Optional[int] = None, **kwargs):
        super().__init__("REINFORCE", action_space_size, input_size)
        if seed is not None:
            np.random.seed(seed)
        self.gamma = discount_factor
        h = hidden_layers or [128, 128]
        dims = [input_size] + h + [action_space_size]
        acts = ['relu'] * len(h) + ['softmax']
        self.policy_net = NeuralNetwork.build_mlp(dims, acts, lr=learning_rate)
        self.ep_states, self.ep_actions, self.ep_rewards = [], [], []

    def select_action(self, state: Any, valid_actions: Optional[np.ndarray] = None) -> int:
        probs = self.policy_net.predict(np.atleast_2d(state))[0]
        probs = _safe_probs(probs, valid_actions, self.action_space_size)
        if not self.training:
            return int(np.argmax(probs))
        action = int(np.random.choice(self.action_space_size, p=probs))
        self.ep_states.append(np.asarray(state, dtype=np.float32))
        self.ep_actions.append(action)
        return action

    def learn(self, state, action, reward, next_state, done):
        if not self.training:
            return
        self.ep_rewards.append(reward)
        if done:
            self._update_policy()

    def _compute_returns(self) -> np.ndarray:
        returns = np.zeros(len(self.ep_rewards))
        G = 0
        for t in reversed(range(len(self.ep_rewards))):
            G = self.ep_rewards[t] + self.gamma * G
            returns[t] = G
        return returns

    def _get_baseline(self, returns: np.ndarray) -> np.ndarray:
        """Override in subclasses."""
        return np.zeros_like(returns)

    def _update_policy(self):
        if len(self.ep_rewards) == 0:
            return
        returns = self._compute_returns()
        baseline = self._get_baseline(returns)
        advantages = returns - baseline
        # Normalize advantages
        if np.std(advantages) > 1e-8:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        states = np.array(self.ep_states, dtype=np.float32)
        actions = np.array(self.ep_actions)
        probs = self.policy_net.forward(states, training=True)

        # Policy gradient: -d/d(theta) sum_t [advantage_t * log pi(a_t|s_t)]
        grad = probs.copy()  # start from probs
        for i in range(len(actions)):
            grad[i, actions[i]] -= 1.0  # d softmax cross-entropy
            grad[i] *= -advantages[i]   # negative because we maximize
        self.policy_net.backward(grad)

        self.ep_states, self.ep_actions, self.ep_rewards = [], [], []

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        self.policy_net.save(filepath + '_policy.pkl')

    def load(self, filepath: str):
        self.policy_net = NeuralNetwork.load(filepath + '_policy.pkl')


class REINFORCEMeanBaselineAgent(REINFORCEAgent):
    """REINFORCE with running mean reward as baseline."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "REINFORCE+MeanBaseline"
        self._running_mean = 0.0
        self._episode_count = 0

    def _get_baseline(self, returns: np.ndarray) -> np.ndarray:
        self._episode_count += 1
        self._running_mean += (np.mean(returns) - self._running_mean) / self._episode_count
        return np.full_like(returns, self._running_mean)


class REINFORCECriticBaselineAgent(REINFORCEAgent):
    """REINFORCE with baseline learned by a critic (value network)."""

    def __init__(self, input_size: int, action_space_size: int,
                 learning_rate: float = 0.001, critic_lr: float = 0.005,
                 discount_factor: float = 0.99,
                 hidden_layers=None, seed: Optional[int] = None, **kwargs):
        super().__init__(input_size, action_space_size,
                         learning_rate=learning_rate,
                         discount_factor=discount_factor,
                         hidden_layers=hidden_layers, seed=seed)
        self.name = "REINFORCE+CriticBaseline"
        h = hidden_layers or [128, 128]
        dims = [input_size] + h + [1]
        acts = ['relu'] * len(h) + ['linear']
        self.value_net = NeuralNetwork.build_mlp(dims, acts, lr=critic_lr)

    def _get_baseline(self, returns: np.ndarray) -> np.ndarray:
        """Use value network as baseline and train it."""
        states = np.array(self.ep_states, dtype=np.float32)
        values = self.value_net.forward(states, training=True)[:, 0]
        # Train value net: MSE loss
        grad = np.zeros((len(states), 1), dtype=np.float32)
        grad[:, 0] = values - returns  # d/dv (v - G)^2 = 2(v - G), factor absorbed by lr
        self.value_net.backward(grad)
        return values

    def save(self, filepath: str):
        super().save(filepath)
        self.value_net.save(filepath + '_value.pkl')

    def load(self, filepath: str):
        super().load(filepath)
        self.value_net = NeuralNetwork.load(filepath + '_value.pkl')
