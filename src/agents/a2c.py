"""A2C (Advantage Actor-Critic) agent."""
import numpy as np
import os
from typing import Optional, Any
from .base_agent import BaseAgent
from .utils import mask_and_normalize as _safe_probs
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from nn.model import NeuralNetwork


class A2CAgent(BaseAgent):
    """Advantage Actor-Critic (A2C).
    
    Unlike PPO, updates happen at the end of each episode without clipping.
    Uses TD(0) or n-step returns for the advantage estimate.
    Actor loss: -log pi(a|s) * A(s,a)
    Critic loss: (V(s) - G_t)^2
    """

    def __init__(self, input_size: int, action_space_size: int,
                 learning_rate: float = 0.001, discount_factor: float = 0.99,
                 entropy_coef: float = 0.01,
                 hidden_layers=None, seed: Optional[int] = None, **kwargs):
        super().__init__("A2C", action_space_size, input_size)
        if seed is not None:
            np.random.seed(seed)
        self.gamma = discount_factor
        self.entropy_coef = entropy_coef

        h = hidden_layers or [128, 128]
        dims_a = [input_size] + h + [action_space_size]
        acts_a = ['relu'] * len(h) + ['softmax']
        self.actor = NeuralNetwork.build_mlp(dims_a, acts_a, lr=learning_rate)
        dims_c = [input_size] + h + [1]
        acts_c = ['relu'] * len(h) + ['linear']
        self.critic = NeuralNetwork.build_mlp(dims_c, acts_c, lr=learning_rate * 3)

        self.ep_states, self.ep_actions, self.ep_rewards = [], [], []

    def select_action(self, state: Any, valid_actions: Optional[np.ndarray] = None) -> int:
        s = np.atleast_2d(np.asarray(state, dtype=np.float32))
        probs = self.actor.predict(s)[0]
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
            self._update()

    def _update(self):
        if len(self.ep_rewards) == 0:
            return
        states = np.array(self.ep_states, dtype=np.float32)
        actions = np.array(self.ep_actions)
        # Compute returns
        returns = np.zeros(len(self.ep_rewards), dtype=np.float32)
        G = 0.0
        for t in reversed(range(len(self.ep_rewards))):
            G = self.ep_rewards[t] + self.gamma * G
            returns[t] = G

        # Critic: compute values and advantage
        values = self.critic.forward(states, training=True)[:, 0]
        advantages = returns - values
        if np.std(advantages) > 1e-8:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # Critic loss
        cgrad = np.zeros((len(states), 1), dtype=np.float32)
        cgrad[:, 0] = values - returns
        self.critic.backward(cgrad)

        # Actor loss
        probs = self.actor.forward(states, training=True)
        agrad = probs.copy()
        for i in range(len(actions)):
            agrad[i, actions[i]] -= 1.0
            agrad[i] *= -advantages[i]
            agrad[i] -= self.entropy_coef * (-np.log(probs[i] + 1e-8) - 1)
        self.actor.backward(agrad)

        self.ep_states, self.ep_actions, self.ep_rewards = [], [], []

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        self.actor.save(filepath + '_actor.pkl')
        self.critic.save(filepath + '_critic.pkl')

    def load(self, filepath: str):
        self.actor = NeuralNetwork.load(filepath + '_actor.pkl')
        self.critic = NeuralNetwork.load(filepath + '_critic.pkl')
