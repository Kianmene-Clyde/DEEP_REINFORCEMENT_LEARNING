"""PPO A2C-style: Proximal Policy Optimization with clipped objective."""
import numpy as np
import os, pickle
from typing import Optional, Any
from .base_agent import BaseAgent
from .utils import mask_and_normalize as _safe_probs
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from nn.model import NeuralNetwork


class PPOAgent(BaseAgent):
    """PPO with clipped surrogate objective (Schulman et al., 2017).
    
    A2C-style: collects a full episode, then does multiple gradient epochs
    with the PPO clipped loss to prevent too-large policy updates.
    
    Key idea: ratio = pi_new(a|s) / pi_old(a|s)
    Loss = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
    """

    def __init__(self, input_size: int, action_space_size: int,
                 learning_rate: float = 0.0003, discount_factor: float = 0.99,
                 gae_lambda: float = 0.95, clip_ratio: float = 0.2,
                 num_epochs: int = 4, entropy_coef: float = 0.01,
                 hidden_layers=None, seed: Optional[int] = None, **kwargs):
        super().__init__("PPO_A2C", action_space_size, input_size)
        if seed is not None:
            np.random.seed(seed)
        self.gamma = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.num_epochs = num_epochs
        self.entropy_coef = entropy_coef

        h = hidden_layers or [128, 128]
        # Actor (policy)
        dims_a = [input_size] + h + [action_space_size]
        acts_a = ['relu'] * len(h) + ['softmax']
        self.actor = NeuralNetwork.build_mlp(dims_a, acts_a, lr=learning_rate)
        # Critic (value)
        dims_c = [input_size] + h + [1]
        acts_c = ['relu'] * len(h) + ['linear']
        self.critic = NeuralNetwork.build_mlp(dims_c, acts_c, lr=learning_rate * 3)

        self.ep_states = []
        self.ep_actions = []
        self.ep_rewards = []
        self.ep_log_probs = []  # log pi_old(a|s)
        self.ep_values = []

    def select_action(self, state: Any, valid_actions: Optional[np.ndarray] = None) -> int:
        s = np.atleast_2d(np.asarray(state, dtype=np.float32))
        probs = self.actor.predict(s)[0]
        value = self.critic.predict(s)[0, 0]
        probs = _safe_probs(probs, valid_actions, self.action_space_size)

        if not self.training:
            return int(np.argmax(probs))

        action = int(np.random.choice(self.action_space_size, p=probs))
        self.ep_states.append(np.asarray(state, dtype=np.float32))
        self.ep_actions.append(action)
        self.ep_log_probs.append(np.log(probs[action] + 1e-8))
        self.ep_values.append(value)
        return action

    def learn(self, state, action, reward, next_state, done):
        if not self.training:
            return
        self.ep_rewards.append(reward)
        if done:
            self._update()

    def _compute_gae(self, rewards, values):
        """Generalized Advantage Estimation."""
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            next_value = values[t]
        return returns, advantages

    def _update(self):
        if len(self.ep_rewards) == 0:
            return
        states = np.array(self.ep_states, dtype=np.float32)
        actions = np.array(self.ep_actions)
        old_log_probs = np.array(self.ep_log_probs, dtype=np.float32)
        values = np.array(self.ep_values, dtype=np.float32)
        rewards = np.array(self.ep_rewards, dtype=np.float32)

        returns, advantages = self._compute_gae(rewards, values)
        if np.std(advantages) > 1e-8:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        for _ in range(self.num_epochs):
            # -- Actor update (PPO clipped) --
            new_probs = self.actor.forward(states, training=True)
            new_log_probs = np.log(new_probs[np.arange(len(actions)), actions] + 1e-8)
            ratio = np.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = np.clip(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            # We want to maximize min(surr1, surr2), so gradient is negative
            # Entropy bonus: -sum(p*log(p))
            entropy = -np.sum(new_probs * np.log(new_probs + 1e-8), axis=1)

            # Gradient computation
            actor_grad = new_probs.copy()
            for i in range(len(actions)):
                # Use surr1 if it's the min, otherwise clipped has zero gradient
                if surr1[i] <= surr2[i]:
                    weight = advantages[i]
                else:
                    # Clipped: gradient is zero if ratio is outside [1-eps, 1+eps]
                    if 1 - self.clip_ratio <= ratio[i] <= 1 + self.clip_ratio:
                        weight = advantages[i]
                    else:
                        weight = 0.0
                actor_grad[i, actions[i]] -= 1.0
                actor_grad[i] *= weight  # correct sign: weight * (p - e_a)
                # Entropy gradient (encourage exploration - subtract to minimize)
                actor_grad[i] += self.entropy_coef * (np.log(new_probs[i] + 1e-8) + 1)
            self.actor.backward(actor_grad)

            # -- Critic update --
            pred_values = self.critic.forward(states, training=True)
            critic_grad = np.zeros_like(pred_values)
            critic_grad[:, 0] = pred_values[:, 0] - returns
            self.critic.backward(critic_grad)

        self.ep_states, self.ep_actions, self.ep_rewards = [], [], []
        self.ep_log_probs, self.ep_values = [], []

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        self.actor.save(filepath + '_actor.pkl')
        self.critic.save(filepath + '_critic.pkl')

    def load(self, filepath: str):
        self.actor = NeuralNetwork.load(filepath + '_actor.pkl')
        self.critic = NeuralNetwork.load(filepath + '_critic.pkl')
