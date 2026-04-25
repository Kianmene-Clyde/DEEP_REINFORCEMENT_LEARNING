import numpy as np
import os
from typing import Optional, Any
from .base_agent import BaseAgent
from .utils import mask_and_normalize as _safe_probs
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from neural_network.model import NeuralNetwork


class A2CAgent(BaseAgent):

    def __init__(self, input_size: int, action_space_size: int, learning_rate: float = 0.0007,
                 discount_factor: float = 0.99, entropy_coef: float = 0.01, tmax: int = 20, hidden_layers=None,
                 seed: Optional[int] = None, **kwargs):

        super().__init__("A2C", action_space_size, input_size)
        if seed is not None:
            np.random.seed(seed)

        self.gamma = discount_factor
        self.entropy_coef = entropy_coef
        self.tmax = tmax

        h = hidden_layers or [128, 128]

        # Réseau actor — sortie softmax
        dims_a = [input_size] + h + [action_space_size]
        acts_a = ['relu'] * len(h) + ['softmax']
        self.actor = NeuralNetwork.build_mlp(dims_a, acts_a, lr=learning_rate)

        # Réseau critic — sortie linéaire
        dims_c = [input_size] + h + [1]
        acts_c = ['relu'] * len(h) + ['linear']
        self.critic = NeuralNetwork.build_mlp(dims_c, acts_c, lr=learning_rate)

        # Buffers de l'épisode courant
        self.ep_states = []
        self.ep_actions = []
        self.ep_rewards = []
        self._last_done = False

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
        self.ep_rewards.append(float(reward))
        self._last_done = bool(done)
        self._last_next_state = np.asarray(next_state, dtype=np.float32)

        # Mise à jour toutes les tmax steps ou en fin d'épisode
        if done or len(self.ep_rewards) >= self.tmax:
            self._update()

    def _compute_returns(self) -> np.ndarray:
        T = len(self.ep_rewards)
        returns = np.zeros(T, dtype=np.float32)

        if self._last_done:
            R = 0.0  # épisode terminal : pas de bootstrap
        else:
            # Bootstrap sur la valeur du dernier état non-terminal
            R = float(
                self.critic.predict(
                    np.atleast_2d(self._last_next_state)
                )[0, 0]
            )

        for t in reversed(range(T)):
            R = self.ep_rewards[t] + self.gamma * R
            returns[t] = R
        return returns

    def _update(self):
        if len(self.ep_rewards) == 0:
            return

        states = np.array(self.ep_states, dtype=np.float32)
        actions = np.array(self.ep_actions, dtype=np.int32)
        returns = self._compute_returns()

        # critic
        # Forward avec cache pour backward
        values = self.critic.forward(states, training=True)[:, 0]
        advantages = returns - values

        # Gradient MSE
        cgrad = np.zeros((len(states), 1), dtype=np.float32)
        cgrad[:, 0] = values - returns
        self.critic.backward(cgrad)

        # Acteur
        # Forward avec cache pour backward
        probs = self.actor.forward(states, training=True)

        # Gradient de la loss acteur :
        agrad = probs.copy()
        for i in range(len(actions)):
            agrad[i, actions[i]] -= 1.0
            agrad[i] *= advantages[i]

        eps = 1e-8
        H = -np.sum(probs * np.log(probs + eps), axis=1, keepdims=True)
        entropy_grad = self.entropy_coef * probs * (np.log(probs + eps) + H)
        agrad += entropy_grad
        self.actor.backward(agrad)

        # Réinitialisation des buffers
        self.ep_states, self.ep_actions, self.ep_rewards = [], [], []

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        self.actor.save(filepath + '_actor.pkl')
        self.critic.save(filepath + '_critic.pkl')

    def load(self, filepath: str):
        self.actor = NeuralNetwork.load(filepath + '_actor.pkl')
        self.critic = NeuralNetwork.load(filepath + '_critic.pkl')
