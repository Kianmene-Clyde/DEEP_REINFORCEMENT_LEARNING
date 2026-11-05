"""Expert Apprenti"""

import numpy as np
import os
from typing import Optional, Any
from .base_agent import BaseAgent
from .utils import mask_and_normalize as _safe_probs, action_mask, masked_batch
from .mcts import MCTSAgent

from ..neural_network.model import NeuralNetwork


class ExpertApprenticeAgent(BaseAgent):
    def __init__(
        self,
        input_size: int,
        action_space_size: int,
        learning_rate: float = 0.001,
        expert_simulations: int = 100,
        batch_size: int = 64,
        hidden_layers=None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        super().__init__("ExpertApprentice", action_space_size, input_size)
        self.rng = np.random.default_rng(seed)

        h = hidden_layers or [128, 128]
        dims = [input_size] + h + [action_space_size]
        acts = ["relu"] * len(h) + ["softmax"]
        self.policy_net = NeuralNetwork.build_mlp(dims, acts, lr=learning_rate, rng=self.rng)

        self.expert = MCTSAgent(
            action_space_size, num_simulations=expert_simulations, seed=int(self.rng.integers(2**32))
        )
        self.batch_size = batch_size
        self.data_states = []
        self.data_actions = []
        self.data_masks = []
        self._env = None
        self._use_expert = True  # Commencer avec l'expert, puis passer à l'apprenti

    def set_env(self, env):
        self._env = env
        self.expert.set_env(env)

    def select_action(self, state: Any, valid_actions: Optional[np.ndarray] = None) -> int:
        if self.training and self._use_expert and self._env is not None:
            # Utiliser l'expert MCTS et collecter les données
            self.expert.set_env(self._env)
            action = self.expert.select_action(state, valid_actions)
            self.data_states.append(np.asarray(state, dtype=np.float32).flatten())
            self.data_actions.append(action)
            self.data_masks.append(action_mask(valid_actions, self.action_space_size))
            return action
        else:
            # Utiliser le réseau de l'apprenti
            probs = self.policy_net.predict(np.atleast_2d(state))[0]
            probs = _safe_probs(probs, valid_actions, self.action_space_size)
            if self.training:
                return int(self.rng.choice(self.action_space_size, p=probs))
            return int(np.argmax(probs))

    def learn(self, state, action, reward, next_state, done):
        if not self.training:
            return
        if done and len(self.data_states) >= self.batch_size:
            self._train_apprentice()

    def _train_apprentice(self):
        """Entraîner l'apprenti sur les données collectées par l'expert."""
        n = len(self.data_states)
        if n < self.batch_size:
            return
        # Échantillonner un lot
        indices = self.rng.choice(n, min(n, self.batch_size * 4), replace=False)
        states = np.array([self.data_states[i] for i in indices], dtype=np.float32)
        actions = np.array([self.data_actions[i] for i in indices])

        # Perte d'entropie croisée
        probs = self.policy_net.forward(states, training=True)
        probs = masked_batch(probs, np.array([self.data_masks[i] for i in indices]))
        grad = probs.copy()
        for i in range(len(actions)):
            grad[i, actions[i]] -= 1.0
        self.policy_net.backward(grad)

        # Après avoir collecté assez de données d'entraînement, commencer à mélanger les décisions de l'apprenti
        if n > 1000:
            self._use_expert = self.rng.random() < max(0.1, 1.0 - n / 5000)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        self.policy_net.save(filepath + "_policy.pkl")
        import pickle

        with open(filepath + "_config.pkl", "wb") as f:
            pickle.dump({"use_expert": self._use_expert, "data_size": len(self.data_states)}, f)

    def load(self, filepath: str):
        self.policy_net = NeuralNetwork.load(filepath + "_policy.pkl")
        self._use_expert = False  # Après le chargement, utiliser l'apprenti
