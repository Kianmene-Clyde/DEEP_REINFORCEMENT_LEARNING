"""MuZero Stochastique: MuZero with stochastic transitions.

Extension of MuZero that handles environments with random elements
(e.g., opponent moves, dice rolls). The dynamics model outputs a
distribution over possible next latent states, and we sample
from it during MCTS to account for stochasticity.

Reference: Antonoglou et al., 2021 - 
  "Planning in Stochastic Environments with a Learned Model"
"""
import numpy as np
import os, math
from typing import Optional, Any, List, Tuple
from .muzero import MuZeroAgent, MuZeroNode
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from nn.model import NeuralNetwork


class StochasticMuZeroAgent(MuZeroAgent):
    """MuZero with stochastic transitions.
    
    Key differences from deterministic MuZero:
    1. Dynamics model outputs mean + log_var for the latent state
       (diagonal Gaussian), plus reward.
    2. During MCTS, we sample from this distribution.
    3. An "afterstate" representation captures the stochastic transition.
    
    This handles environments where the opponent or nature makes random moves.
    """

    def __init__(self, input_size: int, action_space_size: int,
                 learning_rate: float = 0.001, num_chance_outcomes: int = 8,
                 **kwargs):
        """
        Args:
            num_chance_outcomes: Number of discrete chance outcomes to model.
                The dynamics model additionally predicts which chance outcome occurs.
        """
        self.num_chance = num_chance_outcomes
        super().__init__(input_size, action_space_size, learning_rate=learning_rate, **kwargs)
        self.name = "MuZero_Stochastic"

        # Override dynamics network: output = latent_dim (mean) + latent_dim (log_var) + 1 (reward) + num_chance (chance logits)
        h = kwargs.get('hidden_layers', [128])
        dyn_in = self.latent_dim + action_space_size
        dyn_out = self.latent_dim * 2 + 1 + num_chance_outcomes
        self.dyn_net = NeuralNetwork.build_mlp(
            [dyn_in] + h + [dyn_out],
            ['relu'] * len(h) + ['linear'],
            lr=learning_rate)

        # Encoder for chance outcomes (maps chance_onehot + afterstate -> next latent)
        chance_in = self.latent_dim + num_chance_outcomes
        self.chance_net = NeuralNetwork.build_mlp(
            [chance_in] + h + [self.latent_dim],
            ['relu'] * len(h) + ['relu'],
            lr=learning_rate)

    def _dynamics(self, latent: np.ndarray, action: int) -> Tuple[np.ndarray, float]:
        """Stochastic dynamics: predict mean/var, sample, apply chance."""
        a_onehot = np.zeros(self.action_space_size, dtype=np.float32)
        a_onehot[action] = 1.0
        inp = np.concatenate([latent, a_onehot])
        out = self.dyn_net.predict(np.atleast_2d(inp))[0]

        ld = self.latent_dim
        mean = out[:ld]
        log_var = np.clip(out[ld:2*ld], -5, 2)
        reward = float(out[2*ld])
        chance_logits = out[2*ld+1:]

        # Sample latent (reparameterization trick)
        std = np.exp(0.5 * log_var)
        afterstate = mean + std * np.random.randn(ld).astype(np.float32)

        # Sample chance outcome
        chance_probs = np.exp(chance_logits - np.max(chance_logits))
        chance_probs /= chance_probs.sum() + 1e-8
        chance_idx = np.random.choice(self.num_chance, p=chance_probs)
        chance_onehot = np.zeros(self.num_chance, dtype=np.float32)
        chance_onehot[chance_idx] = 1.0

        # Apply chance to get final next latent
        chance_inp = np.concatenate([afterstate, chance_onehot])
        next_latent = self.chance_net.predict(np.atleast_2d(chance_inp))[0]

        return next_latent, reward

    def save(self, filepath: str):
        super().save(filepath)
        self.chance_net.save(filepath + '_chance.pkl')

    def load(self, filepath: str):
        super().load(filepath)
        self.chance_net = NeuralNetwork.load(filepath + '_chance.pkl')
