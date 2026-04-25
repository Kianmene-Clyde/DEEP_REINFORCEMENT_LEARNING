import numpy as np
import os, math
from typing import Optional, Any, List, Tuple
from .muzero import MuZeroAgent, MuZeroNode
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from neural_network.model import NeuralNetwork


class StochasticMuZeroAgent(MuZeroAgent):

    def __init__(self, input_size: int, action_space_size: int,
                 learning_rate: float = 0.001, num_chance_outcomes: int = 8,
                 **kwargs):

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
        log_var = np.clip(out[ld:2 * ld], -5, 2)
        reward = float(out[2 * ld])
        chance_logits = out[2 * ld + 1:]

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

    def _train_networks(self):
        """Override: train stochastic dynamics (mean, log_var, reward, chance)."""
        batch_obs, batch_actions, batch_pis, batch_values, batch_rewards = \
            [], [], [], [], []
        batch_next_obs = []

        for _ in range(min(self.train_batch_size, len(self.buffer))):
            traj = self.buffer[np.random.randint(len(self.buffer))]
            if len(traj) < 2:
                continue
            pos = np.random.randint(max(1, len(traj) - self.K))
            value_target = 0.0
            for t in range(pos, min(pos + self.K, len(traj))):
                value_target += (self.gamma ** (t - pos)) * traj[t].get('reward', 0)
            batch_obs.append(traj[pos]['obs'])
            batch_actions.append(traj[pos]['action'])
            batch_pis.append(traj[pos]['pi'])
            batch_values.append(value_target)
            batch_rewards.append(traj[pos].get('reward', 0))
            if pos + 1 < len(traj):
                batch_next_obs.append(traj[pos + 1]['obs'])
            else:
                batch_next_obs.append(traj[pos]['obs'])

        if len(batch_obs) < 4:
            return

        obs = np.array(batch_obs, dtype=np.float32)
        target_pis = np.array(batch_pis, dtype=np.float32)
        target_vs = np.array(batch_values, dtype=np.float32)
        actions = np.array(batch_actions)
        rewards = np.array(batch_rewards, dtype=np.float32)
        next_obs = np.array(batch_next_obs, dtype=np.float32)

        max_abs_v = max(np.abs(target_vs).max(), 1e-8)
        target_vs_norm = np.clip(target_vs / max_abs_v, -1, 1)

        # === Train prediction network via representation ===
        latents = self.repr_net.forward(obs, training=True)
        pred_policy, pred_value = self.pred_net.forward(latents, training=True)

        policy_grad = pred_policy - target_pis
        value_grad = np.zeros_like(pred_value)
        value_grad[:, 0] = pred_value[:, 0] - target_vs_norm

        self.pred_net.backward_dual(policy_grad, value_grad)

        repr_grad = np.zeros_like(latents)
        repr_grad += value_grad * 0.1
        self.repr_net.backward(repr_grad)

        # === Train stochastic dynamics network ===
        next_latents_target = self.repr_net.predict(next_obs)
        a_onehot = np.zeros((len(actions), self.action_space_size), dtype=np.float32)
        a_onehot[np.arange(len(actions)), actions] = 1.0
        dyn_input = np.concatenate([latents, a_onehot], axis=1)

        # Target for stochastic dynamics: [mean_target, log_var_target(zeros), reward, chance_uniform]
        ld = self.latent_dim
        batch_size = len(actions)
        dyn_target = np.zeros((batch_size, ld * 2 + 1 + self.num_chance), dtype=np.float32)
        dyn_target[:, :ld] = next_latents_target  # mean target = next latent
        dyn_target[:, ld:2 * ld] = 0.0  # log_var target = 0 (low uncertainty)
        dyn_target[:, 2 * ld] = rewards  # reward target
        # chance target: uniform (no supervision signal for chance)
        dyn_target[:, 2 * ld + 1:] = 1.0 / self.num_chance

        dyn_pred = self.dyn_net.forward(dyn_input, training=True)
        dyn_grad = dyn_pred - dyn_target
        self.dyn_net.backward(dyn_grad)

    def save(self, filepath: str):
        super().save(filepath)
        self.chance_net.save(filepath + '_chance.pkl')

    def load(self, filepath: str):
        super().load(filepath)
        self.chance_net = NeuralNetwork.load(filepath + '_chance.pkl')
