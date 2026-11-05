"""Experimental Gaussian latent model, not the published Stochastic MuZero.

One-step Gaussian NLL learns conditional variance from observed successors.
There is no discrete chance encoder, afterstate tree or claim of equivalence
to the original algorithm. Planning averages sampled latent outcomes.
"""

import numpy as np
from .muzero import MuZeroAgent
from .utils import action_mask
from ..neural_network.model import NeuralNetwork


class StochasticMuZeroAgent(MuZeroAgent):
    def __init__(self, input_size, action_space_size, learning_rate=0.001, num_chance_outcomes=8, **kwargs):
        super().__init__(input_size, action_space_size, learning_rate=learning_rate, **kwargs)
        self.name = "Gaussian latent planning (experimental)"
        self.num_chance = max(2, num_chance_outcomes)
        h = kwargs.get("hidden_layers") or [128]
        self.dyn_net = NeuralNetwork.build_mlp(
            [self.latent_dim + action_space_size] + h + [2 * self.latent_dim + 2],
            ["relu"] * len(h) + ["linear"],
            lr=learning_rate,
            rng=self.rng,
        )

    def _model_step(self, latent, action):
        inp = np.r_[latent, np.eye(self.action_space_size)[action]][None, :]
        out = self.dyn_net.predict(inp)[0]
        ld = self.latent_dim
        logvar = np.clip(out[ld : 2 * ld], -5, 2)
        state = np.tanh(out[:ld] + np.exp(0.5 * logvar) * self.rng.standard_normal(ld))
        return state, float(out[-2]), float(NeuralNetwork._activate(out[-1], "sigmoid"))

    def select_action(self, state, valid_actions=None):
        valid = np.flatnonzero(action_mask(valid_actions, self.action_space_size))
        latent = self._get_latent(state)
        scores = np.zeros(self.action_space_size)
        samples = max(2, self.num_simulations // len(valid))
        for a in valid:
            for _ in range(samples):
                nxt, r, term = self._model_step(latent, int(a))
                scores[a] += (r + self.gamma * (1 - term) * self._predict(nxt)[1]) / samples
        if self.training:
            logits = scores[valid] - scores[valid].max()
            weights = np.exp(logits / max(self.temperature, 0.1))
            pi = np.zeros(self.action_space_size)
            pi[valid] = weights / weights.sum()
            pi = 0.9 * pi + 0.1 * action_mask(valid, self.action_space_size) / len(valid)
            action = int(self.rng.choice(self.action_space_size, p=pi))
            self.current_trajectory.append(
                {
                    "obs": np.asarray(state).copy(),
                    "action": action,
                    "pi": pi,
                    "mask": action_mask(valid, self.action_space_size),
                }
            )
        else:
            action = int(valid[np.argmax(scores[valid])])
        return action

    def _train_networks(self):
        data = self._sample_positions()
        items = [traj[t] for traj, t in data]
        obs = np.array([i["obs"] for i in items])
        next_obs = np.array([i["next_obs"] for i in items])
        latent = self.repr_net.forward(obs, training=True)
        policy, value = self.pred_net.forward(latent, training=True)
        pi = np.array([i["pi"] for i in items])
        targets = np.array([i["value_target"] for i in items])
        prediction_grads, repr_dx = self.pred_net.dual_gradients(
            policy - pi, (value[:, 0] - targets)[:, None]
        )
        # Frozen next-representation target, including terminal observed successors.
        next_latent = self.repr_net.predict(next_obs)
        next_raw = np.arctanh(np.clip(next_latent, -0.999, 0.999))
        actions = np.array([i["action"] for i in items])
        inputs = np.concatenate([latent, np.eye(self.action_space_size)[actions]], axis=1)
        output = self.dyn_net.forward(inputs, training=True)
        ld = self.latent_dim
        logvar = np.clip(output[:, ld : 2 * ld], -5, 2)
        variance = np.exp(logvar)
        delta = output[:, :ld] - next_raw
        grad = np.zeros_like(output)
        grad[:, :ld] = delta / variance / ld
        active = (output[:, ld : 2 * ld] > -5) & (output[:, ld : 2 * ld] < 2)
        grad[:, ld : 2 * ld] = 0.5 * (1 - delta**2 / variance) * active / ld
        grad[:, -2] = output[:, -2] - np.array([i["reward"] for i in items])
        grad[:, -1] = NeuralNetwork._activate(output[:, -1], "sigmoid") - np.array(
            [i["terminated"] for i in items]
        )
        dyn_grads, dyn_dx = self.dyn_net.gradients(grad)
        repr_grads, _ = self.repr_net.gradients(repr_dx + dyn_dx[:, :ld])
        self.repr_net.apply_gradients(repr_grads)
        self.pred_net.apply_gradients(prediction_grads)
        self.dyn_net.apply_gradients(dyn_grads)

    def load(self, filepath):
        for suffix in ["repr", "dyn", "pred"]:
            setattr(self, suffix + "_net", NeuralNetwork.load(filepath + "_" + suffix + ".pkl"))
        if self.dyn_net.layers[-1]["b"].size != 2 * self.latent_dim + 2:
            raise ValueError("Legacy stochastic prototype requires retraining")
