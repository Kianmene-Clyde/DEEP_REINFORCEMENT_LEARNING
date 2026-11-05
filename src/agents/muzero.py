"""MuZero-inspired latent planning with exact NumPy K-step backpropagation.

This compact fixed-opponent prototype uses scalar rewards/values, a learned
termination probability and Monte Carlo value targets. It is not a reproduction
of the original large-scale MuZero training/reanalysis protocol.
"""

from collections import deque
from pathlib import Path
import copy
import math
import numpy as np
from .base_agent import BaseAgent
from .utils import action_mask, mask_and_normalize
from .search import visit_policy
from ..neural_network.model import NeuralNetwork


class MinMaxStats:
    def __init__(self):
        self.minimum, self.maximum = float("inf"), float("-inf")

    def update(self, value):
        self.minimum, self.maximum = min(self.minimum, value), max(self.maximum, value)

    def normalize(self, value):
        return (value - self.minimum) / (self.maximum - self.minimum) if self.maximum > self.minimum else 0.0


class MuZeroNode:
    def __init__(self, latent_state, parent=None, action=None, prior=0.0, reward=0.0):
        self.latent_state, self.parent, self.action = latent_state, parent, action
        self.prior, self.reward, self.termination = prior, reward, 0.0
        self.children = {}
        self.visits, self.value_sum = 0, 0.0

    def q_value(self):
        return self.value_sum / self.visits if self.visits else 0.0


class MuZeroAgent(BaseAgent):
    def __init__(
        self,
        input_size,
        action_space_size,
        learning_rate=0.001,
        discount_factor=0.99,
        latent_dim=64,
        num_simulations=50,
        c1=1.25,
        c2=19652,
        temperature=1.0,
        buffer_size=5000,
        unroll_steps=5,
        hidden_layers=None,
        seed=None,
        train_batch_size=32,
        max_search_depth=30,
        **kwargs,
    ):
        super().__init__("MuZero-inspired (scalar prototype)", action_space_size, input_size)
        if min(num_simulations, latent_dim, unroll_steps, train_batch_size, max_search_depth) < 1:
            raise ValueError("Model and search budgets must be positive")
        self.rng = np.random.default_rng(seed)
        self.gamma, self.latent_dim, self.K = discount_factor, latent_dim, unroll_steps
        self.num_simulations, self.c1, self.c2 = num_simulations, c1, c2
        self.temperature, self.max_search_depth = temperature, max_search_depth
        self.train_batch_size = train_batch_size
        h = [128] if hidden_layers is None else hidden_layers
        self.repr_net = NeuralNetwork.build_mlp(
            [input_size] + h + [latent_dim], ["relu"] * len(h) + ["tanh"], lr=learning_rate, rng=self.rng
        )
        self.dyn_net = NeuralNetwork.build_mlp(
            [latent_dim + action_space_size] + h + [latent_dim + 2],
            ["relu"] * len(h) + ["linear"],
            lr=learning_rate,
            rng=self.rng,
        )
        self.pred_net = NeuralNetwork.build_dual_head(
            latent_dim, h, action_space_size, lr=learning_rate, rng=self.rng
        )
        self.buffer = deque(maxlen=buffer_size)
        self.current_trajectory = []

    def _get_latent(self, obs):
        return self.repr_net.predict(np.atleast_2d(obs))[0]

    def _predict(self, latent):
        policy, value = self.pred_net.predict(np.atleast_2d(latent))
        return policy[0], float(value[0, 0])

    def _model_step(self, latent, action):
        onehot = np.eye(self.action_space_size, dtype=np.float32)[action]
        out = self.dyn_net.predict(np.concatenate([latent, onehot])[None, :])[0]
        return (
            np.tanh(out[: self.latent_dim]),
            float(out[self.latent_dim]),
            float(NeuralNetwork._activate(out[-1], "sigmoid")),
        )

    def _dynamics(self, latent, action):
        state, reward, _ = self._model_step(latent, action)
        return state, reward

    def select_action(self, state, valid_actions=None):
        valid = np.flatnonzero(action_mask(valid_actions, self.action_space_size))
        root = MuZeroNode(self._get_latent(np.asarray(state, dtype=np.float32)))
        policy, _ = self._predict(root.latent_state)
        policy = mask_and_normalize(policy, valid, self.action_space_size)
        if self.training:
            policy[valid] = 0.75 * policy[valid] + 0.25 * self.rng.dirichlet(np.full(len(valid), 0.25))
        for a in valid:
            root.children[int(a)] = MuZeroNode(None, root, int(a), float(policy[a]))
        stats = MinMaxStats()
        for _ in range(self.num_simulations):
            node, path = root, [root]
            for _ in range(self.max_search_depth):
                if not node.children:
                    break

                def score(child):
                    q = child.reward + self.gamma * (1 - child.termination) * child.q_value()
                    pb = self.c1 + math.log((node.visits + self.c2 + 1) / self.c2)
                    return stats.normalize(q) + pb * child.prior * math.sqrt(node.visits + 1) / (
                        child.visits + 1
                    )

                node = max(node.children.values(), key=score)
                path.append(node)
                if node.latent_state is None:
                    latent, reward, term = self._model_step(node.parent.latent_state, node.action)
                    node.latent_state, node.reward, node.termination = latent, reward, term
                    policy, value = self._predict(latent)
                    for a in range(self.action_space_size):
                        node.children[a] = MuZeroNode(None, node, a, float(policy[a]))
                    break
                if node.termination > 0.99:
                    break
            value = self._predict(node.latent_state)[1]
            for current in reversed(path):
                current.visits += 1
                current.value_sum += value
                q = current.reward + self.gamma * (1 - current.termination) * value
                stats.update(q)
                value = q
        visits = np.zeros(self.action_space_size, dtype=int)
        for a, child in root.children.items():
            visits[a] = child.visits
        target = visit_policy(visits, valid, 1.0)
        pi = visit_policy(visits, valid, self.temperature if self.training else 0.0)
        action = int(self.rng.choice(self.action_space_size, p=pi))
        if self.training:
            self.current_trajectory.append(
                {
                    "obs": np.asarray(state).copy(),
                    "action": action,
                    "pi": target,
                    "mask": action_mask(valid, self.action_space_size),
                }
            )
        return action

    def learn(self, state, action, reward, next_state, done):
        if not self.training:
            return
        if not self.current_trajectory:
            raise RuntimeError("select_action must precede learn")
        self.current_trajectory[-1].update(
            reward=float(reward),
            next_obs=np.asarray(next_state).copy(),
            terminated=bool(done and not self._transition_truncated),
        )
        if done:
            tail = self._predict(self._get_latent(next_state))[1] if self._transition_truncated else 0.0
            for item in reversed(self.current_trajectory):
                tail = item["reward"] + self.gamma * tail
                item["value_target"] = tail
            self.buffer.append(self.current_trajectory)
            self.current_trajectory = []
            if len(self.buffer) >= 4:
                self._train_networks()

    def _compute_value_target(self, traj, t):
        return float(traj[t]["value_target"])

    @staticmethod
    def _accumulate(total, gradients):
        for i, g in gradients.items():
            if i not in total:
                total[i] = {k: v.copy() for k, v in g.items()}
            else:
                for key in g:
                    total[i][key] += g[key]

    def _sample_positions(self):
        data = []
        for _ in range(self.train_batch_size):
            trajectory = self.buffer[int(self.rng.integers(len(self.buffer)))]
            data.append((trajectory, int(self.rng.integers(len(trajectory)))))
        return data

    def _train_networks(self):
        data = self._sample_positions()
        latent = self.repr_net.forward(np.array([tr[t]["obs"] for tr, t in data]), training=True)
        predictions, dynamics = [], []
        scale = 1.0 / (self.K + 1)
        for k in range(self.K + 1):
            pi, value = self.pred_net.forward(latent, training=True)
            valid = np.array([t + k < len(tr) for tr, t in data], dtype=float)
            targets = np.array(
                [tr[t + k]["pi"] if t + k < len(tr) else np.zeros(self.action_space_size) for tr, t in data]
            )
            vs = np.array([tr[t + k]["value_target"] if t + k < len(tr) else 0.0 for tr, t in data])
            predictions.append(
                (
                    copy.deepcopy(self.pred_net._cache),
                    (pi - targets) * valid[:, None] * scale,
                    (value[:, 0] - vs)[:, None] * valid[:, None] * scale,
                )
            )
            if k == self.K:
                break
            acts = np.array([tr[t + k]["action"] if t + k < len(tr) else 0 for tr, t in data])
            inputs = np.concatenate([latent, np.eye(self.action_space_size)[acts]], axis=1)
            out = self.dyn_net.forward(inputs, training=True)
            next_latent = np.tanh(out[:, : self.latent_dim])
            reward_targets = np.array([tr[t + k]["reward"] if t + k < len(tr) else 0.0 for tr, t in data])
            term_targets = np.array([tr[t + k]["terminated"] if t + k < len(tr) else True for tr, t in data])
            grad = np.zeros_like(out)
            grad[:, self.latent_dim] = (out[:, self.latent_dim] - reward_targets) * valid * scale
            grad[:, -1] = (NeuralNetwork._activate(out[:, -1], "sigmoid") - term_targets) * valid * scale
            dynamics.append((copy.deepcopy(self.dyn_net._cache), next_latent, grad))
            latent = next_latent
        pred_grads, dyn_grads = {}, {}
        carry = np.zeros_like(latent)
        for k in reversed(range(self.K + 1)):
            cache, pg, vg = predictions[k]
            self.pred_net._cache = cache
            grads, dx = self.pred_net.dual_gradients(pg, vg)
            self._accumulate(pred_grads, grads)
            carry = carry + dx
            if k > 0:
                cache, next_latent, grad = dynamics[k - 1]
                grad[:, : self.latent_dim] = carry * (1 - next_latent**2)
                self.dyn_net._cache = cache
                grads, dx = self.dyn_net.gradients(grad)
                self._accumulate(dyn_grads, grads)
                carry = dx[:, : self.latent_dim]
        repr_grads, _ = self.repr_net.gradients(carry)
        self.repr_net.apply_gradients(repr_grads)
        self.pred_net.apply_gradients(pred_grads)
        self.dyn_net.apply_gradients(dyn_grads)

    def save(self, filepath):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        for suffix in ["repr", "dyn", "pred"]:
            getattr(self, suffix + "_net").save(filepath + "_" + suffix + ".pkl")

    def load(self, filepath):
        for suffix in ["repr", "dyn", "pred"]:
            setattr(self, suffix + "_net", NeuralNetwork.load(filepath + "_" + suffix + ".pkl"))
        if self.dyn_net.layers[-1]["b"].size != self.latent_dim + 2:
            raise ValueError("Legacy dynamics checkpoint requires retraining")
