"""DQN variants with legal Bellman targets and explicit replay semantics."""

import os
import pickle
import numpy as np
from .base_agent import BaseAgent
from .utils import action_mask
from ..neural_network.model import NeuralNetwork
from ..training.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class DeepQLearningAgent(BaseAgent):
    def __init__(
        self,
        input_size,
        action_space_size,
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        batch_size=64,
        buffer_size=50000,
        target_update=100,
        hidden_layers=None,
        seed=None,
        double=False,
        use_replay=True,
        prioritized=False,
        alpha=0.6,
        beta=0.4,
        **kwargs,
    ):
        super().__init__("DQN", action_space_size, input_size)
        if batch_size < 1 or target_update < 1:
            raise ValueError("batch_size and target_update must be positive")
        self.rng = np.random.default_rng(seed)
        self.gamma = discount_factor
        self.epsilon, self.epsilon_decay, self.epsilon_min = epsilon, epsilon_decay, epsilon_min
        self.batch_size, self.target_update, self.step_count = batch_size, target_update, 0
        self.double, self.use_replay, self.prioritized = double, use_replay, prioritized
        self.is_value_based = True
        h = [128, 128] if hidden_layers is None else hidden_layers
        self.q_net = NeuralNetwork.build_mlp(
            [input_size] + h + [action_space_size],
            ["relu"] * len(h) + ["linear"],
            lr=learning_rate,
            rng=self.rng,
        )
        self.target_net = self.q_net.copy()
        self.buffer = (
            PrioritizedReplayBuffer(buffer_size, alpha, beta, rng=self.rng)
            if prioritized
            else ReplayBuffer(buffer_size, rng=self.rng)
        )

    def select_action(self, state, valid_actions=None):
        mask = action_mask(valid_actions, self.action_space_size)
        if self.training and self.rng.random() < self.epsilon:
            return int(self.rng.choice(np.flatnonzero(mask)))
        q = self.q_net.predict(np.atleast_2d(state))[0]
        return int(np.argmax(np.where(mask, q, -np.inf)))

    def bellman_targets(self, rewards, next_states, terminated, masks):
        target_q = self.target_net.predict(next_states)
        selection_q = self.q_net.predict(next_states) if self.double else target_q
        best = np.argmax(np.where(masks, selection_q, -np.inf), axis=1)
        bootstrap = target_q[np.arange(len(rewards)), best]
        bootstrap = np.where(np.asarray(terminated) | ~masks.any(axis=1), 0.0, bootstrap)
        return rewards + self.gamma * bootstrap

    def learn(self, state, action, reward, next_state, done):
        if not self.training:
            return
        mask = self.next_action_mask()
        self.step_count += 1
        if self.use_replay:
            self.buffer.push(state, action, reward, next_state, done, mask)
            if len(self.buffer) >= self.batch_size:
                self._train_step()
        else:
            self._update_batch(
                np.atleast_2d(state),
                np.array([action]),
                np.array([reward]),
                np.atleast_2d(next_state),
                np.array([done]),
                mask[None, :],
            )
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if self.step_count % self.target_update == 0:
            self.target_net.copy_weights_from(self.q_net)

    def _train_step(self):
        batch = self.buffer.sample(self.batch_size, include_masks=True)
        td = self._update_batch(*batch[:6], weights=batch[7] if self.prioritized else None)
        if self.prioritized:
            self.buffer.update_priorities(batch[6], td)

    def _update_batch(self, states, actions, rewards, next_states, dones, masks, weights=None):
        targets = self.bellman_targets(rewards, next_states, dones, masks)
        q = self.q_net.forward(states, training=True)
        rows = np.arange(len(actions))
        td = q[rows, actions] - targets
        grad = np.zeros_like(q)
        grad[rows, actions] = td if weights is None else td * weights
        self.q_net.backward(grad)
        return td

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        self.q_net.save(filepath + "_qnet.pkl")
        with open(filepath + "_config.pkl", "wb") as f:
            pickle.dump({"epsilon": self.epsilon, "step_count": self.step_count}, f)

    def load(self, filepath):
        net = NeuralNetwork.load(filepath + "_qnet.pkl")
        if (
            net.layers[0]["W"].shape[0] != self.input_size
            or net.layers[-1]["b"].size != self.action_space_size
        ):
            raise ValueError("Incompatible Q-network dimensions")
        self.q_net, self.target_net = net, net.copy()
        with open(filepath + "_config.pkl", "rb") as f:
            config = pickle.load(f)
        self.epsilon, self.step_count = config["epsilon"], config["step_count"]
