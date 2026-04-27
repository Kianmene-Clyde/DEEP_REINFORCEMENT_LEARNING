"""Double Q-Learning"""
import numpy as np
import os, pickle
from typing import Optional, Any
from .base_agent import BaseAgent
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from neural_network.model import NeuralNetwork
from training.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class DoubleDeepQLearningAgent(BaseAgent):
    """Double DQN sans replay buffer"""

    def __init__(self, input_size: int, action_space_size: int,
                 learning_rate: float = 0.001, discount_factor: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, target_update: int = 100,
                 hidden_layers=None, seed: Optional[int] = None, **kwargs):
        super().__init__("DoubleDeepQLearning", action_space_size, input_size)
        if seed is not None:
            np.random.seed(seed)
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update = target_update
        self.step_count = 0

        h = hidden_layers or [128, 128]
        dims = [input_size] + h + [action_space_size]
        acts = ['relu'] * len(h) + ['linear']
        self.q_net = NeuralNetwork.build_mlp(dims, acts, lr=learning_rate)
        self.target_net = self.q_net.copy()

    def select_action(self, state: Any, valid_actions: Optional[np.ndarray] = None) -> int:
        if self.training and np.random.random() < self.epsilon:
            if valid_actions is not None and len(valid_actions) > 0:
                return int(np.random.choice(valid_actions))
            return np.random.randint(0, self.action_space_size)
        q = self.q_net.predict(np.atleast_2d(state))[0]
        if valid_actions is not None:
            mask = np.full(self.action_space_size, -1e9)
            mask[valid_actions] = 0
            q = q + mask
        return int(np.argmax(q))

    def learn(self, state, action, reward, next_state, done):
        if not self.training:
            return
        self.step_count += 1
        s = np.atleast_2d(np.asarray(state, dtype=np.float32))
        ns = np.atleast_2d(np.asarray(next_state, dtype=np.float32))

        # Propagation avant avec training=True pour mettre en cache pour la rétropropagation
        q_vals = self.q_net.forward(s, training=True)
        # Utiliser des appels predict séparés (pas de cache nécessaire)
        best_a = int(np.argmax(self.q_net.predict(ns)[0]))
        q_target_val = self.target_net.predict(ns)[0, best_a]
        target = reward + (0 if done else self.gamma * q_target_val)

        grad = np.zeros_like(q_vals)
        grad[0, action] = q_vals[0, action] - target
        # backward utilise le cache de forward
        self.q_net.backward(grad)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if self.step_count % self.target_update == 0:
            self.target_net.copy_weights_from(self.q_net)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        self.q_net.save(filepath + '_qnet.pkl')
        with open(filepath + '_config.pkl', 'wb') as f:
            pickle.dump({'epsilon': self.epsilon, 'step_count': self.step_count}, f)

    def load(self, filepath: str):
        self.q_net = NeuralNetwork.load(filepath + '_qnet.pkl')
        self.target_net = self.q_net.copy()
        with open(filepath + '_config.pkl', 'rb') as f:
            cfg = pickle.load(f)
        self.epsilon = cfg['epsilon']
        self.step_count = cfg['step_count']


class DDQNWithERAgent(BaseAgent):
    """Double DQN avec experience replay."""

    def __init__(self, input_size: int, action_space_size: int,
                 learning_rate: float = 0.001, discount_factor: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, target_update: int = 100,
                 batch_size: int = 64, buffer_size: int = 50000,
                 hidden_layers=None, seed: Optional[int] = None, **kwargs):
        super().__init__("DDQN+ExperienceReplay", action_space_size, input_size)
        if seed is not None:
            np.random.seed(seed)
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update = target_update
        self.batch_size = batch_size
        self.step_count = 0

        h = hidden_layers or [128, 128]
        dims = [input_size] + h + [action_space_size]
        acts = ['relu'] * len(h) + ['linear']
        self.q_net = NeuralNetwork.build_mlp(dims, acts, lr=learning_rate)
        self.target_net = self.q_net.copy()
        self.buffer = ReplayBuffer(buffer_size)

    def select_action(self, state: Any, valid_actions: Optional[np.ndarray] = None) -> int:
        if self.training and np.random.random() < self.epsilon:
            if valid_actions is not None and len(valid_actions) > 0:
                return int(np.random.choice(valid_actions))
            return np.random.randint(0, self.action_space_size)
        q = self.q_net.predict(np.atleast_2d(state))[0]
        if valid_actions is not None:
            mask = np.full(self.action_space_size, -1e9)
            mask[valid_actions] = 0
            q = q + mask
        return int(np.argmax(q))

    def learn(self, state, action, reward, next_state, done):
        if not self.training:
            return
        self.buffer.push(np.asarray(state, dtype=np.float32), action, reward,
                         np.asarray(next_state, dtype=np.float32), float(done))
        self.step_count += 1
        if len(self.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        q_vals = self.q_net.forward(states, training=True)
        q_next_online = self.q_net.predict(next_states)
        q_next_target = self.target_net.predict(next_states)
        best_actions = np.argmax(q_next_online, axis=1)

        grad = np.zeros_like(q_vals)
        for i in range(self.batch_size):
            target = rewards[i] + (1 - dones[i]) * self.gamma * q_next_target[i, best_actions[i]]
            grad[i, actions[i]] = q_vals[i, actions[i]] - target
        self.q_net.backward(grad)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if self.step_count % self.target_update == 0:
            self.target_net.copy_weights_from(self.q_net)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        self.q_net.save(filepath + '_qnet.pkl')
        with open(filepath + '_config.pkl', 'wb') as f:
            pickle.dump({'epsilon': self.epsilon, 'step_count': self.step_count}, f)

    def load(self, filepath: str):
        self.q_net = NeuralNetwork.load(filepath + '_qnet.pkl')
        self.target_net = self.q_net.copy()
        with open(filepath + '_config.pkl', 'rb') as f:
            cfg = pickle.load(f)
        self.epsilon = cfg['epsilon']
        self.step_count = cfg['step_count']


class DDQNWithPERAgent(BaseAgent):
    """Double DQN avec Prioritized Experience Replay."""

    def __init__(self, input_size: int, action_space_size: int,
                 learning_rate: float = 0.001, discount_factor: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, target_update: int = 100,
                 batch_size: int = 64, buffer_size: int = 50000,
                 alpha: float = 0.6, beta: float = 0.4,
                 hidden_layers=None, seed: Optional[int] = None, **kwargs):
        super().__init__("DDQN+PrioritizedER", action_space_size, input_size)
        if seed is not None:
            np.random.seed(seed)
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update = target_update
        self.batch_size = batch_size
        self.step_count = 0

        h = hidden_layers or [128, 128]
        dims = [input_size] + h + [action_space_size]
        acts = ['relu'] * len(h) + ['linear']
        self.q_net = NeuralNetwork.build_mlp(dims, acts, lr=learning_rate)
        self.target_net = self.q_net.copy()
        self.buffer = PrioritizedReplayBuffer(buffer_size, alpha, beta)

    def select_action(self, state: Any, valid_actions: Optional[np.ndarray] = None) -> int:
        if self.training and np.random.random() < self.epsilon:
            if valid_actions is not None and len(valid_actions) > 0:
                return int(np.random.choice(valid_actions))
            return np.random.randint(0, self.action_space_size)
        q = self.q_net.predict(np.atleast_2d(state))[0]
        if valid_actions is not None:
            mask = np.full(self.action_space_size, -1e9)
            mask[valid_actions] = 0
            q = q + mask
        return int(np.argmax(q))

    def learn(self, state, action, reward, next_state, done):
        if not self.training:
            return
        self.buffer.push(np.asarray(state, dtype=np.float32), action, reward,
                         np.asarray(next_state, dtype=np.float32), float(done))
        self.step_count += 1
        if len(self.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones, indices, weights = \
            self.buffer.sample(self.batch_size)

        q_vals = self.q_net.forward(states, training=True)
        q_next_online = self.q_net.predict(next_states)
        q_next_target = self.target_net.predict(next_states)
        best_actions = np.argmax(q_next_online, axis=1)

        td_errors = np.zeros(self.batch_size)
        grad = np.zeros_like(q_vals)
        for i in range(self.batch_size):
            target = rewards[i] + (1 - dones[i]) * self.gamma * q_next_target[i, best_actions[i]]
            td_errors[i] = q_vals[i, actions[i]] - target
            grad[i, actions[i]] = td_errors[i] * weights[i]
        self.q_net.backward(grad)

        self.buffer.update_priorities(indices, td_errors)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if self.step_count % self.target_update == 0:
            self.target_net.copy_weights_from(self.q_net)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        self.q_net.save(filepath + '_qnet.pkl')
        with open(filepath + '_config.pkl', 'wb') as f:
            pickle.dump({'epsilon': self.epsilon, 'step_count': self.step_count}, f)

    def load(self, filepath: str):
        self.q_net = NeuralNetwork.load(filepath + '_qnet.pkl')
        self.target_net = self.q_net.copy()
        with open(filepath + '_config.pkl', 'rb') as f:
            cfg = pickle.load(f)
        self.epsilon = cfg['epsilon']
        self.step_count = cfg['step_count']
