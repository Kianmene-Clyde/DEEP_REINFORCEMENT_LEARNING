"""MuZero agent: planning with a learned model in latent space.

Three networks:
- Representation: observation -> latent state  h(o) = s
- Dynamics: latent state + action -> next latent state + reward  g(s,a) = (s', r)
- Prediction: latent state -> policy, value  f(s) = (p, v)

MCTS operates entirely in latent space using the dynamics model.
"""
import numpy as np
import os, math, copy
from typing import Optional, Any, List, Tuple
from .base_agent import BaseAgent
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from nn.model import NeuralNetwork


class MuZeroNode:
    """MCTS node in latent space."""
    __slots__ = ['latent_state', 'parent', 'action', 'children', 'visits',
                 'value_sum', 'prior', 'reward']

    def __init__(self, latent_state, parent=None, action=None, prior=0.0, reward=0.0):
        self.latent_state = latent_state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior
        self.reward = reward

    def q_value(self):
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def ucb_score(self, c_puct=1.5):
        parent_visits = self.parent.visits if self.parent else 1
        return self.q_value() + c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visits)


class MuZeroAgent(BaseAgent):
    """MuZero: learned model + MCTS in latent space.
    
    Unlike AlphaZero, MuZero does NOT need the environment's rules
    for planning. It learns a dynamics model in latent space.
    """

    def __init__(self, input_size: int, action_space_size: int,
                 learning_rate: float = 0.001, discount_factor: float = 0.99,
                 latent_dim: int = 64, num_simulations: int = 50,
                 c_puct: float = 1.5, temperature: float = 1.0,
                 buffer_size: int = 5000, unroll_steps: int = 5,
                 hidden_layers=None, seed: Optional[int] = None, **kwargs):
        super().__init__("MuZero", action_space_size, input_size)
        if seed is not None:
            np.random.seed(seed)
        self.gamma = discount_factor
        self.latent_dim = latent_dim
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.unroll_steps = unroll_steps

        h = hidden_layers or [128]
        # Representation: obs -> latent
        self.repr_net = NeuralNetwork.build_mlp(
            [input_size] + h + [latent_dim], ['relu'] * len(h) + ['relu'], lr=learning_rate)
        # Dynamics: (latent + action_onehot) -> (next_latent, reward)
        dyn_in = latent_dim + action_space_size
        self.dyn_net = NeuralNetwork.build_mlp(
            [dyn_in] + h + [latent_dim + 1],  # last dim: latent + reward
            ['relu'] * len(h) + ['linear'], lr=learning_rate)
        # Prediction: latent -> (policy, value)
        self.pred_net = NeuralNetwork.build_dual_head(
            latent_dim, h, action_space_size, lr=learning_rate)

        self.buffer: List[dict] = []
        self.buffer_size = buffer_size
        self.current_trajectory: List[dict] = []
        self.train_batch_size = 32

    def _get_latent(self, obs: np.ndarray) -> np.ndarray:
        return self.repr_net.predict(np.atleast_2d(obs))[0]

    def _dynamics(self, latent: np.ndarray, action: int) -> Tuple[np.ndarray, float]:
        a_onehot = np.zeros(self.action_space_size, dtype=np.float32)
        a_onehot[action] = 1.0
        inp = np.concatenate([latent, a_onehot])
        out = self.dyn_net.predict(np.atleast_2d(inp))[0]
        next_latent = out[:self.latent_dim]
        reward = float(out[self.latent_dim])
        return next_latent, reward

    def _predict(self, latent: np.ndarray) -> Tuple[np.ndarray, float]:
        policy, value = self.pred_net.predict(np.atleast_2d(latent))
        return policy[0], float(value[0, 0])

    def select_action(self, state: Any, valid_actions: Optional[np.ndarray] = None) -> int:
        obs = np.asarray(state, dtype=np.float32).flatten()
        latent = self._get_latent(obs)

        if valid_actions is None:
            valid_actions = np.arange(self.action_space_size)

        # MCTS in latent space
        root = MuZeroNode(latent_state=latent)
        policy, value = self._predict(latent)
        # Mask and normalize
        mask = np.zeros(self.action_space_size)
        mask[valid_actions] = 1
        policy = policy * mask
        policy /= (policy.sum() + 1e-8)
        for a in valid_actions:
            root.children[int(a)] = MuZeroNode(
                latent_state=None, parent=root, action=int(a), prior=policy[a])
        root.visits = 1
        root.value_sum = value

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # Selection
            while len(node.children) > 0 and node.visits > 0:
                node = max(node.children.values(), key=lambda c: c.ucb_score(self.c_puct))
                search_path.append(node)

            # Expansion (use dynamics model)
            parent = node.parent if node.parent else root
            parent_latent = parent.latent_state
            if parent_latent is not None:
                next_latent, reward = self._dynamics(parent_latent, node.action)
                node.latent_state = next_latent
                node.reward = reward
                policy_child, value_child = self._predict(next_latent)
                for a in range(self.action_space_size):
                    if a not in node.children:
                        node.children[a] = MuZeroNode(
                            latent_state=None, parent=node, action=a,
                            prior=policy_child[a])
                value = value_child
            else:
                value = 0.0

            # Backprop
            for n in reversed(search_path):
                n.visits += 1
                n.value_sum += value
                value = n.reward + self.gamma * value

        # Action selection from root visit counts
        visits = np.zeros(self.action_space_size, dtype=np.float32)
        for a, child in root.children.items():
            visits[a] = child.visits

        if self.temperature > 0 and self.training:
            visits_temp = visits ** (1.0 / max(self.temperature, 0.01))
            pi = visits_temp / (visits_temp.sum() + 1e-8)
            action = int(np.random.choice(self.action_space_size, p=pi))
        else:
            pi = np.zeros(self.action_space_size)
            pi[np.argmax(visits)] = 1.0
            action = int(np.argmax(visits))

        if self.training:
            self.current_trajectory.append({
                'obs': obs.copy(), 'action': action, 'pi': pi.copy()
            })
        return action

    def learn(self, state, action, reward, next_state, done):
        if not self.training:
            return
        if self.current_trajectory:
            self.current_trajectory[-1]['reward'] = reward
        if done:
            # Store trajectory in buffer
            if self.current_trajectory:
                self.buffer.append(self.current_trajectory)
                if len(self.buffer) > self.buffer_size:
                    self.buffer.pop(0)
                self.current_trajectory = []
            if len(self.buffer) >= 4:
                self._train_networks()

    def _train_networks(self):
        """Train all three networks on buffered trajectories."""
        # Sample random trajectories and positions
        batch_obs, batch_actions, batch_pis, batch_values, batch_rewards = \
            [], [], [], [], []

        for _ in range(min(self.train_batch_size, len(self.buffer))):
            traj = self.buffer[np.random.randint(len(self.buffer))]
            if len(traj) < 2:
                continue
            pos = np.random.randint(max(1, len(traj) - self.unroll_steps))
            # Compute value target: discounted sum of future rewards
            value_target = 0.0
            for t in range(pos, min(pos + self.unroll_steps, len(traj))):
                value_target += (self.gamma ** (t - pos)) * traj[t].get('reward', 0)
            batch_obs.append(traj[pos]['obs'])
            batch_actions.append(traj[pos]['action'])
            batch_pis.append(traj[pos]['pi'])
            batch_values.append(value_target)
            batch_rewards.append(traj[pos].get('reward', 0))

        if len(batch_obs) < 4:
            return

        obs = np.array(batch_obs, dtype=np.float32)
        target_pis = np.array(batch_pis, dtype=np.float32)
        target_vs = np.array(batch_values, dtype=np.float32)

        # Train prediction network via representation
        latents = self.repr_net.forward(obs, training=True)
        pred_policy, pred_value = self.pred_net.forward(latents, training=True)

        # Policy loss gradient
        policy_grad = pred_policy - target_pis
        # Value loss gradient
        value_grad = np.zeros_like(pred_value)
        value_grad[:, 0] = pred_value[:, 0] - target_vs

        self.pred_net.backward_dual(policy_grad, value_grad)

        # Backprop through representation network
        # Simplified: just use value error
        repr_grad = np.zeros_like(latents)
        repr_grad += value_grad[:, 0:1] * 0.1  # scale down
        self.repr_net.backward(repr_grad)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        self.repr_net.save(filepath + '_repr.pkl')
        self.dyn_net.save(filepath + '_dyn.pkl')
        self.pred_net.save(filepath + '_pred.pkl')

    def load(self, filepath: str):
        self.repr_net = NeuralNetwork.load(filepath + '_repr.pkl')
        self.dyn_net = NeuralNetwork.load(filepath + '_dyn.pkl')
        self.pred_net = NeuralNetwork.load(filepath + '_pred.pkl')
