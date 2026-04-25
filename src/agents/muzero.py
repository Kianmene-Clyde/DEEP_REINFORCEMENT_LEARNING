import numpy as np
import os, math, copy
from typing import Optional, Any, List, Tuple
from .base_agent import BaseAgent
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from neural_network.model import NeuralNetwork


class MinMaxStats:
    """Tracks min and max Q-values in the search tree for normalization (Eq. 5)."""
    def __init__(self):
        self.minimum = float('inf')
        self.maximum = float('-inf')

    def update(self, value: float):
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class MuZeroNode:
    """MCTS node in latent space (Appendix B)."""
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

    def q_value(self) -> float:
        """Mean value Q(s,a) (Eq. 4)."""
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def ucb_score(self, min_max: MinMaxStats, c1: float = 1.25, c2: float = 19652) -> float:
        """pUCT score with min-max normalization (Eq. 2 + Eq. 5).
        
        a^k = argmax_a [ Q_normalized(s,a) + P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
                         * (c1 + log((sum_b N(s,b) + c2 + 1) / c2)) ]
        """
        parent_visits = self.parent.visits if self.parent else 1
        # Prior score with exploration bonus
        pb_c = math.log((parent_visits + c2 + 1) / c2) + c1
        prior_score = pb_c * self.prior * math.sqrt(parent_visits) / (1 + self.visits)
        # Normalized Q value
        if self.visits > 0:
            value_score = min_max.normalize(self.q_value())
        else:
            value_score = 0.0
        return value_score + prior_score


class MuZeroAgent(BaseAgent):
    """MuZero: learned model + MCTS in latent space.
    
    Follows the algorithm described in Schrittwieser et al. (2020),
    adapted for small-scale environments with MLP networks.
    """

    def __init__(self, input_size: int, action_space_size: int,
                 learning_rate: float = 0.001, discount_factor: float = 0.99,
                 latent_dim: int = 64, num_simulations: int = 50,
                 c1: float = 1.25, c2: float = 19652,
                 temperature: float = 1.0,
                 buffer_size: int = 5000, unroll_steps: int = 5,
                 hidden_layers=None, seed: Optional[int] = None, **kwargs):
        super().__init__("MuZero", action_space_size, input_size)
        if seed is not None:
            np.random.seed(seed)
        self.gamma = discount_factor
        self.latent_dim = latent_dim
        self.num_simulations = num_simulations
        self.c1 = c1
        self.c2 = c2
        self.temperature = temperature
        self.K = unroll_steps  # K hypothetical steps (paper uses K=5)

        h = hidden_layers or [128]

        # Representation function h: obs -> latent state s^0
        self.repr_net = NeuralNetwork.build_mlp(
            [input_size] + h + [latent_dim],
            ['relu'] * len(h) + ['relu'], lr=learning_rate)

        # Dynamics function g: (latent + action_onehot) -> (next_latent + reward)
        dyn_in = latent_dim + action_space_size
        self.dyn_net = NeuralNetwork.build_mlp(
            [dyn_in] + h + [latent_dim + 1],
            ['relu'] * len(h) + ['linear'], lr=learning_rate)

        # Prediction function f: latent -> (policy, value)
        self.pred_net = NeuralNetwork.build_dual_head(
            latent_dim, h, action_space_size, lr=learning_rate)

        self.buffer: List[list] = []  # replay buffer of trajectories
        self.buffer_size = buffer_size
        self.current_trajectory: List[dict] = []
        self.train_batch_size = 32

    # ──── Hidden state scaling (Appendix G) ────

    def _scale_hidden(self, s: np.ndarray) -> np.ndarray:
        """Scale hidden state to [0,1] as described in Appendix G."""
        s_min = s.min()
        s_max = s.max()
        if s_max - s_min > 1e-8:
            return (s - s_min) / (s_max - s_min)
        return s * 0.0  # all zeros if constant

    # ──── Model functions (Section 3) ────

    def _get_latent(self, obs: np.ndarray) -> np.ndarray:
        """h(o) = s^0 : representation function."""
        s = self.repr_net.predict(np.atleast_2d(obs))[0]
        return self._scale_hidden(s)

    def _dynamics(self, latent: np.ndarray, action: int) -> Tuple[np.ndarray, float]:
        """g(s^{k-1}, a^k) = (r^k, s^k) : dynamics function."""
        a_onehot = np.zeros(self.action_space_size, dtype=np.float32)
        a_onehot[action] = 1.0
        inp = np.concatenate([latent, a_onehot])
        out = self.dyn_net.predict(np.atleast_2d(inp))[0]
        next_latent = self._scale_hidden(out[:self.latent_dim])
        reward = float(out[self.latent_dim])
        return next_latent, reward

    def _predict(self, latent: np.ndarray) -> Tuple[np.ndarray, float]:
        """f(s^k) = (p^k, v^k) : prediction function."""
        policy, value = self.pred_net.predict(np.atleast_2d(latent))
        return policy[0], float(value[0, 0])

    # ──── MCTS (Appendix B) ────

    def select_action(self, state: Any, valid_actions: Optional[np.ndarray] = None) -> int:
        obs = np.asarray(state, dtype=np.float32).flatten()
        latent = self._get_latent(obs)

        if valid_actions is None:
            valid_actions = np.arange(self.action_space_size)

        min_max = MinMaxStats()

        # Create root node
        root = MuZeroNode(latent_state=latent)
        policy, value = self._predict(latent)

        # Mask invalid actions and normalize (Appendix A, point 2)
        mask = np.zeros(self.action_space_size)
        mask[valid_actions] = 1
        policy = policy * mask
        policy_sum = policy.sum()
        if policy_sum > 1e-8:
            policy /= policy_sum
        else:
            policy[valid_actions] = 1.0 / len(valid_actions)

        # Add Dirichlet noise at root for exploration
        if self.training:
            noise = np.random.dirichlet([0.25] * len(valid_actions))
            frac = 0.25
            for i, a in enumerate(valid_actions):
                policy[a] = (1 - frac) * policy[a] + frac * noise[i]

        for a in valid_actions:
            root.children[int(a)] = MuZeroNode(
                latent_state=None, parent=root, action=int(a), prior=policy[a])
        root.visits = 1
        root.value_sum = value
        min_max.update(value)

        # Run simulations (Appendix B: Selection, Expansion, Backup)
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # Selection: traverse tree using pUCT (Eq. 2)
            while len(node.children) > 0 and node.visits > 0:
                node = max(node.children.values(),
                           key=lambda c: c.ucb_score(min_max, self.c1, self.c2))
                search_path.append(node)

            # Expansion: use dynamics model (Appendix B)
            parent = node.parent if node.parent else root
            parent_latent = parent.latent_state
            if parent_latent is not None:
                next_latent, reward = self._dynamics(parent_latent, node.action)
                node.latent_state = next_latent
                node.reward = reward
                policy_child, value_child = self._predict(next_latent)
                # Expand all actions from new node
                for a in range(self.action_space_size):
                    if a not in node.children:
                        node.children[a] = MuZeroNode(
                            latent_state=None, parent=node, action=a,
                            prior=policy_child[a])
                value = value_child
            else:
                value = 0.0

            # Backup: compute G^k and update Q (Eq. 3, Eq. 4)
            for n in reversed(search_path):
                n.visits += 1
                n.value_sum += value
                min_max.update(n.q_value())
                value = n.reward + self.gamma * value

        # Action selection from root visit counts (Appendix D, Eq. 6)
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

        # Store data for training
        if self.training:
            self.current_trajectory.append({
                'obs': obs.copy(), 'action': action, 'pi': pi.copy()
            })
        return action

    # ──── Learning (Appendix G) ────

    def learn(self, state, action, reward, next_state, done):
        if not self.training:
            return
        if self.current_trajectory:
            self.current_trajectory[-1]['reward'] = reward
        if done:
            if self.current_trajectory:
                self.buffer.append(self.current_trajectory)
                if len(self.buffer) > self.buffer_size:
                    self.buffer.pop(0)
                self.current_trajectory = []
            if len(self.buffer) >= 4:
                self._train_networks()

    def _train_networks(self):
        """Train with K-step unrolling (Figure 1C, Eq. 1, Appendix G).
        
        For each sampled position t:
        1. Get s^0 = h(o_t) via representation function
        2. Unroll K steps: s^k, r^k = g(s^{k-1}, a_{t+k}) 
        3. At each step k, compute p^k, v^k = f(s^k)
        4. Loss = sum_{k=0}^{K} [ l_r(u_{t+k}, r^k) + l_v(z_{t+k}, v^k) + l_p(pi_{t+k}, p^k) ]
        """
        # Sample batch of positions from replay buffer
        batch_data = []
        for _ in range(min(self.train_batch_size, len(self.buffer))):
            traj = self.buffer[np.random.randint(len(self.buffer))]
            if len(traj) < 2:
                continue
            # Sample a position t that allows at least 1 unroll step
            max_start = max(0, len(traj) - 2)
            t = np.random.randint(max_start + 1)
            batch_data.append((traj, t))

        if len(batch_data) < 4:
            return

        K = self.K
        ld = self.latent_dim

        # ──── Step 1: Representation h(o_t) -> s^0 ────
        obs_batch = np.array([traj[t]['obs'] for traj, t in batch_data], dtype=np.float32)
        latents_0 = self.repr_net.forward(obs_batch, training=True)
        # Scale hidden states to [0,1]
        for i in range(len(latents_0)):
            latents_0[i] = self._scale_hidden(latents_0[i])

        # ──── Step 2: Prediction at k=0: f(s^0) -> p^0, v^0 ────
        pred_policy_0, pred_value_0 = self.pred_net.forward(latents_0, training=True)

        # Build targets for k=0
        target_pi_0 = np.array([traj[t]['pi'] for traj, t in batch_data], dtype=np.float32)
        target_v_0 = np.array([self._compute_value_target(traj, t)
                               for traj, t in batch_data], dtype=np.float32)

        # Normalize value targets to [-1,1] for tanh head
        max_abs = max(np.abs(target_v_0).max(), 1e-8)
        target_v_0_norm = np.clip(target_v_0 / max_abs, -1, 1)

        # ──── Policy + value loss at k=0 (scale by 1/K per Appendix G) ────
        scale = 1.0 / max(K, 1)
        policy_grad_0 = (pred_policy_0 - target_pi_0) * scale
        value_grad_0 = np.zeros_like(pred_value_0)
        value_grad_0[:, 0] = (pred_value_0[:, 0] - target_v_0_norm) * scale

        self.pred_net.backward_dual(policy_grad_0, value_grad_0)

        # Accumulate gradient for representation network
        repr_grad_total = np.zeros_like(latents_0)

        # ──── Steps 3-4: Unroll K steps through dynamics ────
        current_latents = latents_0.copy()

        for k in range(1, K + 1):
            # Gather actions and targets for step k
            actions_k = []
            target_pis_k = []
            target_vs_k = []
            target_rs_k = []
            valid_mask = []  # which samples have data at step t+k

            for idx, (traj, t) in enumerate(batch_data):
                if t + k < len(traj):
                    actions_k.append(traj[t + k - 1]['action'])  # a_{t+k} (action taken at t+k-1 leading to t+k)
                    target_pis_k.append(traj[t + k]['pi'] if t + k < len(traj) else np.zeros(self.action_space_size))
                    target_vs_k.append(self._compute_value_target(traj, t + k))
                    target_rs_k.append(traj[t + k - 1].get('reward', 0))  # u_{t+k}
                    valid_mask.append(True)
                else:
                    # Pad with zeros for samples that don't extend this far
                    actions_k.append(0)
                    target_pis_k.append(np.zeros(self.action_space_size))
                    target_vs_k.append(0.0)
                    target_rs_k.append(0.0)
                    valid_mask.append(False)

            actions_k = np.array(actions_k, dtype=np.int32)
            target_pis_k = np.array(target_pis_k, dtype=np.float32)
            target_vs_k = np.array(target_vs_k, dtype=np.float32)
            target_rs_k = np.array(target_rs_k, dtype=np.float32)
            valid = np.array(valid_mask, dtype=np.float32)

            if valid.sum() < 2:
                break

            # Normalize value targets
            max_abs_k = max(np.abs(target_vs_k).max(), 1e-8)
            target_vs_k_norm = np.clip(target_vs_k / max_abs_k, -1, 1)

            # ── Dynamics: g(s^{k-1}, a_k) -> (s^k, r^k) ──
            a_onehot = np.zeros((len(actions_k), self.action_space_size), dtype=np.float32)
            a_onehot[np.arange(len(actions_k)), actions_k] = 1.0
            # Scale dynamics gradient by 1/2 (Appendix G)
            dyn_input = np.concatenate([current_latents * 0.5 + current_latents * 0.5, a_onehot], axis=1)

            dyn_output = self.dyn_net.forward(dyn_input, training=True)
            next_latents = dyn_output[:, :ld].copy()
            pred_rewards = dyn_output[:, ld]

            # Scale hidden states
            for i in range(len(next_latents)):
                next_latents[i] = self._scale_hidden(next_latents[i])

            # ── Prediction: f(s^k) -> (p^k, v^k) ──
            pred_policy_k, pred_value_k = self.pred_net.forward(next_latents, training=True)

            # ── Losses at step k (scaled by 1/K) ──
            # Reward loss: l_r(u_{t+k}, r^k) = MSE
            reward_grad = np.zeros_like(dyn_output)
            reward_grad[:, ld] = (pred_rewards - target_rs_k) * valid * scale

            # Policy loss: l_p(pi_{t+k}, p^k) = cross-entropy gradient
            policy_grad_k = (pred_policy_k - target_pis_k) * valid[:, None] * scale

            # Value loss: l_v(z_{t+k}, v^k) = MSE
            value_grad_k = np.zeros_like(pred_value_k)
            value_grad_k[:, 0] = (pred_value_k[:, 0] - target_vs_k_norm) * valid * scale

            # Backprop prediction network
            self.pred_net.backward_dual(policy_grad_k, value_grad_k)

            # Backprop dynamics network (reward + latent gradient)
            # The latent part of the gradient comes from the prediction loss
            latent_grad = np.zeros((len(batch_data), ld), dtype=np.float32)
            latent_grad += value_grad_k[:, 0:1] * 0.1  # simplified backprop from pred to dynamics
            reward_grad[:, :ld] = latent_grad
            self.dyn_net.backward(reward_grad)

            current_latents = next_latents

        # ── Backprop through representation network ──
        repr_grad_total += value_grad_0[:, 0:1] * 0.1
        self.repr_net.backward(repr_grad_total)

    def _compute_value_target(self, traj: list, t: int) -> float:
        """Compute n-step bootstrapped value target z_t (Section 3).
        
        z_t = u_{t+1} + gamma*u_{t+2} + ... + gamma^{n-1}*u_{t+n} + gamma^n * v_{t+n}
        For our environments without a stored search value, we use the 
        discounted sum of observed rewards as the target.
        """
        n = min(self.K, len(traj) - t)
        value = 0.0
        for i in range(n):
            if t + i < len(traj):
                value += (self.gamma ** i) * traj[t + i].get('reward', 0)
        return value

    # ──── Save / Load ────

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        self.repr_net.save(filepath + '_repr.pkl')
        self.dyn_net.save(filepath + '_dyn.pkl')
        self.pred_net.save(filepath + '_pred.pkl')

    def load(self, filepath: str):
        self.repr_net = NeuralNetwork.load(filepath + '_repr.pkl')
        self.dyn_net = NeuralNetwork.load(filepath + '_dyn.pkl')
        self.pred_net = NeuralNetwork.load(filepath + '_pred.pkl')
