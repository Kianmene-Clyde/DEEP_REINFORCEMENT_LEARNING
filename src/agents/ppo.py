"""PPO A2C-style — conforme à :
   'Proximal Policy Optimization Algorithms', Schulman et al., 2017.

Éléments clés du papier implémentés ici :

1. Objectif clippé (équation 7) :
   L^CLIP = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]

2. Objectif combiné acteur+critique+entropie (équation 9) :
   L = E[L^CLIP - c1 * L^VF + c2 * S[π]]
   avec L^VF = (V(s) - V_targ)²

3. GAE (équations 11-12) :
   δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
   A_t = Σ_{l≥0} (γλ)^l * δ_{t+l}

4. Algorithme 1 : collecte T timesteps → K époques de SGD par minibatchs
   (minibatch size M ≤ T, tirage aléatoire à chaque époque)

Hyperparamètres du papier (Table 3, MuJoCo) :
   T=2048, lr=3e-4, K=10, M=64, γ=0.99, λ=0.95, ε=0.2

Hyperparamètres du papier (Table 5, Atari) :
   T=128, lr=2.5e-4, K=3, M=256, γ=0.99, λ=0.95, ε=0.1, c1=1, c2=0.01
"""

import numpy as np
import os
import pickle
from typing import Optional, Any
from .base_agent import BaseAgent
from .utils import mask_and_normalize as _safe_probs
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from neural_network.model import NeuralNetwork


class PPOAgent(BaseAgent):
    """PPO avec objectif clippé — Schulman et al. (2017).

    Architecture séparée acteur/critique (sans partage de paramètres),
    conforme à la Section 6.1 du papier ("We don't share parameters
    between the policy and value function").

    Paramètres
    ----------
    clip_ratio : float
        ε dans l'équation 7 du papier. Recommandé : 0.2 (MuJoCo), 0.1 (Atari).
    num_epochs : int
        K dans l'Algorithme 1. Recommandé : 10 (MuJoCo), 3 (Atari).
    minibatch_size : int
        M dans l'Algorithme 1. Recommandé : 64 (MuJoCo), 256 (Atari).
    vf_coef : float
        c1 dans l'équation 9.  Recommandé : 1.0.
    entropy_coef : float
        c2 dans l'équation 9.  Recommandé : 0.01.
    gae_lambda : float
        λ dans l'équation 11.  Recommandé : 0.95.
    """

    def __init__(
            self,
            input_size: int,
            action_space_size: int,
            learning_rate: float = 3e-4,  # Table 3 / Table 5 du papier
            discount_factor: float = 0.99,  # Table 3 / Table 5
            gae_lambda: float = 0.95,  # Table 3 / Table 5
            clip_ratio: float = 0.2,  # ε, Table 3 (meilleur résultat Table 1)
            num_epochs: int = 10,  # K, Table 3 (MuJoCo)
            minibatch_size: int = 64,  # M, Table 3 (MuJoCo)
            vf_coef: float = 1.0,  # c1, équation 9
            entropy_coef: float = 0.01,  # c2, équation 9 / Table 5
            hidden_layers=None,
            seed: Optional[int] = None,
            **kwargs
    ):
        super().__init__("PPO_A2C", action_space_size, input_size)
        if seed is not None:
            np.random.seed(seed)

        self.gamma = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef

        h = hidden_layers or [128, 128]

        # Réseau acteur π(a|s) avec sortie softmax
        dims_a = [input_size] + h + [action_space_size]
        acts_a = ['relu'] * len(h) + ['softmax']
        self.actor = NeuralNetwork.build_mlp(dims_a, acts_a, lr=learning_rate)

        # Réseau critique V(s) avec sortie linéaire
        dims_c = [input_size] + h + [1]
        acts_c = ['relu'] * len(h) + ['linear']
        self.critic = NeuralNetwork.build_mlp(dims_c, acts_c, lr=learning_rate)

        # Buffers de l'épisode courant
        self.ep_states = []
        self.ep_actions = []
        self.ep_rewards = []
        self.ep_log_probs = []  # log π_old(a|s)
        self.ep_values = []  # V(s_t) estimé au moment de la collecte

    # ------------------------------------------------------------------
    # Sélection d'action
    # ------------------------------------------------------------------
    def select_action(
            self, state: Any, valid_actions: Optional[np.ndarray] = None
    ) -> int:
        s = np.atleast_2d(np.asarray(state, dtype=np.float32))
        probs = self.actor.predict(s)[0]
        value = self.critic.predict(s)[0, 0]
        probs = _safe_probs(probs, valid_actions, self.action_space_size)

        if not self.training:
            return int(np.argmax(probs))

        action = int(np.random.choice(self.action_space_size, p=probs))
        self.ep_states.append(np.asarray(state, dtype=np.float32))
        self.ep_actions.append(action)
        self.ep_log_probs.append(float(np.log(probs[action] + 1e-8)))
        self.ep_values.append(float(value))
        return action

    def learn(self, state, action, reward, next_state, done):
        if not self.training:
            return
        self.ep_rewards.append(reward)
        if done:
            self._update()

    # ------------------------------------------------------------------
    # GAE — équations 11-12 du papier
    # δ_t = r_t + γ V(s_{t+1}) - V(s_t)
    # A_t = Σ_{l≥0} (γλ)^l δ_{t+l}
    # V_targ_t = A_t + V(s_t)
    # ------------------------------------------------------------------
    def _compute_gae(self, rewards, values):
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        next_value = 0.0  # V(s_T) = 0 pour un épisode terminal
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae
            next_value = values[t]
        returns = advantages + values  # V_targ = A + V (équation 12 implicite)
        return returns, advantages

    # ------------------------------------------------------------------
    # Gradient de l'objectif clippé (équation 7) par rapport aux logits
    # ------------------------------------------------------------------
    @staticmethod
    def _clip_policy_gradient(new_probs, actions, advantages, old_log_probs, clip_ratio):
        """Gradient de -L^CLIP w.r.t. logits pré-softmax.

        L^CLIP = min(r * A, clip(r, 1-ε, 1+ε) * A)

        Gradient de min(surr1, surr2) :
          • Quand A ≥ 0 et r > 1+ε : gradient = 0 (clippé)
          • Quand A < 0 et r < 1-ε : gradient = 0 (clippé)
          • Sinon                   : gradient = A

        ∂(-L^CLIP)/∂z_j = -effective_weight * (y_j - p_j)
                         = effective_weight * (p_j - y_j)   [pour descente]
        """
        T = len(actions)
        new_log_probs = np.log(new_probs[np.arange(T), actions] + 1e-8)
        ratio = np.exp(new_log_probs - old_log_probs)

        # Gradient effectif selon la règle de dérivation du min clippé
        effective_weight = np.where(
            (advantages >= 0) & (ratio > 1 + clip_ratio), 0.0,  # clippé A>0
            np.where(
                (advantages < 0) & (ratio < 1 - clip_ratio), 0.0,  # clippé A<0
                advantages  # gradient actif
            )
        )

        # Gradient de -log π(a|s) * w w.r.t. logits = (p - one_hot) * w
        grad = new_probs.copy()
        for i in range(T):
            grad[i, actions[i]] -= 1.0  # (p - one_hot)
            grad[i] *= effective_weight[i]  # × advantage effectif
        return grad

    # ------------------------------------------------------------------
    # Gradient de l'entropie bonus +c2*S[π] w.r.t. logits
    #
    # Pour une sortie softmax, ∂H/∂z_j = p_j(-log(p_j) - H)
    # (preuve par la règle de la chaîne sur softmax)
    # Gradient de -c2*H (pour descente) = c2 * p_j * (log(p_j) + H)
    # ------------------------------------------------------------------
    @staticmethod
    def _entropy_gradient(probs, entropy_coef):
        eps = 1e-8
        H = -np.sum(probs * np.log(probs + eps), axis=1, keepdims=True)  # (T,1)
        # ∂(-c2*H)/∂z_j = c2 * p_j * (log(p_j) + H_t)
        return entropy_coef * probs * (np.log(probs + eps) + H)

    # ------------------------------------------------------------------
    # Mise à jour PPO — Algorithme 1 du papier
    # K époques × minibatchs aléatoires
    # ------------------------------------------------------------------
    def _update(self):
        if len(self.ep_rewards) == 0:
            return

        states = np.array(self.ep_states, dtype=np.float32)
        actions = np.array(self.ep_actions, dtype=np.int32)
        old_log_probs = np.array(self.ep_log_probs, dtype=np.float32)
        values = np.array(self.ep_values, dtype=np.float32)
        rewards = np.array(self.ep_rewards, dtype=np.float32)

        returns, advantages = self._compute_gae(rewards, values)

        # Normalisation des advantages (heuristique standard, non explicite
        # dans le papier mais couramment utilisée avec PPO)
        if np.std(advantages) > 1e-8:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        T = len(states)
        minibatch_size = min(self.minibatch_size, T)

        # ── K époques (Algorithme 1 du papier) ─────────────────────────
        for _ in range(self.num_epochs):
            # Mélange aléatoire des indices à chaque époque (minibatch SGD)
            indices = np.random.permutation(T)

            for start in range(0, T, minibatch_size):
                mb_idx = indices[start: start + minibatch_size]
                if len(mb_idx) == 0:
                    continue

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_old_lp = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]

                # ── Acteur : gradient de -L^CLIP + c2*(-S) ──────────────
                new_probs = self.actor.forward(mb_states, training=True)

                # Gradient objectif clippé (équation 7)
                actor_grad = self._clip_policy_gradient(
                    new_probs, mb_actions, mb_adv, mb_old_lp, self.clip_ratio
                )

                # Bonus d'entropie (équation 9 : +c2*S → descente sur -c2*S)
                actor_grad += self._entropy_gradient(new_probs, self.entropy_coef)

                self.actor.backward(actor_grad)

                # ── Critique : gradient de c1 * L^VF = c1*(V - V_targ)² ─
                pred_values = self.critic.forward(mb_states, training=True)
                critic_grad = np.zeros_like(pred_values)
                # dL^VF/dV = 2*(V - V_targ), facteur 2 absorbé dans lr
                critic_grad[:, 0] = self.vf_coef * (pred_values[:, 0] - mb_returns)
                self.critic.backward(critic_grad)

        # Réinitialisation des buffers
        self.ep_states, self.ep_actions, self.ep_rewards = [], [], []
        self.ep_log_probs, self.ep_values = [], []

    # ------------------------------------------------------------------
    # Sauvegarde / Chargement
    # ------------------------------------------------------------------
    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        self.actor.save(filepath + '_actor.pkl')
        self.critic.save(filepath + '_critic.pkl')

    def load(self, filepath: str):
        self.actor = NeuralNetwork.load(filepath + '_actor.pkl')
        self.critic = NeuralNetwork.load(filepath + '_critic.pkl')
