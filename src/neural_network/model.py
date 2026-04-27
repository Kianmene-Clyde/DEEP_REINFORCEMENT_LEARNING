"""Neural network model utilisé par tout nos agents"""
import numpy as np
import pickle
from typing import List, Tuple, Optional, Dict


class NeuralNetwork:
    """feedforward neural network"""

    def __init__(self, lr: float = 0.001, clip_grad: float = 1.0):
        self.lr = lr
        self.clip_grad = clip_grad
        self.layers: List[Dict] = []
        self._cache: List[Dict] = []
        # Adam state per layer
        self._adam_t = 0
        self._adam_m: List[Dict] = []
        self._adam_v: List[Dict] = []

    # Construction
    def add_dense(self, in_dim: int, out_dim: int, activation: str = 'relu'):
        """Ajout d'une couche dense."""
        # Initialisation He pour relu, Xavier pour les autres
        if activation == 'relu':
            scale = np.sqrt(2.0 / in_dim)
        else:
            scale = np.sqrt(1.0 / in_dim)
        W = np.random.randn(in_dim, out_dim).astype(np.float32) * scale
        b = np.zeros(out_dim, dtype=np.float32)
        self.layers.append({'W': W, 'b': b, 'act': activation})
        self._adam_m.append({'W': np.zeros_like(W), 'b': np.zeros_like(b)})
        self._adam_v.append({'W': np.zeros_like(W), 'b': np.zeros_like(b)})

    @classmethod
    def build_mlp(cls, dims: List[int], activations: Optional[List[str]] = None,
                  lr: float = 0.001) -> 'NeuralNetwork':
        """Créer un MLP à partir d'une liste de dimensions.
        Exemple : build_mlp([10, 128, 128, 4], [“relu”,'relu',“linear”])
        """
        net = cls(lr=lr)
        if activations is None:
            activations = ['relu'] * (len(dims) - 2) + ['linear']
        for i in range(len(dims) - 1):
            net.add_dense(dims[i], dims[i + 1], activations[i])
        return net

    @classmethod
    def build_dual_head(cls, input_dim: int, hidden: List[int],
                        policy_dim: int, lr: float = 0.001) -> 'NeuralNetwork':
        """Construire un neural network avec un tronc commun, une tête de politique (softmax) et une tête de valeur (tanh).
        Les deux dernières couches correspondent aux têtes de politique et de valeur.
        """
        net = cls(lr=lr)
        prev = input_dim
        for h in hidden:
            net.add_dense(prev, h, 'relu')
            prev = h
        # Policy head
        net.add_dense(prev, policy_dim, 'softmax')
        # Value head
        net.add_dense(prev, 1, 'tanh')
        net._dual_head = True
        net._trunk_len = len(hidden)  # number of shared layers
        return net

    # Forward
    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        x = np.atleast_2d(x).astype(np.float32)
        if getattr(self, '_dual_head', False):
            return self._forward_dual(x, training)
        if training:
            self._cache = []
        for layer in self.layers:
            z = x @ layer['W'] + layer['b']
            if training:
                self._cache.append({'x': x, 'z': z})
            x = self._activate(z, layer['act'])
        return x

    def _forward_dual(self, x: np.ndarray, training: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Forward for dual-head network."""
        if training:
            self._cache = []
        trunk_layers = self.layers[:self._trunk_len]
        policy_layer = self.layers[self._trunk_len]
        value_layer = self.layers[self._trunk_len + 1]

        for layer in trunk_layers:
            z = x @ layer['W'] + layer['b']
            if training:
                self._cache.append({'x': x, 'z': z})
            x = self._activate(z, layer['act'])

        # Policy head
        zp = x @ policy_layer['W'] + policy_layer['b']
        policy = self._activate(zp, policy_layer['act'])
        if training:
            self._cache.append({'x': x, 'z': zp, 'head': 'policy'})

        # Value head
        zv = x @ value_layer['W'] + value_layer['b']
        value = self._activate(zv, value_layer['act'])
        if training:
            self._cache.append({'x': x, 'z': zv, 'head': 'value'})

        return policy, value

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x, training=False)

    # Backward
    def backward(self, loss_grad: np.ndarray):
        """Rétropropagation et mise à jour des poids avec Adam"""
        if getattr(self, '_dual_head', False):
            raise RuntimeError("Use backward_dual for dual-head networks")
        self._adam_t += 1
        grads = self._compute_grads(loss_grad, list(range(len(self.layers))))
        self._adam_update(grads)

    def backward_dual(self, policy_grad: np.ndarray, value_grad: np.ndarray):
        """Backprop for dual-head network."""
        self._adam_t += 1
        trunk_indices = list(range(self._trunk_len))
        policy_idx = self._trunk_len
        value_idx = self._trunk_len + 1

        # Find cached data
        policy_cache = self._cache[self._trunk_len]
        value_cache = self._cache[self._trunk_len + 1]
        trunk_output = policy_cache['x']  # same as value_cache['x']

        # Policy head grads
        dz_p = policy_grad * self._activate_deriv(policy_cache['z'],
                                                  self.layers[policy_idx]['act'],
                                                  self._activate(policy_cache['z'], self.layers[policy_idx]['act']))
        dW_p = policy_cache['x'].T @ dz_p / len(policy_grad)
        db_p = np.mean(dz_p, axis=0)
        dx_p = dz_p @ self.layers[policy_idx]['W'].T

        # Value head grads
        dz_v = value_grad * self._activate_deriv(value_cache['z'],
                                                 self.layers[value_idx]['act'],
                                                 self._activate(value_cache['z'], self.layers[value_idx]['act']))
        dW_v = value_cache['x'].T @ dz_v / len(value_grad)
        db_v = np.mean(dz_v, axis=0)
        dx_v = dz_v @ self.layers[value_idx]['W'].T

        # Combined gradient flowing into trunk
        dx_trunk = dx_p + dx_v

        # Trunk grads
        trunk_grads = self._compute_grads(dx_trunk, trunk_indices)
        trunk_grads[policy_idx] = {'dW': dW_p, 'db': db_p}
        trunk_grads[value_idx] = {'dW': dW_v, 'db': db_v}
        self._adam_update(trunk_grads)

    def _compute_grads(self, loss_grad: np.ndarray, layer_indices: List[int]) -> Dict:
        """Calculer les gradients pour les couches spécifiées."""
        grads = {}
        dx = loss_grad
        for i in reversed(layer_indices):
            cache = self._cache[i]
            act = self.layers[i]['act']
            out = self._activate(cache['z'], act)
            dz = dx * self._activate_deriv(cache['z'], act, out)
            batch_size = max(1, len(cache['x']))
            grads[i] = {
                'dW': cache['x'].T @ dz / batch_size,
                'db': np.mean(dz, axis=0)
            }
            dx = dz @ self.layers[i]['W'].T
        return grads

    # Adam optimizer
    def _adam_update(self, grads: Dict, beta1=0.9, beta2=0.999, eps=1e-8):
        for i, g in grads.items():
            for key in ('W', 'b'):
                grad = np.clip(g['d' + key], -self.clip_grad, self.clip_grad)
                self._adam_m[i][key] = beta1 * self._adam_m[i][key] + (1 - beta1) * grad
                self._adam_v[i][key] = beta2 * self._adam_v[i][key] + (1 - beta2) * grad ** 2
                m_hat = self._adam_m[i][key] / (1 - beta1 ** self._adam_t)
                v_hat = self._adam_v[i][key] / (1 - beta2 ** self._adam_t)
                self.layers[i][key] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    # Activations
    @staticmethod
    def _activate(z: np.ndarray, act: str) -> np.ndarray:
        if act == 'relu':
            return np.maximum(0, z)
        elif act == 'softmax':
            e = np.exp(z - np.max(z, axis=-1, keepdims=True))
            return e / (np.sum(e, axis=-1, keepdims=True) + 1e-8)
        elif act == 'tanh':
            return np.tanh(z)
        elif act == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        elif act == 'linear':
            return z
        raise ValueError(f"Unknown activation: {act}")

    @staticmethod
    def _activate_deriv(z: np.ndarray, act: str, out: np.ndarray) -> np.ndarray:
        if act == 'relu':
            return (z > 0).astype(np.float32)
        elif act == 'softmax':
            return np.ones_like(z)  # gradient handled externally
        elif act == 'tanh':
            return 1.0 - out ** 2
        elif act == 'sigmoid':
            return out * (1.0 - out)
        elif act == 'linear':
            return np.ones_like(z)
        raise ValueError(f"Unknown activation: {act}")

    # Copy / Save / Load
    def copy(self) -> 'NeuralNetwork':
        """Copie du neural network."""
        import copy
        return copy.deepcopy(self)

    def copy_weights_from(self, other: 'NeuralNetwork'):
        """Copier les poids d'un autre réseau."""
        for i in range(len(self.layers)):
            self.layers[i]['W'] = other.layers[i]['W'].copy()
            self.layers[i]['b'] = other.layers[i]['b'].copy()

    def soft_update(self, other: 'NeuralNetwork', tau: float = 0.01):
        for i in range(len(self.layers)):
            self.layers[i]['W'] = tau * other.layers[i]['W'] + (1 - tau) * self.layers[i]['W']
            self.layers[i]['b'] = tau * other.layers[i]['b'] + (1 - tau) * self.layers[i]['b']

    def save(self, filepath: str):
        data = {
            'layers': [(l['W'].copy(), l['b'].copy(), l['act']) for l in self.layers],
            'lr': self.lr,
            'clip_grad': self.clip_grad,
            'dual_head': getattr(self, '_dual_head', False),
            'trunk_len': getattr(self, '_trunk_len', 0),
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> 'NeuralNetwork':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        net = cls(lr=data['lr'], clip_grad=data['clip_grad'])
        for W, b, act in data['layers']:
            net.layers.append({'W': W, 'b': b, 'act': act})
            net._adam_m.append({'W': np.zeros_like(W), 'b': np.zeros_like(b)})
            net._adam_v.append({'W': np.zeros_like(W), 'b': np.zeros_like(b)})
        if data.get('dual_head'):
            net._dual_head = True
            net._trunk_len = data['trunk_len']
        return net

    def param_count(self) -> int:
        return sum(l['W'].size + l['b'].size for l in self.layers)
