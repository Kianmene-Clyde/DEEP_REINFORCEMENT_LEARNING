"""Small dense NumPy networks with explicit, tested gradient conventions.

The gradient argument is PER SAMPLE (not divided by batch size). Parameter
gradients are averaged exactly once. For a softmax output, logits=True means
the caller supplies dL/dlogits (e.g. p-target for cross entropy). Internal
softmax layers and logits=False use the full softmax Jacobian product.
"""

import copy
import pickle
from pathlib import Path
import numpy as np


class NeuralNetwork:
    FORMAT_VERSION = 2

    def __init__(self, lr=0.001, clip_grad=1.0, rng=None):
        self.lr, self.clip_grad = lr, clip_grad
        self.rng = rng if rng is not None else np.random.default_rng()
        self.layers, self._cache = [], []
        self._adam_t = 0
        self._adam_m, self._adam_v = [], []

    def add_dense(self, in_dim, out_dim, activation="relu"):
        if min(in_dim, out_dim) < 1:
            raise ValueError("Layer dimensions must be positive")
        self._activate(np.zeros((1, out_dim)), activation)
        scale = np.sqrt((2.0 if activation == "relu" else 1.0) / in_dim)
        w = self.rng.standard_normal((in_dim, out_dim)).astype(np.float32) * scale
        b = np.zeros(out_dim, dtype=np.float32)
        self.layers.append({"W": w.astype(np.float32), "b": b, "act": activation})
        self._adam_m.append({"W": np.zeros_like(w), "b": np.zeros_like(b)})
        self._adam_v.append({"W": np.zeros_like(w), "b": np.zeros_like(b)})

    @classmethod
    def build_mlp(cls, dims, activations=None, lr=0.001, rng=None):
        if len(dims) < 2:
            raise ValueError("At least input and output dimensions required")
        acts = activations if activations is not None else ["relu"] * (len(dims) - 2) + ["linear"]
        if len(acts) != len(dims) - 1:
            raise ValueError("One activation required per layer")
        net = cls(lr=lr, rng=rng)
        for a, b, act in zip(dims, dims[1:], acts):
            net.add_dense(a, b, act)
        return net

    @classmethod
    def build_dual_head(cls, input_dim, hidden, policy_dim, lr=0.001, rng=None, value_activation="linear"):
        net = cls(lr=lr, rng=rng)
        prev = input_dim
        for h in hidden:
            net.add_dense(prev, h, "relu")
            prev = h
        net.add_dense(prev, policy_dim, "softmax")
        net.add_dense(prev, 1, value_activation)
        net._dual_head, net._trunk_len = True, len(hidden)
        return net

    def _layer_forward(self, x, index, training):
        layer = self.layers[index]
        z = x @ layer["W"] + layer["b"]
        out = self._activate(z, layer["act"])
        if not np.all(np.isfinite(out)):
            raise FloatingPointError("Non-finite network output")
        if training:
            self._cache.append({"x": x, "z": z, "out": out})
        return out

    def forward(self, x, training=False):
        x = np.atleast_2d(np.asarray(x))
        x = x.astype(np.result_type(x.dtype, np.float32), copy=False)
        if not self.layers or x.shape[1] != self.layers[0]["W"].shape[0]:
            raise ValueError("Input dimension does not match network")
        if not np.all(np.isfinite(x)):
            raise ValueError("Network inputs must be finite")
        if training:
            self._cache = []
        if getattr(self, "_dual_head", False):
            for i in range(self._trunk_len):
                x = self._layer_forward(x, i, training)
            return (
                self._layer_forward(x, self._trunk_len, training),
                self._layer_forward(x, self._trunk_len + 1, training),
            )
        for i in range(len(self.layers)):
            x = self._layer_forward(x, i, training)
        return x

    def predict(self, x):
        return self.forward(x, training=False)

    def _backprop(self, grad, indices, logits=False):
        grad = np.asarray(grad)
        if not np.all(np.isfinite(grad)):
            raise FloatingPointError("Non-finite loss gradient")
        grads = {}
        for index in reversed(indices):
            cache, layer = self._cache[index], self.layers[index]
            if grad.shape != cache["out"].shape:
                raise ValueError("Gradient shape does not match cached output")
            if logits and index == indices[-1] and layer["act"] == "softmax":
                dz = grad
            elif layer["act"] == "softmax":
                p = cache["out"]
                dz = p * (grad - np.sum(grad * p, axis=1, keepdims=True))
            else:
                dz = grad * self._activate_deriv(cache["z"], layer["act"], cache["out"])
            grads[index] = {"dW": cache["x"].T @ dz / len(dz), "db": dz.mean(axis=0)}
            grad = dz @ layer["W"].T
        return grads, grad

    def gradients(self, loss_grad, logits=True):
        if getattr(self, "_dual_head", False):
            raise RuntimeError("Use dual_gradients for a dual-head network")
        return self._backprop(loss_grad, list(range(len(self.layers))), logits)

    def _compute_grads(self, loss_grad, layer_indices):
        return self._backprop(loss_grad, layer_indices, logits=True)[0]

    def dual_gradients(self, policy_grad, value_grad, logits=True):
        p, v = self._trunk_len, self._trunk_len + 1
        gp, dxp = self._backprop(policy_grad, [p], logits)
        gv, dxv = self._backprop(value_grad, [v])
        trunk, dx = self._backprop(dxp + dxv, list(range(self._trunk_len)))
        return {**trunk, **gp, **gv}, dx

    def backward(self, loss_grad, logits=True):
        grads, dx = self.gradients(loss_grad, logits)
        self.apply_gradients(grads)
        return dx

    def backward_dual(self, policy_grad, value_grad, logits=True):
        grads, dx = self.dual_gradients(policy_grad, value_grad, logits)
        self.apply_gradients(grads)
        return dx

    def apply_gradients(self, grads):
        if any(not np.all(np.isfinite(g)) for layer in grads.values() for g in layer.values()):
            raise FloatingPointError("Non-finite parameter gradient")
        self._adam_t += 1
        self._adam_update(grads)

    def _adam_update(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        for i, values in grads.items():
            for key in ("W", "b"):
                grad = values["d" + key]
                if self.clip_grad is not None:
                    grad = np.clip(grad, -self.clip_grad, self.clip_grad)
                self._adam_m[i][key] = beta1 * self._adam_m[i][key] + (1 - beta1) * grad
                self._adam_v[i][key] = beta2 * self._adam_v[i][key] + (1 - beta2) * grad**2
                m = self._adam_m[i][key] / (1 - beta1**self._adam_t)
                v = self._adam_v[i][key] / (1 - beta2**self._adam_t)
                self.layers[i][key] -= self.lr * m / (np.sqrt(v) + eps)

    @staticmethod
    def _activate(z, act):
        if act == "relu":
            return np.maximum(z, 0)
        if act == "softmax":
            e = np.exp(z - z.max(axis=-1, keepdims=True))
            return e / e.sum(axis=-1, keepdims=True)
        if act == "tanh":
            return np.tanh(z)
        if act == "sigmoid":
            return np.exp(-np.logaddexp(0, -z))
        if act == "linear":
            return z
        raise ValueError(f"Unknown activation: {act}")

    @staticmethod
    def _activate_deriv(z, act, out):
        if act == "relu":
            return (z > 0).astype(out.dtype)
        if act == "tanh":
            return 1 - out**2
        if act == "sigmoid":
            return out * (1 - out)
        if act == "linear":
            return np.ones_like(z)
        raise ValueError("Softmax requires a Jacobian-vector product")

    def copy(self):
        return copy.deepcopy(self)

    def copy_weights_from(self, other):
        if [(l["W"].shape, l["act"]) for l in self.layers] != [
            (l["W"].shape, l["act"]) for l in other.layers
        ]:
            raise ValueError("Network architectures differ")
        for layer, source in zip(self.layers, other.layers):
            layer["W"], layer["b"] = source["W"].copy(), source["b"].copy()

    def soft_update(self, other, tau=0.01):
        if not 0 <= tau <= 1:
            raise ValueError("tau must be in [0,1]")
        for layer, source in zip(self.layers, other.layers):
            for key in ("W", "b"):
                layer[key] = tau * source[key] + (1 - tau) * layer[key]

    def save(self, filepath):
        data = {
            "format_version": self.FORMAT_VERSION,
            "layers": [(l["W"], l["b"], l["act"]) for l in self.layers],
            "lr": self.lr,
            "clip_grad": self.clip_grad,
            "dual_head": getattr(self, "_dual_head", False),
            "trunk_len": getattr(self, "_trunk_len", 0),
            "adam_t": self._adam_t,
            "adam_m": self._adam_m,
            "adam_v": self._adam_v,
            "rng_state": self.rng.bit_generator.state,
        }
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as stream:
            data = pickle.load(stream)
        if data.get("format_version", 1) > cls.FORMAT_VERSION:
            raise ValueError("Unsupported network checkpoint version")
        net = cls(lr=data["lr"], clip_grad=data["clip_grad"])
        for w, b, act in data["layers"]:
            w, b = np.asarray(w), np.asarray(b)
            if (
                w.ndim != 2
                or b.shape != (w.shape[1],)
                or not np.all(np.isfinite(w))
                or not np.all(np.isfinite(b))
            ):
                raise ValueError("Invalid checkpoint parameters")
            net.add_dense(*w.shape, act)
            net.layers[-1].update(W=w.copy(), b=b.copy())
        net._dual_head, net._trunk_len = data.get("dual_head", False), data.get("trunk_len", 0)
        if "adam_t" in data:
            net._adam_t, net._adam_m, net._adam_v = data["adam_t"], data["adam_m"], data["adam_v"]
            net.rng.bit_generator.state = data["rng_state"]
        return net

    def param_count(self):
        return sum(l["W"].size + l["b"].size for l in self.layers)
