import numpy as np
import pytest
from src.neural_network.model import NeuralNetwork


def finite_difference(net, loss, grads):
    epsilon = 1e-5
    for i, layer in enumerate(net.layers):
        for key in ["W", "b"]:
            for index in np.ndindex(layer[key].shape):
                original = layer[key][index]
                layer[key][index] = original + epsilon
                plus = loss()
                layer[key][index] = original - epsilon
                minus = loss()
                layer[key][index] = original
                assert grads[i]["d" + key][index] == pytest.approx(
                    (plus - minus) / (2 * epsilon), abs=3e-6, rel=3e-5
                )


def as_double(net):
    for layer in net.layers:
        for key in ["W", "b"]:
            layer[key] = layer[key].astype(float)
    return net


@pytest.mark.parametrize("acts", [["tanh", "linear"], ["softmax", "linear"]])
def test_dense_gradients_and_softmax_jacobian(acts):
    net = as_double(NeuralNetwork.build_mlp([2, 3, 2], acts, rng=np.random.default_rng(1)))
    x = np.array([[0.2, -0.4], [0.7, 0.1]], dtype=float)
    target = np.array([[0.5, 0.1], [-0.2, 0.3]])
    out = net.forward(x, True)
    grads, dx = net.gradients(out - target, logits=False)
    finite_difference(net, lambda: 0.5 * np.sum((net.predict(x) - target) ** 2) / len(x), grads)
    epsilon = 1e-5
    for index in np.ndindex(x.shape):
        original = x[index]
        x[index] = original + epsilon
        plus = 0.5 * np.sum((net.predict(x) - target) ** 2)
        x[index] = original - epsilon
        minus = 0.5 * np.sum((net.predict(x) - target) ** 2)
        x[index] = original
        assert dx[index] == pytest.approx((plus - minus) / (2 * epsilon), abs=3e-6)


def test_dual_head_shared_trunk_gradients():
    net = as_double(NeuralNetwork.build_dual_head(2, [3], 2, rng=np.random.default_rng(2)))
    x = np.array([[0.1, 0.7], [-0.3, 0.5]])
    targets = np.array([[1.0, 0.0], [0.0, 1.0]])
    values = np.array([[2.0], [-1.0]])
    p, v = net.forward(x, True)
    grads, _ = net.dual_gradients(p - targets, v - values)

    def loss():
        p, v = net.predict(x)
        return (-np.sum(targets * np.log(p)) + 0.5 * np.sum((v - values) ** 2)) / len(x)

    finite_difference(net, loss, grads)


def test_batch_is_averaged_once():
    net = NeuralNetwork.build_mlp([2, 3, 1], ["tanh", "linear"], rng=np.random.default_rng(3))
    x = np.array([[0.2, 0.5]])
    p = net.forward(x, True)
    one, _ = net.gradients(p - 1)
    p = net.forward(np.repeat(x, 7, axis=0), True)
    seven, _ = net.gradients(p - 1)
    for i in one:
        for key in one[i]:
            np.testing.assert_allclose(one[i][key], seven[i][key], rtol=1e-6, atol=1e-7)


def test_adam_checkpoint_resumes_exactly(tmp_path):
    net = NeuralNetwork.build_mlp([2, 4, 1], ["tanh", "linear"], rng=np.random.default_rng(5))
    x = np.array([[0.3, 0.2], [-0.1, 0.5]])
    net.backward(net.forward(x, True) - 0.2)
    net.save(tmp_path / "network.pkl")
    loaded = NeuralNetwork.load(tmp_path / "network.pkl")
    assert loaded._adam_t == 1
    for model in [net, loaded]:
        model.backward(model.forward(x, True) - 0.7)
    for first, second in zip(net.layers, loaded.layers):
        np.testing.assert_array_equal(first["W"], second["W"])
        np.testing.assert_array_equal(first["b"], second["b"])


def test_network_learns_xor():
    net = NeuralNetwork.build_mlp([2, 8, 1], ["tanh", "linear"], lr=0.03, rng=np.random.default_rng(7))
    x = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    for _ in range(400):
        net.backward(net.forward(x, True) - y)
    assert np.mean((net.predict(x) - y) ** 2) < 1e-4


def test_nonfinite_gradients_do_not_update_weights():
    net = NeuralNetwork.build_mlp([1, 1], rng=np.random.default_rng(1))
    net.forward([[1.0]], True)
    before = net.layers[0]["W"].copy()
    with pytest.raises(FloatingPointError):
        net.backward(np.array([[np.nan]]))
    np.testing.assert_array_equal(before, net.layers[0]["W"])
