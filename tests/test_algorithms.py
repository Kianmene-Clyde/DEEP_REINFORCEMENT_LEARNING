import pickle
import numpy as np
import pytest
from src.agents import (
    DeepQLearningAgent,
    DDQNWithERAgent,
    PPOAgent,
    A2CAgent,
    MCTSAgent,
    RandomRolloutAgent,
    REINFORCEMeanBaselineAgent,
)
from src.agents.utils import masked_batch
from src.training.replay_buffer import PrioritizedReplayBuffer
from src.training.evaluator import Evaluator
from src.training.trainer import Trainer
from src.training.checkpoints import save_checkpoint, load_checkpoint
from src.environments import LineWorld, TicTacToe
from src.registry import AGENT_ORDER, available_agents_for_env, make_environment, build_agent


@pytest.mark.parametrize("cls,expected", [(DeepQLearningAgent, 5.5), (DDQNWithERAgent, 3.5)])
def test_bellman_targets_mask_illegal_actions(cls, expected):
    agent = cls(1, 3, discount_factor=0.5, hidden_layers=[2], seed=1)

    class Q:
        def __init__(self, values):
            self.values = np.array([values], dtype=float)

        def predict(self, states):
            return np.repeat(self.values, len(states), axis=0)

    agent.q_net, agent.target_net = Q([100, 2, 1]), Q([200, 5, 9])
    masks = np.array([[False, True, True], [False, False, False]])
    result = agent.bellman_targets(np.array([1.0, 7.0]), np.zeros((2, 1)), np.array([False, True]), masks)
    np.testing.assert_allclose(result, [expected, 7.0])


def test_proportional_replay_priority_and_importance_weights():
    replay = PrioritizedReplayBuffer(4, alpha=0.5, beta=0.7, beta_increment=0, rng=np.random.default_rng(5))
    for i in range(3):
        replay.push([i], 0, 0, [i + 1], False)
    replay.update_priorities(np.arange(3), np.array([1.0, 4.0, 9.0]))
    raw = np.array([1.0, 4.0, 9.0]) + 1e-6
    expected = np.sqrt(raw) / np.sqrt(raw).sum()
    np.testing.assert_allclose(replay.probabilities(), expected)
    sample = replay.sample(20000)
    indices, weights = sample[-2:]
    np.testing.assert_allclose(np.bincount(indices, minlength=3) / len(indices), expected, atol=0.012)
    expected_weights = (3 * expected[indices]) ** (-0.7) / (3 * expected.min()) ** (-0.7)
    np.testing.assert_allclose(weights, expected_weights, rtol=1e-6)
    replay.push([4], 0, 0, [5], False)
    assert replay.priorities[3] == pytest.approx(9.0 + 1e-6)


@pytest.mark.parametrize("advantage,old_probability", [(1.0, 0.45), (-1.0, 0.65), (1.0, 0.1), (-1.0, 0.95)])
def test_ppo_clipped_gradient_finite_difference(advantage, old_probability):
    logits = np.array([[0.2, -0.3, 0.5]], dtype=float)
    mask = np.array([[True, False, True]])
    actions = np.array([0])

    def probs(z):
        p = np.exp(z - z.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        return masked_batch(p, mask)

    p = probs(logits)
    old = np.log([old_probability])
    analytic = PPOAgent._clip_policy_gradient(p, actions, np.array([advantage]), old, 0.2)[0]

    def loss(z):
        ratio = probs(z)[0, 0] / old_probability
        return -min(ratio * advantage, np.clip(ratio, 0.8, 1.2) * advantage)

    for j in range(3):
        plus, minus = logits.copy(), logits.copy()
        plus[0, j] += 1e-5
        minus[0, j] -= 1e-5
        assert analytic[j] == pytest.approx((loss(plus) - loss(minus)) / (2e-5), abs=2e-7)


def test_ppo_bootstrapped_gae():
    agent = PPOAgent(2, 2, discount_factor=0.9, gae_lambda=0.95, hidden_layers=[2], seed=1)
    returns, advantages = agent._compute_gae(np.array([1.0, 2.0]), np.array([0.5, 0.25]), 3.0)
    last = 2 + 0.9 * 3 - 0.25
    first = 1 + 0.9 * 0.25 - 0.5 + 0.9 * 0.95 * last
    np.testing.assert_allclose(advantages, [first, last], rtol=1e-6)
    np.testing.assert_allclose(returns, advantages + [0.5, 0.25])


def test_a2c_bootstrap_and_historical_mean_baseline():
    agent = A2CAgent(2, 2, discount_factor=0.5, hidden_layers=[2], seed=1)
    agent.critic.layers[-1]["W"][:] = 0
    agent.critic.layers[-1]["b"][:] = 2
    agent.ep_rewards = [1.0, 1.0]
    agent._last_next_state = np.array([0.0, 1.0])
    agent._last_done = False
    np.testing.assert_allclose(agent._compute_returns(), [2.0, 2.0])
    agent._last_done = True
    np.testing.assert_allclose(agent._compute_returns(), [1.5, 1.0])
    baseline = REINFORCEMeanBaselineAgent(2, 2, hidden_layers=[2], seed=1)
    np.testing.assert_allclose(baseline._get_baseline(np.array([2.0, 4.0])), [0, 0])
    np.testing.assert_allclose(baseline._get_baseline(np.array([10.0])), [3.0])


def test_truncation_preserves_bootstrap_in_replay():
    agent = DeepQLearningAgent(10, 2, batch_size=99, hidden_layers=[2], seed=4)
    metrics = Trainer(
        LineWorld(),
        agent,
        max_episodes=1,
        max_steps_per_episode=1,
        eval_episodes=1,
        checkpoints=[],
        verbose=False,
    ).train()
    assert not agent.buffer.buffer[0][4]
    assert agent.buffer.buffer[0][5].all()
    assert metrics.episode_truncated == [True]


@pytest.mark.parametrize("cls", [MCTSAgent, RandomRolloutAgent])
def test_search_preserves_environment_and_finds_immediate_win(cls):
    env = TicTacToe(seed=9)
    env.board = np.array([1, 1, 0, -1, -1, 0, 0, 0, 0], dtype=np.float32)
    agent = cls(9, num_simulations=40, num_rollouts=20, max_rollout_depth=6, seed=3)
    agent.set_env(env)
    before = pickle.dumps(env)
    assert agent.select_action(env._get_state(), env.get_valid_actions(env._get_state())) == 2
    assert pickle.dumps(env) == before


def test_evaluation_isolated_and_counts_success_with_negative_return():
    class Detour:
        uses_tabular = False
        training = True

        def set_training_mode(self, t):
            self.training = t

        def select_action(self, s, v):
            self.i = getattr(self, "i", 0) + 1
            return 0 if self.i <= 50 else 1

    agent = Detour()
    result = Evaluator(LineWorld(), agent, 1, 100).evaluate()
    assert result["avg_reward"] < 0
    assert result["win_rate"] == 1
    assert agent.training and not hasattr(agent, "i")
    env = LineWorld()
    trained = build_agent("PPO", "LineWorld", 10, 2, {"hidden_layers": [4]}, seed=2)
    before = pickle.dumps(trained)
    initial = pickle.dumps(env)
    first = Evaluator(env, trained, 3, 10, seed=123)
    second = Evaluator(env, trained, 3, 10, seed=123)
    first.evaluate()
    second.evaluate()
    assert first.episodes == second.episodes
    assert pickle.dumps(trained) == before
    assert pickle.dumps(env) == initial


@pytest.mark.parametrize("name", AGENT_ORDER)
@pytest.mark.parametrize("env_name", ["LineWorld", "GridWorld", "TicTacToe", "Quarto"])
def test_every_supported_agent_trains_evaluates_and_checkpoints(name, env_name, tmp_path):
    if name not in available_agents_for_env(env_name):
        pytest.skip("Dense Quarto tabulation is intractable")
    env = make_environment(env_name, seed=7)
    params = {
        "hidden_layers": [8],
        "batch_size": 4,
        "train_batch_size": 4,
        "num_simulations": 2,
        "num_rollouts": 1,
        "expert_simulations": 2,
        "max_rollout_depth": 3,
        "max_search_depth": 3,
        "latent_dim": 4,
        "unroll_steps": 2,
        "num_epochs": 2,
    }
    agent = build_agent(name, env_name, env.observation_space, env.action_space, params, seed=3)
    Trainer(env, agent, 4, 10, eval_episodes=1, checkpoints=[], verbose=False, seed=8).train()
    before = pickle.dumps(agent)
    evaluator = Evaluator(env, agent, 2, 10, seed=19)
    result = evaluator.evaluate()
    assert np.isfinite(result["avg_reward"])
    assert result["unknown_outcome_rate"] == 0
    assert pickle.dumps(agent) == before
    path = tmp_path / "checkpoint.pkl"
    save_checkpoint(
        path,
        agent,
        {"environment": env_name, "observation_size": env.observation_space, "action_size": env.action_space},
    )
    loaded, _ = load_checkpoint(
        path, environment=env_name, observation_size=env.observation_space, action_size=env.action_space
    )
    again = Evaluator(env, loaded, 2, 10, seed=19)
    again.evaluate()
    assert evaluator.episodes == again.episodes


def test_resumption_matches_uninterrupted_training(tmp_path):
    params = {"hidden_layers": [4], "batch_size": 4}
    full = build_agent("DQN", "LineWorld", 10, 2, params, seed=3)
    split = build_agent("DQN", "LineWorld", 10, 2, params, seed=3)
    Trainer(LineWorld(), full, 6, 15, eval_episodes=1, checkpoints=[], verbose=False, seed=5).train()
    Trainer(LineWorld(), split, 3, 15, eval_episodes=1, checkpoints=[], verbose=False, seed=5).train()
    save_checkpoint(tmp_path / "resume.pkl", split, {})
    loaded, _ = load_checkpoint(tmp_path / "resume.pkl")
    Trainer(
        LineWorld(), loaded, 3, 15, eval_episodes=1, checkpoints=[], verbose=False, seed=5, start_episode=3
    ).train()
    for a, b in zip(full.q_net.layers, loaded.q_net.layers):
        np.testing.assert_array_equal(a["W"], b["W"])
    assert full.epsilon == loaded.epsilon


def test_muzero_k_step_gradients_match_full_unrolled_objective(monkeypatch):
    from src.agents import MuZeroAgent

    agent = MuZeroAgent(
        2, 2, hidden_layers=[3], latent_dim=2, unroll_steps=2, num_simulations=2, train_batch_size=2, seed=7
    )
    for net in [agent.repr_net, agent.pred_net, agent.dyn_net]:
        for layer in net.layers:
            for key in ["W", "b"]:
                layer[key] = layer[key].astype(float)
            # Central differences are undefined at a ReLU kink. Avoid the
            # zero-latent/zero-bias states produced by initial zero biases.
            layer["b"] += np.linspace(0.03, 0.07, layer["b"].size)
    trajectory = [
        {
            "obs": np.array([0.2, 0.4]),
            "action": 0,
            "pi": np.array([0.7, 0.3]),
            "reward": 0.3,
            "terminated": False,
            "value_target": 0.8,
        },
        {
            "obs": np.array([0.5, 0.1]),
            "action": 1,
            "pi": np.array([0.2, 0.8]),
            "reward": 0.5,
            "terminated": True,
            "value_target": 0.5,
        },
    ]
    data = [(trajectory, 0), (trajectory, 1)]
    monkeypatch.setattr(agent, "_sample_positions", lambda: data)
    captured = {}
    for name in ["repr_net", "pred_net", "dyn_net"]:
        monkeypatch.setattr(
            getattr(agent, name), "apply_gradients", lambda grads, n=name: captured.update({n: grads})
        )
    agent._train_networks()

    def loss():
        latent = agent.repr_net.predict(np.array([tr[t]["obs"] for tr, t in data]))
        total = 0.0
        for k in range(agent.K + 1):
            p, v = agent.pred_net.predict(latent)
            for row, (tr, t) in enumerate(data):
                if t + k < len(tr):
                    item = tr[t + k]
                    total += (
                        -np.sum(item["pi"] * np.log(p[row])) + 0.5 * (v[row, 0] - item["value_target"]) ** 2
                    )
            if k == agent.K:
                break
            acts = [tr[t + k]["action"] if t + k < len(tr) else 0 for tr, t in data]
            out = agent.dyn_net.predict(np.concatenate([latent, np.eye(2)[acts]], axis=1))
            for row, (tr, t) in enumerate(data):
                if t + k < len(tr):
                    item = tr[t + k]
                    total += 0.5 * (out[row, 2] - item["reward"]) ** 2
                    total += np.logaddexp(0, out[row, 3]) - item["terminated"] * out[row, 3]
            latent = np.tanh(out[:, :2])
        return total / (len(data) * (agent.K + 1))

    for name in captured:
        net = getattr(agent, name)
        for i, layer in enumerate(net.layers):
            for key in ["W", "b"]:
                for index in [tuple(0 for _ in layer[key].shape), tuple(s - 1 for s in layer[key].shape)]:
                    original = layer[key][index]
                    layer[key][index] = original + 1e-5
                    plus = loss()
                    layer[key][index] = original - 1e-5
                    minus = loss()
                    layer[key][index] = original
                    assert captured[name][i]["d" + key][index] == pytest.approx(
                        (plus - minus) / 2e-5, abs=2e-6, rel=1e-4
                    ), (name, i, key, index)


def test_constructing_agents_does_not_reseed_global_numpy():
    np.random.seed(123)
    expected = np.random.random(4)
    np.random.seed(123)
    build_agent("PPO", "LineWorld", 10, 2, {"hidden_layers": [4]}, seed=99)
    make_environment("Quarto", seed=555)
    np.testing.assert_array_equal(np.random.random(4), expected)
