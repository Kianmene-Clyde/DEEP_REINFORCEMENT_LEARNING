import copy
import numpy as np
import pytest
from src.registry import make_environment
from src.environments import LineWorld, GridWorld, TicTacToe, Quarto


@pytest.mark.parametrize("name", ["LineWorld", "GridWorld", "TicTacToe", "Quarto"])
def test_seeded_episodes_and_invalid_actions(name):
    first, second = make_environment(name, 12), make_environment(name, 12)
    a, b = first.reset(seed=51), second.reset(seed=51)
    np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError):
        first.step(-1)
    with pytest.raises(ValueError):
        first.step(first.action_space)
    with pytest.raises(ValueError):
        first.step(0.5)
    for _ in range(30):
        valid = first.get_valid_actions(a)
        action = int(valid[-1])
        a, r, done, info = first.step(action)
        b, s, other, j = second.step(action)
        np.testing.assert_array_equal(a, b)
        assert (r, done, info) == (s, other, j)
        if done:
            assert "outcome" in info
            assert len(first.get_valid_actions(a)) == 0
            with pytest.raises(RuntimeError):
                first.step(action)
            break


def test_lineworld_reference_path():
    env = LineWorld()
    state = env.reset()
    total = 0
    for _ in range(9):
        state, reward, done, info = env.step(1)
        total += reward
    assert done and info["outcome"] == "success"
    assert total == pytest.approx(0.2)
    assert np.argmax(state) == 9


def test_gridworld_pit_and_shortest_safe_path():
    env = GridWorld()
    _, r, done, info = env.step(2)
    assert (r, done, info["outcome"]) == (-10.0, True, "failure")
    env.reset()
    rewards = []
    for action in [3, 3, 3, 1, 1, 1, 1]:
        _, r, done, info = env.step(action)
        rewards.append(r)
    assert done and info["outcome"] == "success"
    assert sum(rewards) == pytest.approx(10.54)


def test_tictactoe_win_and_opponent_macro_step():
    env = TicTacToe(opponent_type="heuristic")
    env.board = np.array([1, 1, 0, -1, -1, 0, 0, 0, 0], dtype=np.float32)
    _, r, done, info = env.step(2)
    assert (r, done, info["outcome"]) == (1.0, True, "win")
    env.reset()
    env.step(0)
    assert np.count_nonzero(env.board == 1) == 1
    assert np.count_nonzero(env.board == -1) == 1
    with pytest.raises(ValueError):
        env.step(0)


def reference_win(board):
    for line in Quarto.LINE_INDICES:
        pieces = board[line]
        if np.any(pieces < 0):
            continue
        for bit in range(4):
            if len(set(((pieces >> bit) & 1).tolist())) == 1:
                return True
    return False


@pytest.mark.parametrize("line", range(10))
def test_quarto_every_winning_line(line):
    env = Quarto()
    env.board[:] = -1
    env.board[env.LINE_INDICES[line]] = [0, 1, 2, 3]
    assert env._check_win()


def test_quarto_vectorized_rules_match_reference_without_mutation():
    rng = np.random.default_rng(7)
    env = Quarto()
    for _ in range(300):
        env.board[:] = -1
        n = int(rng.integers(0, 17))
        positions = rng.choice(16, n, replace=False)
        pieces = rng.permutation(16)
        env.board[positions] = pieces[:n]
        assert env._check_win() == reference_win(env.board)
        if n < 16:
            before = env.board.copy()
            piece = int(pieces[n])
            expected = False
            for pos in np.where(before < 0)[0]:
                candidate = before.copy()
                candidate[pos] = piece
                expected |= reference_win(candidate)
            assert env._would_piece_win_somewhere(piece) == expected
            np.testing.assert_array_equal(env.board, before)


def test_quarto_encoding_and_simulation_randomness():
    env = Quarto(seed=4)
    state = env.reset(seed=19)
    assert state.shape == (101,)
    assert state[80] == 1
    assert state[85:].sum() == 15
    assert np.all(state[:80:5] == 0)
    before = copy.deepcopy(env.rng.bit_generator.state)
    sim = env.clone(seed=999)
    sim.step(0)
    assert env.rng.bit_generator.state == before
    assert np.all(env.board == -1)
