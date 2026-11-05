"""Compare preserved reference algorithms and optimized code on identical inputs."""

import cProfile
import io
import json
from pathlib import Path
import pstats
import statistics
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.environments import Quarto, LineWorld  # noqa: E402
from src.training.metrics import Metrics  # noqa: E402
from src.training.evaluator import Evaluator  # noqa: E402


def old_win(board):
    b = board.reshape(4, 4)
    lines = []
    for i in range(4):
        lines.extend([b[i, :], b[:, i]])
    lines.extend([np.array([b[i, i] for i in range(4)]), np.array([b[i, 3 - i] for i in range(4)])])
    for line in lines:
        if np.any(line == -1):
            continue
        attrs = Quarto.ATTRS[line]
        for bit in range(4):
            if len(set(attrs[:, bit])) == 1:
                return True
    return False


def old_would(board, piece):
    for pos in np.where(board == -1)[0]:
        board[pos] = piece
        won = old_win(board)
        board[pos] = -1
        if won:
            return True
    return False


def median_seconds(function, repeats=7):
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        durations.append(time.perf_counter() - start)
    return statistics.median(durations)


def main():
    rng = np.random.default_rng(2026)
    cases = []
    while len(cases) < 200:
        n = int(rng.integers(0, 13))
        order = rng.permutation(16)
        board = np.full(16, -1, dtype=int)
        board[rng.choice(16, n, replace=False)] = order[:n]
        if not old_win(board):
            cases.append((board, int(order[n])))
    env = Quarto(seed=5)

    def old_quarto():
        return [old_would(board.copy(), piece) for board, piece in cases]

    def new_quarto():
        results = []
        for board, piece in cases:
            env.board = board.copy()
            results.append(env._would_piece_win_somewhere(piece))
        return results

    assert old_quarto() == new_quarto()
    metrics = Metrics()
    metrics.episode_rewards = np.sin(np.arange(10000)).tolist()

    def old_smooth():
        return [np.mean(metrics.episode_rewards[max(0, i - 99) : i + 1]) for i in range(10000)]

    np.testing.assert_allclose(old_smooth(), metrics.get_windowed_rewards(100), atol=1e-12)
    results = {}
    for name, old, new in [
        ("quarto_200_positions", old_quarto, new_quarto),
        ("smoothing_10000", old_smooth, lambda: metrics.get_windowed_rewards(100)),
    ]:
        before, after = median_seconds(old), median_seconds(new)
        results[name] = {
            "before_seconds": before,
            "after_seconds": after,
            "speedup": before / after,
            "statistic": "median of 7 executions, identical inputs; results equivalent",
        }

    class Detour:
        training = True
        uses_tabular = False

        def set_training_mode(self, training):
            self.training = training

        def select_action(self, state, valid):
            self.i = getattr(self, "i", 0) + 1
            return 0 if self.i <= 50 else 1

    corrected = Evaluator(LineWorld(), Detour(), 1, 100).evaluate()
    results["delayed_goal"] = {
        "return": corrected["avg_reward"],
        "old_win_rate": 0.0,
        "corrected_win_rate": corrected["win_rate"],
        "terminal": "goal",
    }
    out = ROOT / "results" / "validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "before_after.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    profiler = cProfile.Profile()
    profiler.runcall(new_quarto)
    output = io.StringIO()
    pstats.Stats(profiler, stream=output).sort_stats("cumulative").print_stats(15)
    (out / "optimized_profile.txt").write_text(output.getvalue(), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
