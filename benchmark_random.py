"""End-to-end random-policy evaluation throughput with explicit terminal outcomes."""

import argparse
import json
import time
from pathlib import Path
from src.registry import ENVIRONMENTS, make_environment, build_agent
from src.training.evaluator import Evaluator


def benchmark(env, agent, num_games=1000, max_steps=200, seed=42):
    Evaluator(env, agent, min(5, num_games), max_steps, seed + 1000000).evaluate()
    start = time.perf_counter()
    result = Evaluator(env, agent, num_games, max_steps, seed).evaluate()
    elapsed = time.perf_counter() - start
    return {
        **result,
        "games_per_sec": num_games / elapsed,
        "total_time": elapsed,
        "avg_steps": result["avg_length"],
        "wins": round(result["win_rate"] * num_games),
        "losses": round(result["loss_rate"] * num_games),
        "draws": round(result["draw_rate"] * num_games),
        "timeouts": round(result["timeout_rate"] * num_games),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--env", nargs="+", choices=list(ENVIRONMENTS), default=list(ENVIRONMENTS))
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    results = {}
    for name in args.env:
        env = make_environment(name, args.seed)
        agent = build_agent("Random", name, env.observation_space, env.action_space, seed=args.seed)
        results[name] = benchmark(env, agent, args.games, args.max_steps, args.seed)
        print(
            f"{name}: {results[name]['games_per_sec']:.1f} games/s; "
            f"success={results[name]['success_rate']:.1%}; timeout={results[name]['timeout_rate']:.1%}"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
