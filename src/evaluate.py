"""Evaluate a versioned checkpoint on independently seeded episodes."""

import argparse
from pathlib import Path
from .registry import make_environment
from .training.checkpoints import load_checkpoint
from .training.evaluator import Evaluator
from .experiments import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3000000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Choose a new output file to preserve previous evaluations")
    agent, metadata = load_checkpoint(args.checkpoint)
    env = make_environment(metadata["environment"], opponent=metadata.get("opponent", "random"))
    evaluator = Evaluator(env, agent, args.episodes, args.max_steps, args.seed)
    result = evaluator.evaluate()
    write_json(
        args.output,
        {
            "checkpoint": str(args.checkpoint),
            "metadata": metadata,
            "evaluation": result,
            "episodes": evaluator.episodes,
        },
    )
    print(
        f"Return={result['avg_reward']:.4f}; success={result['success_rate']:.1%}; "
        f"timeout={result['timeout_rate']:.1%}; output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
