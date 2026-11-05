"""Compatibility entry point for the reproducible experiment runner."""

from src.experiments import main
from src.registry import build_agent, available_agents_for_env, get_observation_size, get_hyperparameter_grid  # noqa: F401

if __name__ == "__main__":
    raise SystemExit(main())
