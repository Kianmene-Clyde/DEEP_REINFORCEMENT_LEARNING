"""Central registry and explicit experimental configurations."""

import itertools
import json
from typing import Any, Dict, List, Optional
import numpy as np
from .agents import (
    RandomAgent,
    TabularQLearningAgent,
    DeepQLearningAgent,
    DoubleDeepQLearningAgent,
    DDQNWithERAgent,
    DDQNWithPERAgent,
    REINFORCEAgent,
    REINFORCEMeanBaselineAgent,
    REINFORCECriticBaselineAgent,
    PPOAgent,
    A2CAgent,
    RandomRolloutAgent,
    MCTSAgent,
    ExpertApprenticeAgent,
    AlphaZeroAgent,
    MuZeroAgent,
    StochasticMuZeroAgent,
)
from .environments import LineWorld, GridWorld, TicTacToe, Quarto

ENVIRONMENTS = {"LineWorld": LineWorld, "GridWorld": GridWorld, "TicTacToe": TicTacToe, "Quarto": Quarto}


def make_environment(name, seed=None, opponent="random"):
    kwargs = {"seed": seed}
    if name in {"TicTacToe", "Quarto"}:
        kwargs["opponent_type"] = opponent
    return ENVIRONMENTS[name](**kwargs)


def available_agents_for_env(name):
    return [a for a in AGENT_ORDER if not (name == "Quarto" and a == "TabularQL")]


def build_agent(name, env_name, obs_size, action_size, params=None, seed=None):
    params = dict(params or {})
    params["seed"] = seed
    classes = {
        "Random": RandomAgent,
        "TabularQL": TabularQLearningAgent,
        "DQN": DeepQLearningAgent,
        "DoubleDQN": DoubleDeepQLearningAgent,
        "DDQN+ER": DDQNWithERAgent,
        "DDQN+PER": DDQNWithPERAgent,
        "REINFORCE": REINFORCEAgent,
        "REINFORCE+Mean": REINFORCEMeanBaselineAgent,
        "REINFORCE+Critic": REINFORCECriticBaselineAgent,
        "PPO": PPOAgent,
        "A2C": A2CAgent,
        "RandomRollout": RandomRolloutAgent,
        "MCTS": MCTSAgent,
        "ExpertApprentice": ExpertApprenticeAgent,
        "AlphaZero": AlphaZeroAgent,
        "MuZero": MuZeroAgent,
        "MuZero_Stochastic": StochasticMuZeroAgent,
    }
    if name in {"RandomRollout", "MCTS"}:
        if "search_budget" in params:
            params["num_rollouts" if name == "RandomRollout" else "num_simulations"] = params.pop(
                "search_budget"
            )
    if name in {"Random", "RandomRollout", "MCTS"}:
        return classes[name](action_size, **params)
    if name == "TabularQL":
        if env_name == "Quarto":
            raise ValueError("Quarto state space is unsuitable for a dense Q table")
        return classes[name](3**9 if env_name == "TicTacToe" else obs_size, action_size, **params)
    return classes[name](obs_size, action_size, **params)


PURE_PLANNING = {"Random", "RandomRollout", "MCTS"}
ALGORITHM_NOTES = {
    "AlphaZero": "AlphaZero-inspired; fixed opponent; no self-play; scalar value",
    "MuZero": "MuZero-inspired; scalar latent model; learned termination; no reanalysis; internal legality not modeled",
    "MuZero_Stochastic": "Experimental Gaussian one-step latent planning; not the published Stochastic MuZero",
    "ExpertApprentice": "Behavioral cloning from UCT; no iterative expert improvement",
    "DoubleDQN": "Online Double DQN ablation without replay",
}


def get_observation_size(env) -> int:
    obs_size = env.observation_space
    if isinstance(obs_size, tuple):
        obs_size = int(np.prod(obs_size))
    return int(obs_size)


AGENT_FAMILIES = {
    "Random": "baseline",
    "TabularQL": "tabular_value_based",
    "DQN": "deep_value_based",
    "DoubleDQN": "deep_value_based",
    "DDQN+ER": "deep_value_based",
    "DDQN+PER": "deep_value_based",
    "REINFORCE": "policy_gradient",
    "REINFORCE+Mean": "policy_gradient",
    "REINFORCE+Critic": "policy_gradient",
    "PPO": "actor_critic_policy_optimization",
    "A2C": "actor_critic",
    "RandomRollout": "planning",
    "MCTS": "planning",
    "ExpertApprentice": "hybrid_planning",
    "AlphaZero": "hybrid_planning",
    "MuZero": "hybrid_planning",
    "MuZero_Stochastic": "hybrid_planning",
}

GRID_SPECS = {
    "compact": {
        "baseline": {
            "fixed": [{}],
        },
        "tabular_value_based": {
            "learning_rate": [0.1, 0.3],
            "discount_factor": [0.95, 0.99],
            "epsilon_decay": [0.995, 0.999],
        },
        "deep_value_based": {
            "learning_rate": [0.001, 0.0005],
            "discount_factor": [0.95, 0.99],
            "epsilon_decay": [0.995, 0.998],
        },
        "policy_gradient": {
            "learning_rate": [0.005, 0.001],
            "discount_factor": [0.95, 0.99],
            "hidden_layers": [[64], [128, 128]],
        },
        "actor_critic_policy_optimization": {
            "learning_rate": [0.003, 0.0003],
            "discount_factor": [0.95, 0.99],
            "clip_ratio": [0.1, 0.2],
        },
        "actor_critic": {
            "learning_rate": [0.003, 0.0007],
            "discount_factor": [0.95, 0.99],
            "entropy_coef": [0.0, 0.01],
        },
        "planning": {
            "search_budget": [10, 30],
            "max_rollout_depth": [25, 50],
            "exploration_constant": [1.0, 1.41],
        },
        "hybrid_planning": {
            "num_simulations": [20, 50],
            "learning_rate": [0.001, 0.003],
            "hidden_layers": [[64], [128]],
        },
    },
    "full": {
        "baseline": {
            "fixed": [{}],
        },
        "tabular_value_based": {
            "learning_rate": [0.05, 0.1, 0.3],
            "discount_factor": [0.95, 0.99],
            "epsilon_decay": [0.99, 0.995, 0.999],
        },
        "deep_value_based": {
            "learning_rate": [0.001, 0.0005, 0.0001],
            "discount_factor": [0.95, 0.99],
            "epsilon_decay": [0.99, 0.995, 0.998],
        },
        "policy_gradient": {
            "learning_rate": [0.005, 0.001, 0.0003],
            "discount_factor": [0.95, 0.99],
            "hidden_layers": [[64], [128], [128, 128]],
        },
        "actor_critic_policy_optimization": {
            "learning_rate": [0.003, 0.001, 0.0003],
            "discount_factor": [0.95, 0.99],
            "clip_ratio": [0.1, 0.2, 0.3],
        },
        "actor_critic": {
            "learning_rate": [0.003, 0.001, 0.0007],
            "discount_factor": [0.95, 0.99],
            "entropy_coef": [0.0, 0.01, 0.02],
        },
        "planning": {
            "search_budget": [10, 30, 50],
            "max_rollout_depth": [25, 50],
            "exploration_constant": [1.0, 1.41, 2.0],
        },
        "hybrid_planning": {
            "num_simulations": [20, 30, 50],
            "learning_rate": [0.001, 0.003],
            "hidden_layers": [[64], [128], [128, 128]],
        },
    },
    "none": {
        "baseline": {"fixed": [{}]},
        "tabular_value_based": {
            "learning_rate": [0.1],
            "discount_factor": [0.99],
            "epsilon_decay": [0.999],
        },
        "deep_value_based": {
            "learning_rate": [0.001],
            "discount_factor": [0.99],
            "epsilon_decay": [0.998],
        },
        "policy_gradient": {
            "learning_rate": [0.001],
            "discount_factor": [0.99],
            "hidden_layers": [[128, 128]],
        },
        "actor_critic_policy_optimization": {
            "learning_rate": [0.0003],
            "discount_factor": [0.99],
            "clip_ratio": [0.2],
        },
        "actor_critic": {
            "learning_rate": [0.0007],
            "discount_factor": [0.99],
            "entropy_coef": [0.01],
        },
        "planning": {
            "search_budget": [30],
            "max_rollout_depth": [50],
            "exploration_constant": [1.41],
        },
        "hybrid_planning": {
            "num_simulations": [30],
            "learning_rate": [0.001],
            "hidden_layers": [[128]],
        },
    },
}

AGENT_ORDER = [
    "Random",
    "TabularQL",
    "DQN",
    "DoubleDQN",
    "DDQN+ER",
    "DDQN+PER",
    "REINFORCE",
    "REINFORCE+Mean",
    "REINFORCE+Critic",
    "PPO",
    "A2C",
    "RandomRollout",
    "MCTS",
    "ExpertApprentice",
    "AlphaZero",
    "MuZero",
    "MuZero_Stochastic",
]


def expand_grid(grid_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Décomposer une spécification de grille en une liste de dictionnaires de paramètres."""
    if "fixed" in grid_spec:
        return [dict(x) for x in grid_spec["fixed"]]

    keys = list(grid_spec.keys())
    values = [grid_spec[k] for k in keys]
    combos = []
    for product in itertools.product(*values):
        combos.append({k: v for k, v in zip(keys, product)})
    return combos


def get_hyperparameter_grid(
    agent_name: str, grid_mode: str, max_configs: Optional[int] = None
) -> List[Dict[str, Any]]:
    family = AGENT_FAMILIES[agent_name]
    grid_spec = GRID_SPECS[grid_mode][family]
    configs = expand_grid(grid_spec)

    if agent_name == "RandomRollout":
        cleaned = []
        seen = set()
        for cfg in configs:
            cfg = {k: v for k, v in cfg.items() if k != "exploration_constant"}
            key = json.dumps(_json_clean(cfg), sort_keys=True)
            if key not in seen:
                cleaned.append(cfg)
                seen.add(key)
        configs = cleaned

    if max_configs is not None and max_configs > 0:
        configs = configs[:max_configs]

    return configs


def _json_clean(v):
    if isinstance(v, dict):
        return {str(k): _json_clean(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_json_clean(x) for x in v]
    if isinstance(v, tuple):
        return [_json_clean(x) for x in v]
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v
