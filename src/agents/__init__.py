"""Agents package - all RL algorithms."""
from .base_agent import BaseAgent
from .random_agent import RandomAgent
from .tabular_q_learning import TabularQLearningAgent
from .deep_q_learning import DeepQLearningAgent
from .double_deep_q_learning import (
    DoubleDeepQLearningAgent,
    DDQNWithERAgent,
    DDQNWithPERAgent,
)
from .reinforce import (
    REINFORCEAgent,
    REINFORCEMeanBaselineAgent,
    REINFORCECriticBaselineAgent,
)
from .ppo import PPOAgent
from .a2c import A2CAgent
from .random_rollout import RandomRolloutAgent
from .mcts import MCTSAgent
from .expert_apprentice import ExpertApprenticeAgent
from .alphazero import AlphaZeroAgent
from .muzero import MuZeroAgent
from .muzero_stochastic import StochasticMuZeroAgent

__all__ = [
    'BaseAgent',
    'RandomAgent',
    'TabularQLearningAgent',
    'DeepQLearningAgent',
    'DoubleDeepQLearningAgent',
    'DDQNWithERAgent',
    'DDQNWithPERAgent',
    'REINFORCEAgent',
    'REINFORCEMeanBaselineAgent',
    'REINFORCECriticBaselineAgent',
    'PPOAgent',
    'A2CAgent',
    'RandomRolloutAgent',
    'MCTSAgent',
    'ExpertApprenticeAgent',
    'AlphaZeroAgent',
    'MuZeroAgent',
    'StochasticMuZeroAgent',
]
