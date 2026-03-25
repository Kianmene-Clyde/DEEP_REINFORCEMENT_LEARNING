#!/usr/bin/env python3
"""
Examples of using the DRL framework.
Run: python examples.py
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from environments import LineWorld, GridWorld, TicTacToe, Quarto
from agents import *
from training.trainer import Trainer
from training.evaluator import Evaluator


def example_1_lineworld():
    """Train multiple agents on LineWorld and compare."""
    print("\n" + "="*60)
    print("Example 1: LineWorld - Agent Comparison")
    print("="*60)

    env = LineWorld(length=10)
    obs = env.observation_space

    agents = {
        'Random': RandomAgent(env.action_space),
        'TabularQL': TabularQLearningAgent(obs, env.action_space, epsilon=0.3),
        'DQN': DeepQLearningAgent(obs, env.action_space, epsilon_decay=0.995),
        'REINFORCE': REINFORCEAgent(obs, env.action_space),
    }

    for name, agent in agents.items():
        print(f"\n  Training {name}...")
        trainer = Trainer(env, agent, max_episodes=1000, max_steps_per_episode=50,
                          checkpoints=[1000], eval_episodes=50, verbose=False)
        m = trainer.train()
        print(f"    Avg reward: {m.get_average_reward():.3f}")
        print(f"    Avg length: {m.get_average_length():.1f}")

        evaluator = Evaluator(env, agent, num_episodes=50)
        res = evaluator.evaluate()
        print(f"    Eval reward: {res['avg_reward']:.3f}")


def example_2_tictactoe():
    """Train DQN on TicTacToe vs random."""
    print("\n" + "="*60)
    print("Example 2: TicTacToe - DQN vs Random")
    print("="*60)

    env = TicTacToe(opponent_type='random')
    agent = DeepQLearningAgent(9, 9, learning_rate=0.001, epsilon_decay=0.998)

    trainer = Trainer(env, agent, max_episodes=2000, checkpoints=[1000, 2000],
                      eval_episodes=100, verbose=True)
    m = trainer.train()

    evaluator = Evaluator(env, agent, num_episodes=200)
    res = evaluator.evaluate()
    print(f"\n  Final win rate: {res['win_rate']:.1%}")
    print(f"  Draw rate: {res['draw_rate']:.1%}")


def example_3_mcts():
    """MCTS on TicTacToe."""
    print("\n" + "="*60)
    print("Example 3: TicTacToe - MCTS")
    print("="*60)

    env = TicTacToe(opponent_type='random')
    agent = MCTSAgent(9, num_simulations=100)

    trainer = Trainer(env, agent, max_episodes=50, eval_episodes=50, 
                      checkpoints=[], verbose=False)
    m = trainer.train()

    evaluator = Evaluator(env, agent, num_episodes=100)
    res = evaluator.evaluate()
    print(f"  Win rate: {res['win_rate']:.1%}")
    print(f"  Avg step time: {res['avg_step_time_ms']:.1f}ms")


def example_4_save_load():
    """Save and load a trained agent."""
    print("\n" + "="*60)
    print("Example 4: Save and Load")
    print("="*60)

    env = LineWorld(length=10)
    agent = TabularQLearningAgent(10, 2, epsilon=0.1)

    trainer = Trainer(env, agent, max_episodes=2000, checkpoints=[], verbose=False)
    trainer.train()
    agent.save('models/example_tabular')
    print("  Model saved")

    agent2 = TabularQLearningAgent(10, 2)
    agent2.load('models/example_tabular')
    print("  Model loaded")

    evaluator = Evaluator(env, agent2, num_episodes=100)
    res = evaluator.evaluate()
    print(f"  Loaded agent reward: {res['avg_reward']:.3f}")


if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    example_1_lineworld()
    example_2_tictactoe()
    example_3_mcts()
    example_4_save_load()
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
