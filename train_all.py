#!/usr/bin/env python3
"""
train_all.py - Train all agents on all environments and collect metrics.

Usage:
    python train_all.py                    # Full training (long)
    python train_all.py --quick            # Quick test (100 episodes)
    python train_all.py --env LineWorld     # Only one environment
    python train_all.py --agent DQN        # Only one agent
"""
import sys, os, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from environments import LineWorld, GridWorld, TicTacToe, Quarto
from agents import *
from training.trainer import Trainer
from training.evaluator import Evaluator
from training.metrics import Metrics
from utils.plotting import plot_learning_curves, plot_comparison, plot_checkpoint_table


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

def get_environments():
    return {
        'LineWorld': LineWorld(length=10),
        'GridWorld': GridWorld(width=5, height=5),
        'TicTacToe': TicTacToe(opponent_type='random'),
        'Quarto': Quarto(opponent_type='random'),
    }


def get_agents(env_name, obs_size, action_size):
    """Create agent instances appropriate for the environment."""
    agents = {}

    # Common agents for all environments
    agents['Random'] = RandomAgent(action_size)

    # Tabular only for small state spaces
    if env_name in ('LineWorld', 'GridWorld'):
        agents['TabularQL'] = TabularQLearningAgent(
            obs_size, action_size, learning_rate=0.1, epsilon=0.3, epsilon_decay=0.999)

    # Deep agents
    agents['DQN'] = DeepQLearningAgent(
        obs_size, action_size, learning_rate=0.001, epsilon_decay=0.998)
    agents['DoubleDQN'] = DoubleDeepQLearningAgent(
        obs_size, action_size, learning_rate=0.001, epsilon_decay=0.998)
    agents['DDQN+ER'] = DDQNWithERAgent(
        obs_size, action_size, learning_rate=0.001, epsilon_decay=0.998)
    agents['DDQN+PER'] = DDQNWithPERAgent(
        obs_size, action_size, learning_rate=0.001, epsilon_decay=0.998)
    agents['REINFORCE'] = REINFORCEAgent(
        obs_size, action_size, learning_rate=0.001)
    agents['REINFORCE+Mean'] = REINFORCEMeanBaselineAgent(
        obs_size, action_size, learning_rate=0.001)
    agents['REINFORCE+Critic'] = REINFORCECriticBaselineAgent(
        obs_size, action_size, learning_rate=0.001)
    agents['PPO'] = PPOAgent(
        obs_size, action_size, learning_rate=0.0003)
    agents['A2C'] = A2CAgent(
        obs_size, action_size, learning_rate=0.001)

    # Planning agents (slower but need fewer episodes)
    agents['RandomRollout'] = RandomRolloutAgent(
        action_size, num_rollouts=20, max_rollout_depth=50)
    agents['MCTS'] = MCTSAgent(
        action_size, num_simulations=50, max_rollout_depth=50)
    agents['ExpertApprentice'] = ExpertApprenticeAgent(
        obs_size, action_size, expert_simulations=30, hidden_layers=[128])
    agents['AlphaZero'] = AlphaZeroAgent(
        obs_size, action_size, num_simulations=30, hidden_layers=[128])
    agents['MuZero'] = MuZeroAgent(
        obs_size, action_size, num_simulations=20, latent_dim=64, hidden_layers=[128])
    agents['MuZero_Stochastic'] = StochasticMuZeroAgent(
        obs_size, action_size, num_simulations=20, latent_dim=64, hidden_layers=[128])

    return agents


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='DRL Training Pipeline')
    parser.add_argument('--quick', action='store_true', help='Quick test mode (100 ep)')
    parser.add_argument('--env', type=str, default=None, help='Train only this environment')
    parser.add_argument('--agent', type=str, default=None, help='Train only this agent')
    parser.add_argument('--episodes', type=int, default=None, help='Number of episodes')
    args = parser.parse_args()

    if args.quick:
        max_episodes = 100
        checkpoints = []
        eval_eps = 20
    else:
        max_episodes = args.episodes or 10000
        checkpoints = [cp for cp in [1000, 10000, 100000] if cp <= max_episodes]
        eval_eps = 100

    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    all_results = {}
    envs = get_environments()

    if args.env:
        envs = {k: v for k, v in envs.items() if k == args.env}

    total_start = time.time()

    for env_name, env in envs.items():
        print(f'\n{"="*70}')
        print(f' Environment: {env_name}')
        print(f'{"="*70}')

        obs_size = env.observation_space
        if isinstance(obs_size, tuple):
            obs_size = int(np.prod(obs_size))

        agents = get_agents(env_name, obs_size, env.action_space)
        if args.agent:
            agents = {k: v for k, v in agents.items() if k == args.agent}

        env_results = {}
        metrics_dict = {}

        for agent_name, agent in agents.items():
            print(f'\n  Training {agent_name}...')
            t0 = time.time()

            # Planning agents: reduce episodes (they're slower per episode)
            is_planning = hasattr(agent, 'set_env') and agent_name not in ['ExpertApprentice']
            ep = min(max_episodes, 500) if is_planning and not args.quick else max_episodes
            cp = [c for c in checkpoints if c <= ep]

            try:
                trainer = Trainer(
                    env, agent,
                    max_episodes=ep,
                    max_steps_per_episode=200,
                    checkpoints=cp,
                    eval_episodes=eval_eps,
                    verbose=True
                )
                metrics = trainer.train()

                elapsed = time.time() - t0
                print(f'  Completed in {elapsed:.1f}s')

                # Save model
                model_path = f'models/{env_name}_{agent_name}'
                try:
                    agent.save(model_path)
                except Exception:
                    pass

                # Save metrics
                metrics.save(f'results/{env_name}_{agent_name}_metrics.json')
                metrics_dict[agent_name] = metrics

                # Final evaluation
                evaluator = Evaluator(env, agent, num_episodes=eval_eps, max_steps=200)
                eval_res = evaluator.evaluate()
                env_results[agent_name] = eval_res
                print(f'  Eval: avg_reward={eval_res["avg_reward"]:.4f}  '
                      f'win_rate={eval_res["win_rate"]:.2%}  '
                      f'step_time={eval_res["avg_step_time_ms"]:.3f}ms')

            except Exception as e:
                print(f'  ERROR: {type(e).__name__}: {e}')
                env_results[agent_name] = {'avg_reward': 0, 'error': str(e)}

        all_results[env_name] = env_results

        # Plot for this environment
        if metrics_dict:
            try:
                plot_learning_curves(metrics_dict, env_name, 'results')
                plot_comparison(env_results, env_name, 'results')
            except Exception as e:
                print(f'  Plot error: {e}')

    # Summary
    print(f'\n{"="*70}')
    print(f' SUMMARY')
    print(f'{"="*70}')

    for env_name, results in all_results.items():
        print(f'\n  {env_name}:')
        sorted_agents = sorted(results.items(),
                               key=lambda x: x[1].get('avg_reward', -999), reverse=True)
        for rank, (agent_name, res) in enumerate(sorted_agents, 1):
            r = res.get('avg_reward', 0)
            t = res.get('avg_step_time_ms', 0)
            print(f'    #{rank:2d} {agent_name:30s}  reward={r:8.4f}  time={t:.3f}ms')

    # Save full results
    serializable = {}
    for env_name, results in all_results.items():
        serializable[env_name] = {}
        for agent_name, res in results.items():
            serializable[env_name][agent_name] = {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in res.items()
            }
    with open('results/all_results.json', 'w') as f:
        json.dump(serializable, f, indent=2)

    # Summary table plot
    try:
        plot_checkpoint_table(all_results, 'results')
    except Exception:
        pass

    total_time = time.time() - total_start
    print(f'\nTotal training time: {total_time:.1f}s ({total_time/60:.1f}min)')
    print('Results saved to results/')
    print('Models saved to models/')


if __name__ == '__main__':
    main()
