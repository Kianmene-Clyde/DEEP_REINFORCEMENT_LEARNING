import argparse
import csv
import itertools
import json
import os
import re
import sys
import time
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from environments import LineWorld, GridWorld, TicTacToe, Quarto
from agents import *
from training.trainer import Trainer
from training.evaluator import Evaluator
from utils.plotting import plot_learning_curves, plot_comparison, plot_checkpoint_table


# Configuration des environnements
def get_environments():
    return {
        'LineWorld': LineWorld(length=10),
        'GridWorld': GridWorld(width=5, height=5),
        'TicTacToe': TicTacToe(opponent_type='random'),
        'Quarto': Quarto(opponent_type='random'),
    }


def get_observation_size(env) -> int:
    obs_size = env.observation_space
    if isinstance(obs_size, tuple):
        obs_size = int(np.prod(obs_size))
    return int(obs_size)


# Définition des familles d'agents et des grilles
AGENT_FAMILIES = {
    'Random': 'baseline',
    'TabularQL': 'tabular_value_based',
    'DQN': 'deep_value_based',
    'DoubleDQN': 'deep_value_based',
    'DDQN+ER': 'deep_value_based',
    'DDQN+PER': 'deep_value_based',
    'REINFORCE': 'policy_gradient',
    'REINFORCE+Mean': 'policy_gradient',
    'REINFORCE+Critic': 'policy_gradient',
    'PPO': 'actor_critic_policy_optimization',
    'A2C': 'actor_critic',
    'RandomRollout': 'planning',
    'MCTS': 'planning',
}

# Chaque famille utilise 3 hyperparamètres maximum.
GRID_SPECS = {
    'compact': {
        'baseline': {
            'fixed': [{}],
        },
        'tabular_value_based': {
            'learning_rate': [0.1, 0.3],
            'discount_factor': [0.95, 0.99],
            'epsilon_decay': [0.995, 0.999],
        },
        'deep_value_based': {
            'learning_rate': [0.001, 0.0005],
            'discount_factor': [0.95, 0.99],
            'epsilon_decay': [0.995, 0.998],
        },
        'policy_gradient': {
            'learning_rate': [0.005, 0.001],
            'discount_factor': [0.95, 0.99],
            'hidden_layers': [[64], [128, 128]],
        },
        'actor_critic_policy_optimization': {
            'learning_rate': [0.003, 0.0003],
            'discount_factor': [0.95, 0.99],
            'clip_ratio': [0.1, 0.2],
        },
        'actor_critic': {
            'learning_rate': [0.003, 0.0007],
            'discount_factor': [0.95, 0.99],
            'entropy_coef': [0.0, 0.01],
        },
        'planning': {
            'search_budget': [10, 30],
            'max_rollout_depth': [25, 50],
            'exploration_constant': [1.0, 1.41],
        },
    },
    'full': {
        'baseline': {
            'fixed': [{}],
        },
        'tabular_value_based': {
            'learning_rate': [0.05, 0.1, 0.3],
            'discount_factor': [0.95, 0.99],
            'epsilon_decay': [0.99, 0.995, 0.999],
        },
        'deep_value_based': {
            'learning_rate': [0.001, 0.0005, 0.0001],
            'discount_factor': [0.95, 0.99],
            'epsilon_decay': [0.99, 0.995, 0.998],
        },
        'policy_gradient': {
            'learning_rate': [0.005, 0.001, 0.0003],
            'discount_factor': [0.95, 0.99],
            'hidden_layers': [[64], [128], [128, 128]],
        },
        'actor_critic_policy_optimization': {
            'learning_rate': [0.003, 0.001, 0.0003],
            'discount_factor': [0.95, 0.99],
            'clip_ratio': [0.1, 0.2, 0.3],
        },
        'actor_critic': {
            'learning_rate': [0.003, 0.001, 0.0007],
            'discount_factor': [0.95, 0.99],
            'entropy_coef': [0.0, 0.01, 0.02],
        },
        'planning': {
            'search_budget': [10, 30, 50],
            'max_rollout_depth': [25, 50],
            'exploration_constant': [1.0, 1.41, 2.0],
        },
    },
    'none': {
        'baseline': {'fixed': [{}]},
        'tabular_value_based': {
            'learning_rate': [0.1],
            'discount_factor': [0.99],
            'epsilon_decay': [0.999],
        },
        'deep_value_based': {
            'learning_rate': [0.001],
            'discount_factor': [0.99],
            'epsilon_decay': [0.998],
        },
        'policy_gradient': {
            'learning_rate': [0.001],
            'discount_factor': [0.99],
            'hidden_layers': [[128, 128]],
        },
        'actor_critic_policy_optimization': {
            'learning_rate': [0.0003],
            'discount_factor': [0.99],
            'clip_ratio': [0.2],
        },
        'actor_critic': {
            'learning_rate': [0.0007],
            'discount_factor': [0.99],
            'entropy_coef': [0.01],
        },
        'planning': {
            'search_budget': [30],
            'max_rollout_depth': [50],
            'exploration_constant': [1.41],
        },
    },
}

AGENT_ORDER = [
    'Random',
    'TabularQL',
    'DQN',
    'DoubleDQN',
    'DDQN+ER',
    'DDQN+PER',
    'REINFORCE',
    'REINFORCE+Mean',
    'REINFORCE+Critic',
    'PPO',
    'A2C',
    'RandomRollout',
    'MCTS',
]


# Construction des agents
def available_agents_for_env(env_name: str) -> List[str]:
    agents = list(AGENT_ORDER)
    if env_name not in ('LineWorld', 'GridWorld'):
        agents.remove('TabularQL')

    return agents


def build_agent(agent_name: str, env_name: str, obs_size: int, action_size: int,
                params: Dict[str, Any], seed: Optional[int] = None):
    """Créer une nouvelle instance d'agent pour une configuration d'hyperparamètres."""
    kwargs = dict(params)
    if seed is not None:
        kwargs['seed'] = seed

    # on met une architecture par défaut allégée pour LineWorld afin de permettre un apprentissage rapide.
    if agent_name in {'REINFORCE', 'REINFORCE+Mean', 'REINFORCE+Critic', 'PPO', 'A2C'}:
        kwargs.setdefault('hidden_layers', [64] if env_name == 'LineWorld' else [128, 128])

    if agent_name == 'Random':
        return RandomAgent(action_size)

    if agent_name == 'TabularQL':
        return TabularQLearningAgent(obs_size, action_size, **kwargs)

    if agent_name == 'DQN':
        return DeepQLearningAgent(obs_size, action_size, **kwargs)

    if agent_name == 'DoubleDQN':
        return DoubleDeepQLearningAgent(obs_size, action_size, **kwargs)

    if agent_name == 'DDQN+ER':
        return DDQNWithERAgent(obs_size, action_size, **kwargs)

    if agent_name == 'DDQN+PER':
        return DDQNWithPERAgent(obs_size, action_size, **kwargs)

    if agent_name == 'REINFORCE':
        return REINFORCEAgent(obs_size, action_size, **kwargs)

    if agent_name == 'REINFORCE+Mean':
        return REINFORCEMeanBaselineAgent(obs_size, action_size, **kwargs)

    if agent_name == 'REINFORCE+Critic':
        kwargs.setdefault('critic_lr', kwargs.get('learning_rate', 0.001))
        return REINFORCECriticBaselineAgent(obs_size, action_size, **kwargs)

    if agent_name == 'PPO':
        return PPOAgent(obs_size, action_size, **kwargs)

    if agent_name == 'A2C':
        return A2CAgent(obs_size, action_size, **kwargs)

    if agent_name == 'RandomRollout':
        rollout_kwargs = {
            'num_rollouts': int(kwargs.get('search_budget', kwargs.get('num_rollouts', 20))),
            'max_rollout_depth': int(kwargs.get('max_rollout_depth', 50)),
        }
        if seed is not None:
            rollout_kwargs['seed'] = seed
        return RandomRolloutAgent(action_size, **rollout_kwargs)

    if agent_name == 'MCTS':
        mcts_kwargs = {
            'num_simulations': int(kwargs.get('search_budget', kwargs.get('num_simulations', 50))),
            'max_rollout_depth': int(kwargs.get('max_rollout_depth', 50)),
            'exploration_constant': float(kwargs.get('exploration_constant', 1.41)),
        }
        if seed is not None:
            mcts_kwargs['seed'] = seed
        return MCTSAgent(action_size, **mcts_kwargs)

    raise ValueError(f'Unknown agent: {agent_name}')


def expand_grid(grid_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Décomposer une spécification de grille en une liste de dictionnaires de paramètres."""
    if 'fixed' in grid_spec:
        return [dict(x) for x in grid_spec['fixed']]

    keys = list(grid_spec.keys())
    values = [grid_spec[k] for k in keys]
    combos = []
    for product in itertools.product(*values):
        combos.append({k: v for k, v in zip(keys, product)})
    return combos


def get_hyperparameter_grid(agent_name: str, grid_mode: str,
                            max_configs: Optional[int] = None) -> List[Dict[str, Any]]:
    family = AGENT_FAMILIES[agent_name]
    grid_spec = GRID_SPECS[grid_mode][family]
    configs = expand_grid(grid_spec)

    if agent_name == 'RandomRollout':
        cleaned = []
        seen = set()
        for cfg in configs:
            cfg = {k: v for k, v in cfg.items() if k != 'exploration_constant'}
            key = json.dumps(_json_clean(cfg), sort_keys=True)
            if key not in seen:
                cleaned.append(cfg)
                seen.add(key)
        configs = cleaned

    if max_configs is not None and max_configs > 0:
        configs = configs[:max_configs]

    return configs


# Score de sélection et sauvegardes

def compute_selection_score(eval_res: Dict[str, Any]) -> float:
    """
    Score utilisé pour sélectionner la meilleure configuration d'hyperparamètres.

    Le classement reste principalement déterminé par avg_reward. win_rate favorise les jeux adversariaux,
    std_reward pénalise l'instabilité, et le temps de décision sert de critère de départage mineur.
    """
    if eval_res.get('status') == 'ERROR':
        return -1e18

    avg_reward = float(eval_res.get('avg_reward', 0.0))
    win_rate = float(eval_res.get('win_rate', 0.0))
    loss_rate = float(eval_res.get('loss_rate', 0.0))
    std_reward = float(eval_res.get('std_reward', 0.0))
    step_time = max(float(eval_res.get('avg_step_time_ms', 0.0)), 0.0)

    return (
            avg_reward
            + 0.25 * win_rate
            - 0.10 * loss_rate
            - 0.01 * std_reward
            - 0.001 * np.log1p(step_time)
    )


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


def sanitize_name(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name)


def flatten_result_row(env_name: str, agent_name: str, config_id: int,
                       family: str, params: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        'environment': env_name,
        'agent': agent_name,
        'family': family,
        'config_id': config_id,
        'hyperparams': json.dumps(_json_clean(params), ensure_ascii=False, sort_keys=True),
    }
    for key in [
        'status', 'selection_score', 'avg_reward', 'std_reward', 'min_reward',
        'max_reward', 'median_reward', 'avg_length', 'avg_step_time_ms',
        'win_rate', 'loss_rate', 'draw_rate', 'num_episodes', 'episodes_trained',
        'training_time_seconds', 'seed', 'error'
    ]:
        row[key] = result.get(key, '')
    return row


def write_csv(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if not rows:
        with open(path, 'w', encoding='utf-8', newline='') as f:
            f.write('')
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _json_clean(v) for k, v in row.items()})


def save_json(path: str, payload: Any):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_json_clean(payload), f, indent=2, ensure_ascii=False)


# Boucle d'entraînement + grid search intégrée

def train_one_configuration(env, env_name: str, agent_name: str, agent,
                            params: Dict[str, Any], config_id: int,
                            episodes: int, checkpoints: List[int], eval_eps: int,
                            max_steps: int, seed: int, verbose: bool):
    t0 = time.time()
    family = AGENT_FAMILIES[agent_name]
    result = None
    metrics = None

    is_planning = hasattr(agent, 'set_env')
    effective_episodes = min(episodes, 500) if is_planning else episodes
    effective_checkpoints = [c for c in checkpoints if c <= effective_episodes]

    try:
        trainer = Trainer(
            env,
            agent,
            max_episodes=effective_episodes,
            max_steps_per_episode=max_steps,
            checkpoints=effective_checkpoints,
            eval_episodes=eval_eps,
            verbose=verbose,
        )
        metrics = trainer.train()
        elapsed = time.time() - t0

        evaluator = Evaluator(env, agent, num_episodes=eval_eps, max_steps=max_steps)
        result = evaluator.evaluate()
        result['status'] = 'OK'
        result['family'] = family
        result['hyperparams'] = deepcopy(params)
        result['config_id'] = config_id
        result['episodes_trained'] = effective_episodes
        result['training_time_seconds'] = elapsed
        result['seed'] = seed
        result['selection_score'] = compute_selection_score(result)

    except Exception as e:
        elapsed = time.time() - t0
        result = {
            'status': 'ERROR',
            'family': family,
            'hyperparams': deepcopy(params),
            'config_id': config_id,
            'avg_reward': -1e9,
            'selection_score': -1e18,
            'error': f'{type(e).__name__}: {e}',
            'episodes_trained': 0,
            'training_time_seconds': elapsed,
            'seed': seed,
        }

    return result, metrics, agent


def main():
    parser = argparse.ArgumentParser(description='DRL training pipeline with integrated grid search')
    parser.add_argument('--quick', action='store_true', help='Quick test mode: 100 episodes, 20 eval episodes')
    parser.add_argument('--env', type=str, default=None, help='Train only this environment')
    parser.add_argument('--agent', type=str, default=None, help='Train only this agent')
    parser.add_argument('--episodes', type=int, default=None, help='Training episodes per configuration')
    parser.add_argument('--eval-episodes', type=int, default=None, help='Evaluation episodes per configuration')
    parser.add_argument('--max-steps', type=int, default=200, help='Maximum steps per episode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--grid', choices=['compact', 'full', 'none'], default='compact',
                        help='Hyperparameter grid size. compact is the recommended default.')
    parser.add_argument('--max-configs-per-agent', type=int, default=None,
                        help='Optional cap on tested configurations per agent; useful for debugging.')
    parser.add_argument('--quiet', action='store_true', help='Reduce per-episode logs')
    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.quick:
        max_episodes = args.episodes or 100
        checkpoints = []
        eval_eps = args.eval_episodes or 20
    else:
        max_episodes = args.episodes or 10000
        checkpoints = [cp for cp in [1000, 10000, 100000] if cp <= max_episodes]
        eval_eps = args.eval_episodes or 100

    os.makedirs('results', exist_ok=True)
    os.makedirs('results/grid_search', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    envs = get_environments()
    if args.env:
        envs = {k: v for k, v in envs.items() if k == args.env}
        if not envs:
            raise ValueError(f'Unknown environment: {args.env}')

    total_start = time.time()

    all_best_results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    all_config_rows: List[Dict[str, Any]] = []
    best_by_agent_rows: List[Dict[str, Any]] = []
    best_agents_rows: List[Dict[str, Any]] = []
    metrics_for_best_agents: Dict[str, Dict[str, Any]] = {}

    print('\n' + '=' * 80)
    print('ENTRAINEMENT AVEC UN GRID SEARCH')
    print('=' * 80)
    print(f'Grid mode           : {args.grid}')
    print(f'Episodes/config     : {max_episodes}')
    print(f'Evaluation episodes : {eval_eps}')
    print(f'Seed                : {args.seed}')
    if args.max_configs_per_agent:
        print(f'Max configs/agent   : {args.max_configs_per_agent}')

    for env_name, env in envs.items():
        print(f'\n{"=" * 80}')
        print(f'ENVIRONNEMENT: {env_name}')
        print(f'{"=" * 80}')

        obs_size = get_observation_size(env)
        action_size = int(env.action_space)
        agent_names = available_agents_for_env(env_name)

        if args.agent:
            agent_names = [a for a in agent_names if a == args.agent]
            if not agent_names:
                print(f' Cet agent {args.agent} n\'est pas disponible pour l\'environnement: {env_name};')
                continue

        env_best_results: Dict[str, Dict[str, Any]] = {}
        env_best_metrics = {}
        env_best_agents = {}

        for agent_name in agent_names:
            family = AGENT_FAMILIES[agent_name]
            configs = get_hyperparameter_grid(
                agent_name,
                args.grid,
                max_configs=args.max_configs_per_agent,
            )

            print(f'\n  Agent: {agent_name} | famille={family} | configs={len(configs)}')

            best_result = None
            best_metrics = None
            best_agent = None

            for config_id, params in enumerate(configs, start=1):
                config_seed = args.seed + config_id
                np.random.seed(config_seed)
                agent = build_agent(agent_name, env_name, obs_size, action_size, params, seed=config_seed)

                print(f'    [{config_id:02d}/{len(configs):02d}] params={params}')

                result, metrics, trained_agent = train_one_configuration(
                    env=env,
                    env_name=env_name,
                    agent_name=agent_name,
                    agent=agent,
                    params=params,
                    config_id=config_id,
                    episodes=max_episodes,
                    checkpoints=checkpoints,
                    eval_eps=eval_eps,
                    max_steps=args.max_steps,
                    seed=config_seed,
                    verbose=not args.quiet,
                )

                all_config_rows.append(
                    flatten_result_row(env_name, agent_name, config_id, family, params, result)
                )

                if result.get('status') == 'OK':
                    print(
                        f'      score={result["selection_score"]:.4f} | '
                        f'avg_reward={result.get("avg_reward", 0):.4f} | '
                        f'win_rate={result.get("win_rate", 0):.2%} | '
                        f'time={result.get("avg_step_time_ms", 0):.3f}ms'
                    )
                else:
                    print(f'      ERROR: {result.get("error", "unknown error")}')

                if best_result is None or result.get('selection_score', -1e18) > best_result.get('selection_score',
                                                                                                 -1e18):
                    best_result = result
                    best_metrics = metrics
                    best_agent = trained_agent

            if best_result is None:
                continue

            env_best_results[agent_name] = best_result
            env_best_metrics[agent_name] = best_metrics
            env_best_agents[agent_name] = best_agent

            best_by_agent_rows.append(
                flatten_result_row(
                    env_name,
                    agent_name,
                    int(best_result.get('config_id', 0)),
                    family,
                    best_result.get('hyperparams', {}),
                    best_result,
                )
            )

            print(
                f'  >>> Meilleur: {agent_name} dans {env_name}: '
                f'config={best_result.get("config_id")} | '
                f'params={best_result.get("hyperparams", {})} | '
                f'score={best_result.get("selection_score", 0):.4f} | '
                f'avg_reward={best_result.get("avg_reward", 0):.4f}'
            )

            # on conserve uniquement le meilleur modèle et les meilleurs hyperparametres pour cet agent/cet environnement.
            safe_agent_name = sanitize_name(agent_name)
            model_path = f'models/{env_name}_{safe_agent_name}_best'
            try:
                if best_agent is not None:
                    best_agent.save(model_path)
            except Exception as e:
                print(f'      Model save warning: {e}')

            if best_metrics is not None:
                try:
                    best_metrics.save(f'results/{env_name}_{safe_agent_name}_best_metrics.json')
                except Exception as e:
                    print(f'      Metrics save warning: {e}')

        all_best_results[env_name] = env_best_results
        metrics_for_best_agents[env_name] = env_best_metrics

        # Classez les meilleurs agents sur cet environnement.
        ranked = sorted(
            env_best_results.items(),
            key=lambda x: x[1].get('selection_score', -1e18),
            reverse=True,
        )

        print(f'\n  Meilleur agents tuner pour l\'environnement {env_name}')
        for rank, (agent_name, res) in enumerate(ranked, start=1):
            print(
                f'    #{rank:02d} {agent_name:20s} | '
                f'score={res.get("selection_score", 0):9.4f} | '
                f'reward={res.get("avg_reward", 0):8.4f} | '
                f'win={res.get("win_rate", 0):6.2%} | '
                f'time={res.get("avg_step_time_ms", 0):8.3f}ms | '
                f'params={res.get("hyperparams", {})}'
            )

            row = flatten_result_row(
                env_name,
                agent_name,
                int(res.get('config_id', 0)),
                AGENT_FAMILIES[agent_name],
                res.get('hyperparams', {}),
                res,
            )
            row['rank'] = rank
            best_agents_rows.append(row)

        # Plots based on the best tuned configuration of each agent.
        if env_best_metrics:
            try:
                plot_learning_curves(env_best_metrics, env_name, 'results')
                plot_comparison(env_best_results, env_name, 'results')
            except Exception as e:
                print(f'  Plot error: {e}')

    # Global summary
    print(f'\n{"=" * 80}')
    print('Sommaire Globale - Meuilleur Agent Tuner par environnement')
    print(f'{"=" * 80}')

    for env_name, results in all_best_results.items():
        if not results:
            continue
        ranked = sorted(results.items(), key=lambda x: x[1].get('selection_score', -1e18), reverse=True)
        best_agent_name, best_res = ranked[0]
        print(
            f'  {env_name:10s} -> {best_agent_name:20s} | '
            f'score={best_res.get("selection_score", 0):.4f} | '
            f'reward={best_res.get("avg_reward", 0):.4f} | '
            f'params={best_res.get("hyperparams", {})}'
        )

    # Save machine-readable outputs.
    save_json('results/all_results.json', all_best_results)
    save_json('results/grid_search/all_config_results.json', all_config_rows)
    save_json('results/grid_search/best_by_agent_env.json', best_by_agent_rows)
    save_json('results/grid_search/best_agents_by_env_after_tuning.json', best_agents_rows)

    write_csv('results/grid_search/all_config_results.csv', all_config_rows)
    write_csv('results/grid_search/best_by_agent_env.csv', best_by_agent_rows)
    write_csv('results/grid_search/best_agents_by_env_after_tuning.csv', best_agents_rows)

    # Summary table plot from the best tuned configurations.
    try:
        plot_checkpoint_table(all_best_results, 'results')
    except Exception:
        pass

    total_time = time.time() - total_start
    print(f'\nTemps total d\'éxécution: {total_time:.1f}s ({total_time / 60:.1f}min)')
    print('Results saved to results/')
    print('Grid-search details saved to results/grid_search/')
    print('Best tuned models saved to models/*_best')


if __name__ == '__main__':
    main()
