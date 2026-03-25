"""Plotting utilities for training metrics."""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Optional


def plot_learning_curves(metrics_dict: Dict[str, 'Metrics'], env_name: str,
                        save_path: str = 'results', window: int = 100):
    """Plot learning curves for multiple agents on same environment."""
    os.makedirs(save_path, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Learning Curves - {env_name}', fontsize=14)

    has_data = False
    for name, m in metrics_dict.items():
        if len(m.episode_rewards) == 0:
            continue
        has_data = True
        smoothed = m.get_windowed_rewards(window)
        axes[0].plot(smoothed, label=name, alpha=0.8)
        if m.episode_lengths:
            lengths = np.array(m.episode_lengths)
            if len(lengths) >= window:
                smooth_len = np.convolve(lengths, np.ones(window)/window, mode='valid')
                axes[1].plot(smooth_len, label=name, alpha=0.8)
            else:
                axes[1].plot(lengths, label=name, alpha=0.8)

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel(f'Avg Reward (window={window})')
    if has_data:
        axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel(f'Avg Episode Length (window={window})')
    if has_data:
        axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(save_path, f'{env_name}_learning_curves.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_comparison(results: Dict[str, Dict], env_name: str,
                   save_path: str = 'results'):
    """Plot bar chart comparison of agents."""
    os.makedirs(save_path, exist_ok=True)
    agents = list(results.keys())
    rewards = [results[a].get('avg_reward', 0) for a in agents]
    times = [results[a].get('avg_step_time_ms', 0) for a in agents]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Agent Comparison - {env_name}', fontsize=14)

    colors = plt.cm.Set3(np.linspace(0, 1, len(agents)))
    x = np.arange(len(agents))

    axes[0].bar(x, rewards, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(agents, rotation=45, ha='right', fontsize=7)
    axes[0].set_ylabel('Average Reward')
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(x, times, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(agents, rotation=45, ha='right', fontsize=7)
    axes[1].set_ylabel('Avg Step Time (ms)')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filepath = os.path.join(save_path, f'{env_name}_comparison.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")


def plot_checkpoint_table(all_results: Dict[str, Dict[str, Dict]],
                         save_path: str = 'results'):
    """Create summary table plot: env x agent x checkpoint."""
    os.makedirs(save_path, exist_ok=True)
    # Build data
    envs = list(all_results.keys())
    agents = set()
    for env_data in all_results.values():
        agents.update(env_data.keys())
    agents = sorted(agents)

    fig, ax = plt.subplots(figsize=(max(12, len(agents) * 1.5), max(4, len(envs) * 1.5)))
    ax.axis('off')

    cell_text = []
    for env_name in envs:
        row = []
        for agent_name in agents:
            r = all_results.get(env_name, {}).get(agent_name, {})
            if r:
                row.append(f"{r.get('avg_reward', 0):.3f}")
            else:
                row.append('-')
        cell_text.append(row)

    table = ax.table(cellText=cell_text, rowLabels=envs, colLabels=agents,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.5)

    plt.title('Results Summary: Average Reward per Agent per Environment', fontsize=12, pad=20)
    plt.tight_layout()
    filepath = os.path.join(save_path, 'summary_table.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filepath}")
