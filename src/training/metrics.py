"""Metrics collection for training and evaluation."""
import numpy as np
import json
import os
from typing import Dict, List, Optional


class Metrics:
    """Collects and stores training metrics."""

    def __init__(self):
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.step_times: List[float] = []
        self.checkpoint_metrics: Dict[int, Dict] = {}
        self.training_losses: List[float] = []

    def add_episode(self, reward: float, length: int, step_time: float = 0.0):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.step_times.append(step_time)

    def add_checkpoint(self, episode: int, eval_reward: float, eval_length: float,
                       avg_step_time: float):
        """Store checkpoint metrics after evaluation."""
        self.checkpoint_metrics[episode] = {
            'avg_reward': eval_reward,
            'avg_length': eval_length,
            'avg_step_time_ms': avg_step_time,
        }

    def get_average_reward(self, last_n: int = 100) -> float:
        if not self.episode_rewards:
            return 0.0
        return float(np.mean(self.episode_rewards[-last_n:]))

    def get_average_length(self, last_n: int = 100) -> float:
        if not self.episode_lengths:
            return 0.0
        return float(np.mean(self.episode_lengths[-last_n:]))

    def get_average_step_time(self) -> float:
        if not self.step_times:
            return 0.0
        return float(np.mean(self.step_times)) * 1000  # ms

    def get_windowed_rewards(self, window: int = 100) -> List[float]:
        """Running average of rewards."""
        if len(self.episode_rewards) < window:
            return [np.mean(self.episode_rewards[:i+1])
                    for i in range(len(self.episode_rewards))]
        return [np.mean(self.episode_rewards[max(0, i-window+1):i+1])
                for i in range(len(self.episode_rewards))]

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'step_times': self.step_times,
            'checkpoint_metrics': {str(k): v for k, v in self.checkpoint_metrics.items()},
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'Metrics':
        with open(filepath, 'r') as f:
            data = json.load(f)
        m = cls()
        m.episode_rewards = data['episode_rewards']
        m.episode_lengths = data['episode_lengths']
        m.step_times = data.get('step_times', [])
        m.checkpoint_metrics = {int(k): v for k, v in data.get('checkpoint_metrics', {}).items()}
        return m
