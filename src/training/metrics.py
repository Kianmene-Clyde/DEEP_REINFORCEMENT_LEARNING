"""Collection des métriques pour l'entrainement et l'évaluation."""

import numpy as np
import json
import os
from typing import Dict, List


class Metrics:
    """Stocker les métriques d'entrainement."""

    def __init__(self):
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.step_times: List[float] = []
        self.checkpoint_metrics: Dict[int, Dict] = {}
        self.training_losses: List[float] = []
        self.episode_outcomes = []
        self.episode_truncated = []
        self.training_time_seconds = 0.0
        self.transitions = 0

    def add_episode(self, reward: float, length: int, step_time: float = 0.0):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.step_times.append(step_time)

    def add_checkpoint(self, episode: int, eval_reward: float, eval_length: float, avg_step_time: float):
        """Stocker les métriques des checkpoints après l'évaluation."""
        self.checkpoint_metrics[episode] = {
            "avg_reward": eval_reward,
            "avg_length": eval_length,
            "avg_step_time_ms": avg_step_time,
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
        """Calculer la moyenne des rewards."""
        if window < 1:
            raise ValueError("window must be positive")
        values = np.asarray(self.episode_rewards, dtype=np.float64)
        if len(values) == 0:
            return []
        cumulative = np.r_[0.0, np.cumsum(values)]
        end = np.arange(1, len(values) + 1)
        start = np.maximum(0, end - window)
        return ((cumulative[end] - cumulative[start]) / (end - start)).tolist()

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        data = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "step_times": self.step_times,
            "episode_outcomes": self.episode_outcomes,
            "episode_truncated": self.episode_truncated,
            "training_time_seconds": self.training_time_seconds,
            "transitions": self.transitions,
            "checkpoint_metrics": {str(k): v for k, v in self.checkpoint_metrics.items()},
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "Metrics":
        with open(filepath, "r") as f:
            data = json.load(f)
        m = cls()
        m.episode_rewards = data["episode_rewards"]
        m.episode_lengths = data["episode_lengths"]
        m.step_times = data.get("step_times", [])
        m.episode_outcomes = data.get("episode_outcomes", [])
        m.episode_truncated = data.get("episode_truncated", [])
        m.training_time_seconds = data.get("training_time_seconds", 0.0)
        m.transitions = data.get("transitions", sum(m.episode_lengths))
        m.checkpoint_metrics = {int(k): v for k, v in data.get("checkpoint_metrics", {}).items()}
        return m
