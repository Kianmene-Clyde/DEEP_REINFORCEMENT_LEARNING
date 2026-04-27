"""Evaluator: evalue un agent entrainé."""
import time
import numpy as np
from typing import Dict


class Evaluator:
    """Evalue un agent entrainé dans un environnement."""

    def __init__(self, env, agent, num_episodes: int = 100,
                 max_steps: int = 200):
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.max_steps = max_steps

    def evaluate(self) -> Dict[str, float]:
        """Execute l'évaluation, retoiurne les stats resumées."""
        if hasattr(self.agent, 'set_training_mode'):
            self.agent.set_training_mode(False)

        rewards, lengths, step_times = [], [], []
        wins, losses, draws = 0, 0, 0

        for _ in range(self.num_episodes):
            state = self.env.reset()
            total_r = 0.0
            steps = 0
            t0 = time.time()
            for _ in range(self.max_steps):
                valid = self.env.get_valid_actions(state)
                s = self._encode(state)
                action = self.agent.select_action(s, valid)
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                total_r += reward
                steps += 1
                if done:
                    break
            elapsed = time.time() - t0
            rewards.append(total_r)
            lengths.append(steps)
            step_times.append(elapsed / max(steps, 1) * 1000)

            # Suivi des victoires/défaites
            # Seuil > 0 : toute récompense totale positive = une victoire
            if total_r > 0:
                wins += 1
            elif total_r < 0:
                losses += 1
            else:
                draws += 1

        return {
            'avg_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'avg_length': float(np.mean(lengths)),
            'avg_step_time_ms': float(np.mean(step_times)),
            'win_rate': wins / self.num_episodes,
            'loss_rate': losses / self.num_episodes,
            'draw_rate': draws / self.num_episodes,
            'num_episodes': self.num_episodes,
        }

    def _encode(self, state):
        if hasattr(self.env, 'state_to_index') and hasattr(self.agent, 'uses_tabular') and self.agent.uses_tabular:
            return self.env.state_to_index(state)
        return np.asarray(state, dtype=np.float32).flatten()
