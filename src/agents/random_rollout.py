"""Random Rollout agent: evaluates actions by simulating random games."""
import numpy as np
import copy
from typing import Optional, Any
from .base_agent import BaseAgent


class RandomRolloutAgent(BaseAgent):

    def __init__(self, action_space_size: int, num_rollouts: int = 50, max_rollout_depth: int = 100,
                 is_two_player: bool = False, seed: Optional[int] = None, **kwargs):

        super().__init__("RandomRollout", action_space_size)
        if seed is not None:
            np.random.seed(seed)
        self.num_rollouts = num_rollouts
        self.max_depth = max_rollout_depth
        self.is_two_player = is_two_player
        self._env = None

    def set_env(self, env):
        self._env = env
        if hasattr(env, 'is_two_player'):
            self.is_two_player = env.is_two_player

    def select_action(self, state: Any, valid_actions: Optional[np.ndarray] = None) -> int:
        if self._env is None:
            if valid_actions is not None and len(valid_actions) > 0:
                return int(np.random.choice(valid_actions))
            return np.random.randint(0, self.action_space_size)

        if valid_actions is None:
            valid_actions = np.arange(self.action_space_size)

        best_action = valid_actions[0]
        best_value = -float('inf')

        for action in valid_actions:
            total_reward = 0.0
            for _ in range(self.num_rollouts):
                sim_env = copy.deepcopy(self._env)
                _, r, done, _ = sim_env.step(int(action))
                total_reward += r
                if not done:
                    # Après notre coup, c'est au tour de l'adversaire :
                    # on commence le rollout depuis la perspective adverse
                    total_reward += self._random_rollout(
                        sim_env, first_move_is_opponent=self.is_two_player
                    )
            avg = total_reward / self.num_rollouts
            if avg > best_value:
                best_value = avg
                best_action = action

        return int(best_action)

    def _random_rollout(self, env, first_move_is_opponent: bool = False) -> float:

        total = 0.0
        sign = -1.0 if first_move_is_opponent else 1.0

        for _ in range(self.max_depth):
            try:
                state = env._get_state() if hasattr(env, '_get_state') else None
                valid = env.get_valid_actions(state)
                action = (int(np.random.choice(valid)) if len(valid) > 0
                          else np.random.randint(0, self.action_space_size))
                _, r, done, _ = env.step(action)
                total += sign * r
                if self.is_two_player:
                    sign = -sign  # alterner la perspective à chaque demi-coup
                if done:
                    break
            except Exception:
                break
        return total

    def learn(self, state, action, reward, next_state, done):
        pass  # Agent de planification pure, pas d'apprentissage

    def save(self, filepath: str):
        import pickle
        import os
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath + '_config.pkl', 'wb') as f:
            pickle.dump({
                'num_rollouts': self.num_rollouts,
                'max_depth': self.max_depth,
                'is_two_player': self.is_two_player,
            }, f)

    def load(self, filepath: str):
        import pickle
        with open(filepath + '_config.pkl', 'rb') as f:
            cfg = pickle.load(f)
        self.num_rollouts = cfg['num_rollouts']
        self.max_depth = cfg['max_depth']
        self.is_two_player = cfg.get('is_two_player', False)
