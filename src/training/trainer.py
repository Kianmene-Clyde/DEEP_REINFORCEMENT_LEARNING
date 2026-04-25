"""Trainer: runs training loops and collects metrics."""
import time
import numpy as np
from typing import Optional, List
from .metrics import Metrics


class Trainer:
    """Train an agent on an environment with metrics collection."""

    CHECKPOINTS = [1000, 10000, 100000]

    def __init__(self, env, agent, max_episodes: int = 10000,
                 max_steps_per_episode: int = 200,
                 eval_episode_interval: int = 0,
                 eval_episodes: int = 100,
                 checkpoints: Optional[List[int]] = None,
                 verbose: bool = True):
        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes
        self.max_steps = max_steps_per_episode
        self.eval_interval = eval_episode_interval
        self.eval_episodes = eval_episodes
        self.checkpoints = checkpoints if checkpoints is not None else self.CHECKPOINTS
        self.verbose = verbose

        # Give env reference to planning agents
        if hasattr(agent, 'set_env'):
            agent.set_env(env)

    def train(self) -> Metrics:
        metrics = Metrics()
        if hasattr(self.agent, 'set_training_mode'):
            self.agent.set_training_mode(True)

        done_checkpoints = set()
        log_interval = max(1, self.max_episodes // 20)

        for ep in range(1, self.max_episodes + 1):
            state = self.env.reset()
            total_reward = 0.0
            steps = 0
            t_start = time.time()

            done = False
            for step in range(self.max_steps):
                valid = self.env.get_valid_actions(state)
                if len(valid) == 0:
                    break
                encoded = self._encode(state)
                action = self.agent.select_action(encoded, valid)
                next_state, reward, done, info = self.env.step(action)

                # Force done on last step (timeout) so episode-based agents
                # (REINFORCE, PPO, A2C) trigger their update
                if step == self.max_steps - 1 and not done:
                    done = True

                self.agent.learn(encoded, action, reward,
                                 self._encode(next_state), done)
                state = next_state
                total_reward += reward
                steps += 1
                if done:
                    break

            elapsed = time.time() - t_start
            avg_step = elapsed / max(steps, 1)
            metrics.add_episode(total_reward, steps, avg_step)

            # Checkpoint evaluation
            for cp in self.checkpoints:
                if ep == cp and cp not in done_checkpoints:
                    done_checkpoints.add(cp)
                    eval_r, eval_l, eval_t = self._evaluate()
                    metrics.add_checkpoint(cp, eval_r, eval_l, eval_t)
                    if self.verbose:
                        print(f"  [Checkpoint {cp:>8d}] "
                              f"avg_reward={eval_r:.4f}  avg_len={eval_l:.1f}  "
                              f"step_time={eval_t:.3f}ms")

            if self.verbose and ep % log_interval == 0:
                avg_r = metrics.get_average_reward(100)
                print(f"  Episode {ep:>8d}/{self.max_episodes}  "
                      f"avg_reward(100)={avg_r:.4f}")

        # Final evaluation
        eval_r, eval_l, eval_t = self._evaluate()
        metrics.add_checkpoint(self.max_episodes, eval_r, eval_l, eval_t)
        if self.verbose:
            print(f"  [Final {self.max_episodes:>8d}] "
                  f"avg_reward={eval_r:.4f}  avg_len={eval_l:.1f}  "
                  f"step_time={eval_t:.3f}ms")
        return metrics

    def _evaluate(self):
        if hasattr(self.agent, 'set_training_mode'):
            self.agent.set_training_mode(False)

        rewards, lengths, times = [], [], []
        for _ in range(self.eval_episodes):
            state = self.env.reset()
            total_r = 0.0
            steps = 0
            t0 = time.time()
            for _ in range(self.max_steps):
                valid = self.env.get_valid_actions(state)
                if len(valid) == 0:
                    break
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
            times.append(elapsed / max(steps, 1) * 1000)

        if hasattr(self.agent, 'set_training_mode'):
            self.agent.set_training_mode(True)

        return float(np.mean(rewards)), float(np.mean(lengths)), float(np.mean(times))

    def _encode(self, state):
        if hasattr(self.env, 'state_to_index') and hasattr(self.agent, 'uses_tabular') and self.agent.uses_tabular:
            return self.env.state_to_index(state)
        s = np.asarray(state, dtype=np.float32).flatten()
        return s
