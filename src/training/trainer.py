"""Training loops with separate termination/truncation and isolated evaluation."""

import time
import numpy as np
from .metrics import Metrics
from .evaluator import Evaluator, encode_state


class Trainer:
    CHECKPOINTS = [1000, 10000, 100000]

    def __init__(
        self,
        env,
        agent,
        max_episodes=10000,
        max_steps_per_episode=200,
        eval_episode_interval=0,
        eval_episodes=100,
        checkpoints=None,
        verbose=True,
        seed=42,
        evaluation_seed=1000000,
        start_episode=0,
        deadline=None,
    ):
        if min(max_episodes, max_steps_per_episode, eval_episodes) < 1:
            raise ValueError("All training and evaluation budgets must be positive")
        self.env, self.agent = env, agent
        self.max_episodes, self.max_steps = max_episodes, max_steps_per_episode
        self.eval_interval, self.eval_episodes = eval_episode_interval, eval_episodes
        self.checkpoints = self.CHECKPOINTS if checkpoints is None else checkpoints
        self.verbose, self.seed, self.evaluation_seed = verbose, int(seed), int(evaluation_seed)
        self.start_episode = start_episode
        self.deadline = deadline
        if hasattr(agent, "set_env"):
            agent.set_env(env)

    def train(self):
        metrics = Metrics()
        self.agent.set_training_mode(True)
        training_seconds = 0.0
        for ep in range(self.start_episode + 1, self.start_episode + self.max_episodes + 1):
            t0 = time.perf_counter()
            episode_seed = int(np.random.SeedSequence([self.seed, ep]).generate_state(1)[0])
            state = self.env.reset(seed=episode_seed)
            total, action_seconds = 0.0, 0.0
            terminated, truncated = False, False
            info = {}
            for step in range(self.max_steps):
                if self.deadline is not None and time.perf_counter() > self.deadline:
                    raise TimeoutError("Training time budget exhausted")
                valid = self.env.get_valid_actions(state)
                if len(valid) == 0:
                    raise RuntimeError("Nonterminal state has no legal actions")
                encoded = self._encode(state)
                before = time.perf_counter()
                action = self.agent.select_action(encoded, valid)
                action_seconds += time.perf_counter() - before
                if action not in valid:
                    raise RuntimeError(f"Agent selected illegal action {action}")
                next_state, reward, terminated, info = self.env.step(action)
                truncated = step + 1 == self.max_steps and not terminated
                next_valid = self.env.get_valid_actions(next_state)
                self.agent.observe_transition(
                    encoded,
                    action,
                    reward,
                    self._encode(next_state),
                    terminated=terminated,
                    truncated=truncated,
                    next_valid_actions=next_valid,
                )
                state, total = next_state, total + reward
                if terminated or truncated:
                    break
            training_seconds += time.perf_counter() - t0
            metrics.add_episode(float(total), step + 1, action_seconds / (step + 1))
            metrics.episode_outcomes.append(info.get("outcome", "unknown") if terminated else "timeout")
            metrics.episode_truncated.append(bool(truncated))
            if ep in self.checkpoints or (self.eval_interval and ep % self.eval_interval == 0):
                metrics.checkpoint_metrics[ep] = self.evaluate()
            if self.verbose and ep % max(1, self.max_episodes // 10) == 0:
                print(f"Episode {ep}: mean training return={metrics.get_average_reward():.4f}")
        final_episode = self.start_episode + self.max_episodes
        if final_episode not in metrics.checkpoint_metrics:
            metrics.checkpoint_metrics[final_episode] = self.evaluate()
        metrics.training_time_seconds = training_seconds
        metrics.transitions = sum(metrics.episode_lengths)
        return metrics

    def evaluate(self):
        return Evaluator(
            self.env,
            self.agent,
            self.eval_episodes,
            self.max_steps,
            seed=self.evaluation_seed,
            deadline=self.deadline,
        ).evaluate()

    def _evaluate(self):
        result = self.evaluate()
        return result["avg_reward"], result["avg_length"], result["avg_step_time_ms"]

    def _encode(self, state):
        return encode_state(state, self.env, self.agent)
