"""Isolated policy evaluation with explicit outcomes and decision-only latency."""

import copy
import time
import numpy as np


def encode_state(state, env, agent):
    if getattr(agent, "uses_tabular", False):
        return env.state_to_index(state)
    return np.asarray(state, dtype=np.float32).flatten()


class Evaluator:
    def __init__(self, env, agent, num_episodes=100, max_steps=200, seed=1000000, deadline=None):
        if num_episodes < 1 or max_steps < 1:
            raise ValueError("Evaluation episode and step budgets must be positive")
        self.env, self.agent = env, agent
        self.num_episodes, self.max_steps, self.seed = num_episodes, max_steps, int(seed)
        self.deadline = deadline

    def evaluate(self):
        # Includes RNGs/buffers: evaluation cannot perturb subsequent training.
        env, agent = copy.deepcopy(self.env), copy.deepcopy(self.agent)
        agent.set_training_mode(False)
        if hasattr(agent, "set_env"):
            agent.set_env(env)
        records, decision_seconds = [], []
        for episode in range(self.num_episodes):
            seed = int(np.random.SeedSequence([self.seed, episode]).generate_state(1)[0])
            state = env.reset(seed=seed)
            agent.rng = np.random.default_rng(np.random.SeedSequence([self.seed, episode, 1]))
            total, terminated, info = 0.0, False, {}
            steps = 0
            for _ in range(self.max_steps):
                if self.deadline is not None and time.perf_counter() > self.deadline:
                    raise TimeoutError("Evaluation time budget exhausted")
                valid = env.get_valid_actions(state)
                if len(valid) == 0:
                    raise RuntimeError("Nonterminal state has no legal actions")
                encoded = encode_state(state, env, agent)
                t0 = time.perf_counter()
                action = agent.select_action(encoded, valid)
                decision_seconds.append(time.perf_counter() - t0)
                if action not in valid:
                    raise RuntimeError(f"Agent selected illegal action {action}")
                state, reward, terminated, info = env.step(action)
                total += reward
                steps += 1
                if terminated:
                    break
            records.append(
                {
                    "seed": seed,
                    "reward": float(total),
                    "length": steps,
                    "terminated": bool(terminated),
                    "truncated": not bool(terminated),
                    "outcome": info.get("outcome", "unknown") if terminated else "timeout",
                }
            )
        rewards = np.array([r["reward"] for r in records])
        outcomes = [r["outcome"] for r in records]

        def rate(choices):
            return sum(o in choices for o in outcomes) / len(outcomes)

        self.episodes = records
        return {
            "avg_reward": float(rewards.mean()),
            "std_reward": float(rewards.std(ddof=1)) if len(rewards) > 1 else 0.0,
            "avg_length": float(np.mean([r["length"] for r in records])),
            "avg_step_time_ms": float(np.mean(decision_seconds)) * 1000,
            "win_rate": rate({"win", "success"}),
            "loss_rate": rate({"loss", "failure"}),
            "draw_rate": rate({"draw"}),
            "success_rate": rate({"success", "win"}),
            "timeout_rate": rate({"timeout"}),
            "unknown_outcome_rate": rate({"unknown"}),
            "num_episodes": len(records),
            "evaluation_seed": self.seed,
            "decision_count": len(decision_seconds),
        }

    def _encode(self, state):
        return encode_state(state, self.env, self.agent)
