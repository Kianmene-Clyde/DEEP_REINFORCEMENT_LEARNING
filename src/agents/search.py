"""Sampled-outcome tree search for environments whose step includes the opponent.

All rewards and values belong to the controlled agent. No sign alternation:
each edge is an agent decision followed by the fixed opponent response.
"""

from dataclasses import dataclass, field
import numpy as np
from .utils import mask_and_normalize


@dataclass
class SearchNode:
    prior: np.ndarray
    valid: np.ndarray
    visits: np.ndarray
    totals: np.ndarray
    children: dict = field(default_factory=dict)


def rollout(env, rng, depth, gamma):
    total, discount = 0.0, 1.0
    for _ in range(depth):
        valid = env.get_valid_actions(env._get_state())
        if not len(valid):
            break
        _, reward, done, _ = env.step(int(rng.choice(valid)))
        total += discount * reward
        discount *= gamma
        if done:
            break
    return total


def tree_search(env, rng, simulations, depth, gamma=0.99, exploration=1.41, evaluator=None, root_noise=None):
    if simulations < 1 or depth < 1:
        raise ValueError("Search budgets must be positive")
    size = env.action_space

    def make_node(sim):
        state = sim._get_state()
        valid = sim.get_valid_actions(state)
        policy, value = evaluator(state) if evaluator is not None else (np.ones(size), 0.0)
        prior = mask_and_normalize(policy, valid, size)
        return SearchNode(prior, valid, np.zeros(size, dtype=int), np.zeros(size)), float(value)

    root, _ = make_node(env)
    if root_noise is not None:
        alpha, fraction = root_noise
        root.prior[root.valid] = (1 - fraction) * root.prior[root.valid] + fraction * rng.dirichlet(
            np.full(len(root.valid), alpha)
        )
    for _ in range(simulations):
        sim = env.clone(seed=int(rng.integers(2**32)))
        node, path, value = root, [], 0.0
        for level in range(depth):
            valid = node.valid
            count = node.visits[valid]
            if evaluator is None and np.any(count == 0):
                action = int(rng.choice(valid[count == 0]))
            else:
                q = np.divide(node.totals, node.visits, out=np.zeros(size), where=node.visits > 0)
                if evaluator is None:
                    bonus = exploration * np.sqrt(np.log(node.visits.sum() + 1) / np.maximum(node.visits, 1))
                else:
                    bonus = exploration * node.prior * np.sqrt(node.visits.sum() + 1) / (1 + node.visits)
                scores = q[valid] + bonus[valid]
                action = int(rng.choice(valid[np.isclose(scores, scores.max(), rtol=0, atol=1e-12)]))
            state, reward, done, _ = sim.step(action)
            path.append((node, action, float(reward)))
            if done:
                value = 0.0
                break
            # Distinct stochastic outcomes never share an observation node.
            key = (action, np.asarray(state).tobytes())
            if key not in node.children:
                child, value = make_node(sim)
                node.children[key] = child
                if evaluator is None:
                    value = rollout(sim, rng, depth - level - 1, gamma)
                break
            node = node.children[key]
            if level == depth - 1:
                value = float(evaluator(state)[1]) if evaluator is not None else 0.0
        for visited, action, reward in reversed(path):
            value = reward + gamma * value
            visited.visits[action] += 1
            visited.totals[action] += value
    return root.visits.copy(), root.totals.copy()


def visit_policy(visits, valid, temperature=1.0):
    out = np.zeros(len(visits), dtype=float)
    counts = visits[valid].astype(float)
    if temperature <= 0:
        out[int(valid[np.argmax(counts)])] = 1.0
    else:
        positive = counts > 0
        logs = np.full(len(counts), -np.inf)
        logs[positive] = np.log(counts[positive]) / temperature
        weights = np.exp(logs - logs.max())
        out[valid] = weights / weights.sum()
    return out
