"""Replay with legal successor masks and proportional importance sampling."""

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=50000, rng=None):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.rng = rng if rng is not None else np.random.default_rng()
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done, next_mask=None):
        item = (
            np.array(state, copy=True),
            int(action),
            float(reward),
            np.array(next_state, copy=True),
            bool(done),
            None if next_mask is None else np.array(next_mask, dtype=bool, copy=True),
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def _batch(self, indices, include_masks):
        batch = [self.buffer[i] for i in indices]
        columns = tuple(
            np.array([b[i] for b in batch], dtype=dtype)
            for i, dtype in enumerate((np.float32, np.int64, np.float32, np.float32, bool))
        )
        if include_masks:
            if any(b[5] is None for b in batch):
                raise ValueError("Successor masks are required for masked replay")
            columns += (np.array([b[5] for b in batch], dtype=bool),)
        return columns

    def sample(self, batch_size, include_masks=False):
        if not 0 < batch_size <= len(self):
            raise ValueError("batch_size must be between 1 and replay size")
        indices = self.rng.choice(len(self), batch_size, replace=False)
        return self._batch(indices, include_masks)

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity=50000, alpha=0.6, beta=0.4, beta_increment=0.001, rng=None):
        super().__init__(capacity, rng)
        if alpha < 0 or not 0 <= beta <= 1 or beta_increment < 0:
            raise ValueError("Invalid prioritized replay parameters")
        self.alpha, self.beta, self.beta_increment = alpha, beta, beta_increment
        self.priorities = np.zeros(capacity, dtype=np.float64)  # raw priorities
        self.max_priority = 1.0

    def push(self, *args, **kwargs):
        index = self.pos
        super().push(*args, **kwargs)
        self.priorities[index] = self.max_priority

    def probabilities(self):
        scaled = self.priorities[: len(self)] ** self.alpha
        return scaled / scaled.sum()

    def sample(self, batch_size, include_masks=False):
        if len(self) == 0 or batch_size < 1:
            raise ValueError("Cannot sample an empty replay buffer")
        probs = self.probabilities()
        indices = self.rng.choice(len(self), batch_size, p=probs, replace=True)
        weights = (len(self) * probs[indices]) ** (-self.beta)
        weights /= (len(self) * probs.min()) ** (-self.beta)
        self.beta = min(1.0, self.beta + self.beta_increment)
        return self._batch(indices, include_masks) + (indices, weights.astype(np.float32))

    def update_priorities(self, indices, td_errors):
        values = np.abs(np.asarray(td_errors, dtype=float)) + 1e-6
        if not np.all(np.isfinite(values)):
            raise ValueError("TD errors must be finite")
        for index in np.unique(indices):
            self.priorities[index] = values[np.asarray(indices) == index].max()
        self.max_priority = max(self.max_priority, float(values.max()))
