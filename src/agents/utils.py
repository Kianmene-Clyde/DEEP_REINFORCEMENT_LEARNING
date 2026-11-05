"""Legal categorical distributions and their logits gradients."""

import numpy as np


def action_mask(valid_actions, action_space_size):
    mask = np.ones(action_space_size, dtype=bool)
    if valid_actions is not None:
        indices = np.asarray(valid_actions, dtype=int)
        if indices.size == 0:
            raise ValueError("Cannot select an action from an empty legal-action set")
        if np.any(indices < 0) or np.any(indices >= action_space_size):
            raise ValueError("Legal-action index outside action space")
        mask[:] = False
        mask[indices] = True
    return mask


def mask_and_normalize(probs, valid_actions, action_space_size):
    probs = np.asarray(probs, dtype=np.float64).copy()
    if probs.shape != (action_space_size,) or not np.all(np.isfinite(probs)):
        raise ValueError("Policy probabilities must be finite and match the action space")
    mask = action_mask(valid_actions, action_space_size)
    probs = np.where(mask, np.maximum(probs, 0.0), 0.0)
    total = probs.sum()
    if total <= np.finfo(float).tiny:
        return mask.astype(float) / mask.sum()
    return probs / total


def masked_batch(probs, masks):
    """Renormalize collected legal-action masks for a batch of policy updates."""
    p = np.where(masks, np.maximum(probs, 0.0), 0.0)
    totals = p.sum(axis=1, keepdims=True)
    return np.divide(
        p,
        totals,
        out=masks.astype(float) / masks.sum(axis=1, keepdims=True),
        where=totals > np.finfo(float).tiny,
    )


def entropy_gradient(probs, coefficient):
    """Per-sample d(-coefficient * entropy)/d logits, including masked zeros."""
    logs = np.log(np.maximum(probs, np.finfo(float).tiny))
    entropy = -np.sum(probs * logs, axis=1, keepdims=True)
    return coefficient * probs * (logs + entropy)
