"""Fonction utilitaire partagée pour les agents."""
import numpy as np
from typing import Optional


def mask_and_normalize(probs: np.ndarray, valid_actions: Optional[np.ndarray],
                       action_space_size: int) -> np.ndarray:
    """Masquer les actions non valides et normaliser la distribution de probabilité.
    Gère les cas limites où le masquage donne lieu à des probabilités proches de zéro ou négatives.
    """
    if valid_actions is not None and len(valid_actions) < action_space_size:
        mask = np.zeros(action_space_size, dtype=np.float32)
        mask[valid_actions] = 1.0
        probs = np.maximum(probs, 0) * mask
    else:
        probs = np.maximum(probs, 0)
    total = probs.sum()
    if total < 1e-10:
        # Uniform over valid actions
        if valid_actions is not None and len(valid_actions) > 0:
            probs = np.zeros(action_space_size, dtype=np.float32)
            probs[valid_actions] = 1.0 / len(valid_actions)
        else:
            probs = np.ones(action_space_size, dtype=np.float32) / action_space_size
    else:
        probs = probs / total
    return probs
