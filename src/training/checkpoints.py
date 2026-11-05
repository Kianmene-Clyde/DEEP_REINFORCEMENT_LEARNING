"""Versioned complete agent checkpoints for trusted local artifacts."""

import pickle
from pathlib import Path
import numpy as np

PROTOCOL_VERSION = 2


def save_checkpoint(path, agent, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"protocol_version": PROTOCOL_VERSION, "agent": agent, "metadata": metadata}
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def load_checkpoint(path, *, environment=None, observation_size=None, action_size=None):
    with Path(path).open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(
            "Checkpoint protocol incompatible: historical models require reevaluation/retraining"
        )
    metadata, agent = payload["metadata"], payload["agent"]
    for key, expected in [
        ("environment", environment),
        ("observation_size", observation_size),
        ("action_size", action_size),
    ]:
        if expected is not None and metadata.get(key) != expected:
            raise ValueError(f"Checkpoint incompatible: {key}")
    expected_input = metadata.get("observation_size", getattr(agent, "input_size", None))
    actions = getattr(agent, "action_space_size", None)
    if metadata.get("action_size", actions) != actions:
        raise ValueError("Checkpoint action space differs from agent")
    for name in [
        "q_net",
        "target_net",
        "policy_net",
        "actor",
        "critic",
        "value_net",
        "net",
        "repr_net",
        "dyn_net",
        "pred_net",
    ]:
        net = getattr(agent, name, None)
        if net is None:
            continue
        if not net.layers or any(
            not np.all(np.isfinite(layer[key])) for layer in net.layers for key in ["W", "b"]
        ):
            raise ValueError("Invalid network parameters in checkpoint")
        if name in {"q_net", "target_net", "policy_net", "actor", "critic", "value_net", "net", "repr_net"}:
            if expected_input is not None and net.layers[0]["W"].shape[0] != expected_input:
                raise ValueError("Checkpoint network observation size incompatible")
        if name in {"q_net", "target_net", "policy_net", "actor"} and net.layers[-1]["b"].size != actions:
            raise ValueError("Checkpoint policy action size incompatible")
        if name == "net" and (
            not getattr(net, "_dual_head", False) or net.layers[net._trunk_len]["b"].size != actions
        ):
            raise ValueError("Checkpoint dual policy head incompatible")
    if hasattr(agent, "q_table") and (
        agent.q_table.shape != (agent.state_space_size, actions) or not np.all(np.isfinite(agent.q_table))
    ):
        raise ValueError("Invalid tabular checkpoint")
    return agent, metadata
