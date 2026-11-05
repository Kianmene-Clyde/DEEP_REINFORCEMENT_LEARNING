import json
import pytest
import gui
from src.registry import build_agent
from src.training.checkpoints import save_checkpoint


@pytest.fixture
def app(tmp_path):
    gui.app.config.update(TESTING=True, MODEL_RUN_DIRECTORY=str(tmp_path))
    gui._games.clear()
    return gui.app


def test_human_game_actions_and_session_isolation(app):
    first, second = app.test_client(), app.test_client()
    assert first.get("/").status_code == 200
    assert first.post("/api/new_game", json={"env": "LineWorld", "agent": "Human"}).status_code == 200
    assert second.post("/api/step").status_code == 409
    assert first.post("/api/step").status_code == 409
    assert first.post("/api/human_action", json={"action": -1}).status_code == 400
    assert first.post("/api/human_action", json={"action": 1.5}).status_code == 400
    for _ in range(9):
        result = first.post("/api/human_action", json={"action": 1})
        assert result.status_code == 200
    assert result.json["outcome"] == "success"
    assert result.json["done"]
    assert first.post("/api/human_action", json={"action": 1}).status_code == 409


def test_missing_model_is_explicit_and_preserves_active_game(app):
    client = app.test_client()
    client.post("/api/new_game", json={"env": "LineWorld", "agent": "Random"})
    response = client.post("/api/new_game", json={"env": "LineWorld", "agent": "PPO"})
    assert response.status_code == 409
    assert "modèle" in response.json["error"]
    assert client.post("/api/step").status_code == 200


def test_versioned_model_loaded_and_identified(app, tmp_path):
    agent = build_agent("PPO", "LineWorld", 10, 2, {"hidden_layers": [4]}, seed=2)
    metadata = {
        "agent": "PPO",
        "environment": "LineWorld",
        "observation_size": 10,
        "action_size": 2,
        "seed": 2,
        "episodes_trained": 12,
    }
    save_checkpoint(tmp_path / "ppo.pkl", agent, metadata)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {"protocol_version": 2, "status": "complete", "models": [{**metadata, "checkpoint": "ppo.pkl"}]}
        ),
        encoding="utf-8",
    )
    client = app.test_client()
    result = client.post("/api/new_game", json={"env": "LineWorld", "agent": "PPO"})
    assert result.status_code == 200
    assert "12 épisodes" in result.json["model_status"]
    assert "seed 2" in result.json["model_status"]
    assert client.post("/api/step").status_code == 200


@pytest.mark.parametrize("env", ["LineWorld", "GridWorld", "TicTacToe", "Quarto"])
def test_random_demo_and_rule_explanation(app, env):
    client = app.test_client()
    new = client.post("/api/new_game", json={"env": env, "agent": "Random"})
    assert new.status_code == 200 and new.json["rules"]
    response = client.post("/api/auto_play")
    assert response.status_code == 200
    assert response.json["steps"][-1]["done"]


def test_invalid_json_and_agent(app):
    client = app.test_client()
    assert client.post("/api/new_game", json={"env": "unknown"}).status_code == 400
    assert client.post("/api/new_game", json={"agent": "invalid"}).status_code == 400
    assert client.post("/api/new_game", json=["invalid"]).status_code == 400
    assert client.post("/api/new_game", json={"env": []}).status_code == 400
