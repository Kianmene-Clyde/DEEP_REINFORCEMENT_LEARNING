import json
import numpy as np
import pytest
from src.experiments import main, summarize


def test_multi_seed_summary_uses_between_training_seed_uncertainty():
    rows = []
    for seed, value in enumerate([1.0, 2.0, 3.0, 4.0, 5.0]):
        rows.append(
            dict(
                environment="LineWorld",
                agent="Random",
                status="OK",
                seed=seed,
                avg_reward=value,
                win_rate=0.5,
                loss_rate=0.2,
                draw_rate=0,
                timeout_rate=0.3,
                avg_length=12.0,
                avg_step_time_ms=0.1,
                training_time_seconds=0.0,
            )
        )
    summary = summarize(rows)[0]
    result = summary["avg_reward"]
    assert summary["n_training_seeds"] == 5
    assert result["mean"] == 3
    assert result["std_between_seeds"] == pytest.approx(np.sqrt(2.5))
    assert result["ci95_mean"] == pytest.approx([1.0367568385, 4.9632431615])
    assert summarize(rows[:1])[0]["avg_reward"]["ci95_mean"] is None


def test_cli_selection_and_independent_final_seeds(tmp_path):
    config = tmp_path / "profile.json"
    config.write_text(
        json.dumps(
            {
                "episodes": 2,
                "eval_episodes": 2,
                "max_steps": 10,
                "seeds": [7, 13],
                "checkpoints": [],
                "time_budget_seconds": 30,
                "agent_params": {"hidden_layers": [4]},
            }
        )
    )
    out = tmp_path / "run"
    assert (
        main(
            [
                "--config",
                str(config),
                "--env",
                "LineWorld",
                "--agent",
                "TabularQL",
                "--grid",
                "compact",
                "--max-configs-per-agent",
                "2",
                "--output",
                str(out),
                "--quiet",
            ]
        )
        == 0
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert {r["seed"] for r in manifest["models"]} == {10007, 10013}
    assert (out / "LineWorld_test_comparison.png").is_file()
    assert (out / "validation_results.csv").is_file()
    assert json.loads((out / "summary.json").read_text())[0]["n_training_seeds"] == 2
    with pytest.raises(ValueError, match="empty"):
        main(["--config", str(config), "--output", str(out)])
