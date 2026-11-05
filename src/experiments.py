"""Reproducible train/validation/test runner with independent final training seeds."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t
from .registry import (
    AGENT_ORDER,
    ENVIRONMENTS,
    PURE_PLANNING,
    ALGORITHM_NOTES,
    available_agents_for_env,
    make_environment,
    build_agent,
    get_hyperparameter_grid,
)
from .training.trainer import Trainer
from .training.evaluator import Evaluator
from .training.metrics import Metrics
from .training.checkpoints import save_checkpoint, load_checkpoint, PROTOCOL_VERSION

SOURCE_ROOT = Path(__file__).resolve().parents[1]
ROOT = SOURCE_ROOT if (SOURCE_ROOT / "configs").is_dir() else Path.cwd()
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".build" / "matplotlib"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summaries = []
    for env, name in dict.fromkeys((r["environment"], r["agent"]) for r in rows if r["status"] == "OK"):
        selected = [r for r in rows if r["status"] == "OK" and r["environment"] == env and r["agent"] == name]
        summary = {"environment": env, "agent": name, "n_training_seeds": len(selected)}
        for metric in [
            "avg_reward",
            "win_rate",
            "loss_rate",
            "draw_rate",
            "timeout_rate",
            "avg_length",
            "avg_step_time_ms",
            "training_time_seconds",
        ]:
            values = np.array([r[metric] for r in selected], dtype=float)
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if len(values) > 1 else None
            half = (
                float(student_t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values)))
                if sd is not None
                else None
            )
            summary[metric] = {
                "mean": mean,
                "std_between_seeds": sd,
                "ci95_mean": None if half is None else [mean - half, mean + half],
            }
        summaries.append(summary)
    return summaries


def plot_summaries(summaries, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for env in dict.fromkeys(s["environment"] for s in summaries):
        data = [s for s in summaries if s["environment"] == env]
        fig, axes = plt.subplots(1, 2, figsize=(max(10, len(data) * 0.65), 5.8))
        for axis, metric, label in [
            (axes[0], "avg_reward", "Test return"),
            (axes[1], "avg_step_time_ms", "Decision latency (ms)"),
        ]:
            means = [s[metric]["mean"] for s in data]
            errors = [
                0
                if s[metric]["ci95_mean"] is None
                else (s[metric]["ci95_mean"][1] - s[metric]["ci95_mean"][0]) / 2
                for s in data
            ]
            axis.errorbar(np.arange(len(data)), means, yerr=errors, fmt="o", capsize=3)
            axis.set_xticks(np.arange(len(data)), [s["agent"] for s in data], rotation=55, ha="right")
            axis.set_ylabel(label)
            axis.grid(axis="y", alpha=0.3)
        n = data[0]["n_training_seeds"]
        fig.suptitle(
            f"{env}: held-out evaluation, {n} training seed(s)"
            + ("; 95% t intervals between seeds" if n > 1 else "; no between-seed interval")
        )
        fig.tight_layout()
        fig.savefig(out / f"{env}_test_comparison.png", dpi=150)
        plt.close(fig)


def source_manifest():
    files = sorted([*ROOT.glob("src/**/*.py"), *ROOT.glob("configs/*.json"), ROOT / "pyproject.toml"])
    return {
        str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
        if p.exists()
    }


def plot_learning_histories(manifest, run_directory, output_directory):
    """Plot learning and independent checkpoint evaluation as separate panels."""
    import matplotlib.pyplot as plt

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    pairs = dict.fromkeys((m["environment"], m["agent"]) for m in manifest["models"])
    for env, name in pairs:
        histories = []
        for model in manifest["models"]:
            if model["environment"] == env and model["agent"] == name:
                path = Path(run_directory) / Path(model["checkpoint"]).parent / "training_metrics.json"
                data = json.loads(path.read_text(encoding="utf-8"))
                if data["episode_rewards"]:
                    histories.append(data)
        if not histories:
            continue
        length = min(len(h["episode_rewards"]) for h in histories)
        values = np.array([h["episode_rewards"][:length] for h in histories])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        window = min(100, length)
        for series in values:
            metrics = Metrics()
            metrics.episode_rewards = series.tolist()
            axes[0].plot(np.arange(1, length + 1), metrics.get_windowed_rewards(window), alpha=0.6)
        axes[0].set_title(f"Training returns, smoothing window {window}")
        axes[0].set_xlabel("Episode (this execution)")
        axes[0].set_ylabel("Return including exploration")
        for history in histories:
            checkpoints = sorted((int(k), v) for k, v in history["checkpoint_metrics"].items())
            axes[1].plot([k for k, _ in checkpoints], [v["avg_reward"] for _, v in checkpoints], "o-")
        axes[1].set_title("Validation checkpoints, frozen policy")
        axes[1].set_xlabel("Total training episodes")
        axes[1].set_ylabel("Validation return")
        for axis in axes:
            axis.grid(alpha=0.25)
        fig.suptitle(f"{env} / {name}: {len(histories)} training seed(s)")
        fig.tight_layout()
        fig.savefig(output_directory / f"{env}_{name.replace('+', '_')}.png", dpi=140)
        plt.close(fig)


def train_candidate(env_name, name, params, seed, config, out, role, deadline, resume=None):
    env = make_environment(env_name, seed, config["opponent"])
    start = 0
    if resume:
        agent, metadata = load_checkpoint(
            resume, environment=env_name, observation_size=env.observation_space, action_size=env.action_space
        )
        if (
            metadata["agent"] != name
            or metadata["seed"] != seed
            or metadata.get("opponent", "random") != config["opponent"]
        ):
            raise ValueError("Resume agent, seed or opponent differs from checkpoint")
        start = metadata["episodes_trained"]
        params = metadata["hyperparams"]
    else:
        agent = build_agent(name, env_name, env.observation_space, env.action_space, params, seed)
    if deadline is not None and time.perf_counter() > deadline:
        raise TimeoutError("Experiment time budget exhausted")
    metrics = Metrics()
    if name not in PURE_PLANNING:
        trainer = Trainer(
            env,
            agent,
            config["episodes"],
            config["max_steps"],
            eval_episodes=config["eval_episodes"],
            checkpoints=config["checkpoints"],
            verbose=not config["quiet"],
            seed=seed,
            evaluation_seed=1000000 + seed,
            start_episode=start,
            deadline=deadline,
        )
        metrics = trainer.train()
    folder = out / role / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    metadata = {
        "environment": env_name,
        "agent": name,
        "display_name": agent.name,
        "seed": seed,
        "observation_size": env.observation_space,
        "action_size": env.action_space,
        "hyperparams": params,
        "episodes_trained": start + (config["episodes"] if name not in PURE_PLANNING else 0),
        "training_time_seconds": metrics.training_time_seconds,
        "opponent": config["opponent"],
        "transitions": metrics.transitions,
        "algorithm_notes": ALGORITHM_NOTES.get(name, ""),
        "source_sha256": source_manifest(),
    }
    checkpoint = folder / "agent.pkl"
    save_checkpoint(checkpoint, agent, metadata)
    write_json(folder / "metadata.json", metadata)
    metrics.save(str(folder / "training_metrics.json"))
    return env, agent, metadata, checkpoint


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    p.add_argument("--quick", action="store_true", help="Alias for --profile smoke")
    p.add_argument("--config", type=Path)
    p.add_argument("--env", nargs="+", choices=list(ENVIRONMENTS))
    p.add_argument("--agent", nargs="+", choices=AGENT_ORDER)
    p.add_argument("--episodes", type=int)
    p.add_argument("--eval-episodes", type=int)
    p.add_argument("--max-steps", type=int)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--seeds", nargs="+", type=int)
    group.add_argument("--seed", type=int)
    p.add_argument("--grid", choices=["none", "compact", "full"], default="none")
    p.add_argument("--max-configs-per-agent", type=int)
    p.add_argument("--opponent", choices=["random", "heuristic"], default="random")
    p.add_argument("--output", type=Path)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--time-budget-seconds", type=float)
    p.add_argument("--resume-checkpoint", type=Path)
    p.add_argument("--no-plots", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    profile = "smoke" if args.quick else args.profile
    bundled = ROOT / "configs" / f"{profile}.json"
    if not bundled.is_file():
        bundled = Path(sys.prefix) / "configs" / f"{profile}.json"
    config = json.loads((args.config or bundled).read_text(encoding="utf-8"))
    for key in ["episodes", "eval_episodes", "max_steps", "time_budget_seconds"]:
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    config.update(opponent=args.opponent, quiet=args.quiet, profile=profile, grid=args.grid)
    if args.seeds is not None or args.seed is not None:
        config["seeds"] = args.seeds if args.seeds is not None else [args.seed]
    if any(config[k] < 1 for k in ["episodes", "eval_episodes", "max_steps"]):
        raise ValueError("Episode and step budgets must be positive")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]) or min(config["seeds"]) < 0:
        raise ValueError("Training seeds must be distinct nonnegative integers")
    if args.max_configs_per_agent is not None and args.max_configs_per_agent < 1:
        raise ValueError("Configuration limit must be positive")
    budget = config["time_budget_seconds"]
    if budget is not None and budget <= 0:
        raise ValueError("Time budget must be positive")
    env_names = args.env or list(ENVIRONMENTS)
    if args.resume_checkpoint and (
        len(env_names) != 1
        or not args.agent
        or len(args.agent) != 1
        or len(config["seeds"]) != 1
        or args.grid != "none"
    ):
        raise ValueError("Resume requires one environment, one agent, one seed, and --grid none")
    for env in env_names:
        if args.agent and any(a not in available_agents_for_env(env) for a in args.agent):
            raise ValueError(f"Agent unavailable for {env}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    out = (args.output or ROOT / "results" / "runs" / stamp).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output directory must be empty; historical runs are preserved")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", config)
    start = time.perf_counter()
    deadline = None if budget is None else start + budget
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": stamp,
        "git_revision": revision,
        "source_sha256": source_manifest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {n: importlib.metadata.version(n) for n in ["numpy", "scipy", "matplotlib", "flask"]},
        "models": [],
        "status": "running",
        "selected_configs": {},
        "interpretation": "smoke validates execution only; full results require sufficient learning budgets",
    }
    final_rows, validation_rows = [], []
    budget_exhausted = False
    for env_name in env_names:
        for name in args.agent or available_agents_for_env(env_name):
            safe = name.replace("+", "_").replace(" ", "_")
            agent_out = out / env_name / safe
            configs = (
                [{}]
                if args.grid == "none"
                else get_hyperparameter_grid(name, args.grid, args.max_configs_per_agent)
            )
            best_score, best_params, best_candidates = -np.inf, None, None
            try:
                for cid, grid_params in enumerate(configs):
                    params = {**config["agent_params"], **grid_params}
                    # Cache trained candidates; final evaluation never participates in selection.
                    candidates = []
                    scores = []
                    for seed in config["seeds"]:
                        env, agent, metadata, checkpoint = train_candidate(
                            env_name,
                            name,
                            params,
                            seed,
                            config,
                            agent_out,
                            f"validation/config_{cid}",
                            deadline,
                            args.resume_checkpoint,
                        )
                        if args.resume_checkpoint:
                            params = metadata["hyperparams"]
                        evaluator = Evaluator(
                            env, agent, config["eval_episodes"], config["max_steps"], 1000000 + seed, deadline
                        )
                        stats = evaluator.evaluate()
                        row = {
                            "environment": env_name,
                            "agent": name,
                            "config_id": cid,
                            "seed": seed,
                            **stats,
                        }
                        validation_rows.append(row)
                        scores.append(stats["avg_reward"])
                        candidates.append((env, agent, metadata, checkpoint))
                    score = float(np.mean(scores))
                    if score > best_score:
                        best_score, best_params, best_candidates = score, params, candidates
                manifest["selected_configs"][env_name + "/" + name] = {
                    "params": best_params,
                    "validation_mean_return": best_score,
                    "number_of_configs": len(configs),
                    "selection_metric": "mean validation return",
                }
                if len(configs) > 1:
                    # Fresh training seeds prevent tuning on the final training seed population.
                    best_candidates = [
                        train_candidate(
                            env_name, name, best_params, seed + 10000, config, agent_out, "final", deadline
                        )
                        for seed in config["seeds"]
                    ]
                for env, agent, metadata, checkpoint in best_candidates:
                    if deadline is not None and time.perf_counter() > deadline:
                        raise TimeoutError("Experiment time budget exhausted")
                    evaluator = Evaluator(
                        env,
                        agent,
                        config["eval_episodes"],
                        config["max_steps"],
                        2000000 + metadata["seed"],
                        deadline,
                    )
                    stats = evaluator.evaluate()
                    row = {
                        "environment": env_name,
                        "agent": name,
                        "seed": metadata["seed"],
                        "status": "OK",
                        **stats,
                        "episodes_trained": metadata["episodes_trained"],
                        "training_time_seconds": metadata["training_time_seconds"],
                        "transitions": metadata["transitions"],
                    }
                    final_rows.append(row)
                    write_json(checkpoint.parent / "test_episodes.json", evaluator.episodes)
                    write_json(checkpoint.parent / "test_summary.json", row)
                    manifest["models"].append(
                        {**metadata, "checkpoint": str(checkpoint.relative_to(out)).replace("\\", "/")}
                    )
                    print(
                        f"{env_name:10} {name:20} seed={metadata['seed']} test={stats['avg_reward']:.4f} "
                        f"success={stats['success_rate']:.1%} timeout={stats['timeout_rate']:.1%}",
                        flush=True,
                    )
            except Exception as exc:
                final_rows.append(
                    {
                        "environment": env_name,
                        "agent": name,
                        "status": "ERROR",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"ERROR {env_name}/{name}: {exc}", flush=True)
                if isinstance(exc, TimeoutError):
                    budget_exhausted = True
            write_csv(out / "test_results.csv", final_rows)
            write_csv(out / "validation_results.csv", validation_rows)
            write_json(out / "manifest.json", manifest)
            if budget_exhausted:
                break
        if budget_exhausted:
            break
    summary = summarize(final_rows)
    write_json(out / "summary.json", summary)
    if not args.no_plots and summary:
        plot_summaries(summary, out)
        plot_learning_histories(manifest, out, out / "learning_curves")
    manifest.update(
        status="incomplete" if any(r["status"] == "ERROR" for r in final_rows) else "complete",
        wall_seconds=time.perf_counter() - start,
        errors=[r for r in final_rows if r["status"] == "ERROR"],
    )
    write_json(out / "manifest.json", manifest)
    # Only complete runs become the GUI's current, verified model source.
    if manifest["status"] == "complete":
        try:
            portable = str(out.relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            portable = str(out)
        write_json(ROOT / "results" / "latest_run.json", {"run_directory": portable})
    print(f"Run: {out}; status={manifest['status']}", flush=True)
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
