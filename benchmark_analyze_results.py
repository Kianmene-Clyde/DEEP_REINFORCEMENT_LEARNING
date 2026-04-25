#!/usr/bin/env python3
"""
benchmark_analyze_results.py

Analyse les résultats générés par train_all.py et produit des fichiers faciles
à utiliser pour comparer les agents en soutenance.

À placer à la racine du projet DEEP_REINFORCEMENT_LEARNING, puis lancer :
    python benchmark_analyze_results.py

Entrées attendues :
    results/all_results.json
    results/<ENV>_<AGENT>_metrics.json

Sorties générées :
    results/benchmark_summary.csv
    results/benchmark_ranking_by_env.csv
    results/best_agents_by_env.csv
    results/benchmark_report.md
    results/benchmark_<ENV>_reward.png
    results/benchmark_<ENV>_composite.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Any, Tuple

KNOWN_ENVS = ["LineWorld", "GridWorld", "TicTacToe", "Quarto"]
GAME_ENVS = {"TicTacToe", "Quarto"}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_metrics_file_name(path: Path) -> Tuple[str | None, str | None]:
    """Retourne (env_name, agent_name) depuis un fichier *_metrics.json."""
    name = path.name
    if not name.endswith("_metrics.json"):
        return None, None

    stem = name[: -len("_metrics.json")]
    for env in KNOWN_ENVS:
        prefix = env + "_"
        if stem.startswith(prefix):
            return env, stem[len(prefix):]
    return None, None


def load_training_metrics(results_dir: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    metrics = {}
    for path in results_dir.glob("*_metrics.json"):
        env, agent = parse_metrics_file_name(path)
        if env is None or agent is None:
            continue
        data = load_json(path, default={}) or {}
        metrics[(env, agent)] = data
    return metrics


def summarize_training_metrics(metric_data: Dict[str, Any]) -> Dict[str, float]:
    rewards = [safe_float(x) for x in metric_data.get("episode_rewards", [])]
    lengths = [safe_float(x) for x in metric_data.get("episode_lengths", [])]
    step_times = [safe_float(x) for x in metric_data.get("step_times", [])]
    checkpoints = metric_data.get("checkpoint_metrics", {}) or {}

    last_100 = rewards[-100:] if rewards else []
    train_episodes = len(rewards)

    # Approximation de l'aire sous la courbe : plus elle est haute, plus l'agent apprend tôt et bien.
    auc_reward = mean(rewards) if rewards else 0.0

    best_cp_episode = ""
    best_cp_reward = None
    final_cp_reward = None
    if checkpoints:
        parsed = []
        for ep_str, vals in checkpoints.items():
            try:
                ep = int(ep_str)
            except Exception:
                continue
            parsed.append((ep, safe_float(vals.get("avg_reward", 0.0))))
        if parsed:
            parsed.sort(key=lambda x: x[0])
            best_ep, best_r = max(parsed, key=lambda x: x[1])
            best_cp_episode = best_ep
            best_cp_reward = best_r
            final_cp_reward = parsed[-1][1]

    return {
        "train_episodes": train_episodes,
        "train_avg_last_100": mean(last_100) if last_100 else 0.0,
        "train_std_last_100": pstdev(last_100) if len(last_100) > 1 else 0.0,
        "train_avg_length": mean(lengths) if lengths else 0.0,
        "train_avg_step_time_ms": (mean(step_times) * 1000.0) if step_times else 0.0,
        "train_auc_reward": auc_reward,
        "best_checkpoint_episode": best_cp_episode,
        "best_checkpoint_reward": best_cp_reward if best_cp_reward is not None else 0.0,
        "final_checkpoint_reward": final_cp_reward if final_cp_reward is not None else 0.0,
    }


def minmax(values: List[float], higher_is_better: bool = True) -> List[float]:
    """Renvoie des scores 0-100. Si toutes les valeurs sont identiques, renvoie 100 partout."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if abs(hi - lo) < 1e-12:
        return [100.0 for _ in values]
    scores = []
    for v in values:
        s = (v - lo) / (hi - lo) * 100.0
        if not higher_is_better:
            s = 100.0 - s
        scores.append(s)
    return scores


def add_scores(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ajoute des scores normalisés et un score composite par environnement."""
    output = []

    for env in sorted(set(row["env"] for row in rows)):
        group = [row for row in rows if row["env"] == env]

        perf_scores = minmax([row["avg_reward"] for row in group], higher_is_better=True)
        speed_scores = minmax([row["avg_step_time_ms"] for row in group], higher_is_better=False)
        stability_scores = minmax([row["std_reward"] for row in group], higher_is_better=False)
        train_scores = minmax([row["train_auc_reward"] for row in group], higher_is_better=True)

        for i, row in enumerate(group):
            row = dict(row)
            row["performance_score"] = round(perf_scores[i], 3)
            row["speed_score"] = round(speed_scores[i], 3)
            row["stability_score"] = round(stability_scores[i], 3)
            row["learning_score"] = round(train_scores[i], 3)

            if env in GAME_ENVS:
                # Pour TicTacToe et Quarto, le taux de victoire est très parlant.
                composite = (
                    0.50 * row["performance_score"]
                    + 0.25 * row["win_rate"] * 100.0
                    + 0.10 * row["stability_score"]
                    + 0.10 * row["speed_score"]
                    + 0.05 * row["learning_score"]
                )
            else:
                # Pour LineWorld/GridWorld, la récompense et la vitesse de convergence sont plus importantes.
                composite = (
                    0.65 * row["performance_score"]
                    + 0.15 * row["learning_score"]
                    + 0.10 * row["stability_score"]
                    + 0.10 * row["speed_score"]
                )
            row["composite_score"] = round(composite, 3)
            output.append(row)

    # Rang final dans chaque environnement.
    ranked = []
    for env in sorted(set(row["env"] for row in output)):
        group = [row for row in output if row["env"] == env]
        group.sort(
            key=lambda r: (
                r["composite_score"],
                r["avg_reward"],
                r["win_rate"],
                -r["avg_step_time_ms"],
            ),
            reverse=True,
        )
        for rank, row in enumerate(group, start=1):
            row = dict(row)
            row["rank"] = rank
            ranked.append(row)
    return ranked


def build_rows(results_dir: Path) -> List[Dict[str, Any]]:
    all_results = load_json(results_dir / "all_results.json", default={}) or {}
    training_metrics = load_training_metrics(results_dir)

    rows = []
    for env, agents in all_results.items():
        for agent, eval_res in agents.items():
            eval_res = eval_res or {}
            train_summary = summarize_training_metrics(training_metrics.get((env, agent), {}))

            error = eval_res.get("error", "")
            row = {
                "env": env,
                "agent": agent,
                "status": "ERROR" if error else "OK",
                "error": error,
                "avg_reward": safe_float(eval_res.get("avg_reward", 0.0)),
                "std_reward": safe_float(eval_res.get("std_reward", 0.0)),
                "win_rate": safe_float(eval_res.get("win_rate", 0.0)),
                "loss_rate": safe_float(eval_res.get("loss_rate", 0.0)),
                "draw_rate": safe_float(eval_res.get("draw_rate", 0.0)),
                "avg_length": safe_float(eval_res.get("avg_length", 0.0)),
                "avg_step_time_ms": safe_float(eval_res.get("avg_step_time_ms", 0.0)),
                "num_eval_episodes": int(safe_float(eval_res.get("num_episodes", 0))),
            }
            row.update(train_summary)
            rows.append(row)

    return add_scores(rows)


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(path: Path, rows: List[Dict[str, Any]]) -> None:
    envs = sorted(set(row["env"] for row in rows))
    lines = []
    lines.append("# Benchmark des agents par environnement")
    lines.append("")
    lines.append("Ce rapport classe les agents après entraînement à partir de `results/all_results.json` et des fichiers `*_metrics.json`.")
    lines.append("")
    lines.append("## Métriques utilisées")
    lines.append("")
    lines.append("- `avg_reward` : performance moyenne finale pendant l'évaluation.")
    lines.append("- `win_rate` : taux de victoire, surtout pertinent pour TicTacToe et Quarto.")
    lines.append("- `std_reward` : stabilité. Plus c'est bas, plus l'agent est régulier.")
    lines.append("- `avg_step_time_ms` : coût d'inférence. Plus c'est bas, plus l'agent décide vite.")
    lines.append("- `train_auc_reward` : moyenne des rewards pendant l'entraînement, utile pour estimer la vitesse d'apprentissage.")
    lines.append("- `composite_score` : score normalisé par environnement, utilisé pour produire un classement synthétique.")
    lines.append("")
    lines.append("> Important : on compare les agents à l'intérieur d'un même environnement. Les rewards ne sont pas directement comparables entre LineWorld, GridWorld, TicTacToe et Quarto.")
    lines.append("")

    for env in envs:
        group = [row for row in rows if row["env"] == env]
        group.sort(key=lambda r: r["rank"])
        best = group[0]
        lines.append(f"## {env}")
        lines.append("")
        lines.append(f"**Meilleur agent selon le score composite : `{best['agent']}`**")
        lines.append("")
        lines.append("| Rang | Agent | Reward moyen | Win rate | Std reward | Temps/step ms | Score composite |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|")
        for row in group[:10]:
            lines.append(
                f"| {row['rank']} | {row['agent']} | {row['avg_reward']:.4f} | "
                f"{row['win_rate']:.2%} | {row['std_reward']:.4f} | "
                f"{row['avg_step_time_ms']:.4f} | {row['composite_score']:.2f} |"
            )
        lines.append("")

        if env in GAME_ENVS:
            lines.append("**Interprétation à défendre :** ici, le taux de victoire compte beaucoup, car l'environnement représente un jeu contre un adversaire. Les agents de planification comme MCTS/RandomRollout peuvent être très performants car ils simulent des coups futurs, mais ils coûtent souvent plus cher en temps de décision.")
        else:
            lines.append("**Interprétation à défendre :** ici, l'environnement est simple et discret. Les méthodes tabulaires ou value-based peuvent très bien performer, car l'espace d'état est petit et la récompense guide directement vers l'objectif.")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def maybe_make_plots(results_dir: Path, rows: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("Matplotlib indisponible : graphiques non générés.")
        return

    for env in sorted(set(row["env"] for row in rows)):
        group = [row for row in rows if row["env"] == env]
        group.sort(key=lambda r: r["avg_reward"], reverse=True)
        agents = [row["agent"] for row in group]
        rewards = [row["avg_reward"] for row in group]

        plt.figure(figsize=(max(12, len(agents) * 0.7), 6))
        plt.bar(range(len(agents)), rewards)
        plt.xticks(range(len(agents)), agents, rotation=45, ha="right")
        plt.ylabel("Average reward")
        plt.title(f"Comparaison des rewards - {env}")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / f"benchmark_{env}_reward.png", dpi=160)
        plt.close()

        group.sort(key=lambda r: r["composite_score"], reverse=True)
        agents = [row["agent"] for row in group]
        scores = [row["composite_score"] for row in group]

        plt.figure(figsize=(max(12, len(agents) * 0.7), 6))
        plt.bar(range(len(agents)), scores)
        plt.xticks(range(len(agents)), agents, rotation=45, ha="right")
        plt.ylabel("Composite score")
        plt.title(f"Classement composite - {env}")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_dir / f"benchmark_{env}_composite.png", dpi=160)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results", help="Dossier contenant all_results.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = build_rows(results_dir)
    if not rows:
        raise SystemExit("Aucun résultat trouvé. Lance d'abord : python train_all.py --episodes 1000")

    summary_cols = [
        "env", "agent", "status", "rank", "avg_reward", "std_reward", "win_rate", "loss_rate", "draw_rate",
        "avg_length", "avg_step_time_ms", "num_eval_episodes", "train_episodes", "train_avg_last_100",
        "train_std_last_100", "train_auc_reward", "best_checkpoint_episode", "best_checkpoint_reward",
        "final_checkpoint_reward", "performance_score", "learning_score", "stability_score", "speed_score",
        "composite_score", "error",
    ]

    rows_sorted = sorted(rows, key=lambda r: (r["env"], r["rank"]))
    write_csv(results_dir / "benchmark_summary.csv", rows_sorted, summary_cols)
    write_csv(results_dir / "benchmark_ranking_by_env.csv", rows_sorted, summary_cols)

    best_rows = [row for row in rows_sorted if row["rank"] == 1]
    write_csv(results_dir / "best_agents_by_env.csv", best_rows, summary_cols)
    write_report(results_dir / "benchmark_report.md", rows_sorted)
    maybe_make_plots(results_dir, rows_sorted)

    print("Benchmark généré avec succès :")
    print(f"- {results_dir / 'benchmark_summary.csv'}")
    print(f"- {results_dir / 'benchmark_ranking_by_env.csv'}")
    print(f"- {results_dir / 'best_agents_by_env.csv'}")
    print(f"- {results_dir / 'benchmark_report.md'}")

    print("\nMeilleurs agents par environnement :")
    for row in best_rows:
        print(
            f"- {row['env']}: {row['agent']} | "
            f"reward={row['avg_reward']:.4f}, win_rate={row['win_rate']:.2%}, "
            f"score={row['composite_score']:.2f}"
        )


if __name__ == "__main__":
    main()
