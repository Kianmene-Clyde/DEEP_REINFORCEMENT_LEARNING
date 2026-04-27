"""
plot_grid_search_results.py

Script de visualisation des résultats produits par train_all.py.

Objectifs :
1. Comparer efficacement les meilleurs agents par environnement à partir de :
   results/grid_search/best_agents_by_env_after_tuning.csv

2. Comparer les performances de chaque agent selon les variations de ses hyperparamètres à partir de :
   results/grid_search/all_config_results.csv

Le script génère des graphiques dans :
   results/plots_grid_search/

Utilisation recommandée depuis la racine du projet :
   python plot_grid_search_results.py

Options utiles :
   python plot_grid_search_results.py --metric composite_score
   python plot_grid_search_results.py --top-k 8
   python plot_grid_search_results.py --show
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_BEST_CSV = Path("results/grid_search/best_agents_by_env_after_tuning.csv")
DEFAULT_ALL_CONFIG_CSV = Path("results/grid_search/all_config_results.csv")
DEFAULT_OUTPUT_DIR = Path("results/plots_grid_search")

PREFERRED_METRICS = [
    "composite_score",
    "avg_reward",
    "win_rate",
    "draw_rate",
    "loss_rate",
    "std_reward",
    "avg_length",
    "avg_step_time_ms",
    "training_time_seconds",
    "episodes_trained",
]

METRIC_ALIASES = {
    "env": "environment",
    "env_name": "environment",
    "agent_name": "agent",
    "mean_reward": "avg_reward",
    "average_reward": "avg_reward",
    "reward_mean": "avg_reward",
    "episode_length": "avg_length",
    "avg_episode_length": "avg_length",
    "decision_time_ms": "avg_step_time_ms",
    "avg_decision_time_ms": "avg_step_time_ms",
    "train_time": "training_time_seconds",
    "training_time": "training_time_seconds",
}

STANDARD_COLUMNS = {
    "environment",
    "agent",
    "rank",
    "config_id",
    "config_name",
    "config_label",
    "hyperparams",
    "hyperparameters",
    "params",
    "config",
    "status",
    "error",
    "model_path",
    "seed",
    "timestamp",
}

METRIC_COLUMNS = set(PREFERRED_METRICS) | {
    "min_reward",
    "max_reward",
    "median_reward",
    "train_auc_reward",
    "eval_episodes",
    "num_episodes",
    "episodes",
    "loss",
    "policy_loss",
    "value_loss",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les noms de colonnes pour rendre le script robuste."""
    df = df.copy()
    renamed = {}
    for col in df.columns:
        clean = str(col).strip()
        clean = re.sub(r"\s+", "_", clean)
        clean = clean.replace("-", "_").lower()
        renamed[col] = METRIC_ALIASES.get(clean, clean)
    return df.rename(columns=renamed)


def ensure_required_columns(df: pd.DataFrame, csv_path: Path) -> None:
    missing = [c for c in ["environment", "agent"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Le fichier {csv_path} doit contenir au minimum les colonnes {missing}. "
            f"Colonnes trouvées : {list(df.columns)}"
        )


def read_csv_safely(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")
    df = pd.read_csv(csv_path)
    df = normalize_columns(df)
    ensure_required_columns(df, csv_path)
    return df


def available_metrics(df: pd.DataFrame) -> List[str]:
    metrics = []
    for metric in PREFERRED_METRICS:
        if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
            metrics.append(metric)
    return metrics


def choose_metric(df: pd.DataFrame, requested_metric: Optional[str]) -> str:
    metrics = available_metrics(df)
    if requested_metric:
        requested_metric = requested_metric.strip().lower()
        if requested_metric in df.columns and pd.api.types.is_numeric_dtype(df[requested_metric]):
            return requested_metric
        raise ValueError(
            f"La métrique demandée '{requested_metric}' est absente ou non numérique. "
            f"Métriques disponibles : {metrics}"
        )
    if metrics:
        return metrics[0]
    raise ValueError("Aucune métrique numérique exploitable trouvée dans le CSV.")


def metric_is_lower_better(metric: str) -> bool:
    return metric in {
        "loss",
        "policy_loss",
        "value_loss",
        "loss_rate",
        "std_reward",
        "avg_length",
        "avg_step_time_ms",
        "training_time_seconds",
    }


def sort_by_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return df.sort_values(metric, ascending=metric_is_lower_better(metric)).reset_index(drop=True)


def sanitize_filename(text: str) -> str:
    text = str(text).replace("+", "plus")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")[:150]


def savefig(path: Path, show: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def annotate_bars(ax, values: Sequence[float], horizontal: bool = False) -> None:
    finite_values = [v for v in values if pd.notna(v) and np.isfinite(v)]
    if not finite_values:
        return
    vmin, vmax = min(finite_values), max(finite_values)
    span = max(abs(vmax - vmin), 1e-9)
    for patch, value in zip(ax.patches, values):
        if pd.isna(value) or not np.isfinite(value):
            continue
        label = f"{value:.3f}"
        if horizontal:
            x = patch.get_width()
            y = patch.get_y() + patch.get_height() / 2
            ax.text(x + 0.01 * span, y, label, va="center", fontsize=8)
        else:
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_height()
            ax.text(x, y + 0.01 * span, label, ha="center", va="bottom", fontsize=8)


def format_title(metric: str) -> str:
    labels = {
        "composite_score": "score composite",
        "avg_reward": "reward moyen",
        "win_rate": "taux de victoire",
        "draw_rate": "taux de match nul",
        "loss_rate": "taux de défaite",
        "std_reward": "écart-type du reward",
        "avg_length": "longueur moyenne des épisodes",
        "avg_step_time_ms": "temps moyen de décision (ms)",
        "training_time_seconds": "temps d'entraînement (s)",
        "episodes_trained": "épisodes entraînés",
    }
    return labels.get(metric, metric.replace("_", " "))


def maybe_convert_percent(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return s
    if s.max() <= 1.0:
        return s * 100.0
    return s


def get_metric_series(df: pd.DataFrame, metric: str) -> pd.Series:
    s = pd.to_numeric(df[metric], errors="coerce")
    if metric in {"win_rate", "draw_rate", "loss_rate"}:
        s = maybe_convert_percent(s)
    return s


def metric_axis_label(metric: str) -> str:
    label = format_title(metric)
    if metric in {"win_rate", "draw_rate", "loss_rate"}:
        label += " (%)"
    return label


def parse_maybe_dict(value) -> Dict:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def expand_hyperparameter_dict_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dict_cols = [c for c in ["hyperparams", "hyperparameters", "params", "config"] if c in df.columns]
    for dict_col in dict_cols:
        extracted_rows = df[dict_col].apply(parse_maybe_dict)
        keys = sorted({k for d in extracted_rows for k in d.keys()})
        for key in keys:
            clean_key = str(key).strip().lower().replace(" ", "_").replace("-", "_")
            if clean_key not in df.columns:
                df[clean_key] = extracted_rows.apply(lambda d: d.get(key, np.nan))
    return df


def infer_hyperparameter_columns(df: pd.DataFrame) -> List[str]:
    known_hp = [
        "learning_rate",
        "lr",
        "discount_factor",
        "gamma",
        "epsilon",
        "epsilon_decay",
        "epsilon_min",
        "batch_size",
        "hidden_layers",
        "clip_ratio",
        "clip_epsilon",
        "entropy_coef",
        "value_coef",
        "search_budget",
        "num_simulations",
        "simulations",
        "max_rollout_depth",
        "rollout_depth",
        "exploration_constant",
        "c_puct",
        "temperature",
        "buffer_size",
        "target_update_freq",
        "planning_steps",
    ]
    hp_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in known_hp or col_lower.startswith("hp_") or col_lower.startswith("param_"):
            hp_cols.append(col)

    excluded = STANDARD_COLUMNS | METRIC_COLUMNS
    for col in df.columns:
        if col in hp_cols or col in excluded or col in {"environment", "agent"}:
            continue
        nunique = df[col].nunique(dropna=True)
        if 1 < nunique <= 20:
            if not pd.api.types.is_numeric_dtype(df[col]) or col not in available_metrics(df):
                hp_cols.append(col)
    return list(dict.fromkeys([c for c in hp_cols if c in df.columns]))


def short_value(value, max_len: int = 18) -> str:
    if isinstance(value, float):
        if abs(value) < 0.01 and value != 0:
            return f"{value:.1e}"
        return f"{value:g}"
    text = str(value).replace("[", "").replace("]", "").replace("'", "")
    text = re.sub(r"\s+", "", text)
    return text[: max_len - 1] + "…" if len(text) > max_len else text


def make_config_label(row: pd.Series, hp_cols: Sequence[str]) -> str:
    if "config_label" in row.index and pd.notna(row["config_label"]):
        return str(row["config_label"])
    parts = []
    for col in hp_cols[:3]:
        value = row.get(col, np.nan)
        if pd.isna(value):
            continue
        clean_col = col.replace("learning_rate", "lr")
        clean_col = clean_col.replace("discount_factor", "gamma")
        clean_col = clean_col.replace("epsilon_decay", "eps_decay")
        parts.append(f"{clean_col}={short_value(value)}")
    if parts:
        return " | ".join(parts)
    if "config_name" in row.index and pd.notna(row["config_name"]):
        return str(row["config_name"])
    if "config_id" in row.index and pd.notna(row["config_id"]):
        return f"config={row['config_id']}"
    return "config"


def plot_best_agents_overview(best_df: pd.DataFrame, output_dir: Path, metric: str, show: bool) -> None:
    rows = []
    for env, group in best_df.groupby("environment"):
        sorted_group = sort_by_metric(group, metric)
        if not sorted_group.empty:
            rows.append(sorted_group.iloc[0])
    if not rows:
        return
    top_df = pd.DataFrame(rows)
    values = get_metric_series(top_df, metric)

    plt.figure(figsize=(11, 6))
    ax = plt.gca()
    bars = ax.bar(top_df["environment"].astype(str), values)
    ax.set_title(f"Meilleur agent par environnement après tuning — {format_title(metric)}", fontsize=14)
    ax.set_xlabel("Environnement")
    ax.set_ylabel(metric_axis_label(metric))
    ax.grid(axis="y", alpha=0.3)
    for bar, (_, row), value in zip(bars, top_df.iterrows(), values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{row['agent']}\n{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    savefig(output_dir / f"01_best_agent_by_environment_{metric}.png", show=show)


def plot_ranking_by_environment(best_df: pd.DataFrame, output_dir: Path, metric: str, top_k: int, show: bool) -> None:
    for env, group in best_df.groupby("environment"):
        group = sort_by_metric(group, metric).head(top_k).copy()
        if group.empty:
            continue
        group = group.iloc[::-1]
        values = get_metric_series(group, metric)
        plt.figure(figsize=(11, max(5, 0.45 * len(group) + 2)))
        ax = plt.gca()
        ax.barh(group["agent"].astype(str), values)
        ax.set_title(f"{env} — classement des agents après tuning ({format_title(metric)})", fontsize=14)
        ax.set_xlabel(metric_axis_label(metric))
        ax.set_ylabel("Agent")
        ax.grid(axis="x", alpha=0.3)
        annotate_bars(ax, list(values), horizontal=True)
        savefig(output_dir / f"02_ranking_{sanitize_filename(env)}_{metric}.png", show=show)


def plot_heatmap_env_agent(best_df: pd.DataFrame, output_dir: Path, metric: str, show: bool) -> None:
    if best_df["environment"].nunique() < 2 or best_df["agent"].nunique() < 2:
        return
    pivot = best_df.pivot_table(
        index="environment",
        columns="agent",
        values=metric,
        aggfunc="max" if not metric_is_lower_better(metric) else "min",
    )
    if pivot.empty:
        return
    data = pivot.to_numpy(dtype=float)
    plt.figure(figsize=(max(10, 0.7 * len(pivot.columns)), max(5, 0.7 * len(pivot.index))))
    ax = plt.gca()
    im = ax.imshow(data, aspect="auto")
    ax.set_title(f"Carte de performance environnement × agent — {format_title(metric)}", fontsize=14)
    ax.set_xlabel("Agent")
    ax.set_ylabel("Environnement")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, label=metric_axis_label(metric))
    savefig(output_dir / f"03_heatmap_environment_agent_{metric}.png", show=show)


def plot_best_agents_secondary_metrics(best_df: pd.DataFrame, output_dir: Path, show: bool) -> None:
    for metric in ["avg_reward", "win_rate", "avg_step_time_ms", "training_time_seconds"]:
        if metric not in best_df.columns or not pd.api.types.is_numeric_dtype(best_df[metric]):
            continue
        try:
            plot_best_agents_overview(best_df, output_dir, metric, show)
            plot_ranking_by_environment(best_df, output_dir, metric, top_k=8, show=show)
        except Exception as exc:
            print(f"  ⚠ Impossible de tracer la métrique {metric}: {exc}")


def plot_top_configs_per_agent_env(
    all_df: pd.DataFrame,
    output_dir: Path,
    metric: str,
    hp_cols: Sequence[str],
    top_k: int,
    show: bool,
) -> None:
    hp_dir = output_dir / "hyperparameters" / "top_configs"
    for (env, agent), group in all_df.groupby(["environment", "agent"]):
        group = group.copy()
        group = group[pd.notna(group[metric])]
        if len(group) <= 1:
            continue
        group["config_label_for_plot"] = group.apply(lambda r: make_config_label(r, hp_cols), axis=1)
        group = sort_by_metric(group, metric).head(top_k).iloc[::-1]
        values = get_metric_series(group, metric)
        plt.figure(figsize=(12, max(5, 0.55 * len(group) + 2)))
        ax = plt.gca()
        ax.barh(group["config_label_for_plot"].astype(str), values)
        ax.set_title(
            f"{env} — {agent}\nMeilleures combinaisons d'hyperparamètres ({format_title(metric)})",
            fontsize=13,
        )
        ax.set_xlabel(metric_axis_label(metric))
        ax.set_ylabel("Configuration")
        ax.grid(axis="x", alpha=0.3)
        annotate_bars(ax, list(values), horizontal=True)
        filename = f"top_configs_{sanitize_filename(env)}_{sanitize_filename(agent)}_{metric}.png"
        savefig(hp_dir / filename, show=show)


def plot_hyperparameter_effects(
    all_df: pd.DataFrame,
    output_dir: Path,
    metric: str,
    hp_cols: Sequence[str],
    show: bool,
) -> pd.DataFrame:
    effect_dir = output_dir / "hyperparameters" / "parameter_effects"
    summary_rows = []
    for (env, agent), group in all_df.groupby(["environment", "agent"]):
        group = group.copy()
        group = group[pd.notna(group[metric])]
        if len(group) <= 1:
            continue
        for hp in hp_cols:
            if hp not in group.columns:
                continue
            hp_group = group[[hp, metric]].dropna()
            if hp_group[hp].nunique() <= 1:
                continue
            aggregated = hp_group.groupby(hp, dropna=True)[metric].agg(["mean", "std", "count"]).reset_index()
            if aggregated.empty:
                continue
            ascending = metric_is_lower_better(metric)
            best_row = aggregated.sort_values("mean", ascending=ascending).iloc[0]
            summary_rows.append(
                {
                    "environment": env,
                    "agent": agent,
                    "hyperparameter": hp,
                    "best_value": best_row[hp],
                    f"best_mean_{metric}": best_row["mean"],
                    "num_values_tested": aggregated[hp].nunique(),
                }
            )
            if pd.api.types.is_numeric_dtype(aggregated[hp]):
                aggregated = aggregated.sort_values(hp)
                x = aggregated[hp].astype(float)
                x_labels = [short_value(v) for v in aggregated[hp]]
            else:
                aggregated = aggregated.sort_values("mean", ascending=ascending)
                x = np.arange(len(aggregated))
                x_labels = [short_value(v, 24) for v in aggregated[hp]]
            plt.figure(figsize=(10, 5.5))
            ax = plt.gca()
            if pd.api.types.is_numeric_dtype(aggregated[hp]):
                ax.plot(x, aggregated["mean"], marker="o")
                if aggregated["std"].notna().any():
                    yerr = aggregated["std"].fillna(0).to_numpy()
                    ax.errorbar(x, aggregated["mean"], yerr=yerr, fmt="none", capsize=4)
                ax.set_xticks(x)
                ax.set_xticklabels(x_labels, rotation=0)
            else:
                ax.bar(x, aggregated["mean"])
                ax.set_xticks(x)
                ax.set_xticklabels(x_labels, rotation=35, ha="right")
            ax.set_title(f"{env} — {agent}\nEffet de {hp} sur {format_title(metric)}", fontsize=13)
            ax.set_xlabel(hp)
            ax.set_ylabel(metric_axis_label(metric))
            ax.grid(axis="y", alpha=0.3)
            filename = f"effect_{sanitize_filename(env)}_{sanitize_filename(agent)}_{sanitize_filename(hp)}_{metric}.png"
            savefig(effect_dir / filename, show=show)
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary.to_csv(output_dir / "hyperparameter_effect_summary.csv", index=False)
    return summary


def plot_best_config_comparison_per_environment(
    all_df: pd.DataFrame,
    output_dir: Path,
    metric: str,
    hp_cols: Sequence[str],
    top_k: int,
    show: bool,
) -> None:
    best_rows = []
    for (env, agent), group in all_df.groupby(["environment", "agent"]):
        group = group.copy()
        group = group[pd.notna(group[metric])]
        if group.empty:
            continue
        best = sort_by_metric(group, metric).iloc[0].copy()
        best["config_label_for_plot"] = make_config_label(best, hp_cols)
        best_rows.append(best)
    if not best_rows:
        return
    best_configs = pd.DataFrame(best_rows)
    for env, group in best_configs.groupby("environment"):
        group = sort_by_metric(group, metric).head(top_k).iloc[::-1]
        values = get_metric_series(group, metric)
        labels = [f"{row['agent']}\n{row.get('config_label_for_plot', '')}" for _, row in group.iterrows()]
        plt.figure(figsize=(12, max(5, 0.7 * len(group) + 2)))
        ax = plt.gca()
        ax.barh(labels, values)
        ax.set_title(
            f"{env} — meilleure configuration trouvée pour chaque agent\n"
            f"Comparaison après tuning ({format_title(metric)})",
            fontsize=13,
        )
        ax.set_xlabel(metric_axis_label(metric))
        ax.set_ylabel("Agent + meilleure configuration")
        ax.grid(axis="x", alpha=0.3)
        annotate_bars(ax, list(values), horizontal=True)
        filename = f"best_config_each_agent_{sanitize_filename(env)}_{metric}.png"
        savefig(output_dir / "hyperparameters" / filename, show=show)


def write_visual_report(
    output_dir: Path,
    best_df: pd.DataFrame,
    all_df: Optional[pd.DataFrame],
    metric: str,
    hp_cols: Sequence[str],
) -> None:
    report_path = output_dir / "visualization_report.md"
    lines = []
    lines.append("# Rapport automatique des visualisations\n")
    lines.append("Ce dossier contient les graphiques générés à partir des résultats de grid search.\n")
    lines.append(f"Métrique principale utilisée : **{metric}** ({format_title(metric)}).\n")
    lines.append("## Meilleurs agents par environnement\n")
    for env, group in best_df.groupby("environment"):
        sorted_group = sort_by_metric(group, metric)
        if sorted_group.empty:
            continue
        best = sorted_group.iloc[0]
        lines.append(f"- **{env}** : meilleur agent = **{best['agent']}**, {metric} = **{best[metric]:.4f}**.")
    if all_df is not None and not all_df.empty:
        lines.append("\n## Hyperparamètres détectés\n")
        if hp_cols:
            for hp in hp_cols:
                lines.append(f"- `{hp}`")
        else:
            lines.append("- Aucun hyperparamètre détecté automatiquement.")
        lines.append("\n## Fichiers importants générés\n")
        lines.append("- `01_best_agent_by_environment_*.png` : meilleur agent par environnement.")
        lines.append("- `02_ranking_<ENV>_*.png` : classement des agents pour chaque environnement.")
        lines.append("- `03_heatmap_environment_agent_*.png` : comparaison environnement × agent.")
        lines.append("- `hyperparameters/top_configs/` : meilleures combinaisons par agent et environnement.")
        lines.append("- `hyperparameters/parameter_effects/` : effet individuel des hyperparamètres.")
        lines.append("- `hyperparameters/best_config_each_agent_*.png` : meilleure configuration de chaque agent par environnement.")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Génère des graphiques simples et pertinents pour comparer les agents et les hyperparamètres."
    )
    parser.add_argument("--best-csv", type=Path, default=DEFAULT_BEST_CSV)
    parser.add_argument("--all-config-csv", type=Path, default=DEFAULT_ALL_CONFIG_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metric", type=str, default=None, help="Exemples : composite_score, avg_reward, win_rate.")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("VISUALISATION DES RÉSULTATS DE GRID SEARCH")
    print("=" * 72)

    print(f"\n[1/3] Lecture du fichier : {args.best_csv}")
    best_df = read_csv_safely(args.best_csv)
    metric = choose_metric(best_df, args.metric)
    print(f"  ✓ {len(best_df)} lignes chargées")
    print(f"  ✓ Métrique principale : {metric}")

    print("\n[2/3] Génération des graphiques de comparaison des meilleurs agents")
    plot_best_agents_overview(best_df, args.output_dir, metric, args.show)
    plot_ranking_by_environment(best_df, args.output_dir, metric, args.top_k, args.show)
    plot_heatmap_env_agent(best_df, args.output_dir, metric, args.show)
    plot_best_agents_secondary_metrics(best_df, args.output_dir, args.show)
    print(f"  ✓ Graphiques sauvegardés dans : {args.output_dir}")

    all_df = None
    hp_cols: List[str] = []
    if args.all_config_csv.exists():
        print(f"\n[3/3] Lecture du fichier des configurations : {args.all_config_csv}")
        all_df = read_csv_safely(args.all_config_csv)
        all_df = expand_hyperparameter_dict_columns(all_df)
        if metric not in all_df.columns or not pd.api.types.is_numeric_dtype(all_df[metric]):
            metric_for_hp = choose_metric(all_df, args.metric)
            print(f"  ⚠ La métrique {metric} n'est pas disponible dans all_config_results.")
            print(f"  → Métrique utilisée pour les hyperparamètres : {metric_for_hp}")
        else:
            metric_for_hp = metric
        hp_cols = infer_hyperparameter_columns(all_df)
        print(f"  ✓ {len(all_df)} configurations chargées")
        print(f"  ✓ Hyperparamètres détectés : {hp_cols if hp_cols else 'aucun'}")
        if hp_cols:
            plot_best_config_comparison_per_environment(all_df, args.output_dir, metric_for_hp, hp_cols, args.top_k, args.show)
            plot_top_configs_per_agent_env(all_df, args.output_dir, metric_for_hp, hp_cols, args.top_k, args.show)
            summary = plot_hyperparameter_effects(all_df, args.output_dir, metric_for_hp, hp_cols, args.show)
            if not summary.empty:
                print(f"  ✓ Résumé des effets sauvegardé : {args.output_dir / 'hyperparameter_effect_summary.csv'}")
            print(f"  ✓ Graphiques hyperparamètres sauvegardés dans : {args.output_dir / 'hyperparameters'}")
        else:
            print("  ⚠ Aucun hyperparamètre détecté. Vérifie les colonnes de all_config_results.csv.")
    else:
        print(f"\n[3/3] Fichier absent : {args.all_config_csv}")
        print("  ⚠ Les graphiques d'hyperparamètres ne peuvent pas être générés sans all_config_results.csv.")

    write_visual_report(args.output_dir, best_df, all_df, metric, hp_cols)

    print("\n" + "=" * 72)
    print("TERMINÉ")
    print("=" * 72)
    print(f"Graphiques disponibles dans : {args.output_dir}")
    print(f"Rapport visuel : {args.output_dir / 'visualization_report.md'}")
    print("\nFichiers clés à regarder en priorité :")
    print(f"  - {args.output_dir / f'01_best_agent_by_environment_{metric}.png'}")
    print(f"  - {args.output_dir / f'03_heatmap_environment_agent_{metric}.png'}")
    print(f"  - {args.output_dir / 'hyperparameters'}")


if __name__ == "__main__":
    main()
