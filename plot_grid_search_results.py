from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

SUMMARY_CANDIDATES = [
    Path("best_agents_summary_from_console.csv"),
]

RANKINGS_CANDIDATES = [
    Path("console_rankings_complete_envs.csv"),
]

QUARTO_CANDIDATES = [
    Path("console_quarto_partial_ranking.csv"),
]

DEFAULT_OUTPUT_DIR = Path("results/plots_console_evidence")
ENV_ORDER = ["LineWorld", "GridWorld", "TicTacToe", "Quarto"]
ENV_COLORS = {
    "LineWorld": "#1971c2",
    "GridWorld": "#2b8a3e",
    "TicTacToe": "#e67700",
    "Quarto": "#c92a2a",
}
BEST_COLOR = "#087f5b"
OTHER_COLOR = "#ced4da"
WARNING_COLOR = "#d9480f"
TEXT_COLOR = "#212529"

COLUMN_ALIASES = {
    "env": "environment",
    "env_name": "environment",
    "agent_name": "agent",
    "selection_score": "score",
    "composite_score": "score",
    "reward": "avg_reward",
    "mean_reward": "avg_reward",
    "average_reward": "avg_reward",
    "win": "win_rate",
    "time": "time_ms",
    "avg_step_time_ms": "time_ms",
    "decision_time_ms": "time_ms",
    "hyperparams": "params",
    "hyperparameters": "params",
    "config_id": "config",
}

METRIC_LABELS = {
    "score": "Score composite",
    "avg_reward": "Reward moyen",
    "win_rate": "Taux de victoire",
    "time_ms": "Temps moyen de décision (ms)",
}


def find_existing(candidates: Sequence[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for col in df.columns:
        clean = str(col).strip().lower().replace("-", "_")
        clean = re.sub(r"\s+", "_", clean)
        renamed[col] = COLUMN_ALIASES.get(clean, clean)
    return df.rename(columns=renamed)


def parse_params(value) -> Dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, float) and math.isnan(value):
        return {}
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return {}
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return {}
    return {}


def read_csv_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_columns(df)
    required = {"environment", "agent"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Le fichier {path} ne contient pas les colonnes obligatoires : {missing}")
    for col in ["rank", "score", "avg_reward", "win_rate", "time_ms", "config"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "params" not in df.columns:
        df["params"] = "{}"
    df["params_dict"] = df["params"].apply(parse_params)
    if "status" not in df.columns:
        df["status"] = "complete"
    return df


def env_sort_key(env: str) -> int:
    try:
        return ENV_ORDER.index(str(env))
    except ValueError:
        return 999


def short_number(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    try:
        value = float(value)
    except Exception:
        return str(value)
    if abs(value) < 0.001 and value != 0:
        return f"{value:.1e}"
    text = f"{value:.{digits}f}"
    return text.rstrip("0").rstrip(".")


def format_percent(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    value = float(value)
    if value <= 1.0:
        value *= 100
    return f"{value:.1f}%"


def pretty_param_name(name: str) -> str:
    mapping = {
        "learning_rate": "lr",
        "discount_factor": "gamma",
        "epsilon_decay": "eps_decay",
        "hidden_layers": "hidden",
        "clip_ratio": "clip",
        "entropy_coef": "entropy",
        "search_budget": "budget",
        "max_rollout_depth": "depth",
        "exploration_constant": "c",
    }
    return mapping.get(str(name), str(name))


def format_value(value) -> str:
    if isinstance(value, float):
        return short_number(value, 4)
    if isinstance(value, list):
        return "[" + ",".join(str(v) for v in value) + "]"
    return str(value)


def format_params(params: Dict, max_items: int = 5, max_len: int = 100) -> str:
    if not params:
        return "aucun"
    parts = []
    for i, (k, v) in enumerate(params.items()):
        if i >= max_items:
            parts.append("…")
            break
        parts.append(f"{pretty_param_name(k)}={format_value(v)}")
    text = " | ".join(parts)
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text


def sanitize_filename(text: str) -> str:
    text = str(text).replace("+", "plus")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")


def sort_by_metric(df: pd.DataFrame, metric: str = "score") -> pd.DataFrame:
    df = df.copy()
    ascending = metric == "time_ms"
    return df.sort_values(metric, ascending=ascending).reset_index(drop=True)


def is_partial(row: pd.Series) -> bool:
    status = str(row.get("status", "")).lower()
    return str(row.get("environment", "")) == "Quarto" or "partial" in status


def savefig(path: Path, show: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def annotate_barh(ax, values: Sequence[float]) -> None:
    finite = [float(v) for v in values if pd.notna(v) and np.isfinite(v)]
    if not finite:
        return
    span = max(max(finite) - min(finite), max(abs(v) for v in finite), 1.0)
    for patch, value in zip(ax.patches, values):
        if pd.isna(value):
            continue
        ax.text(
            patch.get_width() + 0.012 * span,
            patch.get_y() + patch.get_height() / 2,
            short_number(value, 4),
            va="center",
            fontsize=10,
            fontweight="bold",
        )


def load_data(summary_csv: Optional[Path], rankings_csv: Optional[Path], quarto_csv: Optional[Path]) -> Tuple[
    pd.DataFrame, pd.DataFrame, Dict[str, Path]]:
    summary_path = summary_csv or find_existing(SUMMARY_CANDIDATES)
    rankings_path = rankings_csv or find_existing(RANKINGS_CANDIDATES)
    quarto_path = quarto_csv or find_existing(QUARTO_CANDIDATES)
    if rankings_path is None:
        raise FileNotFoundError("Impossible de trouver console_rankings_complete_envs.csv")
    rankings = read_csv_clean(rankings_path)
    used = {"rankings": rankings_path}
    if quarto_path is not None and quarto_path.exists():
        quarto = read_csv_clean(quarto_path)
        rankings = rankings[rankings["environment"].astype(str) != "Quarto"]
        rankings = pd.concat([rankings, quarto], ignore_index=True, sort=False)
        used["quarto"] = quarto_path
    if summary_path is not None and summary_path.exists():
        summary = read_csv_clean(summary_path)
        used["summary"] = summary_path
    else:
        summary = pd.DataFrame([sort_by_metric(g).iloc[0] for _, g in rankings.groupby("environment")])
    quarto_rows = rankings[rankings["environment"].astype(str) == "Quarto"]
    if not quarto_rows.empty:
        summary = summary[summary["environment"].astype(str) != "Quarto"]
        summary = pd.concat([summary, pd.DataFrame([sort_by_metric(quarto_rows).iloc[0]])], ignore_index=True,
                            sort=False)
    summary = summary.sort_values("environment", key=lambda s: s.map(env_sort_key)).reset_index(drop=True)
    rankings = rankings.sort_values(["environment", "rank"],
                                    key=lambda s: s.map(env_sort_key) if s.name == "environment" else s).reset_index(
        drop=True)
    return summary, rankings, used


def plot_00_best_cards(summary: pd.DataFrame, out: Path, show: bool) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17, 10))
    axes = axes.flatten()
    fig.suptitle(
        "PREUVE DIRECTE — MEILLEUR AGENT PAR ENVIRONNEMENT APRÈS TUNING\n"
        "agent gagnant + score + reward + win-rate + hyperparamètres gagnants",
        fontsize=19,
        fontweight="bold",
    )
    for ax, env in zip(axes, ENV_ORDER):
        ax.axis("off")
        color = ENV_COLORS.get(env, "#495057")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#f8f9fa", edgecolor="#dee2e6"))
        ax.add_patch(plt.Rectangle((0, 0.84), 1, 0.16, transform=ax.transAxes, facecolor=color, edgecolor=color))
        ax.text(0.035, 0.92, env, transform=ax.transAxes, fontsize=20, fontweight="bold", color="white", va="center")
        rowdf = summary[summary["environment"].astype(str) == env]
        if rowdf.empty:
            ax.text(0.5, 0.5, "Aucun résultat", ha="center", va="center", fontsize=16)
            continue
        row = rowdf.iloc[0]
        ax.text(0.05, 0.72, "AGENT GAGNANT", transform=ax.transAxes, fontsize=10, fontweight="bold", color="#495057")
        ax.text(0.05, 0.62, str(row["agent"]), transform=ax.transAxes, fontsize=26, fontweight="bold", color=color)
        metrics = (
            f"Score composite : {short_number(row.get('score'), 4)}\n"
            f"Reward moyen    : {short_number(row.get('avg_reward'), 4)}\n"
            f"Win-rate        : {format_percent(row.get('win_rate'))}\n"
            f"Temps décision  : {short_number(row.get('time_ms'), 3)} ms"
        )
        ax.text(0.05, 0.40, metrics, transform=ax.transAxes, fontsize=13, family="monospace", va="top",
                bbox=dict(boxstyle="round,pad=0.55", facecolor="white", edgecolor=color, linewidth=1.7))
        ax.text(0.05, 0.12, 'Params : ' + format_params(row.get("params_dict", {}), max_len=110),
                transform=ax.transAxes, fontsize=11, wrap=True, va="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff9db", edgecolor="#fab005", linewidth=1.4))
        if is_partial(row):
            ax.text(0.95, 0.72, "RÉSULTAT PARTIEL", transform=ax.transAxes, ha="right", fontsize=11, fontweight="bold",
                    color=WARNING_COLOR,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff4e6", edgecolor=WARNING_COLOR))
    fig.text(0.01, 0.01,
             "Lecture : chaque carte présente le meilleur agent observé dans l'environnement correspondant. Quarto est marqué partiel car le classement provient d'une reconstruction depuis la console avant la fin du train_all.py.",
             fontsize=8, color="#495057")
    savefig(out / "00_PREUVE_meilleur_agent_par_environnement.png", show)


def plot_01_winners_score(summary: pd.DataFrame, out: Path, show: bool) -> None:
    df = summary.copy().sort_values("environment", key=lambda s: s.map(env_sort_key))
    labels = [f"{e}\n{a}" for e, a in zip(df["environment"], df["agent"])]
    values = pd.to_numeric(df["score"], errors="coerce")
    colors = [ENV_COLORS.get(str(e), BEST_COLOR) for e in df["environment"]]
    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.bar(labels, values, color=colors, edgecolor="#212529", linewidth=1.0)
    ax.set_title("Comparaison directe des agents gagnants", fontsize=17, fontweight="bold")
    ax.set_ylabel("Score composite")
    ax.grid(axis="y", alpha=0.25)
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{short_number(row.get('score'), 4)}\nwin {format_percent(row.get('win_rate'))}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.text(0.01, 0.01,
             "Attention : les rewards ne sont pas comparables d'un environnement à l'autre. Ce graphique sert uniquement de synthèse des gagnants, pas de comparaison absolue entre tâches différentes.",
             fontsize=8, color="#495057")
    savefig(out / "01_comparaison_agents_gagnants_score.png", show)


def plot_02_summary_table(summary: pd.DataFrame, out: Path, show: bool) -> None:
    rows = []
    for _, row in summary.iterrows():
        rows.append([
            row.get("environment", ""), row.get("agent", ""), short_number(row.get("score"), 4),
            short_number(row.get("avg_reward"), 4), format_percent(row.get("win_rate")),
            f"{short_number(row.get('time_ms'), 3)} ms", format_params(row.get("params_dict", {}), max_len=60)
        ])
    cols = ["Environnement", "Meilleur agent", "Score", "Reward", "Win-rate", "Temps", "Hyperparamètres gagnants"]
    fig, ax = plt.subplots(figsize=(22, max(4.5, len(rows) * 1.1 + 2)))
    ax.axis("off")
    ax.set_title("Résumé exécutif : meilleurs agents et meilleurs hyperparamètres", fontsize=18, fontweight="bold",
                 pad=20)
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False);
    table.set_fontsize(8.7);
    table.scale(1, 1.8)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold", color="white");
            cell.set_facecolor("#343a40")
        else:
            env = rows[r - 1][0]
            if c == 0:
                cell.set_text_props(weight="bold", color="white");
                cell.set_facecolor(ENV_COLORS.get(env, "#495057"))
            elif c == 1:
                cell.set_text_props(weight="bold");
                cell.set_facecolor("#e7f5ff")
            elif c == 6:
                cell.set_facecolor("#fff9db")
            else:
                cell.set_facecolor("#ffffff" if r % 2 else "#f8f9fa")
    savefig(out / "02_tableau_meilleurs_agents_hyperparametres.png", show)


def plot_03_rankings(rankings: pd.DataFrame, out: Path, metric: str, top_k: int, show: bool) -> None:
    for env in sorted(rankings["environment"].dropna().astype(str).unique(), key=env_sort_key):
        group = rankings[rankings["environment"].astype(str) == env].copy()
        group = sort_by_metric(group, metric).head(top_k).iloc[::-1]
        if group.empty: continue
        values = pd.to_numeric(group[metric], errors='coerce')
        labels = [];
        colors = []
        for _, row in group.iterrows():
            rank = int(row["rank"]) if pd.notna(row.get("rank", np.nan)) else "?"
            labels.append(f"#{rank}  {row['agent']}\n{format_params(row.get('params_dict', {}), max_len=60)}")
            colors.append(BEST_COLOR if rank == 1 else OTHER_COLOR)
        fig, ax = plt.subplots(figsize=(15, max(7, len(group) * 0.62 + 3)))
        ax.barh(labels, values, color=colors, edgecolor="#343a40", linewidth=0.8)
        ax.set_title(f"{env} — classement des agents après tuning", fontsize=16, fontweight="bold", pad=28)
        ax.set_xlabel(METRIC_LABELS.get(metric, metric));
        ax.grid(axis='x', alpha=0.25)
        annotate_barh(ax, list(values))
        best = sort_by_metric(group, metric).iloc[0]
        best_text = (
            f"GAGNANT : {best['agent']} | Score={short_number(best.get('score'), 4)} | Reward={short_number(best.get('avg_reward'), 4)} | Win={format_percent(best.get('win_rate'))}\nParams : {format_params(best.get('params_dict', {}), max_len=120)}")
        fig.text(0.015, 0.965, best_text, fontsize=10.5, fontweight='bold', va='top',
                 bbox=dict(boxstyle='round,pad=0.45', facecolor='#e6fcf5', edgecolor=BEST_COLOR, linewidth=1.5))
        if env == 'Quarto' or group['status'].astype(str).str.contains('partial', case=False, na=False).any():
            fig.text(0.985, 0.965, "⚠ Quarto : ranking partiel\nissu de la console", ha='right', va='top', fontsize=9.5,
                     color=WARNING_COLOR, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.35', facecolor='#fff4e6', edgecolor=WARNING_COLOR))
        fig.subplots_adjust(top=0.84)
        savefig(out / f"03_ranking_detaille_{sanitize_filename(env)}_{metric}.png", show)


def plot_04_multimetrics(rankings: pd.DataFrame, out: Path, top_k: int, show: bool) -> None:
    for env in sorted(rankings['environment'].dropna().astype(str).unique(), key=env_sort_key):
        group = rankings[rankings['environment'].astype(str) == env].copy();
        group = sort_by_metric(group, 'score').head(top_k).reset_index(drop=True)
        if group.empty: continue
        agents = group['agent'].astype(str).tolist();
        x = np.arange(len(group))
        fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
        fig.suptitle(f"{env} — lecture multi-métriques des meilleurs agents", fontsize=18, fontweight='bold')
        specs = [('avg_reward', 'Reward moyen', '#1971c2'), ('win_rate', 'Win-rate (%)', '#2b8a3e'),
                 ('time_ms', 'Temps décision (ms)', '#e67700')]
        for ax, (metric, label, color) in zip(axes, specs):
            values = pd.to_numeric(group[metric], errors='coerce') if metric in group else pd.Series(
                [np.nan] * len(group))
            if metric == 'win_rate': values = values.apply(lambda v: v * 100 if pd.notna(v) and v <= 1 else v)
            bars = ax.bar(x, values, color=color, alpha=0.88, edgecolor='#343a40', linewidth=0.6)
            ax.set_ylabel(label);
            ax.grid(axis='y', alpha=0.25)
            for bar, value in zip(bars, values):
                if pd.notna(value):
                    suff = '%' if metric == 'win_rate' else ''
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{short_number(value, 3)}{suff}",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
        axes[-1].set_xticks(x);
        axes[-1].set_xticklabels(agents, rotation=35, ha='right')
        fig.text(0.01, 0.01,
                 "Lecture : un agent peut être performant mais lent. Le temps de décision est affiché séparément pour montrer le compromis performance / coût de calcul.",
                 fontsize=8, color='#495057')
        savefig(out / f"04_multimetriques_{sanitize_filename(env)}.png", show)


def plot_05_heatmap(rankings: pd.DataFrame, out: Path, show: bool) -> None:
    pivot = rankings.pivot_table(index='environment', columns='agent', values='score', aggfunc='max')
    if pivot.empty: return
    ordered = [e for e in ENV_ORDER if e in pivot.index] + [e for e in pivot.index if e not in ENV_ORDER]
    pivot = pivot.reindex(ordered)
    data = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 0.85), max(5, len(pivot.index) * 0.85 + 2)))
    im = ax.imshow(data, aspect='auto', cmap='YlGnBu')
    ax.set_title("Heatmap — score composite par environnement et agent", fontsize=16, fontweight='bold')
    ax.set_xlabel('Agent');
    ax.set_ylabel('Environnement')
    ax.set_xticks(np.arange(len(pivot.columns)));
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(pivot.index)));
    ax.set_yticklabels(pivot.index)
    for i in range(data.shape[0]):
        row = data[i]
        if np.all(np.isnan(row)): continue
        best_j = int(np.nanargmax(row))
        for j in range(data.shape[1]):
            value = data[i, j]
            if np.isfinite(value):
                ax.text(j, i, short_number(value, 2), ha='center', va='center', fontsize=8,
                        fontweight='bold' if j == best_j else 'normal')
                if j == best_j:
                    ax.add_patch(
                        plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='#e03131', linewidth=2.6))
    cbar = plt.colorbar(im, ax=ax);
    cbar.set_label('Score composite')
    fig.text(0.01, 0.01,
             "Les cases encadrées en rouge correspondent au meilleur score observé dans chaque environnement.",
             fontsize=8, color='#495057')
    savefig(out / '05_heatmap_score_environnement_agent.png', show)


def plot_06_hyperparameter_tables(rankings: pd.DataFrame, out: Path, top_k: int, show: bool) -> None:
    for env in sorted(rankings['environment'].dropna().astype(str).unique(), key=env_sort_key):
        group = rankings[rankings['environment'].astype(str) == env].copy();
        group = sort_by_metric(group, 'score').head(top_k)
        if group.empty: continue
        rows = []
        for _, row in group.iterrows():
            rows.append([
                f"#{int(row['rank'])}" if pd.notna(row.get('rank', np.nan)) else '',
                row.get('agent', ''), short_number(row.get('score'), 4), short_number(row.get('avg_reward'), 4),
                format_percent(row.get('win_rate')), f"{short_number(row.get('time_ms'), 3)} ms",
                format_params(row.get('params_dict', {}), max_len=105)
            ])
        title = f"{env} — meilleurs agents et meilleurs hyperparamètres" + (
            " (résultats partiels)" if env == 'Quarto' else '')
        columns = ["Rang", "Agent", "Score", "Reward", "Win", "Temps", "Meilleurs hyperparamètres"]
        fig, ax = plt.subplots(figsize=(19, max(5, len(rows) * 0.62 + 2)));
        ax.axis('off');
        ax.set_title(title, fontsize=17, fontweight='bold', pad=18)
        table = ax.table(cellText=rows, colLabels=columns, loc='center', cellLoc='left', colLoc='left');
        table.auto_set_font_size(False);
        table.set_fontsize(9.4);
        table.scale(1, 1.65)
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor('#343a40');
                cell.set_text_props(weight='bold', color='white')
            else:
                if r == 1:
                    cell.set_facecolor('#d3f9d8' if c != 6 else '#fff3bf');
                    cell.set_text_props(weight='bold')
                elif c == 6:
                    cell.set_facecolor('#fff9db')
                else:
                    cell.set_facecolor('#ffffff' if r % 2 else '#f8f9fa')
        savefig(out / f"06_tableau_hyperparametres_{sanitize_filename(env)}.png", show)


def plot_07_score_time_tradeoff(rankings: pd.DataFrame, out: Path, top_k: int, show: bool) -> None:
    for env in sorted(rankings['environment'].dropna().astype(str).unique(), key=env_sort_key):
        group = rankings[rankings['environment'].astype(str) == env].copy();
        group = sort_by_metric(group, 'score').head(top_k)
        group = group[pd.notna(group['time_ms']) & pd.notna(group['score'])]
        if group.empty: continue
        fig, ax = plt.subplots(figsize=(12, 7))
        sizes = pd.to_numeric(group.get('win_rate', pd.Series([0.5] * len(group))), errors='coerce').fillna(0.5)
        sizes = sizes.apply(lambda v: v if v <= 1 else v / 100)
        sizes = 250 + sizes * 900
        colors = [BEST_COLOR if int(row['rank']) == 1 else ENV_COLORS.get(env, OTHER_COLOR) for _, row in
                  group.iterrows()]
        ax.scatter(group['time_ms'], group['score'], s=sizes, c=colors, alpha=0.75, edgecolors='#212529', linewidth=0.8)
        for _, row in group.iterrows():
            ax.text(row['time_ms'], row['score'], f" {row['agent']}", fontsize=9, va='center')
        ax.set_title(f"{env} — compromis performance / temps de décision", fontsize=16, fontweight='bold')
        ax.set_xlabel('Temps moyen de décision (ms)');
        ax.set_ylabel('Score composite');
        ax.grid(alpha=0.25)
        if group['time_ms'].max() / max(group['time_ms'].min(), 1e-9) > 100:
            ax.set_xscale('log');
            ax.set_xlabel('Temps moyen de décision (ms, échelle log)')
        fig.text(0.01, 0.01,
                 "Taille des points ≈ win-rate. Ce graphique met en évidence les agents performants mais coûteux en temps de décision, en particulier les approches de planning sur Quarto.",
                 fontsize=8, color='#495057')
        savefig(out / f"07_compromis_score_temps_{sanitize_filename(env)}.png", show)


def plot_08_margin_to_second(rankings: pd.DataFrame, out: Path, show: bool) -> None:
    rows = []
    for env, group in rankings.groupby('environment'):
        g = sort_by_metric(group, 'score')
        if len(g) >= 2:
            rows.append({'environment': env, 'winner': g.iloc[0]['agent'],
                         'margin_score': g.iloc[0]['score'] - g.iloc[1]['score'], 'second': g.iloc[1]['agent']})
    if not rows: return
    df = pd.DataFrame(rows).sort_values('environment', key=lambda s: s.map(env_sort_key))
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(df['environment'], df['margin_score'],
                  color=[ENV_COLORS.get(e, BEST_COLOR) for e in df['environment']], edgecolor='#212529')
    ax.set_title('Marge du meilleur agent sur le 2e agent (score composite)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Écart de score');
    ax.grid(axis='y', alpha=0.25)
    for bar, (_, row) in zip(bars, df.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{short_number(row['margin_score'], 4)}\nvs {row['second']}", ha='center', va='bottom', fontsize=9,
                fontweight='bold')
    fig.text(0.01, 0.01,
             "Une marge nulle signifie que plusieurs agents atteignent le même score final ; le classement est alors départagé surtout par le temps de décision et les pénalités secondaires de la fonction de sélection.",
             fontsize=8, color='#495057')
    savefig(out / '08_marge_du_gagnant_sur_le_2e.png', show)


def write_readme(summary: pd.DataFrame, rankings: pd.DataFrame, used: Dict[str, Path], out: Path) -> None:
    lines = ["# Graphiques DRL — meilleurs agents et hyperparamètres", "", "## Fichiers utilisés"]
    for k, p in used.items(): lines.append(f"- `{k}` : `{p}`")
    lines += ["", "## Meilleurs agents par environnement"]
    for env in sorted(summary['environment'].dropna().astype(str).unique(), key=env_sort_key):
        row = summary[summary['environment'].astype(str) == env].iloc[0]
        suffix = ' — **résultat partiel**' if is_partial(row) else ''
        lines.append(
            f"- **{env}** : **{row['agent']}** | score={short_number(row['score'], 4)} | reward={short_number(row['avg_reward'], 4)} | win={format_percent(row['win_rate'])} | temps={short_number(row['time_ms'], 3)} ms | params=`{format_params(row['params_dict'], max_len=160)}`{suffix}")
    lines += ["", "## Graphiques clés", "- `00_PREUVE_meilleur_agent_par_environnement.png`",
              "- `03_ranking_detaille_<ENV>_score.png`", "- `06_tableau_hyperparametres_<ENV>.png`",
              "- `07_compromis_score_temps_<ENV>.png`", "- `08_marge_du_gagnant_sur_le_2e.png`", "", "## Limite",
              "Le classement Quarto est marqué partiel car il provient de la reconstruction console d'un entraînement non terminé jusqu'au résumé global final."]
    (out / 'README_GRAPHIQUES.md').write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Génère des graphiques ultra lisibles à partir des CSV console.')
    parser.add_argument('--summary-csv', type=Path, default=None)
    parser.add_argument('--rankings-csv', type=Path, default=None)
    parser.add_argument('--quarto-csv', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--metric', type=str, default='score', choices=['score', 'avg_reward', 'win_rate', 'time_ms'])
    parser.add_argument('--top-k', type=int, default=13)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print('=' * 80);
    print('GRAPHIQUES DRL — MEILLEURS AGENTS + HYPERPARAMÈTRES');
    print('=' * 80)
    summary, rankings, used = load_data(args.summary_csv, args.rankings_csv, args.quarto_csv)
    print(f"✓ Résumé meilleurs agents : {len(summary)} lignes")
    print(f"✓ Classements fusionnés   : {len(rankings)} lignes")
    print(
        f"✓ Environnements détectés : {', '.join(sorted(rankings['environment'].astype(str).unique(), key=env_sort_key))}")
    print(f"✓ Dossier de sortie       : {args.output_dir}")
    plot_00_best_cards(summary, args.output_dir, args.show)
    plot_01_winners_score(summary, args.output_dir, args.show)
    plot_02_summary_table(summary, args.output_dir, args.show)
    plot_03_rankings(rankings, args.output_dir, args.metric, args.top_k, args.show)
    plot_04_multimetrics(rankings, args.output_dir, args.top_k, args.show)
    plot_05_heatmap(rankings, args.output_dir, args.show)
    plot_06_hyperparameter_tables(rankings, args.output_dir, args.top_k, args.show)
    plot_07_score_time_tradeoff(rankings, args.output_dir, args.top_k, args.show)
    plot_08_margin_to_second(rankings, args.output_dir, args.show)
    write_readme(summary, rankings, used, args.output_dir)
    print('\n' + '=' * 80);
    print('TERMINÉ');
    print('=' * 80);
    print(f'Graphiques disponibles dans : {args.output_dir}')
    print('Fichiers prioritaires :')
    for fn in ['00_PREUVE_meilleur_agent_par_environnement.png', '02_tableau_meilleurs_agents_hyperparametres.png',
               '05_heatmap_score_environnement_agent.png', '08_marge_du_gagnant_sur_le_2e.png', 'README_GRAPHIQUES.md']:
        print('  -', args.output_dir / fn)


if __name__ == '__main__':
    main()
