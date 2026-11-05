"""Analyze a versioned run by environment, without a composite ranking."""

import argparse
import csv
import json
from pathlib import Path
from src.experiments import summarize, plot_summaries, plot_learning_histories, write_json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run", required=True, type=Path, help="Directory containing manifest.json and test_results.csv"
    )
    args = p.parse_args(argv)
    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != 2:
        raise ValueError("Historical results are not validated by this protocol")
    with (args.run / "test_results.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        if row["status"] == "OK":
            for key, value in row.items():
                if key not in {"status", "environment", "agent", "error"} and value:
                    row[key] = float(value)
    summary = summarize(rows)
    out = args.run / "analysis"
    out.mkdir(exist_ok=True)
    write_json(out / "summary.json", summary)
    plot_summaries(summary, out)
    plot_learning_histories(manifest, args.run, out / "learning_curves")
    lines = [
        "# Analyse du test indépendant",
        "",
        "Chaque ligne résume des graines d’entraînement indépendantes. Les intervalles à 95 % utilisent "
        "la loi de Student entre graines. Une seule graine ne permet pas de calculer cet intervalle.",
        "",
        "| Environnement | Agent | Graines | Retour moyen | Réussite | Latence ms |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for result in summary:
        lines.append(
            f"| {result['environment']} | {result['agent']} | {result['n_training_seeds']} | "
            f"{result['avg_reward']['mean']:.4f} | {result['win_rate']['mean']:.1%} | "
            f"{result['avg_step_time_ms']['mean']:.4f} |"
        )
    lines += [
        "",
        "Les résultats du profil smoke vérifient l’exécution et ne démontrent pas la convergence.",
        "Les agents de planification utilisent un simulateur ; leur budget de recherche doit être "
        "considéré dans la comparaison. Aucun score composite ne combine ces critères.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out / "report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
