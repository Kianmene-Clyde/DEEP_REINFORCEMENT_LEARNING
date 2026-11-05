"""Build the corrected report from the actual validation artifacts.

Requires python-docx. Run from the repository root. Original reports are kept.
"""

import json
from pathlib import Path
import xml.etree.ElementTree as ET
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.oxml import OxmlElement

ROOT = Path(__file__).resolve().parents[1]


def main():
    def read(path):
        return json.loads((ROOT / path).read_text(encoding="utf-8"))

    study = read("results/runs/study_five_seeds/summary.json")
    confirmation = read("results/runs/tabular_confirmation/summary.json")
    perf = read("results/validation/before_after.json")
    suites = ET.parse(ROOT / "results/validation/pytest.xml").getroot()
    suite = suites if suites.tag == "testsuite" else suites.find("testsuite")
    passed = (
        int(suite.get("tests"))
        - int(suite.get("failures"))
        - int(suite.get("errors"))
        - int(suite.get("skipped"))
    )
    doc = Document()
    for child in list(doc._element.body):
        if not child.tag.endswith("}sectPr"):
            doc._element.body.remove(child)
    for section in doc.sections:
        section.page_width, section.page_height = Cm(21), Cm(29.7)
        section.top_margin, section.bottom_margin = Cm(1.8), Cm(1.8)
        section.left_margin, section.right_margin = Cm(1.8), Cm(1.8)
        section.header_distance, section.footer_distance = Cm(0.8), Cm(0.8)
        section.header.paragraphs[0].text = "Laboratoire DRL NumPy    Version 0.2    5 novembre 2026"
        section.header.paragraphs[0].style = doc.styles["Normal"]
        footer = section.footer.paragraphs[0]
        footer.text = "Rapport de validation    "
        field = OxmlElement("w:fldSimple")
        field.set("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}instr", "PAGE")
        footer._p.append(field)
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.08
    for style in ["Title", "Heading 1", "Heading 2"]:
        doc.styles[style].font.name = "Calibri"
        doc.styles[style].font.color.rgb = RGBColor(0, 0, 0)
        doc.styles[style].font.size = Pt(25 if style == "Title" else 18 if style == "Heading 1" else 12)

    def p(text, style=None):
        return doc.add_paragraph(text, style)

    def heading(text):
        doc.add_heading(text, level=1)

    def page(title):
        doc.add_page_break()
        heading(title)

    def table(headers, rows):
        t = doc.add_table(rows=1, cols=len(headers))
        t.style = "Light Shading Accent 1"
        for c, value in zip(t.rows[0].cells, headers):
            c.text = value
        for row in rows:
            for cell, value in zip(t.add_row().cells, row):
                cell.text = str(value)
        for row in t.rows:
            trpr = row._tr.get_or_add_trPr()
            trpr.append(OxmlElement("w:cantSplit"))
            for cell in row.cells:
                for para in cell.paragraphs:
                    para.paragraph_format.space_after = Pt(4)
                    for run in para.runs:
                        run.font.name = "Calibri"
                        run.font.size = Pt(9.5)
        return t

    p("Validation du laboratoire de reinforcement learning", "Title")
    p("Projet Python et NumPy    Rapport corrigé    5 novembre 2026")
    p(
        f"La version 0.2 corrige les calculs des agents, l’évaluation et la traçabilité des expériences. "
        f"La validation locale comprend {passed} tests réussis, un cas tabulaire Quarto exclu et "
        "67 combinaisons agent et environnement exécutées. Les résultats historiques sont conservés séparément."
    )
    p(
        "L’étude sur cinq graines atteint les chemins optimaux de LineWorld et GridWorld avec DQN, PPO et A2C "
        "après 1 000 épisodes, dans les conditions testées. Une confirmation sur cinq nouvelles graines rétablit "
        "également ces performances pour Q-Learning tabulaire avec son taux d’apprentissage par défaut."
    )
    table(
        ["Élément", "Preuve et portée"],
        [
            ["Tests automatisés", f"{passed} réussis, 1 cas exclu"],
            ["Vérification courte", "67 combinaisons, 4 épisodes et 3 parties d’évaluation"],
            ["Étude exploratoire", "2 environnements, 5 sélections, 5 graines"],
            ["Confirmation tabulaire", "2 environnements, 5 nouvelles graines"],
            [
                "Optimisations",
                f"Quarto {perf['quarto_200_positions']['speedup']:.1f} fois ; lissage {perf['smoothing_10000']['speedup']:.1f} fois",
            ],
            ["Campagne complète", "10 000 et 100 000 épisodes non exécutés ici"],
        ],
    )
    p(
        "Ces mesures ne prouvent ni une supériorité universelle des agents, ni la convergence des prototypes "
        "hybrides. Le rapport distingue exécution correcte, performance observée et travail restant."
    )
    p(
        "Les fichiers sources des mesures sont les manifests, configurations et résumés de results/runs, "
        "le rapport JUnit results/validation/pytest.xml et les microbenchmarks before_after.json."
    )
    page("Audit et corrections prioritaires")
    p(
        "Le code initial comporte des défauts qui affectent l’interprétation scientifique des classements. "
        "Les corrections ci-dessous portent sur le comportement effectif et disposent de régressions automatisées."
    )
    table(
        ["Composant", "Défaut initial", "Correction"],
        [
            ["Évaluation", "Victoire déduite du signe du retour", "Issue explicite de l’environnement"],
            ["Q-Learning et DQN", "Cibles sur actions interdites", "Masque du successeur stocké et appliqué"],
            [
                "Limite de pas",
                "Fausse terminaison absorbante",
                "Interruption distincte et bootstrap conservé",
            ],
            [
                "PPO",
                "Rapport omis dans le gradient actif",
                "Gradient du surrogate clippé et masques de collecte",
            ],
            [
                "UCT et Rollout",
                "Signes alternés après un step complet",
                "Perspective constante du joueur contrôlé",
            ],
            ["MuZero", "Gradients remplacés par une constante", "Chaîne différentiable sur K transitions"],
            ["Prototype stochastique", "Réseau de hasard non appris", "Modèle gaussien entraîné et renommé"],
            ["Flask", "Modèle absent remplacé silencieusement", "Erreur explicite et provenance affichée"],
        ],
    )
    p(
        "Un test témoin suit un détour de 50 mouvements à gauche puis neuf à droite dans LineWorld. "
        "L’agent atteint le but pour un retour de -4,8. L’ancien évaluateur annonce une défaite ; "
        "l’évaluateur corrigé annonce une réussite. Une récompense cumulée n’est donc pas une issue de partie."
    )
    p(
        "Le syllabus demande des résultats aux jalons 1 000, 10 000 et 100 000 épisodes. "
        "Cette livraison mesure le premier jalon sur deux environnements et ne remplace pas la campagne restante. "
        "Elle conserve le moteur NumPy demandé, sans certifier l’acceptation pédagogique des frameworks choisis."
    )
    page("Protocole expérimental")
    p(
        "Chaque agent et environnement possède un générateur aléatoire local. La graine d’entraînement et "
        "le numéro d’épisode déterminent les resets. Les simulations de recherche utilisent un flux distinct "
        "de celui de l’adversaire réel et ne peuvent pas consulter son futur état aléatoire."
    )
    p(
        "La sélection d’hyperparamètres emploie le retour moyen de validation. Chaque candidat reçoit les mêmes "
        "graines. Si plusieurs candidats sont comparés, la phase finale réentraîne le candidat retenu avec "
        "de nouvelles graines d’entraînement. Les parties de test utilisent encore un domaine distinct."
    )
    p(
        "L’évaluation utilise des copies du modèle et de l’environnement, en mode sans apprentissage. "
        "Elle conserve les poids, les buffers et le générateur de l’entraînement. La latence par coup "
        "chronomètre la sélection d’action seule, tandis que le benchmark de débit mesure une évaluation complète."
    )
    p(
        "Les victoires, défaites, nuls et réussites proviennent des issues terminales. Une limite de pas produit "
        "un timeout séparé. Les méthodes Q et les critiques conservent leur bootstrap lors d’une interruption. "
        "REINFORCE écarte les épisodes censurés ; ce filtrage peut favoriser les épisodes courts et impose "
        "de lire le taux de timeout avant une conclusion."
    )
    table(
        ["Exécution", "Budget", "Graines"],
        [
            ["smoke_verified", "4 épisodes, 3 évaluations, 30 pas maximum", "42"],
            ["study_five_seeds", "1 000 épisodes, 100 évaluations, 100 pas maximum", "11, 23, 37, 53, 71"],
            [
                "tabular_confirmation",
                "1 000 épisodes, 100 évaluations, 100 pas maximum",
                "101, 211, 307, 419, 523",
            ],
        ],
    )
    p(
        "Les résumés donnent un intervalle à 95 % de Student pour la moyenne entre graines. "
        "Les parties d’évaluation d’un même modèle ne sont pas traitées comme des entraînements indépendants. "
        "Avec cinq graines, l’incertitude reste approximative. Avec une seule graine, aucun intervalle "
        "entre entraînements n’est annoncé."
    )
    p(
        "Les checkpoints v2 contiennent poids, Adam, replay, générateurs et buffers. La reprise à une frontière "
        "d’épisode est testée sur DQN. Les anciens fichiers de poids seuls ne permettent pas une reprise exacte."
    )
    page("Résultats sur cinq graines")
    p(
        "L’étude exploratoire utilise un taux d’apprentissage commun de 0,001. Les valeurs ci-dessous "
        "sont les moyennes de test sur cinq graines, après 1 000 épisodes par entraînement. Random "
        "ne reçoit pas de faux budget d’apprentissage."
    )
    rows = []
    for result in study:
        interval = result["avg_reward"]["ci95_mean"]
        rows.append(
            [
                result["environment"],
                result["agent"],
                f"{result['avg_reward']['mean']:.3f}",
                f"{result['win_rate']['mean']:.1%}",
                f"[{interval[0]:.3f} ; {interval[1]:.3f}]",
            ]
        )
    table(["Environnement", "Agent", "Retour", "Réussite", "IC 95 % du retour"], rows)
    p(
        "DQN, PPO et A2C atteignent neuf décisions sur LineWorld et sept sur GridWorld. Leurs retours "
        "de 0,2 et 10,54 correspondent aux chemins de référence. Le Q-Learning tabulaire échoue dans "
        "la première configuration LineWorld : un taux commun n’est pas un réglage équitablement adapté."
    )
    p(
        "Après ce constat exploratoire, la confirmation conserve le taux tabulaire par défaut de 0,1 "
        "et emploie cinq nouvelles graines. Cette décision est explicitement postérieure à la première étude."
    )
    table(
        ["Environnement", "Retour confirmé", "Réussite", "Décisions"],
        [
            [
                r["environment"],
                f"{r['avg_reward']['mean']:.2f}",
                f"{r['win_rate']['mean']:.0%}",
                f"{r['avg_length']['mean']:.0f}",
            ]
            for r in confirmation
        ],
    )
    p(
        "Ces environnements déterministes simples vérifient le fonctionnement et la sensibilité aux paramètres. "
        "Ils ne permettent pas d’extrapoler les performances aux jeux complexes ou à d’autres budgets."
    )
    page("Tests et performances du logiciel")
    p(
        f"La suite finale contient {passed} tests réussis et un cas exclu, la tabulation dense de Quarto "
        "n’étant pas prise en charge. Chaque couple compatible subit entraînement, évaluation, sauvegarde "
        "et rechargement. Le rapport JUnit enregistre le résultat de chaque cas."
    )
    for item in [
        "Gradients des couches, du jacobien softmax et du tronc partagé comparés aux différences finies.",
        "Objectif PPO clippé testé avec avantages positifs et négatifs, régions actives et saturées.",
        "Gradient du déroulement MuZero complet comparé numériquement loin des points non différentiables de ReLU.",
        "Apprentissage d’un XOR et conservation exacte d’Adam après sauvegarde.",
        "Cibles Bellman, retours, GAE et probabilités du replay prioritaire vérifiés.",
        "Règles de Quarto comparées à une référence indépendante sur 300 positions partielles.",
        "RNG, environnement et modèle d’entraînement inchangés par l’évaluation.",
        "Reprise DQN identique au calcul continu et scénarios Flask testés.",
    ]:
        p(item, "List Bullet")
    table(
        ["Microbenchmark", "Avant ms", "Après ms", "Facteur"],
        [
            [
                label,
                f"{perf[key]['before_seconds'] * 1000:.2f}",
                f"{perf[key]['after_seconds'] * 1000:.2f}",
                f"{perf[key]['speedup']:.1f}",
            ]
            for key, label in [
                ("quarto_200_positions", "Quarto sur 200 positions"),
                ("smoothing_10000", "Lissage de 10 000 retours"),
            ]
        ],
    )
    p(
        "Les chiffres sont les médianes de sept exécutions sur les mêmes entrées, avec équivalence des résultats. "
        "Ils ne représentent pas un facteur d’accélération de tout l’entraînement. Le profil initial et le profil "
        "optimisé accompagnent les mesures."
    )
    p(
        "La matrice CI Windows/Linux, Python 3.11/3.12, est fournie mais n’a pas été exécutée sur un service distant. "
        "La validation locale utilise Windows et Python 3.12 avec les versions du fichier requirements-lock.txt."
    )
    page("Portée des prototypes et limites")
    table(
        ["Sélection", "Implémentation et limite"],
        [
            [
                "ExpertApprentice",
                "Imitation d’un expert UCT, sans boucle complète d’amélioration de l’expert",
            ],
            [
                "AlphaZero",
                "Prototype politique et valeur avec PUCT contre un adversaire fixe, sans self-play",
            ],
            [
                "MuZero",
                "Modèle latent scalaire, terminaison apprise, gradients sur K pas ; légalité interne non modélisée",
            ],
            [
                "MuZero_Stochastic",
                "Alias du prototype gaussien à un pas ; ni afterstate discret ni code de hasard appris",
            ],
        ],
    )
    p(
        "Le prototype gaussien apprend les successeurs latents par vraisemblance conditionnelle, avec récompense "
        "et terminaison. Il moyenne des issues échantillonnées pour la décision. Il ne reproduit pas "
        "Stochastic MuZero publié. Les noms affichés et la documentation rendent cette portée explicite."
    )
    p(
        "Dans TicTacToe et Quarto, un step inclut déjà la réponse de l’adversaire fixe. Les valeurs sont "
        "celles du joueur contrôlé et ne changent pas de signe entre steps. Les recherches sur simulateur "
        "connaissent les règles, contrairement aux modèles latents : leurs résultats ne comparent pas "
        "des agents disposant exactement de la même information."
    )
    p(
        "Quarto conserve un espace de 16 placements. La pièce à donner à l’adversaire est choisie par une "
        "heuristique automatique. Une extension au jeu complet doit exposer les deux phases de décision "
        "et réévaluer l’encodage, les masques et les modèles."
    )
    p(
        "Les prochaines expériences doivent atteindre les jalons restants, varier les adversaires, "
        "mesurer l’effet du budget de recherche et effectuer des ablations préspécifiées. "
        "Les anciens classements et les affirmations de supériorité générale ne sont pas reconduits."
    )
    page("Reproduction et démonstration")
    p(
        "Depuis la racine du projet avec Python installé, créer un environnement virtuel puis installer "
        "requirements-dev.txt. Le README contient les variantes PowerShell et Linux/macOS."
    )
    commands = [
        "python -m pip install -r requirements-dev.txt",
        "python -m pytest -q --basetemp=.build/pytest",
        "python train_all.py --profile smoke --quiet",
        "python train_all.py --config configs/study.json --env LineWorld --agent DQN --quiet",
        "python benchmark_analyze_results.py --run results/runs/study_five_seeds",
        "python tools/profile_validation.py",
        "python gui.py",
    ]
    for command in commands:
        para = p(command)
        for run in para.runs:
            run.font.name = "Consolas"
            run.font.size = Pt(9)
    p(
        "L’interface locale est disponible à http://127.0.0.1:5000. Montrer d’abord le chemin humain de LineWorld, "
        "puis le modèle DQN entraîné, en lisant sa graine et son nombre d’épisodes. Sur Quarto, préciser "
        "que la sélection de pièce est automatique. Une absence de modèle donne une erreur explicite."
    )
    p(
        "L’évaluation indépendante d’un checkpoint s’effectue avec evaluate_agent.py. Les arguments "
        "--checkpoint, --episodes, --max-steps, --seed et --output sont documentés dans son aide."
    )
    doc.add_heading("Références primaires", level=2)
    for reference in [
        "Schulman et al. Proximal Policy Optimization Algorithms. https://arxiv.org/abs/1707.06347",
        "Schaul et al. Prioritized Experience Replay. https://arxiv.org/abs/1511.05952",
        "Schrittwieser et al. Mastering Atari Go Chess and Shogi by Planning with a Learned Model. https://arxiv.org/abs/1911.08265",
        "Schrittwieser. Planning in Stochastic Environments with a Learned Model. https://www.julian.ac/blog/2022/05/15/planning-in-stochastic-environments-with-a-learned-model/",
    ]:
        p(reference)
    out = ROOT / "deliverables" / "Rapport_DRL_verifie.docx"
    out.parent.mkdir(exist_ok=True)
    for tree in (doc.element, doc.styles.element):
        for border in list(tree.xpath(".//w:pBdr")):
            border.getparent().remove(border)
    doc.save(out)
    print(out)


if __name__ == "__main__":
    main()
