# Laboratoire de Reinforcement Learning en NumPy

Ce projet compare 17 agents et prototypes sur LineWorld, GridWorld, TicTacToe et une variante de Quarto. Les réseaux et leurs gradients sont implémentés en NumPy. La version 0.2 corrige les calculs d’apprentissage, isole l’évaluation et rend chaque expérience traçable.

Les fichiers historiques de modèles, résultats et présentation sont conservés. Ils ne constituent pas des résultats validés avec le protocole actuel. Les exécutions corrigées sont dans `results/runs/`.

## Installation

Python 3.11 ou ultérieur. En PowerShell, depuis la racine du projet :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.build/pytest
```

Sur Linux/macOS, remplacer `.\.venv\Scripts\python.exe` par `.venv/bin/python`. Une installation Python doit être présente dans le PATH. `requirements-lock.txt` décrit les versions réellement utilisées pour la validation sous Windows/Python 3.12. Une installation éditable optionnelle `python -m pip install -e ".[dev]"` expose `drl-train`.

## Première exécution

```powershell
.\.venv\Scripts\python.exe train_all.py --profile smoke --quiet
.\.venv\Scripts\python.exe gui.py
```

Ouvrir http://127.0.0.1:5000. Les modes Human, Random, RandomRollout et MCTS fonctionnent sans modèle appris. Les autres agents nécessitent un checkpoint versionné compatible. Un modèle absent produit une erreur explicite. Le profil smoke ne fournit que des modèles de démonstration, sans preuve de convergence.

Chaque navigateur dispose de sa propre partie. L’interface est destinée à une démonstration locale, avec au plus 200 actions par partie. Le serveur sérialise les requêtes.

## Entraînement et analyse

```powershell
# Un agent et un environnement
.\.venv\Scripts\python.exe train_all.py --profile full --env LineWorld --agent DQN --quiet

# Étude sur cinq graines avec budget borné
.\.venv\Scripts\python.exe train_all.py --config configs/study.json --env LineWorld GridWorld --agent Random TabularQL DQN PPO A2C --quiet

# Recherche limitée, validation, puis réentraînement sur de nouvelles graines
.\.venv\Scripts\python.exe train_all.py --profile full --env TicTacToe --agent PPO --grid compact --max-configs-per-agent 2 --quiet

# Analyse : remplacer le chemin par celui indiqué à la fin de l’exécution
.\.venv\Scripts\python.exe benchmark_analyze_results.py --run results/runs/smoke_verified

# Débit de simulation, distinct de la latence de décision
.\.venv\Scripts\python.exe benchmark_random.py --games 100
```

Le profil par défaut est smoke : quatre épisodes, trois parties d’évaluation, une graine, recherches courtes, budget de 300 secondes. Le profil full prévoit 10 000 épisodes, 200 évaluations et cinq graines. Il peut être coûteux, surtout pour les recherches arborescentes. Les agents de planification pure ne reçoivent pas de faux épisodes d’entraînement. Aucun plafonnement silencieux à 500 épisodes n’est appliqué.

## Agents et portée scientifique

| Agents | Portée |
|---|---|
| Random | Baseline uniforme sur les actions légales |
| TabularQL | Q-Learning dense, y compris TicTacToe, indisponible sur Quarto |
| DQN, DoubleDQN, DDQN+ER, DDQN+PER | Cibles légales ; DoubleDQN seul est une ablation sans replay |
| REINFORCE, +Mean, +Critic | Retours épisodiques ; baseline moyenne historique indépendante de l’épisode courant |
| PPO | Objectif PPO-Clip, rapport des probabilités, GAE et masques conservés |
| A2C | Variante synchrone à un acteur avec retours n-step |
| RandomRollout, MCTS | Planification sur un simulateur connu, adversaire fixe échantillonné |
| ExpertApprentice | Imitation des décisions UCT, sans boucle complète d’amélioration de l’expert |
| AlphaZero | Prototype inspiré d’AlphaZero, politique/valeur et PUCT, sans self-play |
| MuZero | Prototype scalaire avec modèle latent, terminaison apprise et rétropropagation sur K pas |
| MuZero_Stochastic | Alias historique du prototype gaussien expérimental ; ne reproduit pas Stochastic MuZero publié |

Voir [l’audit](docs/AUDIT.md) et [le protocole](docs/PROTOCOLE.md) pour les limites exactes. Le nombre de combinaisons prises en charge est 67.

## Environnements

| Environnement | Observation | Actions | Convention |
|---|---:|---:|---|
| LineWorld | 10 | 2 | But en 9, départ en 0, coût -0,1 hors but, +1 au but |
| GridWorld | 25 | 4 | Départ (1,0), but (4,4), piège terminal (0,0) |
| TicTacToe | 9 | 9 | X contrôlé contre O fixe, un step inclut la réponse adverse |
| Quarto | 101 | 16 | Placement contrôlé, choix de pièce heuristique automatique |

La politique optimale de LineWorld atteint le but en neuf décisions pour un retour de 0,2. Le plus court chemin sûr de GridWorld compte sept décisions pour un retour de 10,54. Quarto est une variante réduite : les agents n’apprennent pas la phase de choix de la pièce donnée à l’adversaire.

## Organisation

```text
src/agents/          agents et recherche sur issues stochastiques
src/environments/    règles et générateurs aléatoires locaux
src/neural_network/  gradients, activations, Adam et sérialisation
src/training/        transitions, replay, évaluation et checkpoints
src/registry.py      configurations et construction des agents
src/experiments.py   CLI et protocole train/validation/test
configs/             profils smoke, full et study
tests/               règles, gradients, reprise, CLI et interface
tools/               mesures avant/après
docs/                audit, protocole et documentation corrigée
results/runs/        expériences versionnées indépendantes
```

## Contrôles de qualité

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests train_all.py gui.py benchmark_random.py benchmark_analyze_results.py plot_grid_search_results.py tools
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.build/pytest
.\.venv\Scripts\python.exe tools/profile_validation.py
```

Une matrice CI Windows/Linux et Python 3.11/3.12 est fournie. Elle doit encore être exécutée sur le service distant. Les validations locales et leurs limites figurent dans `docs/VALIDATION.md`.
