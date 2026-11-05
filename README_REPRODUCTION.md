# Reproduction des expériences corrigées

Les instructions de ce fichier concernent le protocole version 2. Les anciens CSV, JSON, graphiques et modèles à la racine sont historiques et ne sont pas mélangés aux nouvelles expériences.

## Installation et vérification

Suivre l’installation du README, puis :

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.build/pytest
.\.venv\Scripts\python.exe train_all.py --quick --quiet
```

Les dossiers de sortie portent un identifiant UTC et un suffixe aléatoire. `--output` permet de choisir un dossier vide. Un dossier déjà rempli est refusé.

## Profils et budgets

- `configs/smoke.json` : quatre épisodes, trois parties d’évaluation, une graine, limite de 30 décisions, budget de 300 secondes.
- `configs/study.json` : 1 000 épisodes, 100 parties d’évaluation, cinq graines, limite de 100 décisions, budget de 600 secondes. Les learning rates par défaut propres aux agents sont conservés. L’étude exploratoire archivée dans study_five_seeds utilisait un taux commun de 0,001, défavorable à Q-Learning tabulaire sur LineWorld ; sa configuration est conservée dans configs/exploratory_shared_lr.json.
- `configs/full.json` : 10 000 épisodes, 200 parties d’évaluation et cinq graines. Pas de limite globale de durée par défaut : choisir un sous-ensemble ou ajouter `--time-budget-seconds`.

Les checkpoints d’évaluation sont prévus à 1 000, 10 000 et 100 000 épisodes lorsqu’ils sont atteints. Exemple :

```powershell
.\.venv\Scripts\python.exe train_all.py --profile full --env LineWorld --agent DQN --episodes 100000 --quiet
```

Cette campagne n’a pas été exécutée dans la validation locale. Aucun résultat n’est inventé pour les checkpoints non atteints.

## Séparation des données expérimentales

Chaque candidat est entraîné sur les mêmes graines afin de comparer les hyperparamètres. La sélection utilise uniquement le retour moyen de validation. Si plusieurs configurations sont comparées, le candidat retenu est réentraîné avec les graines augmentées de 10 000. Les parties du test final utilisent un domaine de graines distinct.

L’évaluation travaille sur des copies du modèle et de l’environnement. Elle ne modifie ni poids, ni replay, ni mode ou générateur aléatoire de l’entraînement. La latence mesure seulement la décision de l’agent.

## Fichiers produits

- `config.json` : budget et paramètres déclarés.
- `manifest.json` : versions, empreintes SHA-256, candidats retenus et modèles.
- `validation_results.csv` : scores utilisés pour la sélection.
- `test_results.csv` : mesures indépendantes finales.
- `summary.json` : moyennes et incertitude entre graines.
- `*/seed_*/agent.pkl` : checkpoint complet du protocole v2.
- `training_metrics.json` : trajectoires et mesures d’apprentissage.
- `test_episodes.json` : issues, retours et graines des parties de test.
- `*_test_comparison.png` : figures générées à partir des résultats de test.

Les checkpoints pickle sont des fichiers locaux de confiance. Ils contiennent notamment les poids, Adam, replay, RNG et buffers. Les anciens fichiers `*_qnet.pkl`, `*_actor.pkl`, etc. ne suffisent pas à une reprise exacte et ne sont pas chargés automatiquement par l’interface.

## Reprise en fin d’épisode

```powershell
.\.venv\Scripts\python.exe train_all.py --env LineWorld --agent DQN --seed 42 --episodes 100 --resume-checkpoint CHEMIN/agent.pkl --quiet
```

`--episodes` indique ici le nombre d’épisodes supplémentaires. L’agent, l’environnement, la graine et l’adversaire doivent correspondre au checkpoint. La reprise exacte a été testée sur DQN en fin d’épisode. La reprise au milieu d’une partie n’est pas prise en charge.

## Démonstration

```powershell
.\.venv\Scripts\python.exe gui.py
```

À http://127.0.0.1:5000 :
1. Human sur LineWorld, puis neuf mouvements à droite.
2. Human sur TicTacToe pour illustrer le step comprenant la réponse adverse.
3. RandomRollout ou MCTS sur Quarto, en précisant le choix de pièce automatique.
4. DQN sur LineWorld avec un checkpoint de l’étude : vérifier la graine et le nombre d’épisodes affichés.

`DRL_RUN_DIR` peut sélectionner explicitement un dossier d’expérience pour l’interface. Sinon, elle cherche le modèle compatible le plus récent dans les exécutions terminées de `results/runs/`.
