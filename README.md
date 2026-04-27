# Deep Reinforcement Learning - Projet M2 IABD 2025-2026

## Description

Évaluation comparative de **13 algorithmes de Reinforcement Learning / Deep Reinforcement Learning** sur 4 environnements : LineWorld, GridWorld, TicTacToe et Quarto.

Framework entièrement implémenté en **Python/NumPy**

---

## Installation

```bash
pip install -r requirements.txt
```

Dépendances : numpy, scipy, matplotlib, flask, joblib

---

## Reproduction des résultats

### Étape 1 : Entraînement complet

```bash

python train_all.py --episodes 100000

# Test rapide pour vérifier
python train_all.py --quick

# Un seul environnement
python train_all.py --env TicTacToe

# Un seul agent sur un environnement
python train_all.py --env LineWorld --agent DQN --episodes 5000
```

**Durées estimées** (sur mon PC standard) :
| Commande               | Durée estimée |
|------------------------|---------------|
| `--quick`              | 44 minutes    |
| `--episodes 10000`     | ~48 heures   |
| `--episodes 100000`    | 4 jours au moins |

Les agents de planning (RandomRollout, MCTS) sont automatiquement limités à 500 épisodes car ils n'ont pas besoin d'entraînement long.

### Étape 2 : Vérifier les résultats

Après l'entraînement, les fichiers suivants sont générés :

```
results/
├── all_results.json              # Tableau récapitulatif de tous les résultats
├── LineWorld_DQN_metrics.json    # Métriques détaillées par agent/env
├── LineWorld_learning_curves.png # Courbes d'apprentissage
├── LineWorld_comparison.png      # Comparaison entre agents
├── GridWorld_*.json / *.png
├── TicTacToe_*.json / *.png
└── Quarto_*.json / *.png

models/
├── LineWorld_DQN.pkl             # Modèles entraînés (prêts à être chargés)
├── GridWorld_TabularQL.pkl
└── ...
```

### Étape 3 : Benchmark (parties/seconde avec joueur random)

```bash
python benchmark_random.py
```

Affiche le nombre de parties/seconde pour chaque environnement avec un agent random.

---

## Démonstration rapide (GUI)

```bash
python gui.py
```

Ouvrir **http://localhost:5000** dans un navigateur.

### Fonctionnalités de la GUI :

1. **Choisir un environnement** : LineWorld, GridWorld, TicTacToe, Quarto
2. **Choisir un agent** : Human (jouer soi-même), ou l'un des 13 agents entraînés
3. **Modes de jeu** :
   - **Mode humain** : jouer avec le clavier (flèches pour LineWorld/GridWorld, numéros pour TicTacToe, clic pour Quarto)
   - **Mode observation** : regarder l'agent jouer step-by-step ou en auto-play
4. **Chargement automatique** : la GUI charge les modèles entraînés depuis `models/` quand ils existent

### Scénario de démonstration

1. Lancer `python gui.py` et ouvrir http://localhost:5000
2. Sélectionner **Quarto** + **Human** → jouer quelques coups pour montrer le jeu
3. Sélectionner **Quarto** + **MCTS** → cliquer Auto Play pour voir l'agent jouer
4. Sélectionner **TicTacToe** + **Human** → jouer contre l'adversaire random
5. Sélectionner **LineWorld** + **DQN** → montrer que l'agent atteint l'objectif en 9 steps

---

## Structure du projet

```text
DEEP_REINFORCEMENT_LEARNING/
├── src/
│   ├── agents/                   # 13 agents RL
│   │   ├── base_agent.py         # Interface abstraite
│   │   ├── random_agent.py       # Agent aléatoire (baseline)
│   │   ├── tabular_q_learning.py # Q-Learning tabulaire
│   │   ├── deep_q_learning.py    # DQN avec target network
│   │   ├── double_deep_q_learning.py # DDQN, DDQN+ER, DDQN+PER
│   │   ├── reinforce.py          # REINFORCE, +MeanBaseline, +CriticBaseline
│   │   ├── ppo.py                # PPO (Proximal Policy Optimization)
│   │   ├── a2c.py                # A2C (Advantage Actor-Critic)
│   │   ├── random_rollout.py     # Random Rollout (planning)
│   │   └── mcts.py               # Monte Carlo Tree Search (UCT)
│   ├── environments/             # 4 environnements
│   │   ├── base_env.py           # Interface abstraite
│   │   ├── line_world.py         # Monde 1D (10 positions, 2 actions)
│   │   ├── grid_world.py         # Grille 2D (5x5, 4 actions, piège en (0,0))
│   │   ├── tictactoe.py          # Morpion vs Random/Heuristique
│   │   └── quarto.py             # Quarto vs Random/Heuristique (jeu choisi)
│   ├── neural_network/           # Framework réseau de neurones NumPy
│   │   └── model.py              # MLP, forward/backward, Adam, dual-head
│   ├── training/                 # Infrastructure d'entraînement
│   │   ├── trainer.py            # Boucle d'entraînement + checkpoints
│   │   ├── evaluator.py          # Évaluation post-entraînement
│   │   ├── metrics.py            # Collecte des métriques
│   │   └── replay_buffer.py      # Replay buffer standard + prioritized
│   └── utils/
│       └── plotting.py           # Courbes d'apprentissage, comparaisons
├── models/                       # Modèles entraînés sauvegardés (.pkl)
├── results/                      # Métriques JSON + graphiques PNG
├── gui.py                        # Interface graphique Flask (web)
├── train_all.py                  # Script d'entraînement principal
├── benchmark_random.py           # Benchmark parties/seconde
├── benchmark_analyze_results.py  # Analyse des résultats
├── plot_grid_search_results.py   # Visualisation grid search
├── Specifications_Encodage.docx  # Document d'encodage des états/actions
├── requirements.txt              # Dépendances Python
└── README.md                     # Ce fichier
```

---

## Algorithmes implémentés (13)

| # | Algorithme | Catégorie | Fichier |
|---|-----------|-----------|---------|
| 1 | Random | Baseline | `random_agent.py` |
| 2 | Tabular Q-Learning | Value-based | `tabular_q_learning.py` |
| 3 | Deep Q-Learning (DQN) | Value-based (Deep) | `deep_q_learning.py` |
| 4 | Double DQN | Value-based (Deep) | `double_deep_q_learning.py` |
| 5 | DDQN + Experience Replay | Value-based (Deep) | `double_deep_q_learning.py` |
| 6 | DDQN + Prioritized ER | Value-based (Deep) | `double_deep_q_learning.py` |
| 7 | REINFORCE | Policy gradient | `reinforce.py` |
| 8 | REINFORCE + Mean Baseline | Policy gradient | `reinforce.py` |
| 9 | REINFORCE + Critic Baseline | Actor-Critic | `reinforce.py` |
| 10 | PPO (A2C-style) | Actor-Critic | `ppo.py` |
| 11 | A2C | Actor-Critic | `a2c.py` |
| 12 | Random Rollout | Planning | `random_rollout.py` |
| 13 | MCTS (UCT) | Planning | `mcts.py` |

---

## Environnements (4)

| Environnement | Dim. observation | Encodage | Nb actions | Type |
|--------------|-----------------|----------|------------|------|
| LineWorld | 10 | One-hot | 2 (gauche/droite) | Single-player |
| GridWorld | 25 | One-hot | 4 (haut/bas/gauche/droite) | Single-player |
| TicTacToe | 9 | Valeurs (-1/0/+1) | 9 (positions) | Adversarial |
| Quarto | 101 | Attributs binaires | 16 (positions) | Adversarial |

### Particularités :
- **LineWorld** : reward sparse (+1 en position 9, -0.1 par step)
- **GridWorld** : piège en (0,0) avec reward -10, départ en (1,0), objectif en (4,4) avec reward +10
- **TicTacToe** : agent joue X (premier joueur) vs adversaire aléatoire
- **Quarto** : encodage par attributs (101 dim), heuristique de choix de pièce, symboles visuels ASCII art

---

## Métriques collectées

Pour chaque combinaison agent × environnement :
- Score moyen après N épisodes d'entraînement (checkpoints à 1K, 10K, 100K)
- Taux de victoire (win rate)
- Longueur moyenne d'une partie (nombre de steps)
- Temps moyen par coup (ms)
- Courbes d'apprentissage (reward au cours de l'entraînement)

---

## Hyperparamètres

| Paramètre | Valeur
|-----------|--------
| Learning rate (value-based) | 0.001
| Learning rate (policy gradient) | 0.005 (sparse) / 0.001 (adversarial)
| Epsilon decay | 0.998
| Discount (gamma) | 0.99
| PPO clip ratio | 0.2
| MCTS simulations | 50 
| Batch size | 64

---

## Framework Neural Network (NumPy)

Implémentation from scratch en NumPy :
- Couches Dense avec initialisation He (ReLU) / Xavier (autres)
- Activations : ReLU, Softmax, Tanh, Sigmoid, Linear
- Backpropagation avec gradient clipping
- Optimiseur Adam avec bias correction
- Réseaux dual-head (policy + value) pour Actor-Critic
- Save/Load avec pickle

---

## Outils et technologies

- **Python 3.10+**
- **NumPy** : calcul numérique, réseaux de neurones
- **Matplotlib** : graphiques et visualisations
- **Flask** : interface graphique web
- **SciPy** : fonctions utilitaires
- **Joblib** : parallélisation (grid search)
