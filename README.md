# Deep Reinforcement Learning - Projet M2 IABD

## Description

Évaluation comparative de **16 algorithmes de Deep Reinforcement Learning** sur 4 environnements (LineWorld, GridWorld, TicTacToe, Quarto).

Framework entièrement implémenté en **Python/NumPy** (pas de TensorFlow/PyTorch) — chaque ligne de code est compréhensible et explicable.

## Structure du projet

```
DRL/
├── src/
│   ├── agents/                  # 16 agents RL
│   │   ├── base_agent.py        # Interface abstraite
│   │   ├── random_agent.py      # Agent aléatoire (baseline)
│   │   ├── tabular_q_learning.py # Q-Learning tabulaire
│   │   ├── deep_q_learning.py   # DQN avec target network
│   │   ├── double_deep_q_learning.py  # DDQN, DDQN+ER, DDQN+PER
│   │   ├── reinforce.py         # REINFORCE, +MeanBaseline, +CriticBaseline
│   │   ├── ppo.py               # PPO A2C-style (clipped objective)
│   │   ├── a2c.py               # Advantage Actor-Critic
│   │   ├── random_rollout.py    # Random Rollout planning
│   │   ├── mcts.py              # Monte Carlo Tree Search (UCT)
│   │   ├── expert_apprentice.py # Imitation Learning (MCTS expert)
│   │   ├── alphazero.py         # AlphaZero (MCTS + dual-head NN)
│   │   ├── muzero.py            # MuZero (learned model + MCTS)
│   │   └── muzero_stochastic.py # MuZero Stochastique
│   ├── environments/            # 4 environnements
│   │   ├── base_env.py
│   │   ├── line_world.py        # Monde 1D
│   │   ├── grid_world.py        # Grille 2D (5x5)
│   │   ├── tictactoe.py         # Morpion vs Random/Heuristique
│   │   └── quarto.py            # Quarto vs Random/Heuristique
│   ├── nn/                      # Framework réseau de neurones (NumPy)
│   │   ├── model.py             # MLP, dual-head, forward/backward, Adam
│   │   └── optimizers.py
│   ├── training/                # Infrastructure d'entraînement
│   │   ├── trainer.py           # Boucle d'entraînement + checkpoints
│   │   ├── evaluator.py         # Évaluation post-entraînement
│   │   ├── metrics.py           # Collecte des métriques
│   │   └── replay_buffer.py     # Replay buffer standard + prioritized
│   └── utils/
│       └── plotting.py          # Courbes d'apprentissage, comparaisons
├── models/                      # Modèles entraînés sauvegardés
├── results/                     # Métriques JSON + graphiques PNG
├── train_all.py                 # Script d'entraînement principal
├── gui.py                       # Interface graphique Flask (web)
├── examples.py                  # Exemples d'utilisation
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation rapide

### Entraîner tous les agents

```bash
# Test rapide (100 épisodes)
python train_all.py --quick

# Entraînement complet (10 000 épisodes)
python train_all.py

# Un seul environnement
python train_all.py --env TicTacToe

# Un seul agent
python train_all.py --agent DQN --episodes 5000
```

### Interface graphique

```bash
python gui.py
# Ouvrir http://localhost:5000
```

L'interface permet de :
- **Regarder jouer** n'importe quel agent (step-by-step ou auto-play)
- **Jouer en mode humain** contre l'adversaire (TicTacToe, Quarto)
- Sélectionner l'environnement et l'agent
- Charger automatiquement les modèles entraînés

### Exemples

```bash
python examples.py
```

## Algorithmes implémentés

| # | Algorithme | Type | Fichier |
|---|-----------|------|---------|
| 1 | Random | Baseline | `random_agent.py` |
| 2 | Tabular Q-Learning | Value-based | `tabular_q_learning.py` |
| 3 | Deep Q-Learning (DQN) | Value-based | `deep_q_learning.py` |
| 4 | Double DQN | Value-based | `double_deep_q_learning.py` |
| 5 | DDQN + Experience Replay | Value-based | `double_deep_q_learning.py` |
| 6 | DDQN + Prioritized ER | Value-based | `double_deep_q_learning.py` |
| 7 | REINFORCE | Policy gradient | `reinforce.py` |
| 8 | REINFORCE + Mean Baseline | Policy gradient | `reinforce.py` |
| 9 | REINFORCE + Critic Baseline | Actor-Critic | `reinforce.py` |
| 10 | PPO (A2C-style) | Actor-Critic | `ppo.py` |
| 11 | Random Rollout | Planning | `random_rollout.py` |
| 12 | MCTS (UCT) | Planning | `mcts.py` |
| 13 | Expert Apprentice | Imitation | `expert_apprentice.py` |
| 14 | AlphaZero | Planning+Learning | `alphazero.py` |
| 15 | MuZero | Model-based | `muzero.py` |
| 16 | MuZero Stochastique | Model-based | `muzero_stochastic.py` |

## Environnements

| Environnement | Observation | Actions | Description |
|--------------|-------------|---------|-------------|
| LineWorld | 10 (one-hot) | 2 (gauche/droite) | Atteindre l'extrémité droite |
| GridWorld | 25 (one-hot) | 4 (haut/bas/gauche/droite) | Naviguer vers le coin (4,4) |
| TicTacToe | 9 (board flat) | 9 (positions) | Morpion vs Random/Heuristique |
| Quarto | 33 (board+piece+avail) | 16 (positions) | Jeu Quarto vs Random/Heuristique |

## Encodage des états et actions

### LineWorld
- **État** : vecteur one-hot de taille 10, position courante = 1.0
- **Actions** : 0 = gauche, 1 = droite

### GridWorld  
- **État** : vecteur one-hot de taille 25 (5×5), position = 1.0
- **Actions** : 0=haut, 1=bas, 2=gauche, 3=droite

### TicTacToe
- **État** : vecteur de 9 valeurs (−1=adversaire O, 0=vide, 1=agent X)
- **Actions** : position 0-8 sur le plateau (ligne×3 + colonne)

### Quarto
- **État** : 33 dimensions = plateau normalisé (16) + pièce courante (1) + masque disponibilité (16)
- **Actions** : position 0-15 où placer la pièce courante

## Métriques collectées

Pour chaque combinaison agent × environnement :
- Score moyen après 1 000 / 10 000 / 100 000 épisodes d'entraînement
- Longueur moyenne d'une partie
- Temps moyen par coup (ms)
- Taux de victoire / défaite / match nul (jeux adversariaux)

## Framework Neural Network (NumPy)

Implémentation from scratch d'un MLP en NumPy :
- Couches Dense avec initialisation He/Xavier
- Activations : ReLU, Softmax, Tanh, Sigmoid, Linear
- Backpropagation avec gradient clipping
- Optimiseur Adam intégré
- Réseau dual-head (policy + value) pour AlphaZero/MuZero
- Save/Load avec pickle

## Reproduction des résultats

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Lancer l'entraînement complet
python train_all.py --episodes 10000

# 3. Les résultats sont dans results/
#    - all_results.json : métriques complètes
#    - *_learning_curves.png : courbes d'apprentissage
#    - *_comparison.png : comparaison entre agents

# 4. Les modèles entraînés sont dans models/

# 5. Lancer la GUI pour visualiser
python gui.py
```

## Outils et technologies

- **Python 3.10+**
- **NumPy** : calcul numérique, réseaux de neurones
- **Matplotlib** : graphiques et visualisations
- **Flask** : interface graphique web
- **SciPy** : fonctions utilitaires
