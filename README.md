# Deep Reinforcement Learning - Projet M2 IABD 2025-2026

## Description

Évaluation comparative de **17 algorithmes de Reinforcement Learning / Deep Reinforcement Learning** sur 4 environnements : LineWorld, GridWorld, TicTacToe et Quarto.

Framework entièrement implémenté en **Python/NumPy** (pas de TensorFlow/PyTorch) — chaque ligne de code est compréhensible et explicable.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation rapide

### Entraîner tous les agents

```bash
# Test rapide (~5 min)
python train_all.py --quick

# Entraînement standard (10 000 épisodes, ~1-3h)
python train_all.py

# Entraînement long (100 000 épisodes, ~10-20h)
python train_all.py --episodes 100000

# Un seul environnement
python train_all.py --env TicTacToe
```

### Interface graphique (démonstration)

```bash
python gui.py
# Ouvrir http://localhost:5000
```

L'interface permet de regarder jouer les 17 agents et de jouer en mode humain.

### Benchmark (parties/seconde)

```bash
python benchmark_random.py
```

## Algorithmes implémentés (17)

| # | Algorithme | Catégorie | Fichier |
|---|-----------|-----------|---------|
| 1 | Random | Baseline | random_agent.py |
| 2 | Tabular Q-Learning | Value-based | tabular_q_learning.py |
| 3 | DQN | Value-based (Deep) | deep_q_learning.py |
| 4 | Double DQN | Value-based (Deep) | double_deep_q_learning.py |
| 5 | DDQN + Experience Replay | Value-based (Deep) | double_deep_q_learning.py |
| 6 | DDQN + Prioritized ER | Value-based (Deep) | double_deep_q_learning.py |
| 7 | REINFORCE | Policy gradient | reinforce.py |
| 8 | REINFORCE + Mean Baseline | Policy gradient | reinforce.py |
| 9 | REINFORCE + Critic Baseline | Actor-Critic | reinforce.py |
| 10 | PPO (A2C-style) | Actor-Critic | ppo.py |
| 11 | A2C | Actor-Critic | a2c.py |
| 12 | Random Rollout | Planning | random_rollout.py |
| 13 | MCTS (UCT) | Planning | mcts.py |
| 14 | Expert Apprentice | Imitation Learning | expert_apprentice.py |
| 15 | AlphaZero | Hybrid (NN + MCTS) | alphazero.py |
| 16 | MuZero | Model-based | muzero.py |
| 17 | MuZero Stochastique | Model-based | muzero_stochastic.py |

## Environnements (4)

| Environnement | Dim. observation | Encodage | Actions | Type | Particularité |
|--------------|-----------------|----------|---------|------|---------------|
| LineWorld | 10 | One-hot | 2 | Single-player | Reward sparse (+1 en pos. 9) |
| GridWorld | 25 | One-hot | 4 | Single-player | Piège en (0,0) = -10, départ en (1,0) |
| TicTacToe | 9 | Valeurs (-1/0/+1) | 9 | Adversarial | Agent joue X (premier joueur) |
| Quarto | 101 | Attributs binaires | 16 | Adversarial | 5 valeurs/case, heuristique choix pièce |

## Structure du projet

```text
DEEP_REINFORCEMENT_LEARNING/
├── src/
│   ├── agents/                        # 17 agents RL
│   ├── environments/                  # 4 environnements
│   ├── neural_network/                # Framework NN NumPy
│   └── training/                      # Trainer, Evaluator, Replay Buffer
├── models/                            # Modèles entraînés (.pkl)
├── results/                           # Métriques JSON + graphiques PNG
├── gui.py                             # Interface graphique Flask
├── train_all.py                       # Script d'entraînement principal
├── benchmark_random.py                # Benchmark parties/seconde
├── RAPPORT_EXPERIMENTATIONS_DRL.docx  # Rapport
├── Specifications_Encodage.docx       # Encodage des états/actions
└── requirements.txt
```
