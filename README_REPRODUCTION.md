# Deep Reinforcement Learning - README de reproduction

## 1. Objectif du projet

Ce projet compare **17 agents** de Reinforcement Learning / Deep Reinforcement Learning sur 4 environnements : LineWorld, GridWorld, TicTacToe et Quarto. Le projet est implémenté en Python/NumPy, avec un réseau de neurones écrit from scratch.

Les agents évalués sont :
- **Value-based** : Random, TabularQL, DQN, DoubleDQN, DDQN+ER, DDQN+PER
- **Policy gradient** : REINFORCE, REINFORCE+Mean, REINFORCE+Critic
- **Actor-Critic** : PPO (A2C-style), A2C
- **Planning** : RandomRollout, MCTS (UCT)
- **Hybrides** : ExpertApprentice, AlphaZero, MuZero, MuZero Stochastique

## 2. Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## 3. Vérification rapide

```bash
python train_all.py --quick --max-configs-per-agent 1 --seed 42
```

Vérifie que les 17 agents fonctionnent sur les 4 environnements. Durée : ~5 minutes.

## 4. Lancer l'expérience principale

```bash
# Entraînement standard avec grid search (durée : ~1-3h)
python train_all.py --grid full --episodes 10000 --eval-episodes 200 --seed 42

# Version silencieuse
python train_all.py --grid full --episodes 10000 --eval-episodes 200 --seed 42 --quiet
```

Les agents de planning (RandomRollout, MCTS) et hybrides (AlphaZero, MuZero, ExpertApprentice) sont automatiquement limités à 500 épisodes car ils effectuent une recherche arborescente à chaque coup.

## 5. Fichiers produits

```text
results/
├── all_results.json                    # Tableau récapitulatif (17 agents × 4 envs)
├── grid_search/all_config_results.csv  # Détails du grid search
├── grid_search/best_by_agent_env.csv   # Meilleure config par agent/env
├── *_learning_curves.png               # Courbes d'apprentissage
└── *_comparison.png                    # Comparaisons entre agents

models/
└── *_best/                             # Modèles entraînés prêts à exécuter
```

## 6. Lancer l'interface graphique (démonstration)

```bash
python gui.py
```

Ouvrir **http://localhost:5000** dans un navigateur.

### Scénario de démonstration (~5 min) :

1. Sélectionner **Quarto** + **Human** → jouer quelques coups
2. Sélectionner **Quarto** + **MCTS** → Auto Play pour voir l'agent jouer
3. Sélectionner **TicTacToe** + **AlphaZero** → observer la stratégie
4. Sélectionner **LineWorld** + **DQN** → l'agent atteint l'objectif en 9 steps
5. Sélectionner **GridWorld** + **TabularQL** → l'agent évite le piège en (0,0)

### Contrôles en mode humain :
- **LineWorld** : flèches gauche/droite
- **GridWorld** : flèches directionnelles
- **TicTacToe** : clic sur les cases ou pavé numérique
- **Quarto** : clic sur les cases vides du plateau

## 7. Benchmark (parties/seconde)

```bash
python benchmark_random.py
```

## 8. Environnements

| Environnement | Observation | Actions | Particularité |
|--------------|-------------|---------|---------------|
| LineWorld | 10 dim (one-hot) | 2 (gauche/droite) | Reward sparse, départ pos. 0, objectif pos. 9 |
| GridWorld | 25 dim (one-hot) | 4 (haut/bas/gauche/droite) | Piège en (0,0) = reward -10, départ (1,0), objectif (4,4) |
| TicTacToe | 9 dim (-1/0/+1) | 9 (positions) | Agent = X (premier joueur) vs Random |
| Quarto | 101 dim (attributs) | 16 (positions) | Encodage par attributs binaires (5 val/case), heuristique de choix de pièce |

## 9. Limites de reproduction

- Les résultats peuvent varier selon la machine et la seed
- Les agents de planning sont très coûteux sur Quarto (durée élevée)
- L'implémentation NumPy est 10-100x plus lente que PyTorch/TensorFlow
- Les agents hybrides (AlphaZero, MuZero) nécessitent plus d'épisodes pour converger pleinement
