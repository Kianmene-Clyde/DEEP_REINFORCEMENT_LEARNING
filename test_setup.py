#!/usr/bin/env python3
"""
test_setup.py — Lance ce script en premier dans PyCharm pour vérifier
que tout fonctionne. Corrige les erreurs une par une si besoin.

Usage dans PyCharm :
    Clic droit sur ce fichier → Run 'test_setup'
    
Ou dans le terminal PyCharm :
    python test_setup.py
"""
import sys
import os
import time

# ─── ÉTAPE 0 : Vérifier le répertoire de travail ───
print("=" * 60)
print("ÉTAPE 0 : Répertoire de travail")
print("=" * 60)
cwd = os.getcwd()
print(f"  Répertoire courant : {cwd}")

# Détecter si on est dans le bon dossier
if os.path.exists('src/agents') and os.path.exists('src/environments'):
    print("  ✓ Structure du projet détectée")
    sys.path.insert(0, 'src')
elif os.path.exists('DRL/src/agents'):
    print("  → On est un niveau au-dessus, on entre dans DRL/")
    os.chdir('DRL')
    sys.path.insert(0, 'src')
else:
    print("  ✗ ERREUR : Ce script doit être lancé depuis le dossier DRL/")
    print("    Dans PyCharm : Run → Edit Configurations → Working directory → sélectionne DRL/")
    sys.exit(1)

errors = []

# ─── ÉTAPE 1 : Vérifier les dépendances ───
print("\n" + "=" * 60)
print("ÉTAPE 1 : Dépendances Python")
print("=" * 60)

deps = {
    'numpy': 'pip install numpy',
    'scipy': 'pip install scipy',
    'matplotlib': 'pip install matplotlib',
    'flask': 'pip install flask',
    'joblib': 'pip install joblib',
}

for pkg, install_cmd in deps.items():
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', '?')
        print(f"  ✓ {pkg} ({ver})")
    except ImportError:
        errors.append(f"Module manquant : {pkg}")
        print(f"  ✗ {pkg} MANQUANT → {install_cmd}")

# ─── ÉTAPE 2 : Vérifier les imports du projet ───
print("\n" + "=" * 60)
print("ÉTAPE 2 : Imports du projet")
print("=" * 60)

try:
    from environments import LineWorld, GridWorld, TicTacToe, Quarto
    print("  ✓ Environnements importés (LineWorld, GridWorld, TicTacToe, Quarto)")
except ImportError as e:
    errors.append(f"Import environments : {e}")
    print(f"  ✗ Erreur import environments : {e}")

try:
    from agents import (RandomAgent, TabularQLearningAgent, DeepQLearningAgent,
                        DoubleDeepQLearningAgent, DDQNWithERAgent, DDQNWithPERAgent,
                        REINFORCEAgent, REINFORCEMeanBaselineAgent, REINFORCECriticBaselineAgent,
                        PPOAgent, A2CAgent, RandomRolloutAgent, MCTSAgent,
                        ExpertApprenticeAgent, AlphaZeroAgent, MuZeroAgent,
                        StochasticMuZeroAgent)
    print("  ✓ 17 classes d'agents importées")
except ImportError as e:
    errors.append(f"Import agents : {e}")
    print(f"  ✗ Erreur import agents : {e}")

try:
    from nn.model import NeuralNetwork
    print("  ✓ NeuralNetwork importé")
except ImportError as e:
    errors.append(f"Import nn : {e}")
    print(f"  ✗ Erreur import nn : {e}")

try:
    from training.trainer import Trainer
    from training.evaluator import Evaluator
    from training.metrics import Metrics
    from training.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
    print("  ✓ Training infrastructure importée")
except ImportError as e:
    errors.append(f"Import training : {e}")
    print(f"  ✗ Erreur import training : {e}")

try:
    from utils.plotting import plot_learning_curves, plot_comparison
    print("  ✓ Plotting importé")
except ImportError as e:
    errors.append(f"Import plotting : {e}")
    print(f"  ✗ Erreur import plotting : {e}")

if errors:
    print(f"\n⚠ {len(errors)} erreurs d'import — corrige-les avant de continuer.")
    for e in errors:
        print(f"  → {e}")
    sys.exit(1)

# ─── ÉTAPE 3 : Tester les environnements ───
print("\n" + "=" * 60)
print("ÉTAPE 3 : Environnements")
print("=" * 60)

import numpy as np
import copy

for name, env_cls, kwargs in [
    ('LineWorld', LineWorld, {'length': 10}),
    ('GridWorld', GridWorld, {'width': 5, 'height': 5}),
    ('TicTacToe', TicTacToe, {'opponent_type': 'random'}),
    ('Quarto', Quarto, {'opponent_type': 'random'}),
]:
    try:
        env = env_cls(**kwargs)
        state = env.reset()
        va = env.get_valid_actions(state)
        assert len(va) > 0, "Pas d'actions valides"
        gs = env._get_state()
        assert gs is not None, "_get_state() retourne None"
        s2, r, done, info = env.step(int(va[0]))
        env_copy = copy.deepcopy(env)  # Nécessaire pour MCTS/Rollout
        print(f"  ✓ {name:12s}  obs_dim={env.observation_space}  "
              f"actions={env.action_space}  valid={len(va)}  deepcopy=OK")
    except Exception as e:
        errors.append(f"{name}: {e}")
        print(f"  ✗ {name}: {e}")

# ─── ÉTAPE 4 : Tester chaque agent ───
print("\n" + "=" * 60)
print("ÉTAPE 4 : Agents (test rapide sur LineWorld)")
print("=" * 60)

env = LineWorld(10)
obs = env.observation_space
act = env.action_space

agents_test = [
    ('Random', lambda: RandomAgent(act)),
    ('TabularQL', lambda: TabularQLearningAgent(obs, act)),
    ('DQN', lambda: DeepQLearningAgent(obs, act, epsilon_decay=0.99)),
    ('DoubleDQN', lambda: DoubleDeepQLearningAgent(obs, act, epsilon_decay=0.99)),
    ('DDQN+ER', lambda: DDQNWithERAgent(obs, act, epsilon_decay=0.99)),
    ('DDQN+PER', lambda: DDQNWithPERAgent(obs, act, epsilon_decay=0.99)),
    ('REINFORCE', lambda: REINFORCEAgent(obs, act)),
    ('REINFORCE+Mean', lambda: REINFORCEMeanBaselineAgent(obs, act)),
    ('REINFORCE+Critic', lambda: REINFORCECriticBaselineAgent(obs, act)),
    ('PPO', lambda: PPOAgent(obs, act)),
    ('A2C', lambda: A2CAgent(obs, act)),
    ('RandomRollout', lambda: RandomRolloutAgent(act, num_rollouts=3, max_rollout_depth=10)),
    ('MCTS', lambda: MCTSAgent(act, num_simulations=5, max_rollout_depth=10)),
    ('ExpertApprentice', lambda: ExpertApprenticeAgent(obs, act, expert_simulations=5, hidden_layers=[32])),
    ('AlphaZero', lambda: AlphaZeroAgent(obs, act, num_simulations=5, hidden_layers=[32])),
    ('MuZero', lambda: MuZeroAgent(obs, act, num_simulations=5, latent_dim=16, hidden_layers=[32])),
    ('MuZero_Stochastic', lambda: StochasticMuZeroAgent(obs, act, num_simulations=5, latent_dim=16, hidden_layers=[32])),
]

passed, failed = 0, 0
for name, create_fn in agents_test:
    try:
        agent = create_fn()
        trainer = Trainer(env, agent, max_episodes=10, max_steps_per_episode=15,
                          checkpoints=[], eval_episodes=3, verbose=False)
        m = trainer.train()
        reward = m.get_average_reward()
        passed += 1
        print(f"  ✓ {name:25s}  (reward={reward:.3f})")
    except Exception as e:
        failed += 1
        errors.append(f"Agent {name}: {e}")
        print(f"  ✗ {name:25s}  ERREUR: {e}")

# ─── ÉTAPE 5 : Tester sur TicTacToe et Quarto ───
print("\n" + "=" * 60)
print("ÉTAPE 5 : Agents sur TicTacToe & Quarto")
print("=" * 60)

for env_name, env_cls, kwargs, obs_size in [
    ('TicTacToe', TicTacToe, {'opponent_type': 'random'}, 9),
    ('Quarto', Quarto, {'opponent_type': 'random'}, 33),
]:
    env = env_cls(**kwargs)
    for agent_name, create_fn in [
        ('DQN', lambda o, a: DeepQLearningAgent(o, a, epsilon_decay=0.99)),
        ('PPO', lambda o, a: PPOAgent(o, a)),
        ('MCTS', lambda o, a: MCTSAgent(a, num_simulations=5)),
    ]:
        try:
            agent = create_fn(obs_size, env.action_space)
            trainer = Trainer(env, agent, max_episodes=10, max_steps_per_episode=20,
                              checkpoints=[], eval_episodes=3, verbose=False)
            m = trainer.train()
            passed += 1
            print(f"  ✓ {env_name:12s} × {agent_name:10s}  reward={m.get_average_reward():.3f}")
        except Exception as e:
            failed += 1
            errors.append(f"{env_name}/{agent_name}: {e}")
            print(f"  ✗ {env_name:12s} × {agent_name:10s}  ERREUR: {e}")

# ─── ÉTAPE 6 : Tester save/load ───
print("\n" + "=" * 60)
print("ÉTAPE 6 : Save/Load des modèles")
print("=" * 60)

os.makedirs('models', exist_ok=True)
env = LineWorld(10)

agent = TabularQLearningAgent(10, 2, epsilon=0.1)
trainer = Trainer(env, agent, max_episodes=100, checkpoints=[], verbose=False)
trainer.train()
agent.save('models/test_tabularql')
agent2 = TabularQLearningAgent(10, 2)
agent2.load('models/test_tabularql')
print("  ✓ TabularQL save/load")

agent3 = DeepQLearningAgent(10, 2, epsilon_decay=0.99)
trainer3 = Trainer(env, agent3, max_episodes=50, checkpoints=[], verbose=False)
trainer3.train()
agent3.save('models/test_dqn')
agent4 = DeepQLearningAgent(10, 2)
agent4.load('models/test_dqn')
print("  ✓ DQN save/load")

agent5 = PPOAgent(10, 2)
trainer5 = Trainer(env, agent5, max_episodes=50, checkpoints=[], verbose=False)
trainer5.train()
agent5.save('models/test_ppo')
agent6 = PPOAgent(10, 2)
agent6.load('models/test_ppo')
print("  ✓ PPO save/load")

# Clean up test models
for f in ['models/test_tabularql.pkl', 'models/test_dqn_qnet.pkl',
          'models/test_dqn_config.pkl', 'models/test_ppo_actor.pkl',
          'models/test_ppo_critic.pkl']:
    try: os.remove(f)
    except: pass

# ─── ÉTAPE 7 : Tester les métriques et plotting ───
print("\n" + "=" * 60)
print("ÉTAPE 7 : Métriques & Plotting")
print("=" * 60)

os.makedirs('results', exist_ok=True)
m = Metrics()
for i in range(100):
    m.add_episode(float(i) * 0.01, 10, 0.001)
m.save('results/test_metrics.json')
m2 = Metrics.load('results/test_metrics.json')
assert len(m2.episode_rewards) == 100
print("  ✓ Metrics save/load")

try:
    import matplotlib
    matplotlib.use('Agg')
    plot_learning_curves({'Test': m}, 'Test', 'results')
    print("  ✓ Plotting fonctionne")
    os.remove('results/Test_learning_curves.png')
except Exception as e:
    print(f"  ⚠ Plotting: {e}")

os.remove('results/test_metrics.json')

# ─── ÉTAPE 8 : Tester la GUI ───
print("\n" + "=" * 60)
print("ÉTAPE 8 : GUI Flask")
print("=" * 60)

try:
    sys.path.insert(0, '.')
    import gui
    with gui.app.test_client() as client:
        r = client.get('/')
        assert r.status_code == 200
        print("  ✓ Page d'accueil charge")
        
        r = client.post('/api/new_game',
                        json={'env': 'TicTacToe', 'agent': 'Random'})
        data = r.get_json()
        assert 'board' in data
        print("  ✓ Nouvelle partie TicTacToe + Random")
        
        r = client.post('/api/step')
        data = r.get_json()
        assert 'action' in data
        print("  ✓ Step agent fonctionne")
        
        r = client.post('/api/new_game',
                        json={'env': 'TicTacToe', 'agent': 'Human'})
        data = r.get_json()
        assert data['mode'] == 'human'
        r = client.post('/api/human_action', json={'action': 4})
        data = r.get_json()
        assert 'board' in data
        print("  ✓ Mode humain fonctionne")
except Exception as e:
    errors.append(f"GUI: {e}")
    print(f"  ✗ GUI: {e}")

# ─── RÉSULTAT FINAL ───
elapsed = time.time()
print("\n" + "=" * 60)
print("RÉSULTAT FINAL")
print("=" * 60)
total = passed + failed
print(f"  Agents testés : {passed}/{total} OK")

if errors:
    print(f"\n  ⚠ {len(errors)} ERREURS trouvées :")
    for e in errors:
        print(f"    → {e}")
    print(f"\n  Corrige ces erreurs puis relance ce script.")
else:
    print(f"\n  ✅ TOUT FONCTIONNE PARFAITEMENT !")
    print(f"  Tu peux maintenant :")
    print(f"    1. Lancer l'entraînement :  python train_all.py --quick")
    print(f"    2. Lancer la GUI :          python gui.py")
    print(f"    3. Lancer les exemples :    python examples.py")
