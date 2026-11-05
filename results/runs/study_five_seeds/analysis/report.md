# Analyse du test indépendant

Chaque ligne résume des graines d’entraînement indépendantes. Les intervalles à 95 % utilisent la loi de Student entre graines. Une seule graine ne permet pas de calculer cet intervalle.

| Environnement | Agent | Graines | Retour moyen | Réussite | Latence ms |
|---|---|---:|---:|---:|---:|
| LineWorld | Random | 5 | -5.8746 | 67.6% | 0.0455 |
| LineWorld | TabularQL | 5 | -10.0000 | 0.0% | 0.0376 |
| LineWorld | DQN | 5 | 0.2000 | 100.0% | 0.0525 |
| LineWorld | PPO | 5 | 0.2000 | 100.0% | 0.2751 |
| LineWorld | A2C | 5 | 0.2000 | 100.0% | 0.0674 |
| GridWorld | Random | 5 | -5.5682 | 20.8% | 0.0164 |
| GridWorld | TabularQL | 5 | 10.5400 | 100.0% | 0.0110 |
| GridWorld | DQN | 5 | 10.5400 | 100.0% | 0.0382 |
| GridWorld | PPO | 5 | 10.5400 | 100.0% | 0.0932 |
| GridWorld | A2C | 5 | 10.5400 | 100.0% | 0.0601 |

Les résultats du profil smoke vérifient l’exécution et ne démontrent pas la convergence.
Les agents de planification utilisent un simulateur ; leur budget de recherche doit être considéré dans la comparaison. Aucun score composite ne combine ces critères.
