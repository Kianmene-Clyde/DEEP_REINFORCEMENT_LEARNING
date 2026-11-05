# Analyse du test indépendant

Chaque ligne résume des graines d’entraînement indépendantes. Les intervalles à 95 % utilisent la loi de Student entre graines. Une seule graine ne permet pas de calculer cet intervalle.

| Environnement | Agent | Graines | Retour moyen | Réussite | Latence ms |
|---|---|---:|---:|---:|---:|
| LineWorld | TabularQL | 5 | 0.2000 | 100.0% | 0.0135 |
| GridWorld | TabularQL | 5 | 10.5400 | 100.0% | 0.0164 |

Les résultats du profil smoke vérifient l’exécution et ne démontrent pas la convergence.
Les agents de planification utilisent un simulateur ; leur budget de recherche doit être considéré dans la comparaison. Aucun score composite ne combine ces critères.
