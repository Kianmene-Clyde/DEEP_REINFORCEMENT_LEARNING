# Validation locale du 5 novembre 2026

La validation porte sur le dépôt corrigé sous Windows et Python 3.12.14. Les dépendances exactes sont enregistrées dans requirements-lock.txt. Les originaux sont conservés ; les résultats récents ne les remplacent pas silencieusement.

## Résultats vérifiés

- 119 tests réussis, un cas exclu : Q-Learning tabulaire dense sur Quarto.
- Les 67 combinaisons compatibles subissent entraînement, évaluation et sauvegarde/rechargement dans les tests.
- Une exécution CLI smoke complète produit 67 entrées et modèles dans results/runs/smoke_verified.
- Une étude exploratoire sur cinq graines, cinq sélections et deux environnements produit 50 évaluations de politiques, avec 100 parties de test par politique.
- Une confirmation tabulaire sur cinq nouvelles graines et deux environnements produit dix évaluations supplémentaires.
- Un checkpoint DQN sauvegardé atteint le but de LineWorld en neuf décisions sur 100 nouvelles parties.
- L’interface Flask est testée par son client HTTP et son JavaScript rendu passe la vérification syntaxique de Node.
- Le paquet Python est construit en wheel et son import/exécution est vérifié depuis une installation de contrôle.
- Le rapport de sept pages et les dix diapositives corrigées sont rendus et vérifiés visuellement. Les graphiques et tableaux des diapositives restent natifs et éditables.

Le fichier JUnit est results/validation/pytest.xml. Les paramètres, résultats par partie et checkpoints sont conservés sous results/runs. Les premières expériences de développement ne servent pas à établir les conclusions scientifiques.

## Mesures avant et après

Les références préservées et les fonctions corrigées sont exécutées sur les mêmes données. Les temps sont les médianes de sept exécutions, avec équivalence vérifiée.

| Calcul | Avant | Après | Gain local |
|---|---:|---:|---:|
| Recherche de pièce gagnante sur 200 positions Quarto | 66,27 ms | 4,81 ms | 13,8 fois |
| Lissage de 10 000 retours | 84,01 ms | 0,39 ms | 215,7 fois |

Le gain du projet entier dépend de la part de ces opérations dans l’exécution. Les figures et l’entraînement ne sont pas annoncés comme globalement accélérés par ces facteurs.

Le test témoin du détour LineWorld conserve un retour de -4,8 et une réussite réelle. L’ancien calcul donne 0 % de victoire ; le calcul corrigé donne 100 % pour cette trajectoire témoin.

## Conclusions expérimentales

L’étude exploratoire atteint les chemins optimaux avec DQN, PPO et A2C sur LineWorld et GridWorld après 1 000 épisodes, pour les cinq graines testées. Le Q tabulaire atteint le chemin optimal sur GridWorld mais échoue sur LineWorld avec le taux commun de 0,001.

La confirmation tabulaire rétablit le taux par défaut de 0,1 sur cinq nouvelles graines et atteint neuf décisions sur LineWorld et sept sur GridWorld. Cette révision après observation du test initial est déclarée comme telle. Elle ne démontre pas un classement universel des algorithmes.

## Non exécuté ou hors portée

- Campagnes à 10 000 et 100 000 épisodes pour tous les agents.
- Évaluation statistique complète sur TicTacToe et Quarto, plusieurs adversaires et budgets de recherche.
- Self-play AlphaZero et protocole publié complet de MuZero/Stochastic MuZero.
- Contrôle des deux phases de Quarto.
- Matrice CI distante Windows/Linux et validation native des diapositives dans toutes les applications Office.
- Reprise au milieu d’un épisode.

Ces limites figurent aussi dans le rapport et la présentation. La validation d’un logiciel ne suffit pas à prouver la convergence d’un algorithme.
