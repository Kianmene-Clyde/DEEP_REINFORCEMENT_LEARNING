# Audit et corrections de la version 0 2

L’audit porte sur le code initial présent dans le dépôt et les documents du projet. Les constats ci-dessous sont confirmés par lecture du code et par les régressions ajoutées. Les résultats historiques restent archivés dans leur emplacement initial.

| Priorité | Défaut initial | Correction et preuve |
|---|---|---|
| P0 | Evaluator et benchmark assimilent retour positif à victoire | Issue explicite dans les environnements et test du détour LineWorld à retour négatif |
| P0 | DQN, DDQN et Q tabulaire maximisent sur des actions interdites | Masque du successeur stocké dans le replay et appliqué aux cibles ; tests des cibles Bellman |
| P0 | La limite de pas devient une fausse terminaison | Transition avec terminated/truncated distincts ; bootstrap et tests A2C/PPO/replay |
| P0 | Signes alternés dans UCT/Rollout alors que step joue déjà l’adversaire | Perspective contrôlée constante ; test de victoire immédiate et état réel inchangé |
| P0 | PPO omet le ratio dans son gradient actif | Objectif clippé différencié, masques de collecte, différences finies |
| P0 | AlphaZero ignore la récompense terminale de recherche | Sauvegarde des récompenses et retour actualisé dans PUCT ; même recherche à l’évaluation |
| P0 | MuZero remplace la chaîne de gradients par une constante multipliée par la perte de valeur | Rétropropagation exacte dans le modèle NumPy et les K transitions latentes |
| P0 | La variante stochastique utilise un réseau de hasard jamais appris | Remplacement par un prototype gaussien entraîné, renommé explicitement |
| P1 | np.random global couple agents, adversaires et simulations | Générateurs locaux et flux de simulation distincts |
| P1 | Évaluation sur le même environnement et état aléatoire que l’entraînement | Copies isolées ; modèles, replay et RNG originaux inchangés |
| P1 | Priorité maximale déjà exponentiée puis exponentiée à nouveau | Priorités brutes ; tirages avec remplacement et poids cohérents |
| P1 | Meilleurs scores de recherche présentés comme résultats finaux | Validation séparée, nouvelles graines d’entraînement si sélection, test indépendant |
| P1 | Score composite de classement arbitraire | Comparaison explicite par environnement, métriques séparées |
| P1 | Sauvegardes omettent Adam, replay ou RNG | Checkpoint complet v2 ; reprise DQN identique au calcul continu |
| P1 | Modèle absent remplacé par un agent non entraîné dans Flask | Erreur explicite, provenance du modèle affichée |
| P1 | Partie Flask globale partagée entre utilisateurs | Parties locales indexées par session, tests de séparation |
| P2 | Imports par modifications répétées de sys.path | Package src et registre de construction commun |
| P2 | Vérifications Quarto et lissage coûteux | Opérations NumPy vectorisées ; résultats comparés aux références |
| P2 | Pas de tests/CI/configuration de paquet | Suite pytest, Ruff, pyproject, profils et matrice CI |

## Contradictions documentaires

Le syllabus demande des métriques aux jalons 1 000, 10 000 et 100 000 épisodes, puis davantage si possible. La validation courte ne remplit pas ces jalons scientifiques. La nouvelle CLI permet de les atteindre sans plafonnement caché des hybrides à 500 épisodes.

Les documents décrivent LineWorld comme récompense sparse, alors que le code applique un coût de -0,1 à chaque déplacement hors but. Une réussite après détour peut donc avoir un retour négatif. Le rapport historique affirme aussi des classements et des écarts de vitesse entre frameworks sans protocole de confirmation suffisant. Ces affirmations ne sont pas reconduites.

La présentation de soutenance annonce 13 agents alors que le README en annonce 17. Les supports corrigés doivent employer 17 sélections d’agents/prototypes et préciser leurs limites.

Le syllabus cite des frameworks externes parmi les outils. Cette livraison préserve le choix NumPy demandé dans la mission ; elle ne prétend pas certifier l’acceptation pédagogique de ce choix.

## Limites restant à traiter

- Campagne complète jusqu’à 100 000 épisodes et confirmation sur plusieurs adversaires.
- Quarto complet avec décisions apprises pour les deux phases.
- Self-play et protocole complet d’AlphaZero.
- Légalité interne, reanalysis et entraînement à grande échelle de MuZero.
- Implémentation publiée de Stochastic MuZero avec afterstates et codes de hasard.
- Confirmation de la CI sur les systèmes distants et étude de portabilité des performances.

Les prototypes fonctionnent dans le laboratoire. Leur exécution ne démontre pas leur équivalence aux publications ni leur convergence.
