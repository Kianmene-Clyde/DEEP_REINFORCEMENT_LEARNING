# Protocole scientifique et conventions de calcul

## Objectif expérimental

Comparer les retours, les issues, la longueur des épisodes, la latence et le temps d’apprentissage par environnement. Ces critères restent distincts. Aucun score composite arbitraire ne produit de classement global.

Les environnements adversariaux sont des MDP contre un adversaire fixe. Un step exécute le coup contrôlé puis la réponse adverse. Les récompenses et valeurs restent dans la perspective du joueur contrôlé. La recherche ne change donc pas les signes entre deux steps.

## Transitions et limites de temps

L’environnement conserve son API à quatre valeurs `state, reward, done, info`. `done` signifie une terminaison réelle et `info.outcome` fournit l’issue. Le trainer construit séparément `truncated` à la limite de pas.

`BaseAgent.observe_transition` transmet ces deux informations et les actions légales du successeur. Les cibles Q utilisent `r + gamma * V(next_state)` aux limites de temps et `r` aux terminaisons. A2C et PPO conservent la valeur du successeur lors d’une interruption. Les méthodes Monte Carlo REINFORCE écartent les trajectoires censurées, faute de valeur de bootstrap sans biais. Ce filtrage peut lui-même sélectionner des épisodes courts : rapporter le taux de timeout et augmenter le budget avant une conclusion scientifique.

## Réseaux et gradients

Les gradients fournis au moteur sont des gradients par exemple. Chaque gradient de paramètre est moyenné une fois sur le lot. Pour une sortie softmax, `logits=True` signifie que le gradient est déjà exprimé par rapport aux logits. Les autres cas utilisent le véritable produit jacobien-vecteur.

Les deux têtes politique/valeur contribuent au même tronc. Les gradients d’entrée permettent de différencier la dynamique de MuZero sur K pas, sans constantes de substitution. Les paramètres restent figés pendant le calcul de la trajectoire différentiable, puis Adam reçoit la somme des gradients. Les cibles de valeur scalaires conservent leur échelle de récompense et ne sont plus renormalisées selon le lot.

PPO utilise `min(ratio * A, clip(ratio) * A)`. Le gradient actif contient le facteur ratio. Les masques de collecte sont réutilisés à chaque mise à jour. Le replay prioritaire conserve des priorités brutes, applique alpha une fois, échantillonne avec remplacement et utilise les probabilités correspondantes pour la correction d’importance.

## Simulations et information disponible

UCT et PUCT clonant le simulateur disposent des règles exactes. Chaque simulation reçoit un flux aléatoire indépendant de l’environnement réel. Les issues stochastiques sont distinguées dans l’arbre par leur observation.

Les agents à modèle latent n’utilisent pas le simulateur. Ils reçoivent seulement le masque réel à la racine. MuZero ne modélise pas la légalité à l’intérieur de l’arbre : des branches latentes peuvent représenter des actions impossibles. Les comparaisons avec UCT doivent rappeler cette différence d’information.

Le prototype gaussien apprend une moyenne et une variance latentes par vraisemblance gaussienne sur les successeurs observés, ainsi que récompense et terminaison. Sa planification moyenne des issues latentes échantillonnées à un pas. Il ne possède ni code discret appris, ni arbre alternant afterstates et hasard. Son alias historique `MuZero_Stochastic` sert uniquement à conserver la compatibilité de sélection.

## Graines et sélection

Les agents et environnements possèdent leurs propres générateurs NumPy. Le numéro d’épisode et la graine d’entraînement déterminent la graine du reset. L’évaluation utilise des domaines séparés pour validation et test. Toutes les configurations d’un agent voient les mêmes graines d’entraînement.

La sélection compare le retour moyen sur la validation. Lorsque plusieurs configurations sont effectivement comparées, la phase finale utilise de nouvelles graines d’entraînement. Le test ne sert pas à sélectionner le candidat. Toute révision d’hyperparamètres décidée après lecture du test doit être présentée comme exploratoire et confirmée sur de nouvelles graines.

## Incertitude et lecture des résultats

Le nombre de parties de test mesure l’incertitude conditionnelle d’une politique. Il ne remplace pas plusieurs entraînements indépendants. Les résumés calculent un intervalle à 95 % de Student pour la moyenne entre graines d’entraînement, sans intervalle avec une seule graine.

Avec cinq graines, cet intervalle reste une approximation sensible aux distributions et à la faible taille de l’échantillon. Il n’est pas un intervalle de prédiction d’une future graine et ne garantit pas une probabilité de réussite. La latence dépend de la machine et de la charge. Les plafonds de budget et le nombre de simulations doivent accompagner chaque comparaison.

## Références primaires

- Schulman et al., PPO : https://arxiv.org/abs/1707.06347
- Schaul et al., Prioritized Experience Replay : https://arxiv.org/abs/1511.05952
- Schrittwieser et al., MuZero : https://arxiv.org/abs/1911.08265
- Schrittwieser, description des auteurs de Stochastic MuZero : https://www.julian.ac/blog/2022/05/15/planning-in-stochastic-environments-with-a-learned-model/
