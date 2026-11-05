# États et actions de la version 0 2

Les dimensions historiques restent inchangées. Les environnements sont contrôlés depuis la perspective de l’agent, qui joue toujours le même rôle contre un adversaire fixe.

| Environnement | Observation | Actions | Issue terminale |
|---|---|---|---|
| LineWorld | One-hot de longueur 10, position 0 à 9 | 0 gauche, 1 droite | success au but |
| GridWorld | One-hot de longueur 25, index x + y × 5 | 0 haut, 1 bas, 2 gauche, 3 droite | success au but, failure au piège |
| TicTacToe | 9 valeurs : X=1, O=-1, vide=0 | Position 0 à 8 | win, loss ou draw |
| Quarto | 101 valeurs détaillées ci-dessous | Placement 0 à 15 | win, loss ou draw |

## Quarto

- Indices 0 à 79 : 16 cases de 5 valeurs, présence, taille, couleur, forme et remplissage.
- Indices 80 à 84 : mêmes 5 attributs pour la pièce courante.
- Indices 85 à 100 : disponibilité des 16 pièces.

La disponibilité exclut la pièce courante déjà réservée. Une case vide possède cinq zéros ; la présence distingue donc une pièce dont tous les attributs valent zéro d’une case vide. Une victoire demande une ligne complète de quatre pièces partageant au moins un attribut, y compris un attribut égal à zéro.

L’agent contrôle uniquement le placement. La pièce offerte à l’adversaire est choisie automatiquement par une heuristique de sécurité. L’adversaire place sa pièce et choisit la prochaine pièce de l’agent avant le prochain step. Cette réduction ne correspond pas au contrôle complet de toutes les décisions de Quarto.

## Masques et transitions

`get_valid_actions(state)` retourne les indices d’action légaux. Une fin réelle retourne un ensemble vide. Les actions invalides lèvent une erreur sans modifier le plateau. Un appel à step après terminaison exige un reset.

La boucle conserve séparément la terminaison réelle et la limite de pas. Les masques du successeur sont transmis à l’agent et sauvegardés dans le replay. Les dimensions, les actions et la fin sont vérifiées dans les tests.

LineWorld comporte un coût de -0,1 hors but et +1 au but. GridWorld utilise le shaping historique de distance, avec +10 au but et -10 au piège. Les anciennes descriptions qualifiant LineWorld de récompense uniquement terminale doivent être lues avec cette correction.
