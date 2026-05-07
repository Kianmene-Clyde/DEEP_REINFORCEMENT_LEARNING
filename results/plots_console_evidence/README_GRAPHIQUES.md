# Graphiques DRL — meilleurs agents et hyperparamètres

## Fichiers utilisés
- `rankings` : `console_rankings_complete_envs.csv`
- `quarto` : `console_quarto_partial_ranking.csv`
- `summary` : `best_agents_summary_from_console.csv`

## Meilleurs agents par environnement
- **LineWorld** : **TabularQL** | score=0.45 | reward=0.2 | win=100.0% | temps=0.009 ms | params=`lr=0.3 | gamma=0.95 | eps_decay=0.999`
- **GridWorld** : **TabularQL** | score=10.88 | reward=10.63 | win=100.0% | temps=0.014 ms | params=`lr=0.1 | gamma=0.99 | eps_decay=0.995`
- **TicTacToe** : **REINFORCE+Mean** | score=1.2181 | reward=0.975 | win=98.5% | temps=0.054 ms | params=`lr=0.001 | gamma=0.99 | hidden=[128,128]`
- **Quarto** : **RandomRollout** | score=1.2288 | reward=0.99 | win=99.0% | temps=2245.226 ms | params=`budget=10 | depth=25` — **résultat partiel**

## Graphiques clés
- `00_PREUVE_meilleur_agent_par_environnement.png`
- `03_ranking_detaille_<ENV>_score.png`
- `06_tableau_hyperparametres_<ENV>.png`
- `07_compromis_score_temps_<ENV>.png`
- `08_marge_du_gagnant_sur_le_2e.png`

## Limite
Le classement Quarto est marqué partiel car il provient de la reconstruction console d'un entraînement non terminé jusqu'au résumé global final.