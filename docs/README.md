# Documentation — Connect AI / AML

**Source de vérité (exécution et chiffres)** : [`pipeline_etl_entrainement_evaluation.md`](pipeline_etl_entrainement_evaluation.md)  
**Guide de dépôt (zip livrable)** : [`DEPOT.md`](../DEPOT.md)

## Index

| Document | Contenu |
|----------|---------|
| [`pipeline_etl_entrainement_evaluation.md`](pipeline_etl_entrainement_evaluation.md) | Pipeline ETL → entraînement → hold-out, commandes, métriques, figures |
| [`architecture.md`](architecture.md) | Scripts 01–09, modules `src/`, profils `train` / `unknown` |
| [`soutenance.md`](soutenance.md) | Slides, notes orateur, parcours jury |
| [`guides/concepts_explained.md`](guides/concepts_explained.md) | RDF, GNN, ego-sampling, métriques |
| [`guides/guide_detaille.md`](guides/guide_detaille.md) | Guide pas à pas + glossaire (détail pédagogique) |
| [`reports/process_log.md`](reports/process_log.md) | Journal chronologique (décisions, chiffres) |
| [`reports/final_report.md`](reports/final_report.md) | Rapport final, réponse à la problématique |
| [`analysis/comparaison_modeles_gnn.md`](analysis/comparaison_modeles_gnn.md) | GraphSAGE / GAT / HeteroGNN + hold-out |
| [`analysis/oral_nplus1_gnn_rdf.md`](analysis/oral_nplus1_gnn_rdf.md) | Fiche N+1 : écarts, hold-out, RDF par modèle |
| [`figures/hetero_gnn_architecture.md`](figures/hetero_gnn_architecture.md) | Schémas Mermaid architecture HeteroGNN |

## Données et code (hors `docs/`)

| Emplacement | Rôle |
|-------------|------|
| [`data/README.md`](../data/README.md) | Profils `train_data` / `unknown_data` |
| [`scripts/README.md`](../scripts/README.md) | Scripts CLI 01–09 |
| [`src/README.md`](../src/README.md) | Modules Python |
| [`notebooks/`](../notebooks/) | `metrics_evaluation.ipynb` (train), `holdout_metrics_evaluation.ipynb` |

## Parcours recommandé

1. **Vue d’ensemble** → [`pipeline_etl_entrainement_evaluation.md`](pipeline_etl_entrainement_evaluation.md)
2. **Relancer le code** → [`architecture.md`](architecture.md) + [`data/README.md`](../data/README.md)
3. **Comprendre les choix** → [`guides/concepts_explained.md`](guides/concepts_explained.md)
4. **Soutenance** → [`soutenance.md`](soutenance.md) + [`reports/final_report.md`](reports/final_report.md)
5. **Comparer les modèles** → [`analysis/comparaison_modeles_gnn.md`](analysis/comparaison_modeles_gnn.md)

*Dernière harmonisation des chemins : juin 2026 (`data/train_data/`, `data/unknown_data/`).*
