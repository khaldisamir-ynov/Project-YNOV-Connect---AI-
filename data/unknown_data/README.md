# Données hold-out — test final (LI-Medium)

Jeu **non utilisé à l'entraînement** : évaluation de généralisation des checkpoints dans `data/train_data/results/`.

| Étape | Commande | Sortie |
|-------|----------|--------|
| Sous-graphe ego | `python scripts/03_subgraph_sample.py --data-profile unknown` | `csv/cleaned/LI-Medium_Trans_subgraph.csv` |
| Pipeline complet | `python scripts/08_run_unknown_holdout_pipeline.py --skip-subgraph` | RDF + PyG + métriques |
| Évaluation seule | `python scripts/07_eval_holdout.py` | `results/eval_holdout_metrics.json` |

## Paramètres alignés sur train

- **200 000** transactions (`EGO_TARGET_SIZE`)
- **T-Box** : `data/train_data/rdf/ontology_schema/aml.ttl`
- **Seed** : 42 · **K-hops** : 1
- Inférence RDF identique (`02c_enrich_rdf.py`)

## Entrée

- `csv/raw/LI-Medium_Trans.csv` — **31 251 483** transactions (~2,8 Go)

## Sous-graphe ego — résultats (02/06/2026)

Exécution : `python scripts/03_subgraph_sample.py --data-profile unknown`  
Mode **chunked** (morceaux de **50 000** lignes), sans présélection.

### Corpus brut (LI-Medium)

| Indicateur | Valeur |
|------------|--------|
| Transactions totales | 31 251 483 |
| Fraudes | 16 041 (0,051 %) |
| Comptes graines (touchés par une fraude) | 23 510 |
| Comptes après expansion 1-hop | 250 499 (+226 989) |
| Transactions ego (avant plafond) | 12 126 651 |
| Fraudes dans l'ego (avant plafond) | 16 041 (100 % conservées) |

### Échantillon final (`LI-Medium_Trans_subgraph.csv`)

| Indicateur | Valeur |
|------------|--------|
| **Transactions** | **200 000** |
| **Fraudes** | **16 041** (8,02 %) |
| Comptes uniques | 216 008 |
| Degré moyen | 1,85 |
| Degré médian | 1,00 |
| Degré max | 18 306 |
| % comptes de degré = 1 | 62,72 % |

### Comparaison rapide train (LI-Small) vs hold-out (LI-Medium)

| Métrique | Train `LI-Small` | Hold-out `LI-Medium` |
|----------|------------------|----------------------|
| Transactions finales | 200 000 | 200 000 |
| Fraudes | 3 565 (1,78 %) | 16 041 (8,02 %) |
| Comptes uniques | 121 082 | 216 008 |
| Degré moyen | 3,30 | 1,85 |
| % degré = 1 | 32,69 % | 62,72 % |
| Ego avant plafond | 2 451 392 tx | 12 126 651 tx |

> Le hold-out est **plus dense en fraudes** et **plus fragmenté topologiquement** (plus de comptes isolés) que le jeu d'entraînement — à garder en tête pour l'interprétation des métriques GNN.

## Mémoire / exécution

Le script active le mode **chunked** automatiquement au-delà de ~200 Mo.
**Ne pas** charger les 31 M lignes en RAM.

```powershell
# Recommandé (chunked, morceaux de 50k lignes — pas de présélection par défaut)
python scripts/03_subgraph_sample.py --data-profile unknown

# Si la RAM est saturée, réduire encore les morceaux :
python scripts/03_subgraph_sample.py --data-profile unknown --chunksize 25000

# Présélection optionnelle (2M) — seulement si vous avez assez de RAM libre
python scripts/03_subgraph_sample.py --data-profile unknown --presample-size 2000000

# Suite du pipeline (sous-graphe déjà fait)
python scripts/08_run_unknown_holdout_pipeline.py --skip-subgraph
```

## Pipeline hold-out — statut (02/06/2026)

| Étape | Statut | Sortie |
|-------|--------|--------|
| Sous-graphe ego | ✅ | `csv/cleaned/LI-Medium_Trans_subgraph.csv` |
| RDF + inférence | ✅ | `rdf/ontology_data/aml_data_subgraph_enriched.ttl` |
| PyG (`--holdout-eval`) | ✅ | `pyg/aml_hetero_graph.pt`, `pyg/aml_graph.pt` |
| Test final | ✅ | `results/eval_holdout_metrics.json` |

## Résultats test final (200 000 tx, 16 041 fraudes)

| Modèle | PR-AUC | Recall | Precision | F1 |
|--------|--------|--------|-----------|-----|
| **HeteroGNN** | **0,595** | 0,699 | **0,544** | **0,612** |
| GraphSAGE | 0,229 | 0,781 | 0,133 | 0,228 |
| GAT | 0,150 | 0,643 | 0,114 | 0,194 |

### Visualisations (phase E-hold, soutenance)

```powershell
python scripts/09_visualize_holdout.py
```

| Fichier | Rôle |
|---------|------|
| `holdout_metrics_grid.png` | Grille des 4 métriques |
| `holdout_pr_curves_all_models.png` | Courbes Precision-Recall |
| `holdout_confusion_matrices_all_models.png` | Matrices de confusion |
| `holdout_train_vs_holdout_pr_auc.png` | Train (Small) vs hold-out (Medium) |
| `holdout_metrics_summary.csv` | Tableau exportable |

Notebook : [`notebooks/holdout_metrics_evaluation.ipynb`](../../notebooks/holdout_metrics_evaluation.ipynb)

Détail et interprétation : [`docs/pipeline_etl_entrainement_evaluation.md`](../../docs/pipeline_etl_entrainement_evaluation.md), [`docs/analysis/comparaison_modeles_gnn.md`](../../docs/analysis/comparaison_modeles_gnn.md) §2c.

## Modèles évalués

Checkpoints **train** (`data/train_data/results/`) → graphe **unknown** :

- HeteroGNN (`gnn_hetero_best.pt`)
- GraphSAGE (`gnn_sage_best.pt`)
- GAT (`gnn_gat_best.pt`)

**Note technique** : le build hetero utilise `--holdout-eval` pour aligner les dimensions de features (ex. `bank` 2D comme à l’entraînement, pas 9D sur Medium).
