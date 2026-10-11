# Architecture technique — scripts et modules

## Scripts CLI (`scripts/`)

| # | Script | Profil | Sortie principale (train par défaut) |
|---|--------|--------|--------------------------------------|
| 01 | `01_sample_data.py` | train | `data/train_data/csv/cleaned/LI-Small_Trans_sample.csv` |
| 02 | `02_build_rdf.py` | train | `data/train_data/rdf/ontology_data/aml_data.ttl` |
| 02b | `02b_build_rdf_subgraph.py` | train \| unknown | `aml_data_subgraph.ttl` |
| 02c | `02c_enrich_rdf.py` | train \| unknown | `aml_data_subgraph_enriched.ttl` |
| 03 | `03_subgraph_sample.py` | train \| unknown | `*_Trans_subgraph.csv` (ego 200k) |
| 04 | `04_build_pyg.py` | train \| unknown | `pyg/aml_hetero_graph.pt`, `aml_graph.pt` |
| 05 | `05_train_gnn.py` | train | `data/train_data/results/gnn_*_best.pt` |
| 06 | `06_compare_convergence.py` | train | `data/train_data/results/convergence_*` |
| 07 | `07_eval_holdout.py` | unknown | `data/unknown_data/results/eval_holdout_metrics.json` |
| 08 | `08_run_unknown_holdout_pipeline.py` | unknown | Enchaîne 02b→02c→04→07 |
| 09 | `09_visualize_holdout.py` | unknown | `data/unknown_data/results/holdout_*.png` |

Flags utiles : `--data-profile train|unknown`, `--mode hetero|homo`, `--holdout-eval`, `--model hetero|sage|gat`.

## Modules `src/`

| Module | Fichiers clés |
|--------|----------------|
| `config.py` | Chemins, hyperparamètres GNN, `set_data_profile()` |
| `data_profile.py` | Profils train / unknown |
| `pipeline/` | `sampling.py`, `subgraph_sampling.py` (chunked Medium) |
| `rdf/` | `builder`, `inference`, `features`, `to_hetero_pyg`, `to_homo_pyg` |
| `gnn/` | `graph_loader` ; `homo/` (SAGE, GAT) ; `hetero/` (HeteroConv) |
| `evaluation/` | `metrics_viz`, `convergence_compare`, `holdout_eval`, `holdout_viz` |

## Tâche de classification

| Modèle | Granularité | Label |
|--------|-------------|-------|
| **HeteroGNN** | Nœud `transaction` | `isLaundering` (0/1) |
| **SAGE / GAT** | Arête compte→compte | Fraude sur la transaction modélisée comme arête |

## Hold-out — alignement features

Sur LI-Medium, le build PyG utilise `--holdout-eval` : `hetero_feature_names_from_train()` + `align_feature_bundle()` pour caler les dimensions sur le checkpoint Small (ex. `bank` 2D, pas 9D).

Voir [`pipeline_etl_entrainement_evaluation.md`](pipeline_etl_entrainement_evaluation.md).
