# Scripts — pipeline numéroté

Exécuter depuis la **racine du projet** (`Project/`).

| Script | Étape | Profil | Sortie principale |
|--------|-------|--------|-------------------|
| `01_sample_data.py` | Échantillonnage stratifié 100k | train | `data/train_data/csv/cleaned/LI-Small_Trans_sample.csv` |
| `02_build_rdf.py` | A-Box depuis échantillon | train | `data/train_data/rdf/ontology_data/aml_data.ttl` |
| `03_subgraph_sample.py` | Sous-graphe ego 200k | train \| unknown | `*_Trans_subgraph.csv` |
| `02b_build_rdf_subgraph.py` | A-Box sous-graphe | train \| unknown | `aml_data_subgraph.ttl` |
| `02c_enrich_rdf.py` | Inférence RDF | train \| unknown | `aml_data_subgraph_enriched.ttl` |
| `04_build_pyg.py` | RDF → PyG | train \| unknown | `pyg/aml_hetero_graph.pt` (défaut hetero) |
| `05_train_gnn.py` | Entraînement GNN | train | `data/train_data/results/gnn_*` |
| `06_compare_convergence.py` | Convergence SAGE vs Hetero | train | `data/train_data/results/convergence_*` |
| `07_eval_holdout.py` | Test final (inférence) | unknown | `data/unknown_data/results/eval_holdout_metrics.json` |
| `08_run_unknown_holdout_pipeline.py` | Pipeline hold-out complet | unknown | RDF + PyG + eval |
| `09_visualize_holdout.py` | Figures hold-out | unknown | `data/unknown_data/results/holdout_*.png` |

## Pipeline GNN train (défaut)

```powershell
python scripts/03_subgraph_sample.py --data-profile train
python scripts/02b_build_rdf_subgraph.py --data-profile train
python scripts/02c_enrich_rdf.py --data-profile train
python scripts/04_build_pyg.py --data-profile train --mode hetero
python scripts/05_train_gnn.py --data-profile train --model hetero
```

## Hold-out

```powershell
python scripts/08_run_unknown_holdout_pipeline.py --skip-subgraph
python scripts/09_visualize_holdout.py
```

## Modules associés (`src/`)

- `src/pipeline/` — échantillonnage CSV
- `src/rdf/` — construction et conversion RDF
- `src/gnn/` — modèles et entraînement
- `src/evaluation/` — métriques, convergence, hold-out (`holdout_eval`, `holdout_viz`)

Voir [`docs/architecture.md`](../docs/architecture.md).
