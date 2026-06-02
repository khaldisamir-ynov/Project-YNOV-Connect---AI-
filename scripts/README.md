# Scripts — pipeline numéroté

Exécuter depuis la **racine du projet** (`Project/`).

| Script | Étape | Sortie principale |
|--------|-------|-------------------|
| `01_sample_data.py` | Échantillonnage stratifié | `data/csv/cleaned/LI-Small_Trans_sample.csv` |
| `02_build_rdf.py` | A-Box depuis échantillon | `data/rdf/ontology_data/aml_data.ttl` |
| `03_subgraph_sample.py` | Sous-graphe ego (GNN) | `data/csv/cleaned/LI-Small_Trans_subgraph.csv` |
| `02b_build_rdf_subgraph.py` | A-Box sous-graphe | `aml_data_subgraph.ttl` |
| `02c_enrich_rdf.py` | Inférence RDF | `aml_data_subgraph_enriched.ttl` |
| `04_build_pyg.py` | RDF → PyG | `data/pyg/aml_hetero_graph.pt` (défaut) |
| `05_train_gnn.py` | Entraînement GNN | `data/results/gnn_*` |
| `06_compare_convergence.py` | Convergence SAGE vs Hetero | `data/results/convergence_*` |

## Pipeline GNN (défaut)

```powershell
python scripts/03_subgraph_sample.py
python scripts/02b_build_rdf_subgraph.py
python scripts/02c_enrich_rdf.py
python scripts/04_build_pyg.py
python scripts/05_train_gnn.py
```

## Modules associés (`src/`)

- `src/pipeline/` — échantillonnage CSV
- `src/rdf/` — construction et conversion RDF
- `src/gnn/` — modèles et entraînement
- `src/evaluation/` — métriques et convergence
