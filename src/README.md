# Package `src` — code métier

| Module | Responsabilité |
|--------|----------------|
| [`config.py`](config.py) | Chemins, hyperparamètres, namespaces RDF, `set_data_profile()` |
| [`data_profile.py`](data_profile.py) | Profils `train` / `unknown` → chemins `ACTIVE_DATA_ROOT` |
| [`pipeline/`](pipeline/) | `sampling`, `subgraph_sampling` (ego, mode chunked LI-Medium) |
| [`rdf/`](rdf/) | `builder`, `inference`, `features`, `to_homo_pyg`, `to_hetero_pyg` |
| [`gnn/`](gnn/) | `graph_loader` ; `homo/` (GraphSAGE, GAT) ; `hetero/` (HeteroConv) |
| [`evaluation/`](evaluation/) | `metrics_viz`, `convergence_compare`, `holdout_eval`, `holdout_viz` |

Les scripts CLI dans `scripts/` ajoutent la racine du projet au `sys.path` puis importent ces modules.

Documentation : [`docs/architecture.md`](../docs/architecture.md).
