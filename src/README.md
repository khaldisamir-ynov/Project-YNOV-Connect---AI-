# Package `src` — code métier

| Module | Responsabilité |
|--------|----------------|
| [`config.py`](config.py) | Chemins, hyperparamètres, namespaces RDF |
| [`pipeline/`](pipeline/) | `sampling`, `subgraph_sampling` |
| [`rdf/`](rdf/) | `builder`, `inference`, `features`, `to_homo_pyg`, `to_hetero_pyg` |
| [`gnn/`](gnn/) | `graph_loader` ; `homo/` (GraphSAGE, GAT) ; `hetero/` (HeteroConv) |
| [`evaluation/`](evaluation/) | `metrics_viz`, `convergence_compare` |

Les scripts CLI dans `scripts/` ajoutent la racine du projet au `sys.path` puis importent ces modules.
