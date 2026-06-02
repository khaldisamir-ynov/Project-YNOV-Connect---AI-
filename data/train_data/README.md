# Données — pipeline AML

Tous les chemins sont définis dans [`src/config.py`](../src/config.py).

| Dossier | Rôle | Versionné Git |
|---------|------|---------------|
| `csv/raw/` | CSV IBM AML brut (`LI-Small_Trans.csv`) | Non (`.gitignore`) |
| `csv/cleaned/` | Échantillons et sous-graphe ego | Non |
| `rdf/ontology_schema/` | T-Box (`aml.ttl`) | **Oui** |
| `rdf/ontology_data/` | A-Box Turtle générées | Non |
| `pyg/` | Tenseurs PyTorch Geometric (`.pt`) | Non |
| `results/` | Checkpoints, métriques JSON, figures PNG | Non |

## Fichiers produits par le pipeline

```
csv/cleaned/LI-Small_Trans_sample.csv      ← 01_sample_data.py
csv/cleaned/LI-Small_Trans_subgraph.csv    ← 03_subgraph_sample.py
rdf/ontology_data/aml_data.ttl            ← 02_build_rdf.py
rdf/ontology_data/aml_data_subgraph.ttl   ← 02b_build_rdf_subgraph.py
rdf/ontology_data/aml_data_subgraph_enriched.ttl  ← 02c_enrich_rdf.py
pyg/aml_hetero_graph.pt                   ← 04_build_pyg.py (défaut)
pyg/aml_graph.pt                          ← 04_build_pyg.py --mode homo
results/gnn_{hetero|sage|gat}_*           ← 05_train_gnn.py
results/convergence_sage_vs_hetero.*      ← 06_compare_convergence.py
```
