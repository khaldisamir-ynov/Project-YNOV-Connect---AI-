# Données d'entraînement — LI-Small

Jeu utilisé pour entraîner les GNN et produire les checkpoints de référence.

| Dossier | Contenu |
|---------|---------|
| `csv/raw/` | `LI-Small_Trans.csv` |
| `csv/cleaned/` | `LI-Small_Trans_subgraph.csv` (200k) |
| `rdf/ontology_schema/` | T-Box `aml.ttl` (**partagée avec unknown**) |
| `rdf/ontology_data/` | A-Box Turtle |
| `pyg/` | `aml_hetero_graph.pt`, `aml_graph.pt` |
| `results/` | `gnn_{hetero,sage,gat}_best.pt`, métriques JSON |

### Sous-graphe ego (LI-Small, 01/06/2026)

| Indicateur | Valeur |
|------------|--------|
| Transactions | 200 000 |
| Fraudes | 3 565 (1,78 %) |
| Comptes uniques | 121 082 |
| Degré moyen / médian | 3,30 / 2,00 |
| % degré = 1 | ~36 % |

### Métriques test interne (LI-Small, split 15 %)

| Modèle | PR-AUC | Recall | Precision | F1 |
|--------|--------|--------|-----------|-----|
| HeteroGNN | 0,254 | 94,8 % | 5,9 % | 0,112 |
| GraphSAGE | 0,254 | 92,7 % | 6,0 % | 0,113 |
| GAT | 0,047 | 62,5 % | 3,0 % | 0,056 |

### Hold-out LI-Medium (généralisation, 02/06/2026)

Checkpoints de ce dossier évalués sur `unknown_data/` **sans ré-entraînement** :

| Modèle | PR-AUC hold-out | F1 |
|--------|-----------------|-----|
| HeteroGNN | **0,595** | **0,612** |
| GraphSAGE | 0,229 | 0,228 |
| GAT | 0,150 | 0,194 |

Détail : [`../unknown_data/README.md`](../unknown_data/README.md), [`../../docs/pipeline_etl_entrainement_evaluation.md`](../../docs/pipeline_etl_entrainement_evaluation.md).

Chemins actifs par défaut : `config.set_data_profile("train")`.
