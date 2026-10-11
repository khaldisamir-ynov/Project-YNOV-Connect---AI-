# Données — arborescence du projet AML

Les artefacts ne sont plus sous `data/csv` ou `data/results` à plat : tout est partitionné par **profil**.

## Profils

| Profil | Dossier | Jeu IBM | Rôle |
|--------|---------|---------|------|
| **train** | [`train_data/`](train_data/README.md) | LI-Small | Entraînement, validation, test interne, checkpoints |
| **unknown** | [`unknown_data/`](unknown_data/README.md) | LI-Medium | Hold-out — **jamais vu** à l’entraînement |

Activation dans les scripts : `--data-profile train` (défaut) ou `--data-profile unknown`  
→ implémentation : `src/data_profile.py`, `src/config.py`.

## T-Box partagée (schéma OWL)

Une seule ontologie pour train et hold-out :

`data/train_data/rdf/ontology_schema/aml.ttl`

## Arborescence type

```
data/
├── README.md                 # ce fichier
├── train_data/
│   ├── csv/{raw,cleaned}/
│   ├── rdf/ontology_data/    # A-Box (sous-graphe, enrichie, échantillon 100k)
│   ├── pyg/                  # aml_hetero_graph.pt, aml_graph.pt
│   └── results/              # gnn_*_best.pt, métriques, convergence_*.png
└── unknown_data/
    ├── csv/{raw,cleaned}/
    ├── rdf/ontology_data/
    ├── pyg/
    └── results/              # eval_holdout_metrics.json, holdout_*.png
```

## Fichiers inclus dans le dépôt

| Artefact | Train (LI-Small) | Hold-out (LI-Medium) |
|----------|------------------|----------------------|
| Sous-graphe ego 200k | `train_data/csv/cleaned/LI-Small_Trans_subgraph.csv` | `unknown_data/csv/cleaned/LI-Medium_Trans_subgraph.csv` |
| Graphes PyG | `train_data/pyg/*.pt` | `unknown_data/pyg/*.pt` |
| Checkpoints / métriques | `train_data/results/` | `unknown_data/results/` |
| T-Box OWL | `train_data/rdf/ontology_schema/aml.ttl` | (partagée) |

## CSV IBM bruts (non inclus — ~3,5 Go)

Télécharger sur **Kaggle** (*IBM Transactions for Anti-Money Laundering*) :

- `LI-Small_Trans.csv` → `data/train_data/csv/raw/`
- `LI-Medium_Trans.csv` → `data/unknown_data/csv/raw/`

Puis exécuter `scripts/03_subgraph_sample.py` (ou le pipeline complet dans `README.md`).

## Documentation

- Guide dépôt : [`DEPOT.md`](../DEPOT.md)
- Pipeline complet : [`docs/pipeline_etl_entrainement_evaluation.md`](../docs/pipeline_etl_entrainement_evaluation.md)
- Journal : [`docs/reports/process_log.md`](../docs/reports/process_log.md)
