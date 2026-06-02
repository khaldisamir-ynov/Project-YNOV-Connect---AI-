# Connect AI — Détection de Fraude AML (GNN vs Tabulaire)

> **Problématique** — Dans quelle mesure l'intégration de la topologie d'un
> réseau financier via un réseau de neurones sur graphe (GNN) permet-elle
> d'améliorer la détection de schémas de blanchiment d'argent complexes par
> rapport aux approches d'apprentissage supervisé basées sur des données
> tabulaires ?

## Dataset

IBM Transactions for Anti-Money Laundering (AML) — fichier
`LI-Small_Trans.csv` (~620 Mo, ~6,9 M de transactions synthétiques).

## Documentation

Index complet : [`docs/README.md`](docs/README.md)

| Document | Contenu |
|----------|---------|
| [`docs/guides/guide_detaille.md`](docs/guides/guide_detaille.md) | Guide pédagogique pas à pas, glossaire, préparation jury |
| [`docs/guides/concepts_explained.md`](docs/guides/concepts_explained.md) | RDF, GNN, ego-sampling, métriques |
| [`docs/reports/process_log.md`](docs/reports/process_log.md) | Journal de bord (QUOI / POURQUOI / COMMENT) |
| [`docs/reports/final_report.md`](docs/reports/final_report.md) | Rapport final, réponse à la problématique |
| [`docs/analysis/comparaison_modeles_gnn.md`](docs/analysis/comparaison_modeles_gnn.md) | GraphSAGE / GAT / HeteroGNN + convergence |

## Arborescence

```
Project/
├── README.md
├── requirements.txt
├── docs/                    # Documentation (guides, reports, analysis)
├── notebooks/
│   └── metrics_evaluation.ipynb
├── scripts/                 # Pipeline CLI 01 → 06 (voir scripts/README.md)
├── src/                     # Code métier (voir src/README.md)
│   ├── config.py
│   ├── pipeline/            # Échantillonnage CSV
│   ├── rdf/                 # RDF → PyG
│   ├── gnn/                 # Modèles homo + hetero
│   └── evaluation/          # Métriques et convergence
├── data/                    # Données et artefacts (voir data/README.md)
│   ├── csv/{raw,cleaned}/
│   ├── rdf/{ontology_schema,ontology_data}/
│   ├── pyg/                 # Graphes PyTorch Geometric (.pt)
│   └── results/             # Checkpoints, JSON, figures
└── archive/                 # Travaux YNOV hors périmètre AML
    └── baseline-anomalie-ponctuelle/
```

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Pipeline

### Étape 1 — Sous-échantillonnage stratifié

```powershell
python scripts/01_sample_data.py
```

### Étape 2 — Génération de l'A-Box RDF

```powershell
python scripts/02_build_rdf.py
```

### Pipeline GNN (défaut — multi-relationnel)

```powershell
python scripts/03_subgraph_sample.py
python scripts/02b_build_rdf_subgraph.py
python scripts/02c_enrich_rdf.py
python scripts/04_build_pyg.py
python scripts/05_train_gnn.py
```

Résultats : `data/results/gnn_hetero_best.pt`, `gnn_hetero_metrics.json`.

### Visualisation et convergence

```powershell
pip install matplotlib seaborn
jupyter notebook notebooks/metrics_evaluation.ipynb
python scripts/06_compare_convergence.py --epochs 40 --retrain
```

Détails RDF : [`docs/guides/concepts_explained.md`](docs/guides/concepts_explained.md).

### Pipeline homogène (legacy — comparaison)

```powershell
python scripts/04_build_pyg.py --mode homo
python scripts/05_train_gnn.py --model sage
```

## Modèle de graphe (T-Box)

```
(Account) --sends_via--> (Transaction) --deposits_to--> (Account)
    |                                                       |
    +-------------------- hosted_by --------------------> (Bank)
```

Schéma : `data/rdf/ontology_schema/aml.ttl`

## Statut

Étapes 0 à 6 terminées.  
→ Synthèse : [`docs/reports/final_report.md`](docs/reports/final_report.md)
