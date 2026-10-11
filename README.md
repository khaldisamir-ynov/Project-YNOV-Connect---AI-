# Connect AI — Détection de Fraude AML (GNN vs Tabulaire)

> **Problématique** — Dans quelle mesure l'intégration de la topologie d'un
> réseau financier via un réseau de neurones sur graphe (GNN) permet-elle
> d'améliorer la détection de schémas de blanchiment d'argent complexes par
> rapport aux approches d'apprentissage supervisé basées sur des données
> tabulaires ?

## Dataset

IBM Transactions for Anti-Money Laundering (AML) :

- **LI-Small** — entraînement (`data/train_data/`)
- **LI-Medium** — test final hold-out (`data/unknown_data/`)

## Dépôt du projet

Guide détaillé (structure, fichiers retirés, zip à livrer) : **[`DEPOT.md`](DEPOT.md)**

Le dépôt livrable contient le **code**, la **documentation**, les **sous-graphes 200k**, les **graphes PyG**, les **checkpoints** et les **résultats**. Les CSV IBM bruts (~3,5 Go) ne sont pas inclus — voir `DEPOT.md` pour les télécharger et régénérer le pipeline.

## Documentation

Index : [`docs/README.md`](docs/README.md)

| Document | Contenu |
|----------|---------|
| [`docs/pipeline_etl_entrainement_evaluation.md`](docs/pipeline_etl_entrainement_evaluation.md) | Pipeline complet + hold-out |
| [`docs/architecture.md`](docs/architecture.md) | Scripts 01–09, modules `src/` |
| [`docs/soutenance.md`](docs/soutenance.md) | Slides et notes orateur |
| [`docs/reports/final_report.md`](docs/reports/final_report.md) | Rapport de synthèse |

## Arborescence

```
Project/
├── README.md
├── requirements.txt
├── docs/                       # Documentation (source de vérité)
├── project_presentation/       # Slides soutenance (Marp)
├── notebooks/
│   ├── metrics_evaluation.ipynb
│   └── holdout_metrics_evaluation.ipynb
├── scripts/                    # Pipeline CLI 01 → 09
├── app/                        # Interface Streamlit
├── src/                        # Code métier
├── data/
│   ├── README.md
│   ├── train_data/             # LI-Small — entraînement
│   └── unknown_data/           # LI-Medium — hold-out
└── archive/                    # Baseline tabulaire (hors périmètre AML)
```

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Pipeline train (LI-Small)

```powershell
python scripts/03_subgraph_sample.py --data-profile train
python scripts/02b_build_rdf_subgraph.py --data-profile train
python scripts/02c_enrich_rdf.py --data-profile train
python scripts/04_build_pyg.py --data-profile train --mode hetero
python scripts/05_train_gnn.py --data-profile train --model hetero
python scripts/05_train_gnn.py --data-profile train --model sage
python scripts/05_train_gnn.py --data-profile train --model gat
```

Résultats : `data/train_data/results/gnn_hetero_best.pt`, `gnn_hetero_metrics.json`.

### Visualisation et convergence

```powershell
jupyter notebook notebooks/metrics_evaluation.ipynb
python scripts/06_compare_convergence.py --epochs 40 --retrain
```

## Pipeline hold-out (LI-Medium)

```powershell
python scripts/08_run_unknown_holdout_pipeline.py --skip-subgraph
python scripts/09_visualize_holdout.py
```

Résultats : `data/unknown_data/results/eval_holdout_metrics.json`, `holdout_*.png`.

## Application Streamlit (démo HeteroGNN)

Interface pour scorer le graphe, afficher PR-AUC / recall / precision / F1, courbes PR et tableau des transactions.

```powershell
streamlit run app/streamlit_app.py
```

Voir [`app/README.md`](app/README.md).

## Modèle de graphe (T-Box)

```
(Account) --sends_via--> (Transaction) --deposits_to--> (Account)
    |                                                       |
    +-------------------- hosted_by --------------------> (Bank)
```

Schéma : `data/train_data/rdf/ontology_schema/aml.ttl`

## Statut

Étapes 0 à 6 + **test final hold-out** (LI-Medium, 02/06/2026) terminés.  
→ Synthèse : [`docs/reports/final_report.md`](docs/reports/final_report.md)
