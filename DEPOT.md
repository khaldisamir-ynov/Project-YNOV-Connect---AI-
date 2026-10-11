# Guide de dépôt — Connect AI (AML / GNN)

**Auteur** : Samir KHALDI  
**Projet** : Détection de blanchiment d'argent par GNN (anomalie structurelle)  
**Dataset** : IBM AML — LI-Small (train) / LI-Medium (hold-out)

Ce document décrit la **structure du projet livrable** et ce qui a été **conservé ou retiré** pour un dépôt clair et léger (~500 Mo au lieu de ~5 Go).

---

## Contenu du dépôt (à inclure dans le zip)

| Dossier / fichier | Rôle |
|-------------------|------|
| `README.md` | Point d'entrée — installation et pipelines |
| `DEPOT.md` | Ce guide |
| `requirements.txt` | Dépendances Python |
| `scripts/` | Pipeline CLI `01` → `09` |
| `src/` | Code métier (RDF, PyG, GNN, évaluation, app) |
| `app/` + `.streamlit/` | Démo Streamlit (HeteroGNN) |
| `docs/` | Documentation complète (rapport, pipeline, analyse) |
| `notebooks/` | `metrics_evaluation.ipynb`, `holdout_metrics_evaluation.ipynb` |
| `project_presentation/` | Slides PDF, notes orateur, structure oral |
| `data/train_data/rdf/ontology_schema/aml.ttl` | T-Box OWL partagée |
| `data/*/README.md` | Documentation des jeux de données |
| `data/*/csv/cleaned/*_subgraph.csv` | Sous-graphes ego 200k (train + hold-out) |
| `data/*/pyg/*.pt` | Graphes PyTorch Geometric prêts à l'emploi |
| `data/*/results/` | Checkpoints, métriques JSON, figures clés |
| `archive/` | Référence baseline tabulaire (projet antérieur, hors périmètre AML) |

**Ne pas inclure** dans le zip de soumission : dossier `.git/`, `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`.

---

## Arborescence simplifiée

```
Project/
├── README.md
├── DEPOT.md
├── requirements.txt
├── docs/                    # Rapport final, pipeline, comparaison modèles
├── project_presentation/    # Soutenance (PDF + scripts oraux)
├── notebooks/               # Visualisation des métriques
├── scripts/                 # 01_sample → 09_visualize_holdout
├── src/                     # Modules Python
├── app/                     # Interface Streamlit
├── data/
│   ├── train_data/          # LI-Small — entraînement
│   │   ├── csv/cleaned/LI-Small_Trans_subgraph.csv
│   │   ├── pyg/aml_hetero_graph.pt, aml_graph.pt
│   │   ├── rdf/ontology_schema/aml.ttl
│   │   └── results/         # gnn_*_best.pt, métriques, convergence
│   └── unknown_data/        # LI-Medium — hold-out
│       ├── csv/cleaned/LI-Medium_Trans_subgraph.csv
│       ├── pyg/
│       └── results/         # eval_holdout_metrics.json, figures
└── archive/                 # Baseline tabulaire (contexte)
```

---

## Fichiers volumineux retirés (régénérables)

Ces fichiers ont été **supprimés localement** pour alléger le dépôt. Ils peuvent être recréés avec le pipeline.

| Fichier retiré | Taille approx. | Comment le régénérer |
|----------------|----------------|----------------------|
| `LI-Small_Trans.csv` (brut) | ~620 Mo | Télécharger IBM AML LI-Small → `data/train_data/csv/raw/` |
| `LI-Medium_Trans.csv` (brut) | ~2,8 Go | Télécharger IBM AML LI-Medium → `data/unknown_data/csv/raw/` |
| RDF A-Box intermédiaires (`aml_data*.ttl`) | ~400 Mo | `scripts/02b_build_rdf_subgraph.py` + `02c_enrich_rdf.py` |
| CSV nettoyés / échantillon 100k | ~630 Mo | `scripts/01_sample_data.py` (étape optionnelle, remplacée par ego) |
| Figures hold-out redondantes | ~0,5 Mo | `scripts/09_visualize_holdout.py` |
| Caches `__pycache__`, checkpoints Jupyter | variable | Régénérés à l'exécution |
| `COURS_COMPLET_ML.md` (archive) | ~24 Ko | Matériel pédagogique hors périmètre — retiré |

---

## Obtenir les données IBM AML

1. Télécharger sur **Kaggle** : *IBM Transactions for Anti-Money Laundering (AML)*  
   Jeu **LI** (Lower Illicit) : `LI-Small_Trans.csv` et `LI-Medium_Trans.csv`.
2. Placer les fichiers :
   - `data/train_data/csv/raw/LI-Small_Trans.csv`
   - `data/unknown_data/csv/raw/LI-Medium_Trans.csv`
3. Relancer le pipeline complet (voir `README.md`).

**Sans les CSV bruts**, le dépôt reste **démontrable** :
- notebooks de métriques (JSON + figures présents),
- app Streamlit (graphes `.pt` + checkpoint Hetero),
- hold-out (sous-graphe 200k + métriques).

---

## Démarrage rapide (relecteur / jury)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Démo interface
streamlit run app/streamlit_app.py

# Figures train / hold-out
jupyter notebook notebooks/metrics_evaluation.ipynb
jupyter notebook notebooks/holdout_metrics_evaluation.ipynb
```

---

## Documents essentiels à lire

| Priorité | Fichier |
|----------|---------|
| 1 | [`docs/reports/final_report.md`](docs/reports/final_report.md) |
| 2 | [`docs/pipeline_etl_entrainement_evaluation.md`](docs/pipeline_etl_entrainement_evaluation.md) |
| 3 | [`docs/analysis/comparaison_modeles_gnn.md`](docs/analysis/comparaison_modeles_gnn.md) |
| 4 | [`project_presentation/slides/Connect AI - Soutenance GNN & AML.pdf`](project_presentation/slides/Connect%20AI%20-%20Soutenance%20GNN%20%26%20AML.pdf) |

---

## Résultats clés (rappel)

| Jeu | Modèle | PR-AUC | Recall | Precision | F1 |
|-----|--------|--------|--------|-----------|-----|
| Test LI-Small | HeteroGNN | ~0,43 | ~75 % | ~22 % | ~0,34 |
| Hold-out LI-Medium | HeteroGNN | **0,595** | ~70 % | **54 %** | **0,612** |

Checkpoints : `data/train_data/results/gnn_hetero_best.pt`
