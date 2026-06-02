# Comparaison des modèles GNN — Projet AML

**Auteur** — Samir KHALDI  
**Date** — 01/06/2026  
**Données train** — LI-Small : 200 000 transactions, 3 565 fraudes (1,78 %), seed 42  
**Données hold-out** — LI-Medium : 200 000 transactions, 16 041 fraudes (8,02 %) — voir §2b–2c (résultats ✅ 02/06/2026)

> Document de synthèse comparant **GraphSAGE**, **GAT** et **HeteroGNN**.
> Pour l’analyse **epoch par epoch** (convergence), voir la section 6 et
> `data/results/convergence_sage_vs_hetero.*`.

---

## Sommaire

1. [Les trois architectures](#1-les-trois-architectures)
2. [Conditions expérimentales communes](#2-conditions-expérimentales-communes)
3. [Résultats finaux (jeu TEST)](#3-résultats-finaux-jeu-test)
4. [Analyse par modèle](#4-analyse-par-modèle)
5. [GraphSAGE vs HeteroGNN — lecture comparative](#5-graphsage-vs-heterognn)
6. [Convergence : à partir de quand les courbes divergent ?](#6-convergence)
7. [Recommandations pour la soutenance](#7-soutenance)
8. [Reproduction](#8-reproduction)

---

## 1. Les trois architectures

### GraphSAGE (GNN homogène — legacy)

| Aspect | Détail |
|---|---|
| **Graphe** | `Data` PyG — nœuds = comptes, arêtes = transactions (aplati) |
| **Fichier** | `data/pyg/aml_graph.pt` |
| **Idée** | Agrège les voisins de façon uniforme ; toutes les arêtes ont le **même poids** |
| **Tâche** | Classification d’**arêtes** (`is_laundering` sur chaque transaction) |
| **Module** | `src/gnn/homo/models.py` — `EdgeClassifierGNN` |

```
Compte_A ──(transaction)──▶ Compte_B
         edge_attr = montant, format, devise
```

### GAT (GNN homogène — attention)

| Aspect | Détail |
|---|---|
| **Graphe** | Identique à GraphSAGE |
| **Idée** | Pondère les voisins par **attention** (certains voisins comptent plus) |
| **Tâche** | Edge classification |
| **Module** | Même `EdgeClassifierGNN`, `model_type="gat"` |

### HeteroGNN (GNN hétérogène multi-relationnel — défaut)

| Aspect | Détail |
|---|---|
| **Graphe** | `HeteroData` — nœuds account / transaction / bank |
| **Fichier** | `data/pyg/aml_hetero_graph.pt` |
| **Source** | RDF enrichi (`aml_data_subgraph_enriched.ttl`) |
| **Idée** | **HeteroConv** : transformation **distincte par prédicat RDF** |
| **Tâche** | Classification de nœuds **transaction** |
| **Relations** | `sends_via`, `deposits_to`, `hosted_by`, `transacted_with`, `same_bank_as` (+ inverses) |
| **Module** | `src/gnn/hetero/models.py` — `HeteroTransactionGNN` |

```
Account ──sends_via──▶ Transaction ──deposits_to──▶ Account
   │                        ▲
   └──hosted_by──▶ Bank      │
   └──── transacted_with / same_bank_as ──────────┘
```

---

## 2. Conditions expérimentales communes

| Paramètre | Valeur |
|---|---|
| Échantillon | 200 000 transactions (ego K=1) |
| Fraudes | 3 565 (1,78 %) |
| Split | 70 % train / 15 % val / 15 % test (stratifié, seed 42) |
| `hidden_dim` | 64 |
| `num_layers` | 2 |
| `dropout` | 0,3 |
| `lr` | 0,001 |
| `pos_weight` | ~55 (déséquilibre classes) |
| Loss | `BCEWithLogitsLoss` |
| Critère early stopping | Val **PR-AUC** (patience 15) — sauf étude convergence |

**Note** : HeteroGNN utilise en plus le graphe **enrichi** par inférence RDF
(+224k triplets `transactedWith` / `sameBankAs`). La comparaison mesure donc
**architecture + représentation enrichie** vs **architecture homogène simple**.

### 2b. Jeu hold-out LI-Medium (test final — 02/06/2026)

Sous-graphe produit par `python scripts/03_subgraph_sample.py --data-profile unknown`
(mode chunked, 50k lignes/morceau). Même `EGO_TARGET_SIZE` et T-Box que le train.

| Indicateur | LI-Small (train) | LI-Medium (hold-out) |
|------------|------------------|----------------------|
| Source brute | ~6,9 M tx (échantillon projet) | **31 251 483** tx |
| Transactions finales | 200 000 | 200 000 |
| Fraudes | 3 565 (1,78 %) | **16 041 (8,02 %)** |
| Ego avant plafond | 2 451 392 tx | **12 126 651** tx |
| Comptes uniques | 121 082 | **216 008** |
| Degré moyen | 3,30 | **1,85** |
| Degré médian | 2,00 | **1,00** |
| Degré max | 18 198 | **18 306** |
| % degré = 1 | ~36 % | **62,72 %** |

Fichier : `data/unknown_data/csv/cleaned/LI-Medium_Trans_subgraph.csv`  
Métriques GNN hold-out : `data/unknown_data/results/eval_holdout_metrics.json` ✅ **02/06/2026**

### 2c. Résultats hold-out (inférence, checkpoints train)

**Protocole** : `scripts/07_eval_holdout.py` — modèles entraînés sur LI-Small, graphe LI-Medium (200k tx, 16 041 fraudes, 8,02 %). Features hétérogènes alignées sur le checkpoint (`--holdout-eval`).

| Modèle | PR-AUC | Recall | Precision | F1 | Checkpoint epoch (train) |
|--------|--------|--------|-----------|-----|----------------------------|
| **HeteroGNN** | **0,595** | 0,699 | **0,544** | **0,612** | 40 |
| GraphSAGE | 0,229 | 0,781 | 0,133 | 0,228 | 40 |
| GAT | 0,150 | 0,643 | 0,114 | 0,194 | 3 |

#### Train (test interne 15 %) vs hold-out — HeteroGNN

| Métrique | LI-Small test | LI-Medium hold-out | Commentaire |
|----------|---------------|-------------------|-------------|
| PR-AUC | 0,254 | **0,595** | Meilleur ordonnancement des fraudes sur corpus inconnu |
| Recall | 94,8 % | 69,9 % | Moins de sur-détection qu’au seuil train |
| Precision | 5,9 % | **54,4 %** | Alertes beaucoup plus « propres » en hold-out |
| F1 | 0,112 | **0,612** | Compromis recall/precision nettement meilleur |

#### Message pour la soutenance

1. Sur le **même paradigme** (GNN structurel), le **HeteroGNN** domine SAGE et GAT en hold-out (PR-AUC ×2,6 à ×4 par rapport aux homogènes).
2. Les modèles **homogènes** conservent un recall élevé mais une **precision catastrophique** (~12 %) : inutilisables en production sans recalibrage du seuil.
3. Le hold-out valide que l’**architecture + RDF enrichi** ne sont pas un artefact du seul LI-Small.
4. Ne pas comparer naïvement PR-AUC train vs hold-out sans rappeler le **changement de taux de fraude** (1,78 % → 8,02 %) et de **densité** (degré moy. 3,30 → 1,85).

---

## 3. Résultats finaux (jeu TEST)

| Modèle | PR-AUC | Recall | Precision | F1 | Best epoch |
|---|---|---|---|---|---|
| **HeteroGNN** | **0,254** | **94,8 %** | 5,9 % | 0,112 | 30 |
| **GraphSAGE** | 0,254 | 92,7 % | **6,0 %** | **0,113** | 80 |
| **GAT** | 0,047 | 62,5 % | 3,0 % | 0,056 | 3 |
| *Hasard* | ~0,018 | — | — | — | — |

### Ratio PR-AUC / baseline aléatoire (~0,018)

| Modèle | Facteur vs hasard |
|---|---|
| HeteroGNN | **~14,1×** |
| GraphSAGE | **~14,1×** |
| GAT | **~2,6×** (échec relatif) |

---

## 4. Analyse par modèle

### HeteroGNN — modèle principal

**Points forts**

- **Recall le plus élevé** (94,8 %) : détecte le plus de fraudes sur le test.
- Architecture **alignée sur le RDF** : chaque prédicat = canal d’apprentissage séparé.
- Graphe **enrichi** par inférence (`transactedWith`, `sameBankAs`).
- Convergence plus **rapide** (meilleur epoch = 30 vs 80 pour SAGE).

**Limites**

- Precision ~6 % (nombreuses fausses alertes, comme SAGE).
- PR-AUC finale **quasi identique** à GraphSAGE.
- Coût mémoire / temps supérieur (3 types de nœuds, 10 types d’arêtes).

### GraphSAGE — baseline structurelle homogène

**Points forts**

- Pipeline **simple**, robuste, bien documenté.
- PR-AUC et F1 **équivalents** au HeteroGNN sur le test.
- Meilleure **precision** légère (+0,1 pt) — marginalement moins de faux positifs au seuil 0,5.

**Limites**

- **Aplatit** la sémantique RDF (transaction → arête compte→compte).
- Ne distingue pas `sends_via` vs `transacted_with` vs `same_bank_as`.
- Convergence **plus lente** (80 epochs vs 30).

### GAT — non retenu

**Constat** : PR-AUC ~0,05 sur le test — **proche du hasard**.

**Hypothèses**

- Instabilité sur graphe **sparse** (36 % comptes degré 1).
- Multi-head attention mal adaptée au déséquilibre extrême.
- Early stop très précoce (epoch 3) → sous-apprentissage.

**Décision projet** : GAT documenté comme **contre-exemple**, pas comme baseline.

---

## 5. GraphSAGE vs HeteroGNN — lecture comparative

### Tableau des écarts (TEST, Hetero − SAGE)

| Métrique | GraphSAGE | HeteroGNN | Δ (Hetero − SAGE) |
|---|---|---|---|
| PR-AUC | 0,2543 | 0,2537 | **−0,0006** |
| Recall | 92,7 % | 94,8 % | **+2,1 pts** |
| Precision | 6,0 % | 5,9 % | −0,1 pt |
| F1 | 0,1131 | 0,1115 | −0,0016 |

### Interprétation

1. **Pas de gain massif en PR-AUC** : les deux architectures capturent un signal
   comparable sur ce graphe (limité par la fragmentation et le déséquilibre).

2. **HeteroGNN oriente davantage le recall** : utile en AML si l’on privilégie
   « ne rater aucune fraude » au prix de faux positifs.

3. **L’intérêt du HeteroGNN est surtout architectural** :
   - Fidélité au RDF multi-relationnel.
   - Inférence ontologique intégrée.
   - Extensibilité (nouveaux prédicats / features via T-Box).

4. **Convergence plus rapide** (30 vs 80 epochs) : le graphe enrichi et les
   canaux relationnels peuvent accélérer l’apprentissage du signal de fraude.

### Schéma décisionnel

```
                    Besoin métier
                         │
         ┌───────────────┴───────────────┐
         │                               │
   Maximiser recall AML              Explicabilité RDF /
   (détection)                       évolutivité ontologie
         │                               │
         ▼                               ▼
    HeteroGNN                        HeteroGNN
    (léger +recall)                  (argument soutenance)
         │                               │
         └───────────────┬───────────────┘
                         │
              GraphSAGE suffit si
              simplicité > sémantique RDF
```

---

## 6. Convergence : à partir de quand les courbes divergent ?

### Méthode

Le script `scripts/06_compare_convergence.py` entraîne **GraphSAGE** et **HeteroGNN**
avec les **mêmes hyperparamètres** et **sans early stopping** (ou réutilise
`gnn_sage_history.json` / `gnn_hetero_history.json`), puis :

1. Superpose les courbes **val PR-AUC**, **val F1**, **train loss**.
2. Calcule l’écart **Δ = Hetero − SAGE** à chaque epoch.
3. Détecte la **première epoch** où |Δ PR-AUC| ≥ 0,01.

### Lancer l’analyse

```powershell
python scripts/06_compare_convergence.py --epochs 40 --retrain
```

**Sorties**

| Fichier | Contenu |
|---|---|
| `data/results/convergence_sage_vs_hetero.json` | Métriques par epoch + analyse |
| `data/results/convergence_sage_vs_hetero.png` | Courbes superposées |
| `data/results/convergence_gap_sage_vs_hetero.png` | Écarts Δ par epoch |

### Résultats de l’étude de convergence (40 epochs, sans early stopping)

**Exécution** : `python scripts/06_compare_convergence.py --epochs 40 --retrain`  
**Fichiers** : `data/results/convergence_sage_vs_hetero.json`, `.png`, `_gap_.png`

#### Synthèse automatique

| Indicateur | Valeur | Interprétation |
|---|---|---|
| **1er écart \|Δ PR-AUC\| ≥ 0,01** | **Epoch 1** | Les courbes divergent **dès le départ** |
| **Hetero > SAGE (val PR-AUC)** | **Dès epoch 1** | HeteroGNN jamais dépassé par SAGE sur 40 epochs |
| **Max \|Δ PR-AUC\|** | **0,335 à epoch 40** | Écart maximal en fin de run |
| **Val PR-AUC finale** | SAGE **0,107** / Hetero **0,441** | SAGE encore en sous-apprentissage à 40 epochs |

#### Tableau par epoch (val PR-AUC)

| Epoch | GraphSAGE | HeteroGNN | Δ (Hetero − SAGE) |
|---|---|---|---|
| 1 | 0,022 | 0,085 | +0,063 |
| 5 | 0,055 | 0,147 | +0,092 |
| 10 | 0,067 | 0,183 | +0,115 |
| 15 | ~0,070 | ~0,240 | +0,170 |
| 20 | 0,071 | 0,273 | +0,202 |
| 25 | ~0,080 | ~0,274 | +0,194 |
| 30 | 0,078 | 0,298 | +0,220 |
| 35 | ~0,090 | ~0,370 | +0,280 |
| 40 | 0,107 | 0,441 | +0,335 |

#### À partir de quand la différence devient significative ?

| Seuil \|Δ PR-AUC\| | Première epoch |
|---|---|
| ≥ 0,01 | **1** |
| ≥ 0,05 | **1** |
| ≥ 0,10 | **9** |
| ≥ 0,20 | **19** |

**Point de bascule pratique (epoch ~17–20)** : HeteroGNN atteint **val PR-AUC ≈ 0,25–0,27**
(l’ordre de grandeur du meilleur GraphSAGE avec early stopping à epoch 80),
alors que GraphSAGE n’est encore qu’à **~0,07** sur la validation.

#### Lecture importante — deux niveaux de comparaison

**A. Entraînement aligné 40 epochs (cette étude)**  
→ HeteroGNN **converge beaucoup plus vite** et creuse l’écart progressivement.
GraphSAGE n’a pas atteint son plateau à 40 epochs (il monte encore lentement ;
son meilleur score historique nécessitait **~80 epochs**).

**B. Meilleur checkpoint avec early stopping (métriques section 3)**  
→ Sur le **jeu TEST**, les deux modèles se rejoignent (**PR-AUC ≈ 0,254**).
La différence apparaît surtout en **recall** (+2 pts pour Hetero) et en **vitesse
de convergence**, pas en score final absolu une fois SAGE pleinement entraîné.

#### Graphiques générés

![Convergence superposée](../data/results/convergence_sage_vs_hetero.png)

![Écarts par epoch](../data/results/convergence_gap_sage_vs_hetero.png)

*(Les images sont dans `data/results/` — ouvrir localement si l’aperçu Markdown ne s’affiche pas.)*

---

## 7. Recommandations pour la soutenance

### Message clé

> « Nous comparons trois GNN sur le même sous-graphe AML. GraphSAGE et HeteroGNN
> atteignent un PR-AUC ~0,25 (~14× le hasard). Le HeteroGNN améliore le recall
> (+2 pts) et converge plus vite, au prix d’une complexité RDF multi-relationnelle.
> Le GAT homogène n’est pas adapté à ce graphe sparse. »

### Slides suggérées

1. Tableau section 3 (3 modèles × 4 métriques).
2. Schéma architecture HeteroGNN (section 1).
3. Graphique `convergence_sage_vs_hetero.png`.
4. Graphique `convergence_gap_sage_vs_hetero.png`.
5. Formulation : complémentarité avec le Random Forest tabulaire (projet antérieur).

---

## 8. Reproduction

```powershell
# Graphes
python scripts/04_build_pyg.py --mode homo
python scripts/02c_enrich_rdf.py
python scripts/04_build_pyg.py --mode hetero

# Entraînement
python scripts/05_train_gnn.py --model sage --epochs 80
python scripts/05_train_gnn.py --model hetero --epochs 30
python scripts/05_train_gnn.py --model gat

# Comparaison + convergence
jupyter notebook notebooks/metrics_evaluation.ipynb
python scripts/06_compare_convergence.py --epochs 40 --retrain
```

---

## Documents liés

| Fichier | Rôle |
|---|---|
| [`docs/README.md`](../README.md) | Index de la documentation |
| [`docs/guides/guide_detaille.md`](../guides/guide_detaille.md) | Guide pédagogique complet |
| [`docs/reports/final_report.md`](../reports/final_report.md) | Rapport de synthèse |
| [`notebooks/metrics_evaluation.ipynb`](../../notebooks/metrics_evaluation.ipynb) | Graphiques interactifs (`src/evaluation/`) |
| `data/results/convergence_sage_vs_hetero.json` | Données convergence |

---

*Les métriques finales (section 3–5) proviennent de `gnn_*_metrics.json`.
La section 6 est mise à jour par `06_compare_convergence.py`.*
