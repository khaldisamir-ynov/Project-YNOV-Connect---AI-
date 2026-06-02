# Rapport final — Détection de Fraude AML : Anomalie Ponctuelle vs Anomalie Structurelle

**Projet** — Connect AI / YNOV M1 IA DATA  
**Auteur** — Samir KHALDI  
**Date** — 01/06/2026 — *complété hold-out 02/06/2026*

> Document de synthèse (étape 6). Complète le journal de bord
> (`process_log.md`) et le guide conceptuel (`concepts_explained.md`).

---

## 1. Introduction

La détection de fraude financière recouvre deux réalités distinctes.

D'un côté, des **anomalies ponctuelles** : une transaction isolée trahie par
ses propres attributs (montant aberrant, combinaison improbable de features).
De l'autre, des **anomalies structurelles** : aucune transaction n'est
suspecte prise seule, mais l'**arrangement géométrique** de plusieurs flux
révèle un schéma de blanchiment (fractionnement, relais, cycles).

Ce projet démontre que ces deux familles d'anomalies exigent **deux paradigmes
d'apprentissage radicalement différents** :

| Paradigme | Représentation | Modèle | Illustration dans le parcours |
|---|---|---|---|
| **Anomalie ponctuelle** | Table (lignes indépendantes) | Random Forest, LR | Projet ML antérieur (Credit Card Fraud, Kaggle) |
| **Anomalie structurelle** | Graphe (triplets RDF → HeteroData PyG) | HeteroGNN (HeteroConv) | Présent projet (IBM AML) |

---

## 2. Problématique

> *Dans quelle mesure l'intégration de la topologie d'un réseau financier
> via un réseau de neurones sur graphe (GNN) permet-elle d'améliorer la
> détection de schémas de blanchiment d'argent complexes par rapport aux
> approches d'apprentissage supervisé basées sur des données tabulaires ?*

### Méthode de réponse retenue

La comparaison est **conceptuelle et qualitative**, pas numérique stricte :

- L'approche tabulaire a été couverte dans un **projet antérieur** sur le
  dataset Kaggle *Credit Card Fraud Detection* (`Activite_10.ipynb`).
- L'approche structurelle a été implémentée sur le dataset **IBM AML**
  (`LI-Small_Trans.csv`, ~6,9 M transactions synthétiques).

Les deux datasets étant différents (nature des fraudes, features, volumétrie),
confronter des PR-AUC ou des recall **n'aurait aucun sens scientifique**.
On compare plutôt **ce que chaque paradigme peut voir — et ce qu'il ne peut
pas voir, par construction**.

---

## 3. Approche 1 — Anomalie ponctuelle (référence externe)

### Contexte

Lors du projet de Machine Learning antérieur, un pipeline tabulaire complet
a été entraîné sur `creditcard.csv` (284 807 transactions, ~0,17 % de
fraudes) :

- **Modèles** : Régression Logistique, Random Forest, Isolation Forest,
  Local Outlier Factor, pipeline `StandardScaler + SMOTE + Random Forest`.
- **Métriques** : precision, recall, F1, AUC-ROC, matrice de confusion.
- **Conclusion validée** : le Random Forest surpasse la régression logistique
  sur le recall et l'AUC-ROC ; le pipeline SMOTE + RF est le plus robuste
  face au déséquilibre extrême.

### Ce que cette approche voit

Une transaction est analysée **indépendamment** des autres. Le modèle
détecte des signaux **locaux** :

- Montant anormal pour le profil du titulaire.
- Combinaison improbable de composantes PCA (V1–V28) et de montant.
- Transaction isolée statistiquement aberrante.

### Limite fondamentale

Un Random Forest **ne voit jamais le voisinage** d'une transaction. Un
compte qui reçoit 50 virements de 5 000 € en une semaine puis les renvoie
immédiatement apparaît, pour le modèle, comme 50 transactions normales prises
une par une. **La topologie des flux est invisible.**

C'est précisément cette limite que le présent projet cherche à combler.

---

## 4. Approche 2 — Anomalie structurelle (présent projet)

### Pipeline réalisé

```
CSV sous-graphe ego (200k)
        │
        ▼
aml_data_subgraph.ttl              ← A-Box assertée (02b)
        │
        ▼
aml_data_subgraph_enriched.ttl     ← inférence RDF (02c)
        │   +110 716 paires transactedWith
        │   +3 312 triplets sameBankAs
        ▼
aml_hetero_graph.pt                ← HeteroData (04, mode hetero)
        │
        ▼
HeteroGNN (HeteroConv)             ← entraînement multi-relationnel (05)
```

Le pipeline exploite **trois avantages du RDF** documentés dans
`concepts_explained.md` §0b :

1. **Multi-relationnel** — chaque prédicat a ses propres poids (équivalent R-GCN).
2. **Inférence** — relations métier matérialisées avant le GNN.
3. **Features dynamiques** — extraction automatique depuis la T-Box.

Un mode homogène (GraphSAGE sur `aml_graph.pt`) reste disponible pour comparaison.

### 4.1 Modélisation sémantique (Web Sémantique)

Une **T-Box** (`aml.ttl`) définit le vocabulaire : classes (`Account`,
`Bank`, `Transaction`), relations (`sends_via`, `deposits_to`, `hosted_by`),
attributs (`amountPaid`, `isLaundering`, etc.).

L'**A-Box** générée (`aml_data.ttl`, ~50 Mo) matérialise 100 000 transactions
en triplets RDF, validés par requêtes SPARQL (3 565 fraudes retrouvées à
l'identique du CSV).

**Apport** : la topologie est **explicite et interrogeable** — on peut
suivre des chemins `Source → Transaction → Destination` par SPARQL, ce qui
est impossible avec un CSV brut.

### 4.2 Inférence RDF et construction du graphe hétérogène

Le sampling stratifié naïf (100k transactions) produisait un graphe
**fragmenté** (~70 % de comptes de degré 1). **Correction** : sous-graphe
ego (K=1 hop) — degré moyen **3,30**, 36 % de nœuds de degré 1.

**Enrichissement par inférence** (`02c_enrich_rdf.py`) : matérialisation de
`transactedWith` (comptes ayant transigé) et `sameBankAs` (même banque +
lien transactedWith). Le GNN reçoit un graphe **enrichi** (+224 744 triplets).

Le graphe **HeteroData** résultant préserve la sémantique RDF :

```
HeteroData(
  account.x     = [121 082, 5]     # features structurelles + T-Box
  transaction.x = [200 000, 27]  # features dynamiques (T-Box)
  bank.x        = [4 126, 2]       # bankCode
  relations     = 10 types         # 5 prédicats × 2 directions
  transaction.y = [200 000]        # is_laundering (3 565 positives)
)
```

Relations distinctes : `sends_via`, `deposits_to`, `hosted_by`,
`transacted_with` (inféré), `same_bank_as` (inféré).

### 4.3 Entraînement GNN — HeteroGNN (HeteroConv)

**Architecture** :

```
Account ──sends_via──▶ Transaction ──deposits_to──▶ Account
   │                      ▲                              │
   └──hosted_by──▶ Bank    │         transacted_with ─────┘
                           │         same_bank_as
                    HeteroConv (poids distincts par prédicat)
                           │
                    concat(h_tx, features_T-Box) ──▶ MLP ──▶ fraude ?
```

- **Tâche** : classification de nœuds `transaction` (équivalent sémantique
  à l'edge classification homogène).
- **Loss** : `BCEWithLogitsLoss` avec `pos_weight ≈ 55`.
- **Early stopping** sur val PR-AUC (patience = 15).
- **Comparaison legacy** : GraphSAGE homogène (PR-AUC comparable).

### 4.4 Résultats (jeu TEST — IBM AML, HeteroGNN)

| Métrique | HeteroGNN | GraphSAGE homogène (legacy) |
|---|---|---|
| **PR-AUC** | **0,254** | 0,254 |
| **Recall** | **94,8 %** | 92,7 % |
| **Precision** | 5,9 % | 6,0 % |
| **F1 (fraude)** | 0,112 | 0,113 |

**Lecture AML** : performances comparables au mode homogène, avec une
architecture **sémantiquement plus riche** (multi-relationnel + inférence).
Le modèle privilégie le **recall** — cohérent avec le contexte opérationnel.

### Ce que cette approche voit (et le tabulaire ne voit pas)

Grâce au *message passing*, l'embedding d'un compte intègre l'information
de ses voisins (à 2 sauts avec 2 couches GraphSAGE). Le classificateur
d'arêtes combine alors :

- Le profil **topologique** des comptes source et destination.
- Les **attributs locaux** de la transaction (montant, format, devise).

Motifs théoriquement détectables :

| Motif | Signature topologique | Visible tabulaire ? |
|---|---|---|
| **Fan-in** (regroupement) | Compte avec degré entrant élevé | ❌ |
| **Fan-out** (fractionnement) | Compte avec degré sortant élevé | ❌ |
| **Pass-through** (relais) | Compte avec entrées ≈ sorties rapides | ❌ |
| **Cycle** | Chemin fermé revenant au compte d'origine | ❌ |
| Montant aberrant isolé | Feature locale de l'arête | ✅ (RF) |

---

## 5. Comparaison conceptuelle des deux paradigmes

| Dimension | Anomalie ponctuelle (RF) | Anomalie structurelle (GNN) |
|---|---|---|
| **Représentation** | Table CSV | Graphe RDF enrichi + HeteroData PyG |
| **Unité d'analyse** | Ligne (transaction isolée) | Nœud transaction dans son contexte multi-relationnel |
| **Hypothèse** | Transactions i.i.d. | Structure de réseau exploitable |
| **Force** | Rapide, interprétable, efficace sur anomalies locales | Capture les motifs géométriques (cycles, relais) |
| **Limite** | Aveugle à la topologie | Coût computationnel, sensible à la qualité/connexité du graphe |
| **Cas d'usage** | Fraude carte bancaire (achat suspect isolé) | Blanchiment d'argent (réseau organisé) |
| **Dataset projet** | Kaggle Credit Card | IBM AML |
| **Preuve dans le parcours** | Projet ML antérieur (RF > LR) | HeteroGNN PR-AUC 0,25, recall 95 % |

### Schéma de complémentarité

```
                    SYSTÈME AML MATURE (production)
                              │
              ┌───────────────┴───────────────┐
              │                               │
    ┌─────────▼─────────┐         ┌─────────▼─────────┐
    │  FILTRE 1         │         │  FILTRE 2         │
    │  Tabulaire (RF)   │         │  GNN (HeteroGNN)  │
    │                   │         │                   │
    │  Fraude ponctuelle│         │  Fraude structure │
    │  (montant, profil)│         │  (topologie, cycles│
    │                   │         │   multi-relation) │
    └─────────┬─────────┘         └─────────┬─────────┘
              │                               │
              └───────────────┬───────────────┘
                              │
                    Score de risque consolidé
                    → Revue manuelle par analyste AML
```

Les deux approches **ne sont pas concurrentes** : elles couvrent des
**familles d'anomalies différentes**.

---

## 5b. Test de généralisation (hold-out LI-Medium — 02/06/2026)

### Protocole

Pour compléter la démonstration « structurelle » au-delà du seul LI-Small :

1. **Corpus inconnu** : `LI-Medium_Trans.csv` (~31,3 M transactions) — jamais utilisé à l’entraînement.
2. **Même ETL** : ego K=1, plafond 200 000 tx, 100 % des fraudes conservées, T-Box `aml.ttl` partagée, inférence `transactedWith` / `sameBankAs`.
3. **Inférence seule** : application des checkpoints `gnn_{hetero,sage,gat}_best.pt` entraînés sur Small.
4. **Alignement features** : dimensions hétérogènes calées sur le schéma du checkpoint (`--holdout-eval`).

### Jeu hold-out (chiffres)

| Indicateur | LI-Small (train) | LI-Medium (hold-out) |
|------------|------------------|----------------------|
| Transactions finales | 200 000 | 200 000 |
| Fraudes | 3 565 (1,78 %) | 16 041 (8,02 %) |
| Comptes | 121 082 | 216 008 |
| Degré moyen | 3,30 | 1,85 |

### Résultats

| Modèle | PR-AUC | Recall | Precision | F1 |
|--------|--------|--------|-----------|-----|
| **HeteroGNN** | **0,595** | 0,699 | **0,544** | **0,612** |
| GraphSAGE | 0,229 | 0,781 | 0,133 | 0,228 |
| GAT | 0,150 | 0,643 | 0,114 | 0,194 |

### Interprétation

- Le **HeteroGNN** est le seul modèle à conserver un **bon classement** (PR-AUC) et un **compromis recall/precision** utilisable sur un corpus externe.
- Les GNN **homogènes** sur-alertent (precision ~12 %) : adaptés à un objectif « ne rien manquer », pas à un triage opérationnel sans recalibrage.
- Le PR-AUC hold-out **supérieur** au test interne Small (0,59 vs 0,25) s’explique en partie par un **taux de fraude plus élevé** (8 % vs 1,78 %) et un **graphe plus fragmenté** — ce n’est pas une preuve que Medium est « plus facile », mais que le signal structurel **se transporte** hors échantillon d’entraînement.

**Livrable** : `data/unknown_data/results/eval_holdout_metrics.json`  
**Documentation détaillée** : `docs/pipeline_etl_entrainement_evaluation.md`, `docs/analysis/comparaison_modeles_gnn.md` §2c, `process_log.md` étapes 4b-hold à 5-hold.

---

## 6. Réponse à la problématique

> *Dans quelle mesure l'intégration de la topologie via GNN améliore-t-elle
> la détection des schémas de blanchiment par rapport au tabulaire ?*

### Réponse synthétique

**Dans une mesure significative sur le plan conceptuel et architectural,
mais pas substitutive.**

1. **Le GNN apporte une capacité que le tabulaire n'a pas, par construction.**
   En agrégeant l'information du voisinage via le message passing, il peut
   théoriquement détecter des motifs de blanchiment (fan-in, fan-out, cycles,
   comptes relais) **invisibles** à un Random Forest qui analyse chaque
   transaction isolément.

2. **Cette capacité a été démontrée techniquement de bout en bout** :
   - Modélisation sémantique RDF (T-Box + A-Box, SPARQL).
   - Inférence ontologique (`transactedWith`, `sameBankAs`).
   - Graphe hétérogène multi-relationnel (HeteroData + HeteroConv).
   - Signal de détection réel sur LI-Small (PR-AUC 14× supérieur au hasard, recall 95 %).
   - **Généralisation** sur LI-Medium hold-out (HeteroGNN : PR-AUC **0,59**, F1 **0,61**, precision **54 %**).

3. **Le GNN ne remplace pas le tabulaire** — il le **complète**. Le RF reste
   plus adapté aux anomalies ponctuelles (fraude carte, montant aberrant).
   Le GNN cible une autre classe de menaces (blanchiment organisé).

4. **Les performances absolues restent perfectibles** (PR-AUC 0,25,
   precision 6 %). Axes d'amélioration identifiés :
   - Sous-graphe ego K=2 ou neighborhood sampling PyG.
   - Features temporelles parsées depuis `timestamp`.
   - Inférence OWL-RL avancée (`--owlrl`, règles métier supplémentaires).
   - Optimisation du seuil de décision via courbe PR.

### Formulation pour la soutenance

> *L'intégration de la topologie via GNN ne "bat" pas le tabulaire sur les
> mêmes métriques — elle **adresse une classe d'anomalies différente** que
> le tabulaire ne peut pas voir. Le projet démontre que cette approche est
> techniquement viable (pipeline RDF → inférence → HeteroData → HeteroGNN)
> et produit un signal de détection réel sur le dataset IBM AML, ouvrant la
> voie à un système AML combinant les deux paradigmes en production.*

---

## 7. Limites du projet

| Limite | Impact | Piste d'amélioration |
|---|---|---|
| Datasets différents (Kaggle vs IBM) | Pas de benchmark numérique inter-approches | Comparaison conceptuelle uniquement (choix acté) |
| 36 % de nœuds de degré 1 | Message passing limité | Ego K=2, `NeighborLoader` |
| Pas de features temporelles parsées | Motifs temporels non captés | Parser `timestamp` → heure/jour dans `src/rdf/features.py` |
| Inférence limitée à 2 règles | Graphe enrichi partiel | OWL-RL, règles SWRL, GraphDB en production |
| Precision faible (6 %) | Nombreuses fausses alertes | Tuning du seuil, focal loss |
| CPU uniquement | Limite les recherches d'hyperparamètres | GPU pour itérations plus larges |

---

## 8. Conclusion

Ce projet démontre que la détection de fraude financière ne se réduit pas à
une unique approche. Deux paradigmes coexistent :

- **L'anomalie ponctuelle** (tabulaire / Random Forest) : efficace, rapide,
  interprétable — mais aveugle à la structure des flux.
- **L'anomalie structurelle** (graphe RDF / GNN) : capable de modéliser et
  d'exploiter la topologie d'un réseau financier — mais plus coûteuse et
  sensible à la qualité du graphe.

Le pipeline complet — de la T-Box OWL à l'A-Box enrichie, du HeteroData au
HeteroGNN multi-relationnel — prouve la **faisabilité technique** de
l'approche structurelle sur des données AML réalistes, en exploitant
pleinement la sémantique RDF (prédicats typés, inférence, features dynamiques).

La valeur ajoutée du GNN ne se mesure pas à un écart de PR-AUC face au RF
(sur des datasets différents), mais à sa capacité à **ouvrir un champ de
détection que le tabulaire ne couvre pas** : les schémas de blanchiment
organisé, où l'anomalie vit dans la géométrie du réseau, pas dans une
transaction isolée.

---

## Annexes

| Document | Contenu |
|---|---|
| [`process_log.md`](process_log.md) | Journal de bord détaillé (étapes 0–6, chiffres, décisions) |
| [`concepts_explained.md`](../guides/concepts_explained.md) | Concepts techniques (RDF, GNN, ego-sampling, métriques) |
| [`guide_detaille.md`](../guides/guide_detaille.md) | Guide pas à pas et glossaire soutenance |
| [`comparaison_modeles_gnn.md`](../analysis/comparaison_modeles_gnn.md) | Comparaison GraphSAGE / GAT / HeteroGNN + hold-out §2c |
| [`pipeline_etl_entrainement_evaluation.md`](../pipeline_etl_entrainement_evaluation.md) | Pipeline ETL → évaluation + test final Medium |
| [`README.md`](../../README.md) | Guide d'exécution du pipeline |
| `data/results/gnn_sage_metrics.json` | Métriques GraphSAGE (train/val/test) |
| `data/rdf/ontology_schema/aml.ttl` | T-Box (schéma ontologique) |
| `data/rdf/ontology_data/aml_data.ttl` | A-Box (instances RDF) |
