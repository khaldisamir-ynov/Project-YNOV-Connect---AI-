# Journal de bord — Projet Connect AI / AML

> **Objectif** : suivre l'avancement du projet étape par étape, avec pour chacune :
> *Objectif → Ce qui a été fait → Résultat → Décisions prises.*
>
> Ce document est **mis à jour à chaque action significative**.

---

## Vue d'ensemble du projet

**Titre** — Détection de Fraude Financière (AML) : Anomalie structurelle via GNN
face à l'anomalie ponctuelle tabulaire.

**Problématique** — Dans quelle mesure l'intégration de la topologie d'un
réseau financier via un GNN améliore-t-elle la détection de schémas de
blanchiment d'argent complexes par rapport aux approches d'apprentissage
supervisé basées sur des données tabulaires ?

**Dataset** — IBM Transactions for Anti-Money Laundering, fichier
`LI-Small_Trans.csv` (~620 Mo, ~6,9 M de transactions synthétiques).

**Pipeline cible** :

```
┌───────────────────────────────────┐         ┌─────────────────────────────────────┐
│  APPROCHE 1 — ANOMALIE PONCTUELLE │         │  APPROCHE 2 — ANOMALIE STRUCTURELLE │
│  (déjà couverte — projet ML       │         │  (objet du présent projet)          │
│   antérieur sur Credit Card)      │         │                                     │
│                                   │         │  CSV IBM AML                        │
│  CSV Kaggle                       │         │     │                               │
│     │                             │         │     ▼                               │
│     ▼                             │         │  [1] Sous-échantillonnage stratifié │
│  Features tabulaires              │         │     │                               │
│     │                             │         │     ▼                               │
│     ▼                             │         │  [2] A-Box RDF (Turtle)             │
│  Random Forest / LR / ...         │         │     │                               │
│     │                             │         │     ▼                               │
│     ▼                             │         │  [4] Conversion RDF → PyG           │
│  Prédiction par ligne isolée      │         │     │                               │
└─────────────┬─────────────────────┘         │     ▼                               │
              │                               │  [5] GNN (GraphSAGE / GAT)          │
              │                               │     │                               │
              └────────────────┬──────────────┴─────┘                               │
                               │                                                    │
                               ▼                                                    │
                  [6] COMPARAISON CONCEPTUELLE                                      │
                      des DEUX APPROCHES :                                          │
                      - Quelle approche pour quel type d'anomalie ?                 │
                      - Limites mutuelles                                           │
                      - Complémentarité possible                                    │
```

---

## Tableau de bord

| # | Étape | Statut | Date | Livrable principal |
|---|-------|--------|------|--------------------|
| 0 | Cadrage et schéma de donnée (T-Box) | ✅ Fait | 29/05/2026 | `data/rdf/ontology_schema/aml.ttl` |
| 1 | Structure projet + dépendances | ✅ Fait | 01/06/2026 | `src/`, `scripts/`, `requirements.txt` |
| 2a | Sous-échantillonnage stratifié | ✅ Fait | 01/06/2026 | `data/csv/cleaned/LI-Small_Trans_sample.csv` |
| 2b | Génération A-Box RDF | ✅ Fait | 01/06/2026 | `data/rdf/ontology_data/aml_data.ttl` |
| 3 | Approche "ponctuelle" (tabulaire) | ♻️ **Réutilisée** (projet ML antérieur) | 19/02/2026 | `archive/baseline-anomalie-ponctuelle/.../Activite_10.ipynb` |
| 4a | Échantillonnage par sous-graphe ego — train (LI-Small) | ✅ Fait | 01/06/2026 | `data/train_data/csv/cleaned/LI-Small_Trans_subgraph.csv` |
| 4a-hold | Sous-graphe ego — hold-out (LI-Medium) | ✅ Fait | 02/06/2026 | `data/unknown_data/csv/cleaned/LI-Medium_Trans_subgraph.csv` |
| 4b-hold | A-Box + inférence RDF (unknown) | ✅ Fait | 02/06/2026 | `data/unknown_data/rdf/ontology_data/aml_data_subgraph_enriched.ttl` |
| 4c-hold | PyG hetero + homo hold-out (`--holdout-eval`) | ✅ Fait | 02/06/2026 | `data/unknown_data/pyg/aml_hetero_graph.pt`, `aml_graph.pt` |
| 5-hold | Test final inférence (3 GNN) | ✅ Fait | 02/06/2026 | `data/unknown_data/results/eval_holdout_metrics.json` |
| 5-hold-viz | Visualisations hold-out (phase E) | ✅ Fait | 02/06/2026 | `data/unknown_data/results/holdout_*.png`, `09_visualize_holdout.py` |
| 4b | Inférence RDF (A-Box enrichie) | ✅ Fait | 01/06/2026 | `aml_data_subgraph_enriched.ttl` |
| 4c | Conversion RDF → HeteroData PyG | ✅ Fait | 01/06/2026 | `data/pyg/aml_hetero_graph.pt` |
| 4b-legacy | Conversion RDF → PyG homogène | ✅ Fait | 01/06/2026 | `data/pyg/aml_graph.pt` |
| 5 | Entraînement HeteroGNN multi-relationnel | ✅ Fait | 01/06/2026 | `data/results/gnn_hetero_best.pt` |
| 6 | Comparaison conceptuelle + rapport final | ✅ Fait | 01/06/2026 | `docs/reports/final_report.md` |
| 6b | Étude convergence SAGE vs Hetero (epoch par epoch) | ✅ Fait | 01/06/2026 | `scripts/06_compare_convergence.py` → `data/results/convergence_*` |

---

## Synthèse — Quoi, Pourquoi, Comment

Ce tableau répond en une page à la question : *qu'est-ce qui a été fait,
pourquoi, et comment le relancer ?*

| Étape | **Quoi** (livré) | **Pourquoi** (raison du choix) | **Comment** (commande) |
|---|---|---|---|
| **0** | T-Box `aml.ttl` | Formaliser le vocabulaire **avant** les données ; base pour SPARQL et validation RDF | Fichier statique — pas de script |
| **1** | `src/`, `scripts/`, `config.py` | Code reproductible, config centralisée, séparation logique / exécution | `pip install -r requirements.txt` |
| **2a** | Échantillon stratifié 100k | Réduire le volume pour le RDF ; garder 100 % des fraudes | `python scripts/01_sample_data.py` |
| **2b** | A-Box Turtle ~50 Mo | **Démontrer le Web Sémantique** : matérialiser la topologie en triplets RDF conformes à la T-Box | `python scripts/02_build_rdf.py` |
| **3** | RF / LR (projet ML ant.) | Incarne l'**anomalie ponctuelle** — pas refaite ici (déjà validée) | Notebook externe |
| **4a** | Sous-graphe ego 200k | Le sampling naïf **casse la connexité** ; un GNN a besoin de voisinage pour le message passing | `python scripts/03_subgraph_sample.py` |
| **4b** | A-Box enrichie (inférence) | Matérialiser `transactedWith`, `sameBankAs` avant le GNN | `python scripts/02c_enrich_rdf.py` |
| **4c** | HeteroData `aml_hetero_graph.pt` | Graphe **multi-relationnel** fidèle à la T-Box | `python scripts/04_build_pyg.py` |
| **5** | HeteroGNN entraîné | **Poids distincts par prédicat RDF** (HeteroConv) | `python scripts/05_train_gnn.py` |
| **6** | Rapport final | Répondre à la problématique ; synthèse pour soutenance | `docs/reports/final_report.md` (lecture) |

### Pipeline complet (dans l'ordre)

```powershell
# Installation (une fois)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Exécution (pipeline GNN — défaut)
python scripts/03_subgraph_sample.py
python scripts/02b_build_rdf_subgraph.py
python scripts/02c_enrich_rdf.py          # inférence RDF
python scripts/04_build_pyg.py              # HeteroData (défaut)
python scripts/05_train_gnn.py              # HeteroGNN (défaut)

# Legacy homogène (comparaison)
python scripts/04_build_pyg.py --mode homo
python scripts/05_train_gnn.py --model sage
```

> **Pipeline RDF complet** : A-Box → inférence → HeteroData → HeteroGNN.
> Voir `concepts_explained.md` §0b pour les trois avantages RDF exploités.

### Où lire quoi ?

| Besoin | Document |
|---|---|
| **Guide pas à pas avec glossaire (soutenance)** | **`docs/guides/guide_detaille.md`** |
| Suivi chronologique, décisions, chiffres | `docs/reports/process_log.md` (ce fichier) |
| Comprendre RDF, GNN, ego-sampling, métriques | `docs/guides/concepts_explained.md` |
| **Synthèse finale, réponse à la problématique** | **`docs/reports/final_report.md`** |
| Lancer les scripts rapidement | `README.md` |

---

## Étape 0 — Cadrage & modélisation conceptuelle (T-Box)

**Date** : 29/05/2026
**Statut** : ✅ Terminée

### Objectif

Définir formellement le **vocabulaire** du domaine (classes, relations, attributs)
**avant** de manipuler des données. Cette définition prend la forme d'une
ontologie OWL/RDF — la **T-Box** (Terminological Box).

### Ce qui a été fait

Rédaction de `data/rdf/ontology_schema/aml.ttl` qui définit :

- **Classes** (types de nœuds) : `Account`, `Bank`, `Transaction`.
- **Object Properties** (relations / arêtes du graphe) avec leurs *domain*
  et *range* :
  - `Account --sends_via--> Transaction`
  - `Transaction --deposits_to--> Account`
  - `Account --hosted_by--> Bank`
- **Datatype Properties** (attributs des nœuds) : `amountPaid` (float),
  `paymentCurrency`, `paymentFormat`, `timestamp`, `isLaundering` (entier 0/1).

### Résultat

Un contrat de données strict, qui sert ensuite de **schéma de validation**
pour la génération de l'A-Box (étape 2b).

### Décisions prises

- **Choix du Web Sémantique** plutôt qu'un modèle relationnel pur, pour deux
  raisons :
  1. Permet d'utiliser des requêtes **SPARQL** pour explorer la topologie.
  2. Force une discipline de modélisation (les `domain`/`range` font office
     de "douane" qui empêche d'envoyer une arête au mauvais type d'entité).
- **Identité d'un Account** = `(banque, numéro de compte)` et non le numéro
  seul, parce qu'un même numéro peut exister dans plusieurs banques.
- **`timestamp` laissé en `xsd:string`** pour cette première itération
  (parsing différé en étape 1).

---

## Étape 1 — Structure projet & dépendances

**Date** : 01/06/2026
**Statut** : ✅ Terminée

### Objectif

Mettre en place une base de code propre, réutilisable et reproductible, avec
une séparation nette entre *logique métier* (`src/`) et *points d'entrée*
(`scripts/`).

### Ce qui a été fait

Arborescence créée (mise à jour après étapes 4–5) :

```
Project/
├── docs/{guides,reports,analysis}/
├── data/{csv,rdf,pyg,results}/
├── src/
│   ├── config.py
│   ├── pipeline/    # sampling, subgraph_sampling
│   ├── rdf/         # builder, inference, to_hetero_pyg…
│   ├── gnn/         # homo + hetero + graph_loader
│   └── evaluation/  # metrics_viz, convergence_compare
├── scripts/01 … 06
└── notebooks/metrics_evaluation.ipynb
```

### Résultat

- `requirements.txt` figé avec `pandas`, `numpy`, `rdflib`, `tqdm`, `pyarrow`.
- Dépendances installées et fonctionnelles (Python 3.12.7).
- Configuration centralisée dans `src/config.py` (un seul endroit à toucher
  pour changer un chemin ou une taille d'échantillon).

### Décisions prises

- **`rdflib` plutôt qu'une écriture Turtle "à la main"** : conforme, gère
  l'échappement, permet le rechargement et SPARQL ; coût acceptable à 100k
  transactions.
- **Pas d'utilisation de `python -m scripts.xx`** : Python n'accepte pas
  les noms de module commençant par un chiffre. Les scripts s'exécutent
  directement avec `python scripts/01_sample_data.py`.
- **Pattern `sys.path.insert(...)`** en haut de chaque script pour que
  `from src import ...` fonctionne sans avoir à installer le projet en
  mode `pip install -e .`.

---

## Étape 2a — Sous-échantillonnage stratifié

**Date** : 01/06/2026
**Statut** : ✅ Terminée
**Script** : `scripts/01_sample_data.py`
**Module métier** : `src/pipeline/sampling.py`

### Objectif

Réduire le dataset brut (~6,9 M lignes) à un échantillon **manipulable sur
laptop** (~100 k lignes), tout en **préservant la totalité de la classe
minoritaire** (les transactions de blanchiment).

### Ce qui a été fait

1. Chargement du CSV brut avec **renommage des colonnes dupliquées**
   (les deux colonnes `Account` du CSV original deviennent `from_account`
   et `to_account`).
2. Séparation `fraud` / `legit`.
3. Conservation de **100 %** des transactions `is_laundering == 1`.
4. Tirage aléatoire (`random_state=42`) de transactions légitimes pour
   compléter à 100 000 lignes.
5. Re-tri chronologique sur `timestamp` (utile pour les analyses
   temporelles futures).
6. Écriture dans `data/csv/cleaned/LI-Small_Trans_sample.csv`.

### Résultat

| Métrique | Dataset brut | Échantillon |
|---|---|---|
| Lignes | 6 924 049 | 100 000 |
| Fraudes | 3 565 | 3 565 (100 % conservées) |
| % fraude | 0.0515 % | 3.57 % |

Le ratio de fraude est **~70× plus favorable** dans l'échantillon, ce qui
facilitera l'entraînement supervisé sans techniques lourdes de
ré-équilibrage.

### Décisions prises

- **Stratification "à plat" plutôt que par sous-graphe** : approche pragmatique
  pour cette première itération. Permet de générer rapidement le RDF.
  ⚠️ Une approche **par sous-graphe ego** sera nécessaire pour l'étape GNN
  (voir limite ci-dessous).
- **`random_state=42`** centralisé dans `config.RANDOM_SEED` pour garantir
  la reproductibilité.
- **Tri par `timestamp`** en mode `mergesort` (stable) pour conserver
  l'ordre des fraudes en cas d'ex aequo.

### Limite identifiée

Le tirage aléatoire **casse la connexité** du graphe original : la majorité
des comptes n'ont qu'une seule transaction dans l'échantillon. Pour le GNN,
cela limitera la capacité d'apprendre des motifs (cycles, fan-in/fan-out).
**À corriger à l'étape 4** : échantillonnage par expansion de voisinage
autour des transactions frauduleuses.

---

## Étape 2b — Génération de l'A-Box RDF

**Date** : 01/06/2026
**Statut** : ✅ Terminée
**Script** : `scripts/02_build_rdf.py`
**Module métier** : `src/rdf/builder.py`

### Objectif

Matérialiser l'échantillon CSV sous forme de **graphe RDF** (A-Box, Assertional
Box) **strictement conforme** à la T-Box `aml.ttl` rédigée à l'étape 0.

### Ce qui a été fait

1. Définition des URI builders :
   - `inst:bank-<bank_id>`
   - `inst:acct-<bank_id>-<account_id>`
   - `inst:tx-<index>`
2. Pour chaque ligne du CSV échantillonné :
   - Création (si nouvelle) des nœuds `Bank` et `Account` avec le triple
     `Account --hosted_by--> Bank`.
   - Création du nœud `Transaction`.
   - Ajout des arêtes topologiques `Account --sends_via--> Transaction`
     et `Transaction --deposits_to--> Account`.
   - Ajout des **5 attributs typés** sur la transaction (`amountPaid`,
     `paymentCurrency`, `paymentFormat`, `timestamp`, `isLaundering`).
3. Sérialisation en Turtle dans `data/rdf/ontology_data/aml_data.ttl`.

### Résultat

| Métrique | Valeur |
|---|---|
| Triples produits | **1 204 787** |
| Banques distinctes | 6 039 |
| Comptes distincts | 130 903 |
| Transactions | 100 000 |
| Taille du fichier Turtle | ~50 Mo |
| Temps de génération | ~25 s |
| Temps de sérialisation | ~50 s |

### Validation effectuée

- **Rechargement complet** du Turtle avec `rdflib` → 1 204 787 triples
  retrouvés (cohérence parfaite).
- **Requête SPARQL** `COUNT(?tx) WHERE { ?tx aml:isLaundering 1 }`
  → renvoie **3 565** (identique au CSV : zéro perte).
- **Requête SPARQL topologique** `?src aml:sends_via ?tx . ?tx aml:deposits_to ?dst`
  → renvoie correctement les triplets `(Source → Transaction → Destination)`
  pour les fraudes.

### Décisions prises

- **`rdflib.Graph` en mémoire** : à 100k lignes ça reste sous le Go de RAM
  et c'est le plus simple. Si on passe à un échantillon plus large, on
  basculera sur un export streaming N-Triples.
- **Labels `rdfs:label` ajoutés** sur les comptes et banques pour
  faciliter l'inspection humaine du Turtle.
- **`timestamp` conservé en `xsd:string`** comme prévu à l'étape 0
  (parsing à venir).

---

## Étape 3 — Approche tabulaire / Anomalie ponctuelle (RÉFÉRENCE)

**Date** : 19/02/2026 (projet antérieur)
**Statut** : ♻️ **Réutilisée — pas de ré-implémentation**
**Source** : `archive/baseline-anomalie-ponctuelle/Evaluation final - Fraudes et segmentation - 11/activité 10/Activite_10.ipynb`

### Pourquoi on ne refait pas cette étape

Le cahier des charges du projet précise dès son introduction :

> *"Lors de notre précédent projet durant le cours de machine learning sur
> la détection de fraude à la carte bancaire (Credit Card Fraud Detection),
> l'approche par apprentissage supervisé classique (modèles tabulaires de
> type Random Forest) a démontré son efficacité pour identifier des
> anomalies ponctuelles."*

Le présent projet **part de ce constat acquis** pour explorer une seconde
famille d'anomalies — les anomalies **structurelles** — qui exigent une
représentation des données radicalement différente (graphes plutôt que
tables).

### Ce qui a été couvert (rappel)

- **Dataset** : `creditcard.csv` (Kaggle).
- **Modèles** : `LogisticRegression`, `RandomForestClassifier`,
  `IsolationForest`, `LocalOutlierFactor`, pipeline `StandardScaler + SMOTE + RF`.
- **Conclusion validée** : le tabulaire fonctionne **bien** sur des
  anomalies qui sont **détectables ligne par ligne** (montant aberrant,
  combinaison improbable de features, etc.).

### Positionnement dans le présent projet

Cette étape n'est **pas une baseline à battre numériquement** — c'est
l'**incarnation de l'approche "anomalie ponctuelle"** qu'on va opposer
**conceptuellement** à l'approche "anomalie structurelle" développée
ici. Voir la section sur l'étape 6 pour le détail de cette comparaison.

---

## Étape 4a — Échantillonnage par sous-graphe ego (préalable GNN)

**Date** : 01/06/2026
**Statut** : ✅ Terminée
**Script** : `scripts/03_subgraph_sample.py`
**Module métier** : `src/pipeline/subgraph_sampling.py`

### Objectif

Corriger la limite identifiée à l'étape 2a : le sampling stratifié naïf
**casse la connexité** du graphe (~37 % des comptes étaient orphelins).
Un GNN n'apprend rien sur des nœuds isolés — il faut un voisinage
exploitable pour que le *message passing* fonctionne.

### Méthode

1. **Comptes "graines"** : tous les comptes touchés par au moins une
   transaction frauduleuse.
2. **Expansion K hops** (K=1 dans cette itération) : on ajoute tous les
   voisins directs des graines.
3. **Récolte** : toutes les transactions impliquant au moins un compte
   du sous-graphe résultant.
4. **Plafonnement** à `EGO_TARGET_SIZE = 200 000` transactions, en
   gardant **100 % des fraudes** et en sous-échantillonnant aléatoirement
   les transactions légitimes excédentaires.
5. **Tri chronologique stable**.

### Résultats

| Étape | Valeur |
|---|---|
| Comptes graines (impliqués dans une fraude) | 5 304 |
| Comptes ajoutés en 1-hop | +70 535 |
| Comptes dans le sous-graphe (avant plafond) | 75 839 |
| Transactions du sous-graphe (avant plafond) | 2 451 392 |
| **Transactions conservées (après plafond)** | **200 000** |
| **Fraudes conservées** | **3 565** (100 %) |
| **Taux de fraude** | **1.78 %** |

### Statistiques de connexité (sortie finale)

| Métrique | Valeur | Comparaison vs sampling naïf (étape 2a) |
|---|---|---|
| Comptes uniques | 121 082 | 130 903 (légèrement moins) |
| Degré moyen | **3.30** | ~1.5 |
| Degré médian | 2.00 | 1 |
| % comptes de degré 1 | **36 %** | ~70 % (estimé) |
| Degré max | 18 198 | — |

Lecture : on a multiplié par ~2 la connectivité moyenne, et plus de la
moitié des comptes a maintenant au moins 2 transactions — ce qui rend le
*message passing* utile.

### Décisions prises

- **K=1 hop suffit** pour cette itération. K=2 aurait fait exploser le
  nombre de transactions (estimation > 5M) sans bénéfice avéré pour
  cette taille d'échantillon.
- **Plafond à 200k** plutôt que 100k pour préserver davantage de
  contexte topologique tout en restant manipulable sur laptop.
- **Pas de "fond légitime extérieur"** (`EGO_INCLUDE_LEGIT_BACKGROUND=False`)
  pour garder la concentration sur le voisinage des fraudes.
- **Le sampling stratifié naïf reste disponible** (`scripts/01_sample_data.py`)
  pour des tests rapides ou si on veut nourrir l'A-Box RDF d'une autre
  distribution.

---

## Étape 4a-hold — Sous-graphe ego hold-out (LI-Medium, test final)

**Date** : 02/06/2026  
**Statut** : ✅ Terminée  
**Script** : `python scripts/03_subgraph_sample.py --data-profile unknown`  
**Module** : `src/pipeline/subgraph_sampling.py` (mode **chunked**, `chunksize=50 000`)  
**Livrable** : `data/unknown_data/csv/cleaned/LI-Medium_Trans_subgraph.csv`

### Objectif

Produire **200 000 transactions** sur **LI-Medium** (~31,3 M lignes) avec la
même méthode que l'entraînement (ego K=1, seed 42, 100 % des fraudes au
plafonnement), pour l'évaluation hold-out des GNN entraînés sur LI-Small.

### Corpus brut scanné

| Indicateur | Valeur |
|------------|--------|
| Transactions | 31 251 483 |
| Fraudes | 16 041 (0,051 %) |
| Comptes graines | 23 510 |
| Comptes après 1-hop | 250 499 |
| Transactions ego (avant plafond) | 12 126 651 |
| Fraudes ego (avant plafond) | 16 041 |

### Échantillon final et connexité

| Indicateur | Valeur |
|------------|--------|
| Transactions | **200 000** |
| Fraudes | **16 041** (8,02 %) |
| Comptes uniques | 216 008 |
| Degré moyen | 1,85 |
| Degré médian | 1,00 |
| Degré max | 18 306 |
| % degré = 1 | 62,72 % |

### Comparaison train vs hold-out

| Métrique | LI-Small (train) | LI-Medium (hold-out) |
|----------|------------------|----------------------|
| Fraudes finales | 3 565 (1,78 %) | 16 041 (8,02 %) |
| Comptes | 121 082 | 216 008 |
| Degré moyen | 3,30 | 1,85 |
| % degré = 1 | 36 % (log 4a) | 62,72 % |

Lecture : le hold-out est **plus riche en fraudes** mais **moins favorable au
message passing** (plus de comptes quasi isolés). Interpréter les métriques
GNN en conséquence.

**Suite** : pipeline hold-out → voir étapes 4b-hold à 5-hold ci-dessous.

---

## Étape 4b-hold — RDF hold-out (LI-Medium)

**Date** : 02/06/2026  
**Statut** : ✅ Terminée  
**Scripts** : `02b_build_rdf_subgraph.py`, `02c_enrich_rdf.py` (`--data-profile unknown`)  
**T-Box** : `data/train_data/rdf/ontology_schema/aml.ttl` (partagée)

| Livrable | Emplacement |
|----------|-------------|
| A-Box sous-graphe | `data/unknown_data/rdf/ontology_data/aml_data_subgraph.ttl` |
| A-Box enrichie | `data/unknown_data/rdf/ontology_data/aml_data_subgraph_enriched.ttl` |

Inférence identique au train : `transactedWith`, `sameBankAs` (règles matérialisées sur le sous-graphe Medium).

---

## Étape 4c-hold — PyG hold-out + alignement features

**Date** : 02/06/2026  
**Statut** : ✅ Terminée  
**Script** : `04_build_pyg.py --data-profile unknown --holdout-eval`

| Graphe | Dimensions clés | Fraudes |
|--------|-----------------|---------|
| HeteroData | account (216008, 5), transaction (200000, 27), bank (8135, 2) | 16 041 / 200 000 |
| Data homogène | x (216008, 4), edge_attr (200000, 17) | 16 041 / 200 000 |

**Problème rencontré** : au premier run, `bank` avait 9 dimensions (vocabulaire Medium) vs 2 dans le checkpoint train → `ValueError` à l’évaluation.

**Correction** : `align_feature_bundle()` + `hetero_feature_names_from_train()` — colonnes alignées sur `hetero_info` du checkpoint avant sauvegarde du `.pt`.

---

## Étape 5-hold — Test final (généralisation)

**Date** : 02/06/2026  
**Statut** : ✅ Terminée  
**Script** : `07_eval_holdout.py` (ou `08_run_unknown_holdout_pipeline.py --skip-subgraph`)  
**Module** : `src/evaluation/holdout_eval.py`  
**Livrable** : `data/unknown_data/results/eval_holdout_metrics.json`

### Protocole

- Checkpoints **train** : `data/train_data/results/gnn_{hetero,sage,gat}_best.pt`
- Graphes **unknown** : `data/unknown_data/pyg/`
- **Aucun ré-entraînement** sur Medium
- Métriques sur les **200 000** transactions (évaluation plein graphe)

### Résultats hold-out

| Modèle | PR-AUC | Recall | Precision | F1 |
|--------|--------|--------|-----------|-----|
| **HeteroGNN** | **0,5947** | 0,6993 | **0,5437** | **0,6117** |
| GraphSAGE | 0,2286 | 0,7809 | 0,1335 | 0,2280 |
| GAT | 0,1497 | 0,6434 | 0,1143 | 0,1941 |

### Lecture (soutenance)

- **HeteroGNN** : meilleur compromis PR-AUC / precision ; adapté au **triage** d’alertes (pas décision finale seule).
- **SAGE / GAT** : recall élevé mais precision ~11–13 % → **sur-alerte** ; PR-AUC faible = mauvais ranking des cas suspects.
- Comparaison test interne LI-Small (Hetero PR-AUC **0,254**, recall **95 %**, precision **6 %**) : le hold-out montre une **généralisation du signal** (PR-AUC ↑) avec un seuil plus conservateur (recall ↓, precision ↑).

**Commande de reproduction** :

```powershell
python scripts/08_run_unknown_holdout_pipeline.py --skip-subgraph
python scripts/09_visualize_holdout.py
```

### Étape 5-hold-viz — Graphiques et CSV (équivalent phase E)

**Script** : `scripts/09_visualize_holdout.py` — **Module** : `src/evaluation/holdout_viz.py`  
**Notebook** : `notebooks/holdout_metrics_evaluation.ipynb`

Livrables principaux dans `data/unknown_data/results/` :

- `holdout_metrics_grid.png`, `holdout_pr_auc_bars.png`, `holdout_architecture_*.png`
- `holdout_pr_curves_all_models.png`, `holdout_confusion_matrices_all_models.png`
- `holdout_train_vs_holdout_pr_auc.png` (généralisation)
- `holdout_train_convergence_reference.png` (référence entraînement LI-Small)
- `gnn_*_holdout_metrics.json`, `holdout_metrics_summary.csv`, `holdout_viz_manifest.json`

---

### Étape 4b — Inférence RDF (A-Box enrichie)

**Script** : `scripts/02c_enrich_rdf.py` — **Module** : `src/rdf/inference.py`

Règles matérialisées :
- `transactedWith` : 110 716 paires (comptes ayant transigé)
- `sameBankAs` : 3 312 triplets (même banque + transactedWith)
- Total : +224 744 triplets → `aml_data_subgraph_enriched.ttl` (~90 Mo)

### Étape 4c — Conversion RDF enrichi → HeteroData

**Scripts** : `04_build_pyg.py --mode hetero` (défaut)
**Modules** : `src/rdf/to_hetero_pyg.py`, `src/rdf/features.py`

- 3 types de nœuds : account, transaction, bank
- 10 types d'arêtes (5 prédicats × 2 directions)
- Features extraites dynamiquement depuis la T-Box
- Livrable : `data/pyg/aml_hetero_graph.pt`

### Étape 5 — HeteroGNN (HeteroConv)

**Script** : `05_train_gnn.py` (défaut `--model hetero`)
**Modules** : `src/gnn/hetero/models.py`, `src/gnn/hetero/training.py`

Résultats TEST : PR-AUC **0,254** | Recall **94,8 %** | F1 **0,112**

> Le mode GraphSAGE homogène reste documenté ci-dessous comme référence legacy.

### Étape 4b-legacy — Conversion RDF → PyG homogène

**Modules** : `src/rdf/to_homo_pyg.py`, `src/gnn/graph_loader.py`
**Commande** : `python scripts/04_build_pyg.py --mode homo`

Graphe homogène `aml_graph.pt` — GraphSAGE/GAT via `--model sage|gat`.
Conservé pour comparaison avec le HeteroGNN.

### Features encodées

**Features de nœuds** (4 dimensions, normalisées z-score après log) :

| Feature | Sens |
|---|---|
| `log_out_deg` | Nombre de transactions envoyées (log). |
| `log_in_deg` | Nombre de transactions reçues (log). |
| `log_amount_sent` | Montant total envoyé (log). |
| `log_amount_received` | Montant total reçu (log). |

**Features d'arêtes** (17 dimensions) :

- `amount_log_z` : montant log-normalisé z-score.
- One-hot du **format de paiement** (7) : `ACH`, `Bitcoin`, `Cash`, `Cheque`,
  `Credit Card`, `Reinvestment`, `Wire`.
- One-hot de la **devise** top-8 + Other (9) : `US Dollar`, `Euro`, `Yuan`,
  `Rupee`, `Bitcoin`, `Saudi Riyal`, `Australian Dollar`, `Brazil Real`,
  `Other`.

### Splits stratifiés sur le label d'arête

| Split | Taille | Fraudes (≈) |
|---|---|---|
| Train | 140 000 (70 %) | ~2 495 |
| Validation | 30 000 (15 %) | ~535 |
| Test | 30 000 (15 %) | ~535 |

Le split est **stratifié sur `is_laundering`** pour préserver la proportion
de fraudes dans chaque partition.

### Résultats

```
Data(
    x         = Tensor[121 082, 4]
    edge_index= Tensor[2, 200 000]
    edge_attr = Tensor[200 000, 17]
    y         = Tensor[200 000]      # 3 565 positives
    train_mask, val_mask, test_mask  # 140k / 30k / 30k
)
```

- Fichier produit : `data/pyg/aml_graph.pt` (~21 Mo)
- Sanity check : rechargement et inspection → cohérent.
- Métadonnées des features stockées dans `FeatureInfo` pour interprétation
  ultérieure.

### Décisions prises

- **`torch` CPU + `torch_geometric` 2.7** installés (cf. `requirements.txt`).
  GPU pas indispensable à 200k arêtes.
- **One-hot pour les variables catégorielles** plutôt qu'embeddings appris :
  les vocabulaires sont petits (7 formats, 8 devises + Other).
- **`Data` standard** plutôt qu'objet personnalisé : compatibilité maximale
  avec tous les *loaders* et modèles de PyG.
- **Sauvegarde au format `.pt`** (`torch.save`) : plus simple que ré-encoder
  à chaque entraînement.

### Limites identifiées

- 36 % des nœuds restent de degré 1 → le GNN sera limité sur ces nœuds-là.
  Si les performances sont décevantes à l'étape 5, envisager un sampling
  ego K=2 avec budget plus grand.
- Features d'arête volontairement minimales (pas de temporalité fine).
  Si besoin, ajouter `hour_of_day`, `day_of_week` extraits du `timestamp`.

---

## Étape 5 — Entraînement GNN (classification d'arêtes)

**Date** : 01/06/2026
**Statut** : ✅ Terminée
**Script** : `scripts/05_train_gnn.py`
**Modules métier** : `src/gnn/homo/models.py`, `src/gnn/homo/training.py`

### Objectif

Entraîner un GNN capable de prédire si une **transaction** (arête du graphe)
est frauduleuse (`is_laundering = 1`), en exploitant :
1. Les **embeddings topologiques** des comptes source et destination
   (via message passing).
2. Les **attributs locaux** de la transaction (`amount`, format, devise).

### Ce qui a été fait

1. Implémentation de `EdgeClassifierGNN` dans `src/gnn/homo/models.py` :
   - encodeur GraphSAGE (2 couches, hidden=64) **ou** GAT (4 têtes d'attention),
   - tête MLP : `concat(h[src], h[dst], edge_attr) → logit`.
2. Boucle d'entraînement dans `src/gnn/homo/training.py` :
   - forward sur le **graphe entier** (message passing global),
   - loss calculée **uniquement** sur les arêtes du train_mask,
   - early stopping sur val PR-AUC (patience=15).
3. Comparaison GraphSAGE vs GAT ; sauvegarde des checkpoints séparés.
4. Métriques exportées en JSON (`gnn_sage_metrics.json`).

### Comment lancer

```powershell
python scripts/05_train_gnn.py              # GraphSAGE (défaut)
python scripts/05_train_gnn.py --model gat  # GAT
```

Hyperparamètres modifiables dans `src/config.py` (`GNN_*`).

### Architecture implémentée

```
Comptes (nœuds)  ──▶  GraphSAGE / GAT  ──▶  embeddings h[v]
                                                    │
Transaction (arête) ──▶  concat(h[src], h[dst], edge_attr)  ──▶  MLP  ──▶  logit fraude
```

| Composant | Détail |
|---|---|
| Encodeur nœuds | 2 couches GraphSAGE ou GAT, hidden=64, dropout=0.3 |
| Tête d'arête | MLP 3 couches (145 → 64 → 32 → 1) |
| Loss | `BCEWithLogitsLoss` avec `pos_weight=55.09` (ratio négatif/positif train) |
| Optimiseur | Adam (lr=1e-3, weight_decay=1e-4) |
| Early stopping | Patience=15 epochs sur **val PR-AUC** |
| Device | CPU (121k nœuds, 200k arêtes — ~1 s/epoch) |

### Résultats — jeu TEST (30 000 arêtes, 534 fraudes)

| Modèle | PR-AUC | Recall | F1 | Precision | Meilleur epoch |
|---|---|---|---|---|---|
| **GraphSAGE** ✅ retenu | **0.254** | **0.927** | **0.113** | 0.060 | 80 |
| GAT | 0.047* | 0.626 | 0.056 | 0.030 | 3 (early stop) |

\* Le GAT a convergé prématurément lors du run final (early stopping à l'epoch 3).
Un run complet antérieur donnait PR-AUC=0.198 / Recall=0.916 — GraphSAGE reste
nettement plus stable et performant sur ce graphe.

**Référence aléatoire** : PR-AUC ≈ 0.018 (taux de fraude du dataset = 1.78 %).
Le GraphSAGE fait **~14× mieux** qu'un classificateur aléatoire.

### Lecture des métriques

| Observation | Interprétation |
|---|---|
| **Recall élevé (~93 %)** | Le modèle attrape la grande majorité des fraudes — cohérent avec la pondération `pos_weight` et le contexte AML (mieux vaut une alerte en trop qu'une fraude ratée). |
| **Precision faible (~6 %)** | Beaucoup de faux positifs. Attendu avec un déséquilibre extrême et un seuil à 0.5. En production, on abaisserait/ajusterait le seuil via une courbe PR. |
| **PR-AUC modeste (0.25)** | Le modèle apprend un signal réel mais le graphe reste partiellement fragmenté (36 % de nœuds de degré 1). Des gains sont possibles avec K=2 ego, features temporelles parsées, inférence OWL-RL avancée. |
| **GraphSAGE > GAT** | Sur ce graphe homogène dense, l'agrégation mean de SAGE suffit ; GAT (attention) surfit ou converge mal sans tuning supplémentaire. |

### Livrables produits

| Fichier | Contenu |
|---|---|
| `data/results/gnn_sage_best.pt` | Checkpoint GraphSAGE (meilleur epoch) |
| `data/results/gnn_sage_metrics.json` | Métriques train/val/test |
| `data/results/gnn_gat_best.pt` | Checkpoint GAT (run de comparaison) |
| `data/results/gnn_gat_metrics.json` | Métriques GAT |

### Décisions prises

- **GraphSAGE retenu** comme modèle principal pour l'étape 6 (comparaison conceptuelle).
- **`pos_weight` plutôt que SMOTE** : le sur-échantillonnage n'a pas de sens sur des
  arêtes dans un graphe ; la pondération de la loss est la méthode standard.
- **Seuil fixe à 0.5** pour cette première itération. L'optimisation du seuil
  (courbe PR) pourra être ajoutée à l'étape 6 si nécessaire.
- **Checkpoints séparés par modèle** (`gnn_sage_best.pt`, `gnn_gat_best.pt`) pour
  éviter l'écrasement lors des comparaisons.

### Limites identifiées

- 36 % des nœuds restent de degré 1 → message passing limité sur ces comptes.
- Pas de features temporelles dans `edge_attr` (timestamp non exploité).
- Entraînement sur CPU uniquement — acceptable à cette échelle mais limitant
  pour des hyperparam searches plus larges.

---

---

## Étape 6 — Rapport final (comparaison conceptuelle)

**Date** : 01/06/2026
**Statut** : ✅ Terminée
**Livrable** : `docs/reports/final_report.md`

### Objectif

Synthétiser l'ensemble du projet et **répondre formellement à la
problématique** par une comparaison **conceptuelle** des deux paradigmes
(anomalie ponctuelle vs structurelle), sans confrontation numérique
inter-datasets.

### Ce qui a été rédigé

1. **Introduction** — dualité ponctuel / structurel.
2. **Approche 1** — rappel du projet ML antérieur (RF sur Credit Card).
3. **Approche 2** — pipeline complet du présent projet (RDF → PyG → GNN).
4. **Tableau comparatif** des deux paradigmes (représentation, forces, limites).
5. **Schéma de complémentarité** en production AML.
6. **Réponse à la problématique** — formulation directe pour la soutenance.
7. **Limites** et pistes d'amélioration.
8. **Conclusion** — les deux approches ne sont pas concurrentes.

### Décisions actées (rappel)

- Comparaison **conceptuelle uniquement** (datasets Kaggle vs IBM AML différents).
- Pas de mini-RF sur IBM AML pour cette itération.
- Le GNN démontre une **capacité architecturale** que le tabulaire n'a pas,
  pas une supériorité numérique directe.

---

*Dernière mise à jour : 01/06/2026 — **projet terminé** (étapes 0 à 6).*
