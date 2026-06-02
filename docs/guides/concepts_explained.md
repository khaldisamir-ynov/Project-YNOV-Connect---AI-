# Concepts, outils & méthodes du projet AML

> **But** : comprendre **pourquoi** chaque brique est utilisée et **comment**
> elle fonctionne. Le ton est volontairement pédagogique mais rigoureux ;
> chaque section est autonome et peut être lue indépendamment.

---

## Sommaire

0. [Deux chemins de données dans le projet (RDF vs PyG)](#0-deux-chemins)
0b. [Trois avantages RDF exploités par le GNN](#0b-avantages-rdf)
1. [Le domaine métier — AML & blanchiment d'argent](#1-domaine-aml)
2. [La dualité fondamentale — Anomalie ponctuelle vs structurelle](#1bis-dualite)
3. [Approche tabulaire — Random Forest](#2-tabulaire-random-forest)
4. [Le Web Sémantique — RDF, OWL, T-Box, A-Box, Turtle](#3-web-semantique)
5. [Requêter le graphe — SPARQL](#4-sparql)
6. [La théorie des graphes appliquée](#5-theorie-des-graphes)
7. [Les Graph Neural Networks (GNN)](#6-gnn)
8. [Pourquoi les GNN sont pertinents pour l'AML](#7-gnn-pour-aml)
9. [Échantillonnage par sous-graphe ego](#9-ego-sampling)
10. [Le format `Data` / `HeteroData` de PyTorch Geometric](#9b-pyg-data)
11. [Outils & librairies du projet](#11-outils)
12. [Évaluation — métriques pour classes déséquilibrées](#12-metriques)

---

<a id="0-deux-chemins"></a>
## 0. Deux chemins de données dans le projet (RDF vs PyG)

Le projet utilise **deux représentations** du même réseau transactionnel.
Ce n'est pas une contradiction — chacune sert un objectif différent.

```
CSV brut
   │
   ├──▶ [Chemin A — Web Sémantique démo]  CSV 100k → aml_data.ttl
   │
   └──▶ [Chemin B — GNN multi-relationnel]  CSV sous-graphe ego 200k
              │
              ▼
         aml_data_subgraph.ttl              ← A-Box assertée (02b)
              │
              ▼
         aml_data_subgraph_enriched.ttl    ← inférence RDF (02c)
              │     transactedWith, sameBankAs
              ▼
         aml_hetero_graph.pt               ← HeteroData (04, mode hetero)
              │
              ▼
         HeteroGNN (HeteroConv)            ← 1 poids par prédicat (05)
```

| | Chemin A (`aml_data.ttl`) | Chemin B (pipeline GNN) |
|---|---|---|
| **Pourquoi** | Démo Web Sémantique (100k) | Entraînement GNN (200k, sous-graphe connecté) |
| **Alimente le GNN ?** | ❌ Non | ✅ Oui |
| **Scripts** | `02_build_rdf.py` | `02b` → `02c` → `04` → `05` |
| **Format PyG** | — | `HeteroData` (défaut) ou `Data` homogène (`--mode homo`) |

Le mode **homogène** (GraphSAGE sur `aml_graph.pt`) reste disponible pour
comparaison ; le mode **hétérogène** est le chemin officiel du projet.

---

<a id="0b-avantages-rdf"></a>
## 0b. Trois avantages RDF exploités par le GNN

Le GNN consomme des tenseurs PyG — pas du Turtle directement. L'intérêt du
RDF est **en amont** : modélisation, inférence et extensibilité.

### 1. Multi-relationnel — le GNN comprend le *pourquoi* du lien

Dans un graphe homogène, toutes les arêtes sont traitées pareil. En RDF,
chaque prédicat a une sémantique (`sends_via` ≠ `transactedWith` ≠
`sameBankAs`). Le **HeteroGNN** (`HeteroConv`) applique une transformation
**distincte par type de relation** — équivalent pratique du **R-GCN**.

| Prédicat RDF | Rôle sémantique |
|---|---|
| `sends_via` | Compte émetteur → transaction |
| `deposits_to` | Transaction → compte destinataire |
| `hosted_by` | Compte → banque hébergeuse |
| `transactedWith` | Comptes ayant déjà transigé (**inféré**) |
| `sameBankAs` | Même banque + lien transactedWith (**inféré**) |

Module : `src/gnn/hetero/models.py` — `src/rdf/to_hetero_pyg.py`

### 2. Inférence — pre-processing intelligent

Avant l'entraînement, `src/rdf/inference.py` enrichit l'A-Box :

```
Compte_A ──hosted_by──▶ Banque_X ◀──hosted_by── Compte_B
Compte_A ──transactedWith──▶ Compte_B   (inféré depuis les transactions)
Compte_A ──sameBankAs──▶ Compte_B       (inféré si même banque + transactedWith)
```

Sur le sous-graphe 200k : **+224 744 triplets** (110 716 paires
`transactedWith`, 3 312 triplets `sameBankAs`). Le GNN reçoit un graphe
**enrichi** sans deviner ces relations métier.

Script : `python scripts/02c_enrich_rdf.py` — option `--owlrl` si `owlrl` installé.

### 3. Features dynamiques depuis la T-Box

`src/rdf/features.py` parcourt les `DatatypeProperty` de `aml.ttl` et
construit automatiquement les tenseurs de features. Ajouter une propriété
ontologique (ex. `bankCountry` sur `Bank`) suffit pour l'inclure au GNN —
sans `ALTER TABLE` ni refonte manuelle du CSV.

| Type de nœud | Features extraites (exemple) |
|---|---|
| Transaction | amountPaid, paymentCurrency, paymentFormat, timestamp |
| Bank | bankCode |
| Account | degrés sends_via, deposits_to, transactedWith, sameBankAs |

---
## 1. Le domaine métier — AML & blanchiment d'argent

### Qu'est-ce que l'AML ?

**AML** = *Anti-Money Laundering*. Ensemble des dispositifs (légaux, humains,
techniques) qui visent à **détecter et empêcher** le blanchiment d'argent
dans le système financier.

### Le blanchiment, schématiquement

Le blanchiment se décompose classiquement en **trois étapes** :

| Étape | Nom | Description |
|---|---|---|
| 1 | **Placement** | Faire entrer l'argent illicite dans le système (dépôts, achats). |
| 2 | **Empilage** (*layering*) | Multiplier les transactions pour **brouiller la traçabilité** : virements en cascade, comptes relais, conversions de devises, fractionnement (*smurfing*). |
| 3 | **Intégration** | Réintroduire l'argent "propre" via des activités légitimes (immobilier, sociétés écrans). |

Le **layering** est la phase où la **topologie** du réseau devient essentielle :
ce qui paraît normal ligne par ligne devient suspect quand on regarde la
**forme** des flux (cycles, comptes "passe-plats", arbres en éventail).

### Pourquoi c'est dur à détecter

- Les transactions individuelles sont **conformes** prises isolément
  (montants raisonnables, comptes existants).
- L'anomalie est **structurelle** : elle vit dans les *relations* entre
  transactions, pas dans les transactions elles-mêmes.
- Les fraudeurs **adaptent leurs schémas** quand des règles statiques
  sont en place.

---

<a id="1bis-dualite"></a>
## 2. La dualité fondamentale — Anomalie ponctuelle vs structurelle

**C'est le cœur du présent projet.** La fraude financière se décline en
**deux grandes familles** d'anomalies, qui exigent **deux paradigmes
d'apprentissage radicalement différents**.

### Anomalie ponctuelle

> Une transaction est suspecte **par elle-même**, indépendamment de toute
> autre transaction.

**Exemples concrets** :

- Achat de 12 000 € dans un pays où le titulaire de la carte n'a jamais
  voyagé.
- Trois retraits successifs au plafond journalier en moins d'une heure.
- Combinaison improbable de features (marchand + montant + heure).

**Représentation des données adaptée** : **table** (chaque ligne =
une transaction, traitée indépendamment des autres).

**Modèles adaptés** : Random Forest, Logistic Regression, XGBoost,
Isolation Forest, réseaux de neurones denses (MLP).

**C'est le sujet du projet ML antérieur** sur le dataset Credit Card Fraud.

### Anomalie structurelle

> Aucune transaction n'est suspecte prise isolément. Ce qui est suspect,
> c'est **l'arrangement géométrique** de plusieurs transactions entre elles.

**Exemples concrets** :

- Un compte reçoit 100 petits virements de 5 000 € en une semaine
  (*fan-in* / regroupement final).
- Un compte fait suivre instantanément chaque réception vers un autre
  compte avec une décote de 1 % (*pass-through*).
- L'argent revient au compte d'origine après 4 sauts entre comptes
  intermédiaires (*cycle*).

**Représentation des données adaptée** : **graphe** (les nœuds sont les
comptes, les arêtes sont les transactions ; ce qu'on analyse, c'est la
topologie des flux).

**Modèles adaptés** : Graph Neural Networks (GNN), motifs détectés par
SPARQL, algorithmes de détection de communautés, autoencodeurs sur
graphes.

**C'est le sujet du présent projet** sur le dataset IBM AML.

### Synthèse visuelle

```
ANOMALIE PONCTUELLE                    ANOMALIE STRUCTURELLE
───────────────────                    ─────────────────────

   ┌─────┐                                 A─►B─►C
   │ TX  │  ← un seul point                   │
   │ ❗   │     dans l'espace                  ▼
   └─────┘     des features                D─►E
                                             ▲
                                             │
                                             A   ← cycle invisible
                                                   ligne par ligne

Forêt aléatoire                       Graph Neural Network
voit ce point                         voit le motif
isolément                             dans son ensemble
```

### Pourquoi ces deux approches ne sont pas concurrentes

Elles couvrent des **familles d'anomalies différentes**. Un système AML
mature de niveau industriel **combine les deux** :

- Un modèle tabulaire en première ligne pour les fraudes simples.
- Un GNN en seconde ligne pour les schémas de blanchiment organisé.
- Les sorties des deux modèles alimentent un score de risque consolidé.

Le présent projet ne cherche **pas à départager** les deux approches en
performance brute, mais à **démontrer** que la seconde existe, qu'elle
est techniquement implémentable, et qu'elle adresse une classe
d'anomalies que la première ne peut pas voir, **par construction**.

---

<a id="2-tabulaire-random-forest"></a>
## 3. Approche tabulaire — Random Forest

> 📌 **Statut dans ce projet** : l'approche tabulaire n'est **pas réimplémentée
> ici**. Elle a déjà été couverte dans le projet de Machine Learning antérieur
> (`archive/baseline-anomalie-ponctuelle/Evaluation final - Fraudes et segmentation - 11/activité 10/Activite_10.ipynb`)
> sur le dataset *Credit Card Fraud Detection* (Kaggle), avec
> `LogisticRegression`, `RandomForestClassifier`, `IsolationForest` et un
> pipeline `StandardScaler + RF` validé en stratified K-fold.
>
> Cette section reste utile pour **comprendre conceptuellement** ce à quoi
> on compare le GNN.

### L'idée d'un modèle tabulaire

Un modèle **tabulaire** considère chaque ligne du CSV **indépendamment des
autres**. Pour notre dataset, ça revient à prédire `is_laundering`
uniquement à partir des colonnes de la transaction (montant, devise,
format de paiement, banque source / destination).

### Random Forest, en bref

C'est un ensemble de **N arbres de décision** entraînés indépendamment, dont
on **agrège les votes**.

```
            ┌─── Arbre 1 ──▶ vote_1 ──┐
Features  ──┼─── Arbre 2 ──▶ vote_2 ──┼──▶ majorité ──▶ prédiction
            └─── Arbre N ──▶ vote_N ──┘
```

Chaque arbre :
- Est entraîné sur un **échantillon bootstrap** (tirage avec remise) des données.
- Choisit ses splits sur un **sous-ensemble aléatoire de features**.

Cette double randomisation (lignes + colonnes) **réduit la variance** par
rapport à un seul arbre profond.

### Pourquoi Random Forest comme baseline ?

| Argument | Détail |
|---|---|
| **Robuste** | Peu sensible aux outliers, gère valeurs manquantes raisonnablement. |
| **Pas de feature scaling** | Pas besoin de normaliser les montants. |
| **Donne une importance des variables** | Utile pour expliquer ce que le modèle "voit". |
| **Référence reconnue** | C'est la baseline standard pour la détection de fraude tabulaire (cf. dataset Kaggle Credit Card Fraud). |
| **Rapide à entraîner** | Pas besoin de GPU. |

### Limite fondamentale (qui motive le GNN)

Un Random Forest ne **voit jamais le voisinage d'une transaction**. Une
transaction de 10 000 $ vers un compte qui en a déjà reçu 50 d'autres dans
la même minute ressemble, pour le modèle, à une transaction de 10 000 $
isolée. **La topologie est invisible**.

---

<a id="3-web-semantique"></a>
## 4. Le Web Sémantique — RDF, OWL, T-Box, A-Box, Turtle

### L'intuition centrale

Le Web Sémantique cherche à **rendre les données auto-descriptives** : au
lieu d'avoir une table "Transaction" avec des colonnes implicites, on
**explicite le sens** de chaque entité et de chaque relation, dans un
vocabulaire partagé.

### RDF — la grammaire de base

**RDF** (*Resource Description Framework*) représente toute information sous
forme de **triplets** :

```
( sujet , prédicat , objet )
```

Exemple concret tiré de notre A-Box :

```
inst:acct-011-8000ECA90   aml:sends_via       inst:tx-42
inst:tx-42                aml:amountPaid      3195403.00
inst:tx-42                aml:isLaundering    0
```

Un ensemble de triplets forme un **graphe** : les sujets et objets sont des
nœuds, les prédicats sont les arêtes étiquetées.

### OWL — le vocabulaire structurant

**OWL** (*Web Ontology Language*) est bâti par-dessus RDF et permet de
définir un **schéma** : quelles classes existent, quelles relations sont
valides, sur quels types de valeurs.

C'est OWL qui nous permet d'écrire dans `aml.ttl` :

```turtle
aml:sends_via rdf:type owl:ObjectProperty ;
    rdfs:domain aml:Account ;       # part TOUJOURS d'un Account
    rdfs:range  aml:Transaction .   # pointe TOUJOURS vers une Transaction
```

Cette contrainte agit comme une **douane** : un outil de validation
(*reasoner*) rejettera tout triple `(Transaction, sends_via, Bank)`.

### T-Box vs A-Box

| Terme | Signification | Rôle | Notre fichier |
|---|---|---|---|
| **T-Box** | *Terminological Box* | Décrit le **vocabulaire** (classes, propriétés, contraintes). C'est le **schéma**. | `aml.ttl` |
| **A-Box** | *Assertional Box* | Décrit les **instances réelles** conformes à la T-Box. Ce sont les **données**. | `aml_data.ttl` |

Analogie SQL : la T-Box correspond au `CREATE TABLE`, l'A-Box correspond aux
`INSERT INTO`.

### Turtle — la syntaxe lisible

**Turtle** (*Terse RDF Triple Language*) est une notation textuelle de RDF.
On peut **factoriser** plusieurs triples partageant le même sujet avec `;`,
et plusieurs objets pour le même prédicat avec `,` :

```turtle
inst:acct-000-800062D90 a aml:Account ;
    rdfs:label "Account 800062D90 @ 000" ;
    aml:hosted_by inst:bank-000 ;
    aml:sends_via inst:tx-22453,
                  inst:tx-28152,
                  inst:tx-29336 .
```

### Pourquoi RDF/OWL pour ce projet plutôt qu'une base SQL ?

1. **Sémantique explicite** : un humain ou une autre application peut lire
   `aml.ttl` et comprendre ce qu'est une Transaction sans documentation
   externe.
2. **Topologie native** : RDF *est* un graphe par construction — pas de
   `JOIN` à écrire pour suivre les flux.
3. **Interopérabilité** : on peut fusionner facilement avec d'autres
   ontologies (FIBO en finance par exemple).
4. **Bridge vers le GNN** : un graphe RDF se convertit naturellement en
   structure exploitable par PyTorch Geometric.

---

<a id="4-sparql"></a>
## 5. Requêter le graphe — SPARQL

**SPARQL** est le **SQL des graphes RDF**. On y écrit des **patterns de
triplets** que le moteur cherche à apparier dans le graphe.

### Exemple — compter les fraudes

```sparql
PREFIX aml: <http://escience.datafactory.com/ontology/aml#>
SELECT (COUNT(?tx) AS ?n)
WHERE {
    ?tx a aml:Transaction ;
        aml:isLaundering 1 .
}
```

Lecture : *"trouve tous les `?tx` qui sont une Transaction ET dont
`isLaundering` vaut 1, puis compte-les."*

### Exemple — détecter un motif topologique simple

```sparql
PREFIX aml: <http://escience.datafactory.com/ontology/aml#>
SELECT ?src ?tx ?dst WHERE {
    ?src aml:sends_via ?tx .
    ?tx  aml:deposits_to ?dst .
    ?tx  aml:isLaundering 1 .
}
LIMIT 10
```

Lecture : *"trouve tous les chemins Source → Transaction → Destination où
la transaction est frauduleuse."*

C'est cette capacité à raisonner **sur des chemins** qui rend le RDF/SPARQL
intéressant en amont du GNN.

---

<a id="5-theorie-des-graphes"></a>
## 6. La théorie des graphes appliquée

### Vocabulaire essentiel

| Terme | Définition | Dans notre projet |
|---|---|---|
| **Nœud** (*node*) | Une entité du graphe. | Un Account, une Bank, une Transaction. |
| **Arête** (*edge*) | Une relation entre deux nœuds. | `sends_via`, `deposits_to`, `hosted_by`. |
| **Degré** | Nombre d'arêtes connectées à un nœud. | Un compte qui envoie beaucoup → degré entrant élevé sur `sends_via`. |
| **Voisinage** | Ensemble des nœuds directement connectés à un nœud donné. | Les transactions issues / reçues par un compte. |
| **Chemin** | Suite d'arêtes reliant deux nœuds. | Un parcours `acct1 → tx1 → acct2 → tx2 → acct3`. |
| **Composante connexe** | Sous-graphe dans lequel tous les nœuds sont reliés. | Un "îlot" de comptes liés par des transactions. |

### Motifs typiques du blanchiment

- **Fan-out** : un compte envoie de petits montants à beaucoup d'autres
  (fractionnement). Visible comme un nœud de degré sortant élevé.
- **Fan-in** : un compte reçoit de petits montants depuis beaucoup d'autres
  (regroupement final).
- **Cycle** : l'argent revient au compte d'origine après plusieurs sauts
  (auto-blanchiment).
- **Pass-through** : un compte "relais" reçoit puis renvoie immédiatement
  vers un autre compte avec peu de différence de montant.

Ces motifs ont une **signature géométrique** qu'un modèle tabulaire ne
peut pas voir.

### Graphe homogène vs hétérogène

- **Homogène** : un seul type de nœud, un seul type d'arête. On projette
  la Transaction sur une arête `Account → Account`. Modèle : GraphSAGE.
- **Hétérogène** : plusieurs types de nœuds et d'arêtes, fidèle à la T-Box.
  Modèle : **HeteroGNN** (`HeteroConv`) — **implémenté et retenu par défaut**.

Le graphe hétérogène préserve la sémantique de chaque prédicat RDF et
permet au GNN d'apprendre des poids distincts par type de relation.

---

<a id="6-gnn"></a>
## 7. Les Graph Neural Networks (GNN)

### L'intuition fondamentale — *message passing*

Un GNN apprend une **représentation vectorielle** (*embedding*) pour chaque
nœud en **agrégeant l'information de ses voisins**, puis en répétant
l'opération sur plusieurs couches.

À chaque couche :

```
       voisin_1 ──┐
       voisin_2 ──┼──▶ AGRÉGATION (somme, moyenne, attention…) ──▶ nouvel embedding du nœud
       voisin_k ──┘                                   ▲
                                                      │
                                          ancien embedding du nœud
```

Après *L* couches, l'embedding d'un nœud contient l'information de tout son
voisinage à distance *L* — c'est ainsi que le modèle "voit" la topologie.

### Trois architectures GNN courantes

| Modèle | Mécanisme d'agrégation | Quand l'utiliser |
|---|---|---|
| **GCN** *(Graph Convolutional Network)* | Moyenne pondérée par l'inverse du degré (équivalent d'une convolution spectrale). Simple, classique. | Premier essai, données homogènes. |
| **GraphSAGE** | Échantillonne un **sous-ensemble** des voisins, applique une agrégation apprise (mean / max / LSTM). Passe bien à l'échelle. | Grands graphes — adapté à notre cas. |
| **GAT** *(Graph Attention Network)* | Attribue un **poids d'attention appris** à chaque voisin, comme un Transformer. | Quand certains voisins sont bien plus informatifs que d'autres. |
| **HeteroConv / R-GCN** | **Transformation distincte par type de relation** (multi-relationnel). | Graphe RDF avec prédicats typés — **notre modèle par défaut**. |

### Tâches typiques pour un GNN

- **Node classification** : prédire un label par nœud (ex : compte
  frauduleux ou non).
- **Edge classification** : prédire un label par arête (ex : transaction
  frauduleuse ou non) — **c'est notre cas**.
- **Link prediction** : prédire si une arête manquante devrait exister.
- **Graph classification** : prédire un label pour un graphe entier.

### Différence clef avec un modèle tabulaire

| Aspect | Tabulaire | GNN |
|---|---|---|
| Unité d'entrée | Une ligne | Un nœud + son voisinage |
| Information sur les voisins | Aucune | Centrale (par message passing) |
| Sensibilité à l'ordre des lignes | Aucune | Aucune (mais sensible à la **structure**) |
| Coût d'entraînement | Faible | Élevé (souvent GPU) |
| Explicabilité | Bonne (importance des features) | Plus difficile (mais GNNExplainer existe) |

---

<a id="7-gnn-pour-aml"></a>
## 8. Pourquoi les GNN sont pertinents pour l'AML

### Reformulation de la problématique

Dans un schéma de blanchiment :
- **Chaque transaction**, prise isolément, semble normale.
- **L'ensemble des transactions**, par leur **arrangement géométrique**,
  trahit une intention frauduleuse.

Un GNN, en construisant la représentation d'une transaction à partir des
**comptes qui l'entourent** (et de leurs propres voisins), capture
exactement cette information structurelle.

### Trois apports concrets attendus

1. **Détection des motifs de fractionnement (*smurfing*)** : un nœud
   destinataire avec beaucoup d'entrées récentes voit son embedding
   "tagué" comme suspect, ce qui rejaillit sur les transactions
   nouvelles qui l'alimentent.
2. **Détection des comptes relais** : un compte qui n'a que des arêtes
   "reçoit puis renvoie immédiatement" produit un embedding très
   particulier, qu'on peut associer à la classe `isLaundering=1`.
3. **Détection des cycles** : avec deux ou trois couches de message
   passing, un compte impliqué dans un cycle voit son propre embedding
   se "boucler" — un GNN peut apprendre à reconnaître cette signature.

### Hypothèse de recherche du projet

> Le GNN devrait améliorer **principalement le rappel** (*recall*) sur la
> classe frauduleuse, en détectant des cas que le Random Forest manque
> parce qu'ils ne se distinguent pas dans leurs features unitaires.
> En contrepartie, on peut s'attendre à une **précision** au moins
> comparable, voire légèrement inférieure (plus de fraudes détectées
> implique souvent plus de faux positifs).

C'est précisément ce que la **comparaison de l'étape 6** cherchera à
vérifier.

---

<a id="9-ego-sampling"></a>
## 9. Échantillonnage par sous-graphe ego

### Le problème

Notre dataset brut contient 6,9 M de transactions — impossible à entraîner
en l'état sur un laptop. Un sous-échantillonnage est donc obligatoire,
mais **pas n'importe lequel**.

Un tirage aléatoire stratifié (cf. étape 2a) donne :
- ✅ une bonne proportion de fraudes (3.5 % au lieu de 0.05 %),
- ❌ un graphe **fragmenté** : ~70 % des comptes finissent avec une seule
  transaction, sans voisin atteignable.

Or un GNN apprend **par agrégation du voisinage**. Sans voisin → pas
d'information à agréger → le GNN se réduit à un MLP sur les features
d'arête. On perd le bénéfice central du graphe.

### La solution — *Ego-network sampling*

Un **sous-graphe ego** centré sur un nœud `v` est constitué :
- du nœud `v` (l'ego),
- de tous ses voisins jusqu'à `K` sauts (les *alters*),
- et de **toutes les arêtes** reliant ces nœuds entre eux.

Schéma pour K=1 autour d'un compte frauduleux :

```
        légitime_A
            │
            ▼
    ┌── FRAUDE_1 ──────┐
    │       │          │
    ▼       ▼          ▼
 légitime_B  FRAUDE_2  légitime_C
```

L'arête frauduleuse `FRAUDE_1` est conservée **avec son contexte**, ce qui
permet au GNN de "voir" la topologie qui l'entoure.

### Algorithme appliqué dans le projet

1. **Comptes graines** = tous les comptes touchés par une fraude.
2. **Expansion 1-hop** : ajout des voisins directs.
3. **Récolte** : toutes les transactions entre comptes du sous-graphe.
4. **Plafonnement** à `EGO_TARGET_SIZE` en gardant 100 % des fraudes.

Résultat sur **LI-Small (train)** : densité multipliée par ~2 (degré moyen
~1,5 → **3,30**), 200 000 transactions, **3 565** fraudes (1,78 %), 121 082
comptes, ~36 % de degré 1.

### Hold-out LI-Medium (02/06/2026)

Même algorithme (`--data-profile unknown`, lecture **chunked** sur 31,3 M
lignes) :

| Indicateur | LI-Medium hold-out |
|------------|-------------------|
| Transactions finales | 200 000 |
| Fraudes | 16 041 (8,02 %) |
| Ego avant plafond | 12 126 651 tx |
| Comptes | 216 008 |
| Degré moyen / médian | 1,85 / 1,00 |
| % degré = 1 | 62,72 % |

Le hold-out est **plus frauduleux** mais **moins connecté** que le train :
interpréter les métriques de généralisation avec prudence.

### Test final — métriques GNN (02/06/2026)

Évaluation **sans ré-entraînement** : checkpoints LI-Small → graphe LI-Medium (`07_eval_holdout.py`).

| Modèle | PR-AUC | Recall | Precision | F1 |
|--------|--------|--------|-----------|-----|
| **HeteroGNN** | **0,595** | 0,699 | **0,544** | **0,612** |
| GraphSAGE | 0,229 | 0,781 | 0,133 | 0,228 |
| GAT | 0,150 | 0,643 | 0,114 | 0,194 |

**Lecture** :

- **PR-AUC** : qualité du *classement* (quelle transaction est plus suspecte qu’une autre) — prioritaire en déséquilibre extrême.
- **Recall** : part des fraudes détectées au seuil choisi.
- **Precision** : part des alertes qui sont de vraies fraudes.
- HeteroGNN : bon compromis ; SAGE/GAT : recall élevé mais precision ~12 % → **sur-alerte**.

Fichier source : `data/unknown_data/results/eval_holdout_metrics.json`.  
Voir aussi [`pipeline_etl_entrainement_evaluation.md`](../pipeline_etl_entrainement_evaluation.md).

### Pourquoi pas K=2 ou K=3 ?

À chaque saut supplémentaire, le nombre de transactions explose
exponentiellement (effet "petit monde" du réseau financier). Sur ce
dataset, K=1 collecte déjà 2,4 M transactions (Small) ou **12,1 M** (Medium) ;
K=2 dépasserait
probablement le dataset entier. On peut envisager K=2 plus tard avec un
**neighborhood sampling** (par exemple `NeighborLoader` de PyG) qui
échantillonne *à la volée* un sous-ensemble aléatoire de voisins lors
de l'entraînement.

---

<a id="9b-pyg-data"></a>
## 10. Le format `Data` / `HeteroData` de PyTorch Geometric

PyG centralise l'information d'un graphe dans `Data` (homogène) ou
`HeteroData` (multi-types). C'est le format d'entrée standard pour tous
ses modèles.

### Anatomie d'un `Data` homogène (legacy)

| Attribut | Forme | Sens |
|---|---|---|
| `x` | `Tensor[N_nodes, F_node]` | Features par nœud |
| `edge_index` | `Tensor[2, N_edges]` (long) | Liste des arêtes |
| `edge_attr` | `Tensor[N_edges, F_edge]` | Features par arête |
| `y` | `Tensor[N_edges]` | Labels (edge classification) |

Fichier produit : `data/pyg/aml_graph.pt` — mode `--mode homo`.

### Anatomie d'un `HeteroData` (défaut du projet)

| Attribut | Exemple | Sens |
|---|---|---|
| `data['account'].x` | `[121 082, 5]` | Features comptes |
| `data['transaction'].x` | `[200 000, 27]` | Features transactions (T-Box dynamique) |
| `data['bank'].x` | `[4 126, 2]` | Features banques |
| `data['account', 'sends_via', 'transaction'].edge_index` | `[2, 200 000]` | Topologie assertée |
| `data['account', 'transacted_with', 'account'].edge_index` | `[2, 221 432]` | Lien **inféré** |
| `data['transaction'].y` | `[200 000]` | Label fraude (node classification) |

Fichier produit : `data/pyg/aml_hetero_graph.pt` — mode `--mode hetero` (défaut).

10 types d'arêtes (5 prédicats × 2 directions pour le message passing).

### Pourquoi `edge_index` et pas une matrice d'adjacence dense ?

Une matrice d'adjacence `N × N` est :
- coûteuse en mémoire (`N²` pour 121k nœuds = ~14 milliards de cases),
- creuse à 99,99 % (peu de paires sont connectées).

`edge_index` au format **COO** (*coordinate*) stocke uniquement les
arêtes existantes — c'est `O(E)` au lieu de `O(N²)`. Tous les modèles
PyG (GCN, GraphSAGE, GAT…) consomment ce format directement.

### Choix homogène vs hétérogène — implémentation actuelle

Notre T-Box décrit **3 types de nœuds** et **5 prédicats** (3 assertés +
2 inférés). Le projet utilise désormais **`HeteroData` par défaut** :

- Chaque prédicat RDF = type d'arête distinct dans `HeteroConv`.
- Les relations inférées (`transactedWith`, `sameBankAs`) enrichissent le
  message passing entre comptes.
- La tâche cible devient une **classification de nœuds** sur `transaction`
  (équivalent sémantique à l'edge classification homogène).

Le mode homogène (GraphSAGE) reste disponible via
`04_build_pyg.py --mode homo` pour comparaison.

### Classification — HeteroGNN (modèle par défaut)

**Étape 1 — Message passing multi-relationnel**
`HeteroConv` propage l'information le long de chaque type de prédicat
avec des poids **distincts** (sends_via ≠ transacted_with ≠ same_bank_as).

**Étape 2 — Tête de classification**
Pour chaque nœud `transaction` : concat(embedding GNN, features brutes T-Box)
→ MLP → logit fraude.

> **Mode homogène (legacy)** : la transaction est projetée sur une arête
> `Account → Account` ; le classificateur utilise
> `concat(h[src], h[dst], edge_attr)` — voir `src/gnn/homo/models.py`.

**Gestion du déséquilibre**
Avec seulement 1.78 % de fraudes, on utilise `BCEWithLogitsLoss` avec
`pos_weight = n_negatif / n_positif ≈ 55`. Chaque fraude manquée coûte
55× plus cher qu'un faux positif dans la loss — le modèle apprend à
privilégier le **recall**.

**Résultats obtenus (GraphSAGE, test set)**
PR-AUC = 0.254 | Recall = 93 % | Precision = 6 %

→ Le modèle détecte ~93 % des fraudes mais génère beaucoup de fausses
alertes. En AML, c'est souvent acceptable en première ligne (un analyste
filtre ensuite). La PR-AUC (~14× mieux que le hasard) confirme que le
signal topologique est **réel**, même si le modèle reste perfectible.

---

<a id="11-outils"></a>
## 11. Outils & librairies du projet

| Outil | Version | Pourquoi ? |
|---|---|---|
| **Python** | 3.12 | Standard scientifique, support de toutes les libs. |
| **pandas** | ≥ 2.2 | Lecture / manipulation du CSV (6,9 M lignes tient en RAM). |
| **numpy** | ≥ 1.26 | Calcul vectorisé sous-jacent. |
| **rdflib** | ≥ 7.0 | Construction, validation et requêtage SPARQL du graphe RDF. |
| **tqdm** | ≥ 4.66 | Barres de progression pour les boucles longues. |
| **pyarrow** | ≥ 15.0 | I/O Parquet rapide (anticipation d'éventuels caches). |
| **scikit-learn** | à venir | Random Forest, splits stratifiés, métriques. |
| **xgboost** | à venir (option) | Alternative gradient-boosting plus performante que RF en pratique. |
| **PyTorch** | à venir | Backend tensoriel pour le GNN. |
| **PyTorch Geometric (PyG)** | à venir | Implémentations prêtes des GNN (GCN, GraphSAGE, GAT). |

### Pourquoi PyTorch Geometric et pas DGL ?

| Critère | PyG | DGL |
|---|---|---|
| Adoption recherche | Très forte | Forte |
| Documentation francophone / tutoriels | Plus abondants | Moins |
| Intégration native avec l'écosystème PyTorch | Excellente | Excellente |
| Choix de modèles AML / fraude publiés | Souvent en PyG | Présents aussi |

→ PyG choisi pour la richesse des tutoriels publics et la simplicité
d'API.

---

<a id="12-metriques"></a>
## 12. Évaluation — métriques pour classes déséquilibrées

### Pourquoi l'*accuracy* est piégeuse

Avec **3.57 %** de fraudes dans notre échantillon, un modèle qui prédit
"jamais de fraude" obtient déjà **96.43 %** d'accuracy. Ce chiffre est
inutile en AML.

### Métriques pertinentes

| Métrique | Définition | Pourquoi utile en AML |
|---|---|---|
| **Precision** | Parmi les transactions prédites frauduleuses, quelle proportion l'est vraiment ? | Coût opérationnel des faux positifs (alertes à investiguer). |
| **Recall** | Parmi les fraudes réelles, quelle proportion est détectée ? | **Critique** — une fraude ratée a un coût bien supérieur à une alerte vaine. |
| **F1 (classe fraude)** | Moyenne harmonique precision / recall. | Compromis si on n'a pas de préférence forte. |
| **PR-AUC** | Aire sous la courbe Precision-Recall. | **La référence** en classes déséquilibrées : ne se laisse pas berner par la classe majoritaire. |
| **ROC-AUC** | Aire sous la courbe ROC. | À utiliser avec précaution : optimiste en classes déséquilibrées. |
| **Matrice de confusion** | TP / FP / TN / FN. | Lecture humaine immédiate. |

### Conventions pour le projet

- **Métrique principale de comparaison** : **PR-AUC**.
- **Métrique secondaire** : **Recall@Precision=0.5** (combien de fraudes
  on attrape en gardant au moins 1 alerte sur 2 vraies).
- **Split** : train / validation / test stratifié sur `is_laundering`,
  avec un test mis de côté **avant** toute exploration.

---

## Pour aller plus loin

- **Hamilton, W. L.** — *Graph Representation Learning*, 2020 (gratuit en
  ligne) : référence pédagogique pour les GNN.
- **PyTorch Geometric** — [https://pytorch-geometric.readthedocs.io](https://pytorch-geometric.readthedocs.io)
- **W3C — RDF Primer** — [https://www.w3.org/TR/rdf11-primer/](https://www.w3.org/TR/rdf11-primer/)
- **IBM AML dataset paper** — *Realistic Synthetic Financial Transactions
  for Anti-Money Laundering Models* (Altman et al., NeurIPS 2023).

---

*Dernière mise à jour : 01/06/2026 — complété avec la classification
d'arêtes (§10) et les résultats GraphSAGE de l'étape 5. Sera complété
à l'étape 6 (comparaison conceptuelle des deux approches).*
