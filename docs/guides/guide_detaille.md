# Guide détaillé du projet AML — Explications pas à pas

**Auteur** — Samir KHALDI  
**Date** — 01/06/2026  
**Public visé** — soutenance, relecture pédagogique, auto-révision  

> Ce document reprend **tout ce qui a été réalisé** sur le projet, avec des
> explications accessibles. Chaque terme technique est défini au moment où
> il apparaît, puis regroupé dans un glossaire en fin de document.

---

## Sommaire

1. [De quoi parle ce projet ?](#1-de-quoi-parle-ce-projet)
2. [Glossaire rapide (à lire en premier si besoin)](#2-glossaire-rapide)
3. [La grande idée : deux types de fraude, deux approches](#3-deux-types-de-fraude)
4. [Ce qui existait avant aujourd'hui](#4-ce-qui-existait-avant)
5. [Ce qui a été fait aujourd'hui — résumé chronologique](#5-ce-qui-a-ete-fait-aujourdhui)
6. [Le pipeline complet, expliqué étape par étape](#6-pipeline-complet)
7. [Les trois avantages du RDF, expliqués simplement](#7-trois-avantages-rdf)
8. [Le modèle HeteroGNN — comment il apprend](#8-heterognn)
9. [Les résultats obtenus et comment les interpréter](#9-resultats)
10. [Fichiers produits et leur rôle](#10-fichiers-produits)
11. [Comment présenter ce travail au jury](#11-presentation-jury)
12. [Glossaire complet](#12-glossaire-complet)

---

<a id="1-de-quoi-parle-ce-projet"></a>
## 1. De quoi parle ce projet ?

### Le problème métier

Les banques doivent détecter le **blanchiment d'argent** (*Anti-Money Laundering*,
abréviation **AML**). Concrètement : repérer des flux financiers suspects
avant qu'ils ne traversent tout le système bancaire.

Le blanchiment ne ressemble pas toujours à une transaction « bizarre » prise
seule. Souvent, **chaque virement semble normal**, mais c'est **l'ensemble
des virements**, leur **enchaînement** et leur **topologie** (qui paie qui,
via quels comptes relais) qui trahissent une fraude.

### La question de recherche

> *Est-ce qu'un modèle qui « voit » le réseau de transactions (un **GNN**)
> détecte mieux certains schémas de blanchiment qu'un modèle classique qui
> analyse chaque transaction **indépendamment** (comme un Random Forest) ?*

On ne compare pas les chiffres entre deux datasets différents. On montre que
**les deux approches ne voient pas la même chose**, et qu'elles se complètent.

### Les deux briques du projet

| Brique | Rôle | Outils |
|---|---|---|
| **Web Sémantique** | Modéliser le réseau financier de façon formelle, lisible, interrogeable | RDF, OWL, Turtle, SPARQL |
| **Intelligence artificielle graphe** | Apprendre à prédire la fraude en exploitant la structure du réseau | PyTorch Geometric, HeteroGNN |

---

<a id="2-glossaire-rapide"></a>
## 2. Glossaire rapide

Si vous êtes perdu, lisez ce mini-glossaire avant de continuer.

| Terme | Explication en une phrase |
|---|---|
| **AML** | *Anti-Money Laundering* — lutte contre le blanchiment d'argent. |
| **RDF** | Format standard (W3C) pour décrire des faits sous forme de **triplets** : `(sujet, prédicat, objet)`. |
| **Triplet** | Plus petite unité d'information RDF. Ex. : `(Compte_A, envoie_via, Transaction_42)`. |
| **Turtle (.ttl)** | Syntaxe texte lisible pour écrire des triplets RDF. |
| **T-Box** | Le **schéma** / vocabulaire : quels types d'entités et de relations existent. |
| **A-Box** | Les **données concrètes** : les instances réelles (comptes, transactions). |
| **Inférence** | Déduire automatiquement de **nouveaux faits** à partir de faits existants + règles. |
| **GNN** | *Graph Neural Network* — réseau de neurones qui apprend sur un graphe. |
| **PyG** | *PyTorch Geometric* — bibliothèque Python pour manipuler des graphes et entraîner des GNN. |
| **Tenseur** | Tableau multidimensionnel de nombres — format consommé par PyTorch / PyG. |
| **HeteroData** | Format PyG pour un graphe avec **plusieurs types** de nœuds et d'arêtes. |
| **Prédicat RDF** | Le **nom de la relation** dans un triplet (ex. `sends_via`, `sameBankAs`). |
| **HeteroConv** | Couche GNN qui applique une transformation **différente par type de relation**. |
| **PR-AUC** | Métrique de qualité quand les fraudes sont **très rares** (meilleur que l'accuracy). |
| **Recall** | Parmi les vraies fraudes, combien le modèle en a détecté. |
| **Precision** | Parmi les alertes du modèle, combien étaient de vraies fraudes. |

---

<a id="3-deux-types-de-fraude"></a>
## 3. La grande idée : deux types de fraude, deux approches

### Anomalie ponctuelle

**Définition** : une transaction est suspecte **toute seule**, sans regarder
les autres.

**Exemple** : achat de 12 000 € dans un pays où le titulaire n'a jamais voyagé.

**Approche adaptée** : tableau CSV, une ligne = une transaction. Modèles :
Random Forest, régression logistique.

**Dans votre parcours** : déjà fait dans le projet ML antérieur (Credit Card
Fraud, notebook `Activite_10.ipynb`).

### Anomalie structurelle

**Définition** : aucune transaction n'est suspecte isolément, mais **le schéma
global** l'est (cycle de virements, compte relais, fractionnement).

**Exemple** : 50 virements de 4 900 € entre les mêmes comptes — chaque montant
passe inaperçu, mais la **forme du réseau** est typique du *layering*
(empilement / brouillage).

**Approche adaptée** : graphe — nœuds = entités, arêtes = relations. Modèle :
GNN.

**Dans le présent projet** : dataset IBM AML, pipeline RDF → HeteroGNN.

```
┌─────────────────────────────┐     ┌─────────────────────────────┐
│  ANOMALIE PONCTUELLE        │     │  ANOMALIE STRUCTURELLE      │
│  « Cette ligne est bizarre »│     │  « Ce réseau est bizarre »  │
│  Random Forest / CSV        │     │  GNN / Graphe RDF           │
└─────────────────────────────┘     └─────────────────────────────┘
         │                                    │
         └─────────── COMPLÉMENTAIRES ────────┘
              (pas concurrentes en production)
```

---

<a id="4-ce-qui-existait-avant"></a>
## 4. Ce qui existait avant aujourd'hui

Avant la session du 01/06/2026, le projet disposait déjà de :

| Élément | Description |
|---|---|
| **T-Box** `aml.ttl` | Ontologie OWL : classes Account, Bank, Transaction + relations |
| **A-Box démo** `aml_data.ttl` | 100 000 transactions en RDF (Web Sémantique) |
| **Sous-graphe ego** | 200 000 transactions connectées autour des fraudes |
| **GraphSAGE homogène** | GNN simple : comptes = nœuds, transactions = arêtes aplatis |
| **Rapport final** | Comparaison conceptuelle ponctuel vs structurel |

### Limites de l'ancienne version

1. **Le GNN contournait parfois le RDF** (CSV → PyG direct) — le Web Sémantique
   n'était pas la source unique du graphe d'entraînement.
2. **Graphe homogène** : toutes les connexions étaient traitées pareil par le
   GNN — la sémantique des relations RDF (`sends_via` vs `hosted_by`) était perdue.
3. **Pas d'inférence** : le modèle devait « deviner » des relations métier
   évidentes (ex. deux comptes qui ont déjà transigé).
4. **Features codées en dur** : ajouter une nouvelle caractéristique demandait
   de modifier manuellement le code Python.

---

<a id="5-ce-qui-a-ete-fait-aujourdhui"></a>
## 5. Ce qui a été fait aujourd'hui — résumé chronologique

### Phase A — Clarifications et alignement RDF

1. **Confirmation** : le GNN consomme des **tenseurs PyG**, pas du Turtle brut.
   Le RDF est la **source canonique amont**, pas le format d'entrée du modèle.
2. **Pipeline RDF → PyG** formalisé : `src/rdf/to_homo_pyg.py` extrait les transactions
   depuis l'A-Box Turtle.
3. **Discussion** sur les avantages théoriques du RDF pour le GNN (multi-relationnel,
   inférence, extensibilité).

### Phase B — Implémentation des trois avantages RDF

4. **Extension de l'ontologie** : nouveaux prédicats `transactedWith`, `sameBankAs`,
   attribut `bankCode`.
5. **Module d'inférence** `src/rdf/inference.py` + script `02c_enrich_rdf.py`.
6. **Extraction dynamique de features** `src/rdf/features.py` (lecture auto de la T-Box).
7. **Graphe hétérogène** `src/rdf/to_hetero_pyg.py` → `aml_hetero_graph.pt`.
8. **Modèle HeteroGNN** `src/gnn/hetero/models.py` + entraînement `src/gnn/hetero/training.py`.
9. **Entraînement et validation** : PR-AUC 0,254, recall 94,8 %.
10. **Documentation** : mise à jour de `docs/guides/`, `docs/reports/`, `docs/analysis/`, `README.md`.

---

<a id="6-pipeline-complet"></a>
## 6. Le pipeline complet, expliqué étape par étape

Voici le chemin que suivent vos données, de la source brute jusqu'au modèle
entraîné. Chaque étape est expliquée avec son **objectif**, sa **commande** et
**ce qu'elle produit**.

```
Fichier IBM AML (6,9 M transactions)
            │
            ▼
    ┌───────────────────┐
    │ 03_subgraph_sample │  Échantillonnage « ego »
    └─────────┬─────────┘
              ▼
    LI-Small_Trans_subgraph.csv  (200 000 lignes)
              │
              ▼
    ┌──────────────────────┐
    │ 02b_build_rdf_subgraph│  CSV → triplets RDF
    └─────────┬────────────┘
              ▼
    aml_data_subgraph.ttl  (A-Box « assertée »)
              │
              ▼
    ┌───────────────────┐
    │ 02c_enrich_rdf     │  Inférence : nouveaux liens
    └─────────┬─────────┘
              ▼
    aml_data_subgraph_enriched.ttl  (A-Box enrichie)
              │
              ▼
    ┌───────────────────┐
    │ 04_build_pyg       │  RDF → tenseurs PyG
    └─────────┬─────────┘
              ▼
    aml_hetero_graph.pt  (HeteroData)
              │
              ▼
    ┌───────────────────┐
    │ 05_train_gnn       │  Entraînement HeteroGNN
    └─────────┬─────────┘
              ▼
    gnn_hetero_best.pt  (modèle entraîné)
```

---

### Étape 0 — La T-Box (`aml.ttl`) — le dictionnaire du projet

**Qu'est-ce que c'est ?**  
La **T-Box** (*Terminological Box*) est le **schéma** : elle dit quels types
d'entités existent et comment elles peuvent se relier, **avant** d'avoir des
données réelles.

**Analogie** : c'est le plan d'architecte. Il dit « un immeuble a des étages,
des portes, des couloirs » — sans dire où se trouve l'immeuble rue de Rivoli.

**Contenu de notre T-Box** :

| Classe OWL | Signification |
|---|---|
| `Account` | Compte bancaire |
| `Bank` | Établissement bancaire |
| `Transaction` | Opération financière |

| Propriété | Signification | Type |
|---|---|---|
| `sends_via` | Un compte **initie** une transaction | Relation (Account → Transaction) |
| `deposits_to` | Une transaction **crédite** un compte | Relation (Transaction → Account) |
| `hosted_by` | Un compte est **hébergé par** une banque | Relation (Account → Bank) |
| `transactedWith` | Deux comptes ont **déjà transigé** | Relation inférée (Account → Account) |
| `sameBankAs` | Deux comptes partagent la **même banque** et ont transigé | Relation inférée |
| `amountPaid` | Montant de la transaction | Attribut numérique |
| `isLaundering` | Label fraude (0 ou 1) | Attribut entier (cible du modèle) |

**Fichier** : `data/rdf/ontology_schema/aml.ttl`

---

### Étape 03 — Sous-graphe ego (échantillonnage intelligent)

**Commande** :
```powershell
python scripts/03_subgraph_sample.py
```

**Pourquoi cette étape ?**  
Un GNN apprend en **échangeant de l'information entre voisins** (*message
passing*). Si un compte n'a aucun voisin (degré = 1), le GNN ne voit que
les features locales — il se comporte comme un simple MLP.

Un échantillon aléatoire de 100 000 transactions produisait un graphe
**fragmenté** : ~70 % des comptes étaient isolés.

**Solution — échantillonnage « ego »** :  
On part des comptes impliqués dans des **fraudes connues**, on inclut leurs
**voisins à 1 saut** (K=1 hop), puis toutes les transactions entre ces comptes.

**Résultat** :

| Indicateur | Avant (naïf) | Après (ego) |
|---|---|---|
| Transactions | 100 000 | 200 000 |
| Degré moyen | ~1,5 | **3,30** |
| Comptes degré 1 | ~70 % | **36 %** |
| Fraudes conservées | 3 565 | 3 565 (100 %) |

**Fichier produit** : `data/csv/cleaned/LI-Small_Trans_subgraph.csv`

---

### Étape 02b — Construction de l'A-Box RDF

**Commande** :
```powershell
python scripts/02b_build_rdf_subgraph.py
```

**Qu'est-ce qu'une A-Box ?**  
L'**A-Box** (*Assertional Box*) contient les **instances réelles** conformes
à la T-Box. Chaque ligne du CSV devient une constellation de triplets RDF.

**Exemple concret** pour une transaction :

```turtle
# Le compte émetteur initie la transaction
inst:acct-BNP-12345  aml:sends_via      inst:tx-42 .

# La transaction crédite le compte destinataire
inst:tx-42             aml:deposits_to    inst:acct-SG-67890 .

# Attributs de la transaction
inst:tx-42             aml:amountPaid     "5000.0"^^xsd:float .
inst:tx-42             aml:isLaundering   "1"^^xsd:integer .
```

**Pourquoi Turtle et pas rester en CSV ?**

- La **topologie est explicite** : on voit clairement qui est relié à qui.
- Le format est **standard W3C** : interopérable, versionnable, auditable.
- On peut **interroger** le graphe en SPARQL (langage de requête sur RDF).
- C'est la **source de vérité** pour tout le pipeline aval.

**Fichier produit** : `data/rdf/ontology_data/aml_data_subgraph.ttl`  
(~80 Mo, ~1,97 million de triplets)

---

### Étape 02c — Inférence RDF (enrichissement du graphe)

**Commande** :
```powershell
python scripts/02c_enrich_rdf.py
```

**Qu'est-ce que l'inférence ?**  
L'inférence, c'est le fait de **déduire de nouveaux faits** à partir de faits
déjà connus, sans les lire explicitement dans le CSV d'origine.

**Analogie** : si vous savez que « Alice habite Paris » et « Bob habite Paris »,
vous pouvez inférer qu'« Alice et Bob habitent la même ville » — même si
personne ne l'a écrit noir sur blanc.

**Règles implémentées dans le projet** :

#### Règle 1 — `transactedWith`

```
SI  Compte_A ──sends_via──▶ Transaction_X ──deposits_to──▶ Compte_B
ALORS  Compte_A ──transactedWith──▶ Compte_B  (et symétriquement)
```

**Sens métier** : « ces deux comptes ont déjà eu affaire ensemble ».  
**Volume** : 110 716 paires → 221 432 triplets (relation symétrique).

#### Règle 2 — `sameBankAs`

```
SI  Compte_A ──hosted_by──▶ Banque_X
ET  Compte_B ──hosted_by──▶ Banque_X
ET  Compte_A ──transactedWith──▶ Compte_B
ALORS  Compte_A ──sameBankAs──▶ Compte_B
```

**Sens métier** : « virement interne entre comptes de la même banque ».  
**Volume** : 3 312 triplets (volontairement **sparse** pour ne pas exploser
le graphe).

**Pourquoi c'est utile pour le GNN ?**  
Le modèle reçoit un graphe **pré-enrichi**. Il n'a pas besoin de « deviner »
via plusieurs sauts qu deux comptes sont liés — la relation est **matérialisée**
directement.

**Fichier produit** : `data/rdf/ontology_data/aml_data_subgraph_enriched.ttl`  
(~90 Mo, ~2,2 millions de triplets, +224 744 triplets inférés)

**Option avancée** : `python scripts/02c_enrich_rdf.py --owlrl` active
l'expansion OWL-RL (nécessite `pip install owlrl`) pour inférer automatiquement
des propriétés comme la symétrie des relations.

---

### Étape 04 — Conversion RDF → HeteroData (tenseurs PyG)

**Commande** :
```powershell
python scripts/04_build_pyg.py
```

**Qu'est-ce qu'un tenseur ?**  
Un **tenseur** est un tableau de nombres que PyTorch manipule efficacement.
PyTorch Geometric convertit le graphe RDF en tenseurs que le GNN peut consommer.

**Qu'est-ce qu'un HeteroData ?**  
Par opposition au graphe **homogène** (un seul type de nœud, un seul type
d'arête), un **HeteroData** (*graphe hétérogène*) préserve **plusieurs types**
de nœuds et d'arêtes — exactement comme le modèle RDF.

**Structure produite** :

| Composant | Taille | Signification |
|---|---|---|
| `account.x` | [121 082, 5] | Features de chaque compte |
| `transaction.x` | [200 000, 27] | Features de chaque transaction |
| `bank.x` | [4 126, 2] | Features de chaque banque |
| `account → transaction` (sends_via) | 200 000 arêtes | Lien émetteur |
| `transaction → account` (deposits_to) | 200 000 arêtes | Lien destinataire |
| `account → bank` (hosted_by) | ~121 082 arêtes | Hébergement bancaire |
| `account → account` (transacted_with) | 221 432 arêtes | **Inféré** |
| `account → account` (same_bank_as) | 3 312 arêtes | **Inféré** |
| `transaction.y` | [200 000] | Label fraude (0/1) |

**Features dynamiques** : le module `src/rdf/features.py` lit la T-Box et extrait
automatiquement les attributs déclarés (`amountPaid`, `paymentCurrency`, etc.).
Si demain vous ajoutez `bankCountry` dans l'ontologie, il suffit de regénérer
le graphe — pas de refonte du code CSV.

**Fichier produit** : `data/pyg/aml_hetero_graph.pt`

---

### Étape 05 — Entraînement du HeteroGNN

**Commande** :
```powershell
python scripts/05_train_gnn.py
```

**Qu'est-ce qu'un GNN ?**  
Un **Graph Neural Network** (*réseau de neurones sur graphe*) apprend une
représentation numérique (*embedding*) pour chaque nœud en **agrégant
l'information de ses voisins**, couche après couche.

**Mécanisme — message passing (passage de messages)** :

```
Couche 1 : chaque compte « écoute » ses transactions directes
Couche 2 : l'information a propagé sur 2 sauts (comptes voisins des voisins)
```

**Pourquoi HeteroConv et pas GraphSAGE classique ?**  
GraphSAGE traite **toutes les arêtes pareil**. HeteroConv applique une
**transformation différente** selon le type de relation :

- recevoir un message via `sends_via` ≠ recevoir via `transacted_with`
- recevoir via `same_bank_as` ≠ recevoir via `hosted_by`

C'est l'équivalent pratique du **R-GCN** (*Relational Graph Convolutional
Network*) : le modèle apprend que certains types de liens sont plus
suspects que d'autres.

**Tâche du modèle** : prédire `isLaundering` pour chaque nœud `transaction`
(classification binaire : fraude ou non).

**Fichiers produits** :
- `data/results/gnn_hetero_best.pt` — poids du meilleur modèle
- `data/results/gnn_hetero_metrics.json` — métriques train/val/test

---

<a id="7-trois-avantages-rdf"></a>
## 7. Les trois avantages du RDF, expliqués simplement

### Avantage 1 — Multi-relationnel : le GNN comprend le *pourquoi* du lien

**Problème sans RDF typé** :  
Dans un graphe homogène, « Compte A est connecté à Compte B » — point. Le
modèle ne sait pas **pourquoi** ils sont connectés (virement ? même banque ?
même propriétaire ?).

**Avec RDF + HeteroConv** :  
Chaque prédicat est un **canal distinct**. Le modèle apprend des poids
séparés :

```
Poids_W1 pour sends_via       → « émetteur d'une transaction »
Poids_W2 pour transacted_with → « contrepartie historique »
Poids_W3 pour same_bank_as    → « co-localisation bancaire »
```

**Formulation jury** : *« Le RDF force la donnée à être typée ; le HeteroGNN
exploite cette typologie via des convolutions relationnelles distinctes. »*

---

### Avantage 2 — Inférence : un pré-traitement intelligent

**Problème sans inférence** :  
Pour savoir que deux comptes ont transigé, le GNN devrait « parcourir » le
graphe sur 2 sauts (compte → transaction → compte) à chaque couche. C'est
coûteux et indirect.

**Avec inférence** :  
La relation `transactedWith` est **matérialisée en amont**. Le GNN accède
directement au lien en 1 saut.

**Formulation jury** : *« L'inférence RDF joue le rôle d'un feature engineering
sémantique automatique, comparable à un triplestore qui pré-calcule des
jointures métier. »*

---

### Avantage 3 — Features dynamiques : extensibilité sans migration

**Problème avec un schéma SQL/CSV figé** :  
Ajouter « pays de la banque » = modifier la table, migrer les données,
mettre à jour le code d'encodage.

**Avec RDF** :  
Une feature = un triplet de plus :

```turtle
inst:bank-BNP  aml:bankCountry  "FR"^^xsd:string .
```

Le module `src/rdf/features.py` lit la T-Box, détecte `bankCountry` sur `Bank`,
et l'inclut automatiquement dans le tenseur `bank.x`.

**Formulation jury** : *« L'ontologie OWL sert de contrat d'interface entre
la modélisation métier et le pipeline ML — ajouter une propriété suffit
pour l'exposer au GNN. »*

---

<a id="8-heterognn"></a>
## 8. Le modèle HeteroGNN — comment il apprend

### Architecture simplifiée

```
┌─────────────────────────────────────────────────────────────┐
│                    GRAPHE HÉTÉROGÈNE                          │
│                                                               │
│   Account ──sends_via──▶ Transaction ──deposits_to──▶ Account │
│      │                                              │         │
│      └──hosted_by──▶ Bank                           │         │
│      │                                              │         │
│      └──── transacted_with / same_bank_as ──────────┘         │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │  Projection linéaire     │  Chaque type de nœud
              │  (account, tx, bank)   │  → espace commun (64 dim)
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  HeteroConv × 2 couches  │  Message passing
              │  (poids par prédicat)    │  multi-relationnel
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  Pour chaque transaction:│
              │  concat(embedding_GNN,   │
              │         features_T-Box)  │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │  MLP (3 couches)         │  → score de fraude
              └────────────┬────────────┘
                           │
                           ▼
                    fraude ? (0 ou 1)
```

### Hyperparamètres utilisés

| Paramètre | Valeur | Signification |
|---|---|---|
| `hidden_dim` | 64 | Taille des embeddings internes |
| `num_layers` | 2 | Nombre de couches de message passing |
| `dropout` | 0,3 | Régularisation (30 % de neurones « coupés » à l'entraînement) |
| `lr` | 0,001 | Vitesse d'apprentissage |
| `epochs` | 80 (early stop ~30) | Nombre max d'itérations |
| `pos_weight` | ~55 | Poids des fraudes dans la loss (classes déséquilibrées) |

### Gestion du déséquilibre des classes

Seulement **1,78 %** des transactions sont frauduleuses (3 565 / 200 000).
Sans correction, le modèle apprendrait à toujours prédire « pas fraude »
(accuracy ~98 % mais recall = 0 %).

**Solution** : `BCEWithLogitsLoss` avec `pos_weight = 55` — chaque fraude
manquée est **55× plus coûteuse** qu'un faux positif. Le modèle privilégie
le **recall** (détecter un maximum de fraudes).

---

<a id="9-resultats"></a>
## 9. Les résultats obtenus et comment les interpréter

### Métriques sur le jeu de TEST (30 000 transactions, dont 534 fraudes)

| Métrique | HeteroGNN | GraphSAGE homogène | Signification |
|---|---|---|---|
| **PR-AUC** | **0,254** | 0,254 | Qualité globale de ranking des fraudes |
| **Recall** | **94,8 %** | 92,7 % | 95 % des fraudes détectées |
| **Precision** | 5,9 % | 6,0 % | 6 % des alertes sont de vraies fraudes |
| **F1** | 0,112 | 0,113 | Compromis recall/precision |

### Comment lire ces chiffres ?

**PR-AUC = 0,254**  
- Baseline aléatoire ≈ 0,018 (proportion de fraudes).  
- Le modèle fait **~14× mieux que le hasard**.  
- Ce n'est pas « excellent » en absolu, mais c'est un **signal réel** sur un
  problème très difficile (graphe partiellement fragmenté, fraude rare).

**Recall = 94,8 %**  
- Sur 534 fraudes du test, le modèle en détecte **~506**.  
- En contexte AML, rater une fraude coûte souvent plus qu'une fausse alerte.

**Precision = 5,9 %**  
- Sur 100 alertes, ~6 sont de vraies fraudes.  
- Attendu avec `pos_weight` élevé : on « sur-alerté » volontairement.  
- En production, on **ajusterait le seuil** (pas forcément 0,5) via la courbe
  Precision-Recall pour équilibrer selon le coût métier.

**Comparaison HeteroGNN vs GraphSAGE**  
Les métriques sont **quasi identiques**. L'apport du HeteroGNN n'est pas
un gain de score immédiat, mais une **architecture sémantiquement plus riche**
et **défendable devant un jury** (multi-relationnel + inférence + extensibilité).

---

<a id="10-fichiers-produits"></a>
## 10. Fichiers produits et leur rôle

### Données

| Fichier | Taille approx. | Rôle |
|---|---|---|
| `LI-Small_Trans_subgraph.csv` | — | Sous-graphe ego 200k (CSV intermédiaire) |
| `aml_data_subgraph.ttl` | ~80 Mo | A-Box RDF assertée |
| `aml_data_subgraph_enriched.ttl` | ~90 Mo | A-Box enrichie par inférence |
| `aml_hetero_graph.pt` | ~50 Mo | Tenseurs PyG (HeteroData) |
| `aml_graph.pt` | — | Tenseurs PyG homogène (legacy) |

### Code source (nouveaux modules du 01/06)

| Fichier | Rôle en une phrase |
|---|---|
| `src/rdf/inference.py` | Applique les règles d'inférence sur l'A-Box |
| `src/rdf/features.py` | Extrait les features depuis la T-Box automatiquement |
| `src/rdf/to_hetero_pyg.py` | Convertit le RDF enrichi en HeteroData |
| `src/rdf/to_homo_pyg.py` | Convertit le RDF en graphe homogène (legacy) |
| `src/gnn/hetero/models.py` | Architecture HeteroConv |
| `src/gnn/hetero/training.py` | Boucle d'entraînement + métriques |
| `scripts/02c_enrich_rdf.py` | Script CLI pour l'inférence |

### Modèles entraînés

| Fichier | Modèle |
|---|---|
| `gnn_hetero_best.pt` | HeteroGNN (défaut) |
| `gnn_hetero_metrics.json` | Métriques JSON |
| `gnn_sage_best.pt` | GraphSAGE homogène (legacy) |

---

<a id="11-presentation-jury"></a>
## 11. Comment présenter ce travail au jury

### Pitch de 2 minutes

> « Notre projet démontre que la détection de fraude financière ne se résume
> pas à analyser des lignes CSV. Certaines fraudes — le blanchiment organisé —
> vivent dans la **structure du réseau** : cycles, comptes relais, fractionnement.
>
> Nous avons modélisé un réseau de 200 000 transactions IBM AML en **RDF** —
> un standard du Web Sémantique — puis enrichi ce graphe par **inférence**
> (relations `transactedWith`, `sameBankAs`).
>
> Un **GNN hétérogène** (HeteroConv) apprend ensuite avec un **poids distinct
> par type de relation**, ce qui préserve la sémantique du modèle RDF.
>
> Résultat : PR-AUC 0,25, recall 95 % — signal réel de détection, complémentaire
> de l'approche tabulaire (Random Forest) déjà validée sur un autre dataset. »

### Questions fréquentes du jury — réponses préparées

**Q : Pourquoi RDF si le GNN lit des tenseurs ?**  
R : Le RDF est la **source canonique** amont. Il formalise la topologie,
permet l'inférence et l'extensibilité. PyG n'est qu'un **format d'exécution**
pour le modèle — comme un compilateur transforme du code source en binaire.

**Q : Pourquoi les métriques ne sont pas meilleures avec le HeteroGNN ?**  
R : Le graphe reste partiellement fragmenté (36 % de comptes isolés), les
features temporelles ne sont pas encore parsées, et le dataset est très
déséquilibré. L'apport du HeteroGNN est **architectural et sémantique**,
pas uniquement un gain de score immédiat.

**Q : Quelle différence avec le Random Forest du projet ML antérieur ?**  
R : Le RF analyse chaque transaction **indépendamment**. Il ne voit pas les
cycles, les comptes relais, les fan-in/fan-out. Les deux approches couvrent
des **familles d'anomalies différentes** et se complètent en production.

**Q : C'est utilisable en production ?**  
R : C'est une **preuve de concept** académique. En production, il faudrait
un flux temps réel, un GPU, un tuning de seuil, et une intégration avec
le filtre tabulaire existant.

---

<a id="12-glossaire-complet"></a>
## 12. Glossaire complet

### Web Sémantique

| Terme | Définition détaillée |
|---|---|
| **RDF** | *Resource Description Framework* — cadre W3C pour représenter des connaissances sous forme de triplets `(sujet, prédicat, objet)`. |
| **Triplet** | Unité atomique RDF. Ex. : `(Compte_A, sends_via, Transaction_1)`. |
| **Turtle** | Syntaxe texte pour écrire/lire des triplets RDF (extension `.ttl`). |
| **OWL** | *Web Ontology Language* — langage pour définir des ontologies avec classes, propriétés, contraintes logiques. |
| **Ontologie** | Spécification formelle d'un domaine : quels concepts existent et comment ils se relient. |
| **T-Box** | Partie **terminologique** de l'ontologie = le schéma (vocabulaire). Fichier : `aml.ttl`. |
| **A-Box** | Partie **assertionnelle** = les instances concrètes (données). Fichiers : `aml_data*.ttl`. |
| **Prédicat** | Le « verbe » d'un triplet RDF — nom de la relation (ex. `sends_via`). |
| **SPARQL** | Langage de requête pour interroger des graphes RDF (équivalent du SQL pour les graphes). |
| **Inférence** | Déduction automatique de nouveaux triplets à partir de triplets existants + règles/axiomes. |
| **OWL-RL** | Profil OWL déductif implémentable efficacement (symétrie, transitivité, domaine/portée). |
| **Triplestore** | Base de données optimisée pour stocker et interroger des triplets RDF (ex. GraphDB). |

### Graphes et GNN

| Terme | Définition détaillée |
|---|---|
| **Graphe** | Structure mathématique G = (V, E) : ensemble de nœuds V et d'arêtes E. |
| **Nœud** | Entité du graphe (compte, transaction, banque). |
| **Arête** | Lien orienté ou non entre deux nœuds. |
| **Degré** | Nombre d'arêtes connectées à un nœud. |
| **Graphe homogène** | Un seul type de nœud, un seul type d'arête. |
| **Graphe hétérogène** | Plusieurs types de nœuds et/ou d'arêtes. |
| **Message passing** | Mécanisme GNN : chaque nœud agrège l'information de ses voisins pour mettre à jour son embedding. |
| **Embedding** | Vecteur numérique appris représentant un nœud ou une arête. |
| **GNN** | *Graph Neural Network* — famille de réseaux de neurones opérant sur des graphes. |
| **GraphSAGE** | Architecture GNN qui échantillonne et agrège les voisins (scalable). |
| **GAT** | *Graph Attention Network* — GNN avec mécanisme d'attention sur les voisins. |
| **HeteroConv** | Couche PyG appliquant une convolution **distincte par type d'arête** dans un graphe hétérogène. |
| **R-GCN** | *Relational GCN* — GNN multi-relationnel ; HeteroConv en est l'équivalent pratique dans PyG. |
| **PyTorch Geometric (PyG)** | Bibliothèque Python pour deep learning sur graphes, extension de PyTorch. |
| **HeteroData** | Classe PyG encapsulant un graphe hétérogène (plusieurs types de nœuds/arêtes). |
| **Data** | Classe PyG pour un graphe homogène. |
| **edge_index** | Tenseur `[2, num_arêtes]` listant les connexions (format COO). |
| **Tenseur** | Tableau multidimensionnel de nombres (type de données PyTorch). |

### Machine Learning

| Terme | Définition détaillée |
|---|---|
| **Classification binaire** | Prédire entre deux classes (ici : fraude / pas fraude). |
| **Label** | Valeur cible à prédire (`isLaundering` : 0 ou 1). |
| **Feature** | Variable d'entrée du modèle (montant, devise, degré du compte…). |
| **Train / Val / Test** | Découpage des données : entraînement (70 %), validation (15 %), test (15 %). |
| **Early stopping** | Arrêt de l'entraînement quand la validation cesse de s'améliorer (évite le surapprentissage). |
| **Loss** | Fonction de coût mesurant l'erreur du modèle (`BCEWithLogitsLoss` ici). |
| **pos_weight** | Poids des exemples positifs (fraudes) dans la loss — compense le déséquilibre. |
| **PR-AUC** | Aire sous la courbe Precision-Recall — métrique adaptée aux classes rares. |
| **Recall (rappel)** | TP / (TP + FN) — proportion de fraudes réelles détectées. |
| **Precision** | TP / (TP + FP) — proportion d'alertes correctes. |
| **F1-score** | Moyenne harmonique de precision et recall. |
| **Surapprentissage** | Le modèle mémorise le train mais généralise mal au test. |

### Domaine AML

| Terme | Définition détaillée |
|---|---|
| **AML** | *Anti-Money Laundering* — ensemble des mesures contre le blanchiment. |
| **Blanchiment** | Processus de dissimuler l'origine illicite de fonds via le système financier. |
| **Layering** | Phase d'empilement : multiplier les transactions pour brouiller la traçabilité. |
| **Fan-in** | Motif où un compte reçoit de nombreux virements (regroupement). |
| **Fan-out** | Motif où un compte envoie de nombreux virements (fractionnement). |
| **Pass-through** | Compte relais : reçoit puis renvoie rapidement les fonds. |
| **Cycle** | Chemin fermé dans le graphe (l'argent revient à son point de départ). |
| **Sous-graphe ego** | Sous-ensemble du graphe centré sur un nœud et ses voisins à K sauts. |

---

## Commandes de reproduction

```powershell
# Installation (une fois)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Pipeline complet (défaut — HeteroGNN)
python scripts/03_subgraph_sample.py
python scripts/02b_build_rdf_subgraph.py
python scripts/02c_enrich_rdf.py
python scripts/04_build_pyg.py
python scripts/05_train_gnn.py

# Mode legacy (comparaison)
python scripts/04_build_pyg.py --mode homo
python scripts/05_train_gnn.py --model sage

# Visualisation + convergence alignée (optionnel)
jupyter notebook notebooks/metrics_evaluation.ipynb
python scripts/06_compare_convergence.py --epochs 40 --retrain
```

---

## Documents connexes

| Document | Contenu |
|---|---|
| [`docs/guides/concepts_explained.md`](concepts_explained.md) | Référence conceptuelle (RDF, GNN, métriques…) |
| [`docs/reports/final_report.md`](../reports/final_report.md) | Rapport de synthèse pour la soutenance |
| [`docs/reports/process_log.md`](../reports/process_log.md) | Journal chronologique des décisions et chiffres |
| [`docs/analysis/comparaison_modeles_gnn.md`](../analysis/comparaison_modeles_gnn.md) | Comparaison modèles + convergence |
| [`README.md`](../../README.md) | Guide rapide d'exécution |

---

*Document généré le 01/06/2026 — reflète l'état du projet après implémentation
du pipeline RDF enrichi + HeteroGNN multi-relationnel.*
