# Le Forward Pass du HeteroTransactionGNN — Guide & Schéma

**Projet** — Connect AI / AML — Samir KHALDI  
**Public visé** — Tout public ayant une notion de base en réseau de neurones  
**Fichier source** — `src/gnn/hetero/models.py` — `HeteroTransactionGNN`

> Ce document explique pas à pas comment le modèle GNN hétérogène traite
> les données du graphe et produit un **score de fraude** pour chaque transaction.
> Les dimensions indiquées sont celles du modèle entraîné sur LI-Small.

---

## Sommaire

1. [Ce que le modèle reçoit en entrée](#1-entrees)
2. [Le schéma du réseau de neurones](#2-schema)
3. [Étape 1 — La projection linéaire](#3-projection)
4. [Étape 2 & 3 — Les couches HeteroConv (passage de messages)](#4-heteroconv)
5. [Étape 4 — La skip connection et la tête de classification](#5-tete)
6. [L'entraînement — comment le modèle apprend](#6-entrainement)
7. [Résumé visuel du flux complet](#7-resume)

---

## 1. Ce que le modèle reçoit en entrée

Le modèle ne reçoit pas un tableau de transactions — il reçoit un **graphe structuré** avec trois types de nœuds et dix types de relations.

### Les nœuds et leurs dimensions de features

| Type de nœud | Nombre de features | Exemples de features |
|---|---|---|
| **Account** (compte bancaire) | **5** | Degré (nbre de transactions), features structurelles |
| **Transaction** | **27** | Montant, format, devise, horodatage, codes banques (one-hot) |
| **Bank** (banque) | **2** | Code banque (one-hot) |

> **Pourquoi des dimensions différentes ?** Chaque entité du monde réel a ses propres attributs. Un compte bancaire n'a pas les mêmes propriétés qu'une transaction financière. C'est précisément ce que le modèle hétérogène sait gérer — contrairement au modèle homogène qui force tout dans le même moule.

### Les relations entre nœuds (10 types d'arêtes)

| Relation | De → Vers | Origine |
|---|---|---|
| `sends_via` | Account → Transaction | Données brutes |
| `deposits_to` | Transaction → Account | Données brutes |
| `hosted_by` | Account → Bank | Données brutes |
| `transacted_with` | Account → Account | **Inférée par OWL** |
| `same_bank_as` | Account → Account | **Inférée par OWL** |
| `rev_sends_via` | Transaction → Account | Arête inverse (PyG) |
| `rev_deposits_to` | Account → Transaction | Arête inverse (PyG) |
| `rev_hosted_by` | Bank → Account | Arête inverse (PyG) |
| `rev_transacted_with` | Account → Account | Arête inverse |
| `rev_same_bank_as` | Account → Account | Arête inverse |

> Les 5 premières relations viennent du graphe RDF enrichi. Les 5 relations inverses sont générées automatiquement par PyTorch Geometric pour permettre au message de circuler dans les deux sens.

---

## 2. Le schéma du réseau de neurones

```
╔══════════════════════════════════════════════════════════════════════════╗
║                         HÉTÉROGÉNN — FORWARD PASS                        ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  ENTRÉES                                                                 ║
║  ┌─────────────────┐  ┌──────────────────────┐  ┌─────────────────┐    ║
║  │  Account         │  │  Transaction          │  │  Bank           │    ║
║  │  [N_a × 5]       │  │  [N_t × 27]           │  │  [N_b × 2]      │    ║
║  └────────┬─────────┘  └──────────┬────────────┘  └────────┬────────┘    ║
║           │                       │                         │             ║
║  ═════════╪═══════════════════════╪═════════════════════════╪═════════    ║
║  ÉTAPE 1  │  Projection Linéaire  │                         │             ║
║  ═════════╪═══════════════════════╪═════════════════════════╪═════════    ║
║           │                       │                         │             ║
║      Linear(5→64)           Linear(27→64)             Linear(2→64)       ║
║      + ReLU                 + ReLU                    + ReLU             ║
║           │                       │                         │             ║
║  ┌────────▼─────────┐  ┌──────────▼────────────┐  ┌────────▼────────┐    ║
║  │  Account emb.    │  │  Transaction emb.      │  │  Bank emb.      │    ║
║  │  [N_a × 64]      │  │  [N_t × 64]            │  │  [N_b × 64]     │    ║
║  └──────────────────┘  └───────────────────────┘  └─────────────────┘    ║
║                  │               │              │                          ║
║                  │    + 10 types d'arêtes RDF   │                          ║
║                  └───────────────┬──────────────┘                          ║
║                                  │                                         ║
║  ═══════════════════════════════╪═════════════════════════════════════    ║
║  ÉTAPE 2  HeteroConv — Couche 1 │                                         ║
║  ═══════════════════════════════╪═════════════════════════════════════    ║
║                                  │                                         ║
║  ┌───────────────────────────────▼─────────────────────────────────────┐  ║
║  │  10 × SAGEConv indépendants (un par type de relation)               │  ║
║  │                                                                     │  ║
║  │  W_sends_via        · Account  ──────►  Transaction                 │  ║
║  │  W_deposits_to      · Transaction ───►  Account                     │  ║
║  │  W_hosted_by        · Account  ──────►  Bank                        │  ║
║  │  W_transacted_with  · Account  ──────►  Account                     │  ║
║  │  W_same_bank_as     · Account  ──────►  Account                     │  ║
║  │  ... + 5 inverses                                                   │  ║
║  │                                                                     │  ║
║  │  Agrégation : sum des messages reçus par chaque nœud                │  ║
║  └─────────────────────────────────────────────────────────────────────┘  ║
║            │                                                               ║
║      LayerNorm + ReLU + Dropout(0.3)                                       ║
║            │                                                               ║
║  [Tous les nœuds mis à jour — 64 dim]                                      ║
║                                  │                                         ║
║  ═══════════════════════════════╪═════════════════════════════════════    ║
║  ÉTAPE 3  HeteroConv — Couche 2 │   (même architecture, nouveaux poids)    ║
║  ═══════════════════════════════╪═════════════════════════════════════    ║
║                                  │                                         ║
║  [même opération, profondeur 2 — voisins des voisins]                      ║
║            │                                                               ║
║      LayerNorm + ReLU + Dropout(0.3)                                       ║
║            │                                                               ║
║  ════════════════════════════════════════════════════════════════════════  ║
║  ÉTAPE 4  Tête de classification — Transaction uniquement                  ║
║  ════════════════════════════════════════════════════════════════════════  ║
║                                                                            ║
║  Transaction emb. [N_t × 64]  ──┐                                         ║
║                                  ├──► concat ──► [N_t × 91]               ║
║  Transaction raw  [N_t × 27]  ──┘     (64 + 27 = 91)                      ║
║                    ↑ skip connection (features brutes préservées)          ║
║                                  │                                         ║
║              Linear(91 → 64) + ReLU + Dropout(0.3)                        ║
║              Linear(64 → 32) + ReLU + Dropout(0.3)                        ║
║              Linear(32 → 1)                                                ║
║                                  │                                         ║
║              ┌───────────────────▼───────────────────┐                     ║
║              │   Logit fraude  [N_t × 1]             │                     ║
║              │   sigmoid(logit) → proba ∈ [0, 1]     │                     ║
║              └───────────────────────────────────────┘                     ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 3. Étape 1 — La projection linéaire

### Le problème

Chaque type de nœud a un nombre **différent** de features :
- Account : **5** colonnes
- Transaction : **27** colonnes
- Bank : **2** colonnes

Le message passing nécessite que tous les nœuds parlent le **même langage** (même dimension vectorielle) pour pouvoir s'échanger des informations.

### La solution : une couche linéaire par type de nœud

Une couche linéaire est une simple multiplication matricielle suivie d'un biais :

```
h = ReLU( W × x + b )

Account :     W ∈ ℝ^{64×5}   — "traduit" 5 features en 64 dimensions
Transaction : W ∈ ℝ^{64×27}  — "traduit" 27 features en 64 dimensions
Bank :        W ∈ ℝ^{64×2}   — "traduit" 2 features en 64 dimensions
```

Après cette étape, tous les nœuds sont représentés par un vecteur de **64 nombres** — un espace commun où les comparaisons et échanges sont possibles.

> **Analogie** : imaginez que trois témoins d'un crime parlent chacun une langue différente (anglais, arabe, mandarin). Avant de les confronter, un interprète traduit leurs témoignages dans un formulaire commun de 64 questions. Ensuite, on peut comparer et croiser les informations.

---

## 4. Étapes 2 & 3 — Les couches HeteroConv

### Le principe du message passing

Chaque nœud **écoute ses voisins** et met à jour sa représentation en fonction de ce qu'il entend. Ce processus se répète pour chaque type de relation, avec des **poids distincts** pour chaque prédicat RDF.

```
Pour chaque nœud n de type "transaction" :

  Message reçu via sends_via    = W₁ · h(compte émetteur)
  Message reçu via rev_deposits_to = W₇ · h(compte bénéficiaire)
  ...

  Nouveau h(transaction) = LayerNorm( ReLU( Somme de tous ces messages ) )
```

### Pourquoi des poids W distincts par relation ?

Dans un GNN homogène (GraphSAGE), **un seul W** traite toutes les relations :

```
Message d'un compte émetteur    → W · h(voisin)
Message d'une banque hébergeuse → W · h(voisin)   ← MÊME W !
```

Le modèle ne sait pas faire la différence entre "je reçois un message de mon émetteur" et "je reçois un message de ma banque" — toute l'information sémantique est perdue.

Dans le HeteroGNN, chaque relation a son propre W :

```
Message d'un compte émetteur    → W_sends_via · h(voisin)
Message d'une banque hébergeuse → W_hosted_by · h(voisin)  ← W DIFFÉRENT
```

Le modèle peut apprendre que la relation `transacted_with` (deux comptes ayant déjà échangé) est **différente** de `same_bank_as` (deux comptes dans la même banque), et leur attribuer des importances différentes.

### Deux couches = deux degrés de voisinage

```
Couche 1 : chaque transaction apprend de ses voisins directs
           (les comptes émetteur et bénéficiaire, leur banque)

Couche 2 : chaque transaction apprend des voisins de ses voisins
           (les autres transactions de ces comptes, leurs réseaux)
```

Après 2 couches, une transaction "connaît" non seulement ses contreparties directes, mais aussi leur **comportement passé** (via `transacted_with`) et leur **appartenance bancaire** (via `same_bank_as`).

### LayerNorm, ReLU, Dropout

Après chaque couche HeteroConv, trois opérations de régularisation :

| Opération | Rôle |
|-----------|------|
| **LayerNorm** | Normalise les valeurs pour que l'entraînement reste stable — évite les explosions de gradient |
| **ReLU** | Active uniquement les signaux positifs — introduit la non-linéarité nécessaire à l'apprentissage |
| **Dropout(0.3)** | Désactive aléatoirement 30 % des neurones — force le modèle à ne pas mémoriser mais à généraliser |

---

## 5. Étape 4 — La skip connection et la tête de classification

### La skip connection — pourquoi ne pas oublier le point de départ

Après 2 couches de message passing, l'embedding d'une transaction est **riche en contexte graphe** — il a "digéré" l'information de son voisinage. Mais les features **brutes** de la transaction (montant exact, devise, format) pourraient avoir été diluées au fil des couches.

La **skip connection** résout ce problème en concaténant :

```
h_final = concat( embedding_GNN [64 dim] , features_brutes_transaction [27 dim] )
        = vecteur de 91 dimensions
```

Le modèle dispose ainsi **simultanément** de :
- Ce que le graphe dit de la transaction (son contexte structurel, ses relations)
- Ce que la transaction dit d'elle-même (ses attributs propres)

> **Analogie** : un juge qui a écouté tous les témoins (contexte graphe) ne jette pas pour autant les pièces matérielles du dossier (features brutes). Il combine les deux pour rendre son verdict.

### Le MLP — le décideur final

Un réseau de neurones classique (MLP = Multi-Layer Perceptron) réduit progressivement le vecteur de 91 dimensions jusqu'à **un seul nombre** :

```
[91 dim] → Linear(91→64) → ReLU → Dropout
[64 dim] → Linear(64→32) → ReLU → Dropout
[32 dim] → Linear(32→1)
[1 dim]  → logit de fraude
```

Ce logit est ensuite transformé en probabilité via une **sigmoïde** :

```
sigmoid(logit) → probabilité ∈ [0, 1]

  logit = −3  →  proba ≈ 0,05  →  probablement légitime
  logit =  0  →  proba = 0,50  →  incertain
  logit = +3  →  proba ≈ 0,95  →  probablement fraude
```

---

## 6. L'entraînement — comment le modèle apprend

### La fonction de perte : BCEWithLogitsLoss

Pendant l'entraînement, chaque transaction a un **label réel** : `isLaundering = 0` (légitime) ou `1` (fraude).

Le modèle compare sa prédiction au label réel et calcule une **erreur** (la loss) :

```
Si la transaction est une fraude (label = 1) :
  → Le modèle prédit 0,05  →  erreur GRANDE
  → Le modèle prédit 0,95  →  erreur petite

Si la transaction est légitime (label = 0) :
  → Le modèle prédit 0,95  →  erreur GRANDE
  → Le modèle prédit 0,05  →  erreur petite
```

### Le déséquilibre des classes : pos_weight ≈ 55

Dans LI-Small, il y a **~55 transactions légitimes pour 1 fraude**. Sans correction, le modèle apprendrait à tout prédire "légitime" (98,2 % d'accuracy — inutile).

Le paramètre `pos_weight = 55` dit au modèle :

> *« Une erreur sur une fraude compte 55 fois plus qu'une erreur sur une transaction légitime. »*

Cela force le modèle à prêter beaucoup plus d'attention aux rares cas de fraude.

### La rétropropagation

Après chaque batch d'exemples, l'erreur est **propagée à rebours** à travers tout le réseau (du logit final jusqu'aux couches de projection initiales). Chaque poids W est ajusté d'un petit pas pour réduire l'erreur — c'est ce qu'on appelle la **descente de gradient** (lr = 0,001).

Ce processus se répète sur 40 epochs (passages complets sur les données d'entraînement).

---

## 7. Résumé visuel du flux complet

```
Données brutes CSV (LI-Small)
         │
         ▼
Graphe RDF + inférence OWL
(+225 000 triplets : transacted_with, same_bank_as)
         │
         ▼
HeteroData PyG
┌────────────────────────────────────────────────┐
│  Account [N_a × 5]                             │
│  Transaction [N_t × 27]   ← label isLaundering │
│  Bank [N_b × 2]                                │
│  10 types d'arêtes                             │
└────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  [1] Projection linéaire    │  5 / 27 / 2  →  64 dim (tous)
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  [2] HeteroConv couche 1   │  10 SAGEConv × poids distincts
│      + LayerNorm + ReLU    │  Voisins directs (profondeur 1)
│      + Dropout 0.3         │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  [3] HeteroConv couche 2   │  10 SAGEConv × poids distincts
│      + LayerNorm + ReLU    │  Voisins des voisins (profondeur 2)
│      + Dropout 0.3         │
└─────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  [4] Skip connection                        │
│      concat(emb_GNN [64], raw_tx [27])      │  = 91 dim
│                                             │
│      MLP  91 → 64 → 32 → 1                 │
│      + ReLU + Dropout à chaque couche       │
└─────────────────────────────────────────────┘
         │
         ▼
  Score fraude ∈ [0, 1]  par transaction
  ─────────────────────────────────────
  > 0,5  →  Alerte fraude
  ≤ 0,5  →  Transaction légitime
```

### Nombre total de paramètres appris

| Bloc | Paramètres estimés |
|------|-------------------|
| Projection Account (5→64) | 5×64 + 64 = **384** |
| Projection Transaction (27→64) | 27×64 + 64 = **1 792** |
| Projection Bank (2→64) | 2×64 + 64 = **192** |
| HeteroConv couche 1 (10 SAGEConv) | ~10 × 2×64×64 = **81 920** |
| HeteroConv couche 2 (10 SAGEConv) | ~10 × 2×64×64 = **81 920** |
| MLP tête (91→64→32→1) | ~8 100 |
| **Total estimé** | **~174 000 paramètres** |

Un modèle compact et efficace — entraîné en 40 epochs sur LI-Small.

---

*Document rédigé dans le cadre du projet Connect AI — Master Data & IA — YNOV*  
*Auteur : Samir KHALDI*  
*Fichier source : `src/gnn/hetero/models.py`*
