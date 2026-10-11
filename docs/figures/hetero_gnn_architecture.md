# Architecture HeteroGNN — schémas visuels

**Projet** — Connect AI / AML — Samir KHALDI  
**Module** — `src/gnn/hetero/models.py` (`HeteroTransactionGNN`)  
**Graphe** — `HeteroData` PyG depuis RDF enrichi (`src/rdf/to_hetero_pyg.py`)

Ce document regroupe les diagrammes Mermaid pour illustrer l'architecture HeteroGNN
(graphe hétérogène + réseau de neurones). Les schémas sont versionnables et
exportables en PNG pour slides ou rapport.

**Documents liés** : [`guide_detaille.md`](../guides/guide_detaille.md) (§8),
[`comparaison_modeles_gnn.md`](../analysis/comparaison_modeles_gnn.md),
[`pipeline_etl_entrainement_evaluation.md`](../pipeline_etl_entrainement_evaluation.md).

---

## 1. Graphe hétérogène (RDF → PyG)

Chaque prédicat RDF devient un type d'arête distinct dans PyG. Les relations
`transacted_with` et `same_bank_as` sont **inférées** par `src/rdf/inference.py`
à partir des transactions observées.

```mermaid
flowchart LR
  subgraph nodes ["Types de nœuds"]
    A["Account<br/>(compte bancaire)"]
    T["Transaction<br/>(label : isLaundering)"]
    B["Bank<br/>(banque hébergeuse)"]
  end

  A -->|"sends_via<br/>(émetteur)"| T
  T -->|"deposits_to<br/>(bénéficiaire)"| A
  A -->|"hosted_by"| B
  A -.->|"transacted_with<br/>(inféré)"| A
  A -.->|"same_bank_as<br/>(inféré)"| A

  T -.->|"tâche ML"| F["Score de fraude<br/>(classification nœud)"]

  style T fill:#fff3e0,stroke:#e65100
  style A fill:#e3f2fd,stroke:#1565c0
  style B fill:#e8f5e9,stroke:#2e7d32
  style F fill:#fce4ec,stroke:#c2185b
```

| Élément | Détail |
|---------|--------|
| **Nœuds** | `account`, `transaction`, `bank` |
| **Relations observées** | `sends_via`, `deposits_to`, `hosted_by` |
| **Relations inférées** | `transacted_with`, `same_bank_as` (+ arêtes inverses dans PyG) |
| **Label** | `isLaundering` sur chaque nœud `transaction` (0/1) |
| **Fichier PyG** | `data/train_data/pyg/aml_hetero_graph.pt` |

**Message clé pour la soutenance** : contrairement au graphe homogène (compte→compte),
la sémantique de chaque prédicat RDF est **préservée** — le modèle peut apprendre
des transformations distinctes par type de relation.

---

## 2. Forward pass du modèle `HeteroTransactionGNN`

Architecture implémentée dans `HeteroTransactionGNN` : projection linéaire par type
de nœud, puis 2 couches `HeteroConv` (chaque prédicat → `SAGEConv` dédié),
concaténation skip avec les features brutes transaction, et MLP de classification.

```mermaid
flowchart TB
  subgraph input ["Entrées HeteroData"]
    XA["x account<br/>(features T-Box + structurelles)"]
    XT["x transaction<br/>(montant, format, devise…)"]
    XB["x bank<br/>(features T-Box dynamiques)"]
    E["edge_index_dict<br/>(une matrice par prédicat RDF)"]
  end

  subgraph proj ["Étape 1 — Projection linéaire"]
    LA["Linear → 64 dim"]
    LT["Linear → 64 dim"]
    LB["Linear → 64 dim"]
  end

  subgraph layer1 ["Étape 2 — Couche 1"]
    HC1["HeteroConv<br/>SAGEConv par relation<br/>aggr = sum"]
    LN1["LayerNorm + ReLU + Dropout 0.3"]
    HC1 --> LN1
  end

  subgraph layer2 ["Étape 3 — Couche 2"]
    HC2["HeteroConv<br/>SAGEConv par relation<br/>aggr = sum"]
    LN2["LayerNorm + ReLU + Dropout 0.3"]
    HC2 --> LN2
  end

  subgraph head ["Étape 4 — Tête de classification"]
    SK["Skip connection<br/>concat(embedding_GNN, x_raw transaction)"]
    MLP["MLP 3 couches<br/>64 → 32 → 16 → 1"]
    OUT["logit fraude<br/>(BCEWithLogitsLoss)"]
    SK --> MLP --> OUT
  end

  XA --> LA
  XT --> LT
  XB --> LB
  LA --> HC1
  LT --> HC1
  LB --> HC1
  E --> HC1
  LN1 --> HC2
  E --> HC2
  LN2 --> SK
  XT --> SK

  style XT fill:#fff3e0,stroke:#e65100
  style SK fill:#f3e5f5,stroke:#7b1fa2
  style OUT fill:#fce4ec,stroke:#c2185b
```

### Détail d'une couche HeteroConv

Pour chaque type d'arête `(src, relation, dst)`, le message passing utilise un
`SAGEConv` **indépendant** — équivalent pratique du R-GCN sur graphe RDF :

```mermaid
flowchart LR
  subgraph relations ["Poids distincts par prédicat"]
    R1["sends_via → W₁"]
    R2["deposits_to → W₂"]
    R3["hosted_by → W₃"]
    R4["transacted_with → W₄"]
    R5["same_bank_as → W₅"]
  end

  MSG["Agrégation des messages<br/>voisins (sum)"] --> UPD["Mise à jour embedding<br/>par type de nœud"]
  R1 --> MSG
  R2 --> MSG
  R3 --> MSG
  R4 --> MSG
  R5 --> MSG
```

| Bloc | Rôle |
|------|------|
| `lin_dict` | Projette chaque type de nœud vers un espace commun (64 dim) |
| `HeteroConv × 2` | Message passing multi-relationnel ; un SAGEConv par prédicat |
| `LayerNorm` | Stabilise l'entraînement après chaque couche |
| Skip connection | Préserve les features T-Box brutes de la transaction (extensibilité) |
| `head` (MLP) | Produit un logit binaire par transaction |

---

## 3. Position dans le pipeline global

```mermaid
flowchart TB
  CSV["CSV nettoyé<br/>(LI-Small / LI-Medium)"]
  RDF["RDF A-Box<br/>aml_data_subgraph.ttl"]
  ENR["RDF enrichi<br/>+ transacted_with / same_bank_as"]
  PYG["HeteroData PyG<br/>aml_hetero_graph.pt"]
  GNN["HeteroTransactionGNN<br/>entraînement / inférence"]
  SCORE["Score fraude<br/>par transaction"]

  CSV --> RDF --> ENR --> PYG --> GNN --> SCORE

  TBox["T-Box aml.ttl<br/>(contrat features)"] -.-> ENR
  TBox -.-> PYG

  style GNN fill:#e8eaf6,stroke:#3949ab
  style SCORE fill:#fce4ec,stroke:#c2185b
```

Pipeline détaillé (train + hold-out) : [`pipeline_etl_entrainement_evaluation.md`](../pipeline_etl_entrainement_evaluation.md).

---

## 4. Hyperparamètres

| Paramètre | Valeur | Fichier / config |
|-----------|--------|------------------|
| `hidden_channels` | 64 | `src/config.py` |
| `num_layers` | 2 | `src/config.py` |
| `dropout` | 0,3 | `src/config.py` |
| `head_hidden` | 64 | `src/config.py` |
| `lr` | 0,001 | `src/config.py` |
| `pos_weight` | ~55 | `src/gnn/hetero/training.py` (déséquilibre classes) |
| Loss | `BCEWithLogitsLoss` | Early stopping sur Val PR-AUC |

---

## 5. Comparaison rapide Homo vs Hetero

```mermaid
flowchart TB
  subgraph homo ["GraphSAGE / GAT (homogène)"]
    HA["Nœud = Account"]
    HE["Arête = Transaction<br/>(montant en edge_attr)"]
    HT["Tâche = classification d'arête"]
    HA --- HE --- HA
  end

  subgraph hetero ["HeteroGNN (hétérogène — défaut)"]
    EA["Account"]
    ET["Transaction"]
    EB["Bank"]
    ER["5 prédicats RDF<br/>poids distincts"]
    ETH["Tâche = classification de nœud transaction"]
    EA --> ER
    ET --> ER
    EB --> ER
  end

  style hetero fill:#f9fbe7,stroke:#827717
  style homo fill:#fafafa,stroke:#9e9e9e
```

---

## 6. Exporter en PNG (slides / rapport)

### Option A — Mermaid Live Editor

1. Copier un bloc ` ```mermaid ` ci-dessus.
2. Coller sur [mermaid.live](https://mermaid.live).
3. Exporter en PNG ou SVG.

### Option B — CLI Mermaid

```powershell
npm install -g @mermaid-js/mermaid-cli

# Exemple : extraire le diagramme §2 dans un fichier .mmd puis :
mmdc -i hetero_forward.mmd -o hetero_forward.png -b transparent
```

### Option C — draw.io / PowerPoint

Reproduire manuellement les schémas ci-dessus en appliquant la palette suggérée :

| Type | Couleur suggérée |
|------|------------------|
| Account | Bleu clair |
| Transaction | Orange clair |
| Bank | Vert clair |
| Relations inférées | Traits pointillés |
| Score / label | Rose |

---

## 7. Références code

| Fichier | Contenu |
|---------|---------|
| `src/gnn/hetero/models.py` | `HeteroTransactionGNN` |
| `src/gnn/hetero/training.py` | Boucle d'entraînement, checkpoint |
| `src/rdf/to_hetero_pyg.py` | Construction `HeteroData` |
| `src/rdf/inference.py` | Inférence `transacted_with`, `same_bank_as` |
| `src/app/hetero_inference.py` | Inférence unitaire (app Streamlit) |
