"""Configuration centrale du projet AML.

Centralise les chemins, les constantes du dataset et les namespaces RDF
pour rester cohérent avec la T-Box partagée `data/train_data/rdf/ontology_schema/aml.ttl`.
"""
from __future__ import annotations

from pathlib import Path

from .data_profile import apply_data_profile

# --------------------------------------------------------------------------- #
# Chemins — racines et profil actif
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
TRAIN_DATA_DIR: Path = DATA_DIR / "train_data"
UNKNOWN_DATA_DIR: Path = DATA_DIR / "unknown_data"
TRAIN_RESULTS_DIR: Path = TRAIN_DATA_DIR / "results"

# T-Box unique (partagée train + unknown)
SHARED_RDF_SCHEMA_DIR: Path = TRAIN_DATA_DIR / "rdf" / "ontology_schema"

ACTIVE_DATA_PROFILE: str = "train"
ACTIVE_DATA_ROOT: Path = TRAIN_DATA_DIR

# Chemins actifs (mis à jour par `apply_data_profile` / `--data-profile`)
CSV_RAW_DIR: Path = TRAIN_DATA_DIR / "csv" / "raw"
CSV_CLEANED_DIR: Path = TRAIN_DATA_DIR / "csv" / "cleaned"
RDF_SCHEMA_DIR: Path = SHARED_RDF_SCHEMA_DIR
RDF_DATA_DIR: Path = TRAIN_DATA_DIR / "rdf" / "ontology_data"
PYG_DIR: Path = TRAIN_DATA_DIR / "pyg"

RAW_CSV: Path = TRAIN_DATA_DIR / "csv" / "raw" / "LI-Small_Trans.csv"
CLEANED_CSV: Path = TRAIN_DATA_DIR / "csv" / "cleaned" / "LI-Small_Trans_cleaned.csv"
SAMPLE_CSV: Path = TRAIN_DATA_DIR / "csv" / "cleaned" / "LI-Small_Trans_sample.csv"
SUBGRAPH_CSV: Path = TRAIN_DATA_DIR / "csv" / "cleaned" / "LI-Small_Trans_subgraph.csv"

ONTOLOGY_TTL: Path = SHARED_RDF_SCHEMA_DIR / "aml.ttl"
ABOX_TTL: Path = RDF_DATA_DIR / "aml_data.ttl"
ABOX_SUBGRAPH_TTL: Path = RDF_DATA_DIR / "aml_data_subgraph.ttl"
ABOX_SUBGRAPH_ENRICHED_TTL: Path = RDF_DATA_DIR / "aml_data_subgraph_enriched.ttl"

PYG_DATA: Path = PYG_DIR / "aml_graph.pt"
PYG_HETERO_DATA: Path = PYG_DIR / "aml_hetero_graph.pt"
RESULTS_DIR: Path = TRAIN_DATA_DIR / "results"
GNN_CHECKPOINT: Path = RESULTS_DIR / "gnn_best.pt"
GNN_METRICS_JSON: Path = RESULTS_DIR / "gnn_metrics.json"


def set_data_profile(profile: str) -> Path:
    """Active le profil `train` ou `unknown` (chemins dérivés + T-Box partagée)."""
    return apply_data_profile(
        profile,  # type: ignore[arg-type]
        project_root=PROJECT_ROOT,
        train_data_dir=TRAIN_DATA_DIR,
        unknown_data_dir=UNKNOWN_DATA_DIR,
        rdf_schema_dir=SHARED_RDF_SCHEMA_DIR,
        globals_dict=globals(),
    )


# Profil par défaut au chargement du module
set_data_profile("train")

# --------------------------------------------------------------------------- #
# Schéma CSV
# --------------------------------------------------------------------------- #
# Les colonnes `Account` du CSV brut sont dupliquées (source + destination).
# On les renomme dès la lecture pour éviter toute ambiguïté.
RAW_COLUMNS: list[str] = [
    "Timestamp",
    "From Bank",
    "Account",        # 1re occurrence -> compte émetteur
    "To Bank",
    "Account",        # 2e occurrence -> compte destinataire
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]

CANONICAL_COLUMNS: list[str] = [
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
]

# --------------------------------------------------------------------------- #
# Sous-échantillonnage
# --------------------------------------------------------------------------- #
SAMPLE_TARGET_SIZE: int = 100_000     # objectif total (sampling stratifié naïf)
KEEP_ALL_LAUNDERING: bool = True      # garde 100% des frauduleuses (rares)
RANDOM_SEED: int = 42

# Échantillonnage par sous-graphe ego (pour le GNN)
EGO_NUM_HOPS: int = 1                 # K-hops d'expansion autour des comptes frauduleux
EGO_TARGET_SIZE: int = 200_000        # plafond du nombre de transactions
EGO_INCLUDE_LEGIT_BACKGROUND: bool = False  # ajouter des tx légitimes "pures" pour bruit

# Lecture CSV volumineux (ex. LI-Medium ~31M lignes) — évite de charger tout en RAM
CSV_CHUNKSIZE: int = 50_000          # petit morceau = moins de RAM (LI-Medium)
CSV_AUTO_CHUNK_MIN_MB: float = 200.0   # mode chunked auto au-delà de cette taille
# Présélection avant ego (0 = désactivé) — activer seulement si la RAM est confortable
UNKNOWN_RAW_PRESAMPLE_SIZE: int = 0

# --------------------------------------------------------------------------- #
# Inférence RDF (A-Box enrichie)
# --------------------------------------------------------------------------- #
RDF_USE_OWLRL: bool = False           # True si owlrl installé (SymmetricProperty…)

# --------------------------------------------------------------------------- #
# Entraînement GNN
# --------------------------------------------------------------------------- #
GNN_MODEL_TYPE: str = "hetero"        # "hetero" (défaut), "sage" ou "gat" (homogène legacy)
GNN_HIDDEN_DIM: int = 64
GNN_NUM_LAYERS: int = 2
GNN_DROPOUT: float = 0.3
GNN_LR: float = 1e-3
GNN_WEIGHT_DECAY: float = 1e-4
GNN_EPOCHS: int = 80
GNN_PATIENCE: int = 15                # early stopping sur val PR-AUC
GNN_EDGE_HIDDEN_DIM: int = 64         # MLP de classification d'arêtes

# --------------------------------------------------------------------------- #
# Namespaces RDF (alignés sur la T-Box aml.ttl)
# --------------------------------------------------------------------------- #
AML_ONTOLOGY_NS: str = "http://escience.datafactory.com/ontology/aml#"
AML_INSTANCE_NS: str = "http://escience.datafactory.com/aml/instance#"
