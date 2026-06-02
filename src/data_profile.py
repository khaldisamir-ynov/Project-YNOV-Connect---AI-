"""Profils de chemins `train` / `unknown` pour le pipeline AML."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

DataProfile = Literal["train", "unknown"]

_PROFILE_FILES: dict[DataProfile, tuple[str, str]] = {
    "train": ("LI-Small_Trans.csv", "LI-Small_Trans_subgraph.csv"),
    "unknown": ("LI-Medium_Trans.csv", "LI-Medium_Trans_subgraph.csv"),
}


def apply_data_profile(
    profile: DataProfile,
    *,
    project_root: Path,
    train_data_dir: Path,
    unknown_data_dir: Path,
    rdf_schema_dir: Path,
    globals_dict: dict,
) -> Path:
    """Met à jour les chemins actifs dans `src.config` selon le profil."""
    if profile not in _PROFILE_FILES:
        raise ValueError(f"Profil inconnu : {profile!r} (attendu train|unknown)")

    raw_name, subgraph_name = _PROFILE_FILES[profile]
    root = train_data_dir if profile == "train" else unknown_data_dir

    g = globals_dict
    g["ACTIVE_DATA_PROFILE"] = profile
    g["ACTIVE_DATA_ROOT"] = root
    g["CSV_RAW_DIR"] = root / "csv" / "raw"
    g["CSV_CLEANED_DIR"] = root / "csv" / "cleaned"
    g["RDF_DATA_DIR"] = root / "rdf" / "ontology_data"
    g["PYG_DIR"] = root / "pyg"
    g["RESULTS_DIR"] = root / "results"
    g["RDF_SCHEMA_DIR"] = rdf_schema_dir

    g["RAW_CSV"] = g["CSV_RAW_DIR"] / raw_name
    if profile == "train":
        g["CLEANED_CSV"] = g["CSV_CLEANED_DIR"] / "LI-Small_Trans_cleaned.csv"
        g["SAMPLE_CSV"] = g["CSV_CLEANED_DIR"] / "LI-Small_Trans_sample.csv"
    else:
        g["CLEANED_CSV"] = g["CSV_CLEANED_DIR"] / "LI-Medium_Trans_cleaned.csv"
        g["SAMPLE_CSV"] = g["CSV_CLEANED_DIR"] / "LI-Medium_Trans_sample.csv"
    g["SUBGRAPH_CSV"] = g["CSV_CLEANED_DIR"] / subgraph_name

    g["ONTOLOGY_TTL"] = rdf_schema_dir / "aml.ttl"
    g["ABOX_TTL"] = g["RDF_DATA_DIR"] / "aml_data.ttl"
    g["ABOX_SUBGRAPH_TTL"] = g["RDF_DATA_DIR"] / "aml_data_subgraph.ttl"
    g["ABOX_SUBGRAPH_ENRICHED_TTL"] = g["RDF_DATA_DIR"] / "aml_data_subgraph_enriched.ttl"
    g["PYG_DATA"] = g["PYG_DIR"] / "aml_graph.pt"
    g["PYG_HETERO_DATA"] = g["PYG_DIR"] / "aml_hetero_graph.pt"
    g["GNN_CHECKPOINT"] = g["RESULTS_DIR"] / "gnn_best.pt"
    g["GNN_METRICS_JSON"] = g["RESULTS_DIR"] / "gnn_metrics.json"

    return root


def add_profile_arg(parser) -> None:
    """Ajoute `--data-profile {train,unknown}` à un ArgumentParser."""
    parser.add_argument(
        "--data-profile",
        choices=["train", "unknown"],
        default="train",
        help="Jeu de données actif (défaut : train).",
    )
