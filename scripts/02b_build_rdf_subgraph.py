"""Génère l'A-Box RDF du sous-graphe ego (pour entraînement GNN via RDF).

Produit `data/rdf/ontology_data/aml_data_subgraph.ttl` à partir de
`LI-Small_Trans_subgraph.csv` — même T-Box, échantillon connecté pour le GNN.

Usage :
    python scripts/02b_build_rdf_subgraph.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.rdf.builder import build_graph, serialize_graph  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("build_rdf_subgraph")


def main() -> None:
    if not config.SUBGRAPH_CSV.exists():
        raise FileNotFoundError(
            f"Sous-graphe introuvable : {config.SUBGRAPH_CSV}\n"
            "→ Lance d'abord `python scripts/03_subgraph_sample.py`."
        )

    logger.info("Lecture du sous-graphe ego : %s", config.SUBGRAPH_CSV)
    df = pd.read_csv(
        config.SUBGRAPH_CSV,
        dtype={
            "from_bank": str,
            "from_account": str,
            "to_bank": str,
            "to_account": str,
            "payment_format": str,
            "payment_currency": str,
            "receiving_currency": str,
        },
    )
    logger.info("Sous-graphe chargé : %s lignes", f"{len(df):,}")

    graph = build_graph(df)
    serialize_graph(graph, out_path=config.ABOX_SUBGRAPH_TTL)
    logger.info("A-Box sous-graphe écrite : %s", config.ABOX_SUBGRAPH_TTL)
    logger.info("Terminé.")


if __name__ == "__main__":
    main()
