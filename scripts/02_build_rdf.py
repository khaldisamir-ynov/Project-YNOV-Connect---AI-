"""Étape 2 : génération de l'A-Box RDF (Turtle) à partir de l'échantillon CSV.

Usage :
    python scripts/02_build_rdf.py
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
logger = logging.getLogger("build_rdf")


def main() -> None:
    if not config.SAMPLE_CSV.exists():
        raise FileNotFoundError(
            f"Échantillon introuvable : {config.SAMPLE_CSV}\n"
            "→ Lance d'abord `python scripts/01_sample_data.py`."
        )

    logger.info("Lecture de l'échantillon : %s", config.SAMPLE_CSV)
    df = pd.read_csv(
        config.SAMPLE_CSV,
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
    logger.info("Échantillon chargé : %s lignes", f"{len(df):,}")

    graph = build_graph(df)
    serialize_graph(graph, out_path=config.ABOX_TTL)
    logger.info("Terminé.")


if __name__ == "__main__":
    main()
