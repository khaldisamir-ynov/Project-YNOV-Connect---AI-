"""Étape 1 : sous-échantillonnage stratifié du dataset AML.

Usage :
    python scripts/01_sample_data.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Permet l'exécution directe via `python scripts/01_sample_data.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.pipeline.sampling import save_sample, stratified_sample  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("sample_data")


def main() -> None:
    if not config.RAW_CSV.exists():
        raise FileNotFoundError(
            f"Le fichier brut est introuvable : {config.RAW_CSV}"
        )

    sample = stratified_sample(
        csv_path=config.RAW_CSV,
        target_size=config.SAMPLE_TARGET_SIZE,
        keep_all_laundering=config.KEEP_ALL_LAUNDERING,
        random_seed=config.RANDOM_SEED,
    )
    save_sample(sample, out_path=config.SAMPLE_CSV)
    logger.info("Terminé.")


if __name__ == "__main__":
    main()
