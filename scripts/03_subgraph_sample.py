"""Échantillonnage par sous-graphe ego — préalable au GNN.

Remplace le sampling stratifié naïf de `01_sample_data.py` par un sampling
qui **préserve la connexité** : pour chaque transaction frauduleuse, on
récupère ses comptes impliqués et leurs voisins à K hops.

Usage :
    python scripts/03_subgraph_sample.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data_profile import add_profile_arg  # noqa: E402
from src.pipeline.subgraph_sampling import (  # noqa: E402
    connectivity_stats,
    ego_subgraph_sample,
    presample_raw_csv,
    save_subgraph_sample,
    should_use_chunked_read,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("subgraph_sample")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sous-graphe ego pour le GNN.")
    add_profile_arg(p)
    p.add_argument(
        "--chunked",
        action="store_true",
        default=None,
        help="Force la lecture par morceaux (recommandé pour LI-Medium).",
    )
    p.add_argument(
        "--no-chunked",
        action="store_true",
        help="Charge tout le CSV en RAM (déconseillé au-delà de ~200 Mo).",
    )
    p.add_argument(
        "--presample-size",
        type=int,
        default=0,
        help=(
            "Présélection stratifiée avant ego (ex. 2000000). "
            f"Défaut profil unknown : {config.UNKNOWN_RAW_PRESAMPLE_SIZE or 'auto-chunked seul'}."
        ),
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=config.CSV_CHUNKSIZE,
        help="Lignes par morceau en mode chunked.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config.set_data_profile(args.data_profile)

    if not config.RAW_CSV.exists():
        raise FileNotFoundError(f"CSV brut introuvable : {config.RAW_CSV}")

    chunked: bool | None = True if args.chunked else (False if args.no_chunked else None)
    presample_size = args.presample_size
    if presample_size <= 0 and args.data_profile == "unknown":
        presample_size = config.UNKNOWN_RAW_PRESAMPLE_SIZE

    if args.data_profile == "unknown" and presample_size <= 0:
        logger.info(
            "LI-Medium : mode chunked (chunksize=%s). Fermez Cursor/navigateur si RAM faible.",
            f"{args.chunksize:,}",
        )

    csv_in = config.RAW_CSV
    if presample_size > 0:
        presample_path = config.CSV_CLEANED_DIR / f"{config.RAW_CSV.stem}_presample.csv"
        presample_raw_csv(
            csv_in,
            presample_path,
            target_size=presample_size,
            random_seed=config.RANDOM_SEED,
            chunksize=args.chunksize,
        )
        csv_in = presample_path

    use_chunked = should_use_chunked_read(csv_in, chunked)
    logger.info(
        "Source : %s | mode %s | presample=%s",
        csv_in.name,
        "CHUNKED" if use_chunked else "mémoire",
        presample_size if presample_size > 0 else "non",
    )

    df = ego_subgraph_sample(
        csv_path=csv_in,
        num_hops=config.EGO_NUM_HOPS,
        target_size=config.EGO_TARGET_SIZE,
        include_legit_background=config.EGO_INCLUDE_LEGIT_BACKGROUND,
        random_seed=config.RANDOM_SEED,
        chunked=chunked,
        chunksize=args.chunksize,
    )
    save_subgraph_sample(df, out_path=config.SUBGRAPH_CSV)

    stats = connectivity_stats(df)
    logger.info("=" * 60)
    logger.info("Statistiques de connexité du sous-graphe :")
    logger.info("  Transactions   : %s", f"{stats['n_transactions']:,}")
    logger.info("  Comptes uniques: %s", f"{stats['n_accounts']:,}")
    logger.info("  Degré moyen    : %.2f", stats["avg_degree"])
    logger.info("  Degré médian   : %.2f", stats["median_degree"])
    logger.info("  Degré max      : %s", f"{stats['max_degree']:,}")
    logger.info("  %% degré = 1   : %.2f %%", stats["pct_degree_one"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
