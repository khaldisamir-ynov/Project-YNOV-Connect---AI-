"""Étape 2c : inférence RDF — enrichissement de l'A-Box avant le GNN.

Déduit transactedWith et sameBankAs, matérialise aml_data_subgraph_enriched.ttl.

Usage :
    python scripts/02c_enrich_rdf.py
    python scripts/02c_enrich_rdf.py --owlrl
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data_profile import add_profile_arg  # noqa: E402
from src.rdf.inference import enrich_rdf_file  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("enrich_rdf")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enrichit l'A-Box RDF par inférence.")
    add_profile_arg(p)
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="A-Box source (défaut : aml_data_subgraph.ttl du profil actif).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="A-Box enrichie de sortie (défaut : profil actif).",
    )
    p.add_argument(
        "--owlrl",
        action="store_true",
        help="Active l'expansion OWL-RL (nécessite pip install owlrl).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config.set_data_profile(args.data_profile)
    input_path = args.input or config.ABOX_SUBGRAPH_TTL
    output_path = args.output or config.ABOX_SUBGRAPH_ENRICHED_TTL
    if not input_path.exists():
        raise FileNotFoundError(
            f"A-Box introuvable : {input_path}\n"
            "→ Lance d'abord `python scripts/02b_build_rdf_subgraph.py`."
        )

    logger.info("Inférence RDF : %s → %s", input_path.name, output_path.name)
    enrich_rdf_file(
        abox_path=input_path,
        out_path=output_path,
        use_owlrl=args.owlrl or config.RDF_USE_OWLRL,
    )
    logger.info("Terminé.")


if __name__ == "__main__":
    main()
