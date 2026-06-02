"""Étape 4 : conversion vers PyTorch Geometric.

Modes :
    hetero (défaut) — A-Box enrichie → HeteroData multi-relationnel
    homo            — graphe homogène legacy (RDF ou CSV)

Usage :
    python scripts/04_build_pyg.py
    python scripts/04_build_pyg.py --mode hetero
    python scripts/04_build_pyg.py --mode homo --source rdf
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src import config  # noqa: E402
from src.data_profile import add_profile_arg  # noqa: E402
from src.evaluation.holdout_eval import (  # noqa: E402
    hetero_feature_names_from_train,
    homo_feature_info_from_train,
)
from src.gnn.graph_loader import build_pyg_data, load_pyg_data, save_pyg_data  # noqa: E402
from src.rdf.to_hetero_pyg import build_hetero_pyg_from_rdf, load_hetero_data  # noqa: E402
from src.rdf.to_homo_pyg import build_pyg_from_rdf  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("build_pyg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construit le tenseur PyG.")
    add_profile_arg(parser)
    parser.add_argument(
        "--mode",
        choices=["hetero", "homo"],
        default="hetero",
        help="hetero = HeteroData multi-relationnel (défaut) ; homo = legacy.",
    )
    parser.add_argument(
        "--source",
        choices=["rdf", "csv"],
        default="rdf",
        help="Source homo uniquement : rdf ou csv.",
    )
    parser.add_argument(
        "--holdout-eval",
        action="store_true",
        help="100 %% des labels en masque test (évaluation finale, pas de split train/val).",
    )
    return parser.parse_args()


def _sanity_hetero() -> None:
    data, info = load_hetero_data(config.PYG_HETERO_DATA)
    logger.info("=" * 60)
    logger.info("Sanity check — HeteroData :")
    for ntype in info.node_types:
        logger.info("  [%s] x : %s | features : %s", ntype, tuple(data[ntype].x.shape), info.node_feature_names.get(ntype, []))
    logger.info("  relations : %s", len(info.edge_types))
    logger.info("  transacted_with : %s | same_bank_as : %s", info.n_inferred_transacted_with, info.n_inferred_same_bank_as)
    logger.info(
        "  labels transaction : %s fraudes / %s",
        info.n_fraud_transactions,
        info.n_nodes.get("transaction", 0),
    )
    logger.info("=" * 60)


def _sanity_homo() -> None:
    data, info = load_pyg_data(config.PYG_DATA)
    logger.info("=" * 60)
    logger.info("Sanity check — Data homogène :")
    logger.info("  x : %s | edge_index : %s | edge_attr : %s", tuple(data.x.shape), tuple(data.edge_index.shape), tuple(data.edge_attr.shape))
    logger.info("  fraudes : %s / %s", int(data.y.sum()), data.y.size(0))
    logger.info("=" * 60)


def main() -> None:
    args = parse_args()
    config.set_data_profile(args.data_profile)

    if args.mode == "hetero":
        if not config.ABOX_SUBGRAPH_ENRICHED_TTL.exists():
            raise FileNotFoundError(
                f"A-Box enrichie introuvable : {config.ABOX_SUBGRAPH_ENRICHED_TTL}\n"
                "→ Lance d'abord `python scripts/02c_enrich_rdf.py`."
            )
        logger.info("Mode HETERO — source : %s", config.ABOX_SUBGRAPH_ENRICHED_TTL.name)
        hetero_ref = None
        if args.holdout_eval and args.data_profile == "unknown":
            hetero_ref = hetero_feature_names_from_train(config.TRAIN_RESULTS_DIR)
            if hetero_ref is None:
                logger.warning(
                    "hetero_info absent du checkpoint — dimensions peuvent différer du train."
                )
            else:
                logger.info("Alignement des features sur le checkpoint train (hold-out).")
        build_hetero_pyg_from_rdf(
            ttl_path=config.ABOX_SUBGRAPH_ENRICHED_TTL,
            pyg_path=config.PYG_HETERO_DATA,
            holdout_eval=args.holdout_eval,
            reference_feature_names=hetero_ref,
        )
        _sanity_hetero()
    else:
        feat_ref = None
        if args.holdout_eval and args.data_profile == "unknown":
            feat_ref = homo_feature_info_from_train("sage", config.TRAIN_RESULTS_DIR)
            if feat_ref is None:
                logger.warning(
                    "feature_info absent du checkpoint sage — vocabulaire dérivé du CSV Medium."
                )
        if args.source == "rdf":
            if not config.ABOX_SUBGRAPH_TTL.exists():
                raise FileNotFoundError(
                    f"A-Box RDF introuvable : {config.ABOX_SUBGRAPH_TTL}\n"
                    "→ Lance d'abord `python scripts/02b_build_rdf_subgraph.py`."
                )
            build_pyg_from_rdf(
                ttl_path=config.ABOX_SUBGRAPH_TTL,
                pyg_path=config.PYG_DATA,
                feature_info_ref=feat_ref,
                holdout_eval=args.holdout_eval,
            )
        else:
            if not config.SUBGRAPH_CSV.exists():
                raise FileNotFoundError(f"Sous-graphe introuvable : {config.SUBGRAPH_CSV}")
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
            data, info = build_pyg_data(
                df,
                feature_info_ref=feat_ref,
                holdout_eval=args.holdout_eval,
            )
            save_pyg_data(data, info, out_path=config.PYG_DATA)
        _sanity_homo()

    logger.info("Terminé.")


if __name__ == "__main__":
    main()
