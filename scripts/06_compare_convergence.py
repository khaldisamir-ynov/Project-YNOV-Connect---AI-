"""Compare la convergence GraphSAGE vs HeteroGNN epoch par epoch.

Usage :
    python scripts/06_compare_convergence.py
    python scripts/06_compare_convergence.py --epochs 40 --retrain
    python scripts/06_compare_convergence.py --use-existing
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.evaluation.convergence_compare import (  # noqa: E402
    CONVERGENCE_GAP_PNG,
    CONVERGENCE_JSON,
    CONVERGENCE_PNG,
    run_convergence_study,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("compare_convergence")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convergence SAGE vs HeteroGNN.")
    p.add_argument("--epochs", type=int, default=40, help="Nombre d'epochs (entraînement aligné).")
    p.add_argument(
        "--retrain",
        action="store_true",
        help="Force le ré-entraînement des deux modèles.",
    )
    p.add_argument(
        "--use-existing",
        action="store_true",
        help="Utilise gnn_*_history.json si présents (sans ré-entraîner).",
    )
    p.add_argument("--threshold", type=float, default=0.01, help="Seuil |Δ PR-AUC| significatif.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    use_existing = args.use_existing and not args.retrain

    if not config.PYG_DATA.exists():
        raise FileNotFoundError(f"Graphe homogène manquant : {config.PYG_DATA}")
    if not config.PYG_HETERO_DATA.exists():
        raise FileNotFoundError(f"HeteroData manquant : {config.PYG_HETERO_DATA}")

    report = run_convergence_study(
        epochs=args.epochs,
        use_existing_history=use_existing,
        pr_auc_threshold=args.threshold,
    )
    a = report["analysis"]

    logger.info("=" * 60)
    logger.info("Analyse de divergence (GraphSAGE vs HeteroGNN)")
    logger.info("  Epochs comparées        : %s", report["epochs_compared"])
    logger.info("  1er écart PR-AUC ≥ %.2f : epoch %s", args.threshold, a.get("first_epoch_gap_pr_auc_0.01"))
    logger.info("  1er Hetero > SAGE (PR)  : epoch %s", a.get("first_epoch_hetero_ahead_pr_auc"))
    logger.info("  1er SAGE > Hetero (PR)  : epoch %s", a.get("first_epoch_sage_ahead_pr_auc"))
    logger.info("  Max |Δ PR-AUC|          : %.4f à epoch %s", a.get("max_abs_gap_pr_auc"), a.get("epoch_max_abs_gap_pr_auc"))
    logger.info("  Final Δ PR-AUC (val)    : %+.4f", a.get("final_gap_val_pr_auc"))
    logger.info("  Final SAGE / Hetero PR  : %.4f / %.4f", a.get("final_sage_val_pr_auc"), a.get("final_hetero_val_pr_auc"))
    logger.info("Graphiques : %s", CONVERGENCE_PNG)
    logger.info("             %s", CONVERGENCE_GAP_PNG)
    logger.info("JSON       : %s", CONVERGENCE_JSON)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
