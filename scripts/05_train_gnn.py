"""Étape 5 : entraînement GNN pour classification de fraude AML.

Usage :
    python scripts/05_train_gnn.py                    # HeteroGNN (défaut)
    python scripts/05_train_gnn.py --model hetero
    python scripts/05_train_gnn.py --model sage       # homogène legacy
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.gnn.homo.training import train_gnn  # noqa: E402
from src.gnn.hetero.training import train_hetero_gnn  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("train_gnn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraîne un GNN AML.")
    parser.add_argument(
        "--model",
        choices=["hetero", "sage", "gat"],
        default=config.GNN_MODEL_TYPE,
        help="hetero = multi-relationnel (défaut) ; sage/gat = homogène legacy.",
    )
    parser.add_argument("--epochs", type=int, default=config.GNN_EPOCHS)
    parser.add_argument("--hidden-dim", type=int, default=config.GNN_HIDDEN_DIM)
    parser.add_argument("--lr", type=float, default=config.GNN_LR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Modèle sélectionné : %s", args.model.upper())

    if args.model == "hetero":
        if not config.PYG_HETERO_DATA.exists():
            raise FileNotFoundError(
                f"HeteroData introuvable : {config.PYG_HETERO_DATA}\n"
                "→ Lance d'abord `python scripts/04_build_pyg.py --mode hetero`."
            )
        result = train_hetero_gnn(
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            epochs=args.epochs,
        )
    else:
        if not config.PYG_DATA.exists():
            raise FileNotFoundError(
                f"Tenseur homogène introuvable : {config.PYG_DATA}\n"
                "→ Lance d'abord `python scripts/04_build_pyg.py --mode homo`."
            )
        result = train_gnn(
            model_type=args.model,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            epochs=args.epochs,
        )

    logger.info("=" * 60)
    logger.info("Résultats finaux (meilleur epoch = %d)", result.best_epoch)
    for split_name, metrics in [
        ("TRAIN", result.train),
        ("VAL", result.val),
        ("TEST", result.test),
    ]:
        logger.info(
            "%s | PR-AUC=%.4f | Recall=%.4f | F1=%.4f | Precision=%.4f | n=%s (fraudes=%s)",
            split_name,
            metrics.pr_auc,
            metrics.recall,
            metrics.f1,
            metrics.precision,
            f"{metrics.n_samples:,}",
            metrics.n_fraud,
        )
    logger.info("=" * 60)
    logger.info("Terminé.")


if __name__ == "__main__":
    main()
