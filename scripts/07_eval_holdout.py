"""Test final (option A) : évalue hetero / sage / gat sur unknown avec checkpoints train.

Usage :
    python scripts/07_eval_holdout.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.evaluation.holdout_eval import run_holdout_evaluation  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("eval_holdout")


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation hold-out des 3 GNN.")
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=None,
        help="Dossier des checkpoints train (défaut : data/train_data/results).",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Génère aussi les graphiques (équivalent scripts/09_visualize_holdout.py).",
    )
    args = parser.parse_args()

    config.set_data_profile("unknown")
    checkpoints = args.checkpoints_dir or config.TRAIN_RESULTS_DIR

    if not checkpoints.exists():
        raise FileNotFoundError(f"Checkpoints train introuvables : {checkpoints}")

    report = run_holdout_evaluation(checkpoints_dir=checkpoints)
    logger.info("=" * 60)
    for model, payload in report.get("models", {}).items():
        if payload.get("status") != "ok":
            logger.info("%s : %s", model.upper(), payload.get("reason", payload.get("status")))
            continue
        m = payload["holdout"]
        logger.info(
            "%s | PR-AUC=%.4f | Recall=%.4f | F1=%.4f | Precision=%.4f",
            model.upper(),
            m["pr_auc"],
            m["recall"],
            m["f1"],
            m["precision"],
        )
    logger.info("=" * 60)

    if args.viz:
        from src.evaluation.holdout_viz import export_holdout_visualizations  # noqa: WPS433

        logger.info("Génération des visualisations hold-out…")
        export_holdout_visualizations(
            report=report,
            output_dir=config.RESULTS_DIR,
            checkpoints_dir=checkpoints,
        )


if __name__ == "__main__":
    main()
