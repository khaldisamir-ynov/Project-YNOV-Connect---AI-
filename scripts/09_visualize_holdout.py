"""Visualisations hold-out (équivalent phase E / metrics_evaluation.ipynb sur LI-Medium).

Usage :
    python scripts/09_visualize_holdout.py
    python scripts/09_visualize_holdout.py --report data/unknown_data/results/eval_holdout_metrics.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.evaluation.holdout_viz import (  # noqa: E402
    export_holdout_visualizations,
    holdout_report_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("visualize_holdout")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Génère graphiques et CSV d'évaluation hold-out (soutenance)."
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Chemin vers eval_holdout_metrics.json (défaut : unknown_data/results/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Dossier de sortie des figures (défaut : unknown_data/results/).",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=None,
        help="Checkpoints train (défaut : train_data/results/).",
    )
    args = parser.parse_args()

    config.set_data_profile("unknown")
    output_dir = args.output_dir or config.RESULTS_DIR
    checkpoints_dir = args.checkpoints_dir or config.TRAIN_RESULTS_DIR

    report_path = args.report or holdout_report_path(output_dir)
    if not report_path.exists():
        raise FileNotFoundError(
            f"{report_path} introuvable. Lancez : python scripts/07_eval_holdout.py"
        )

    with report_path.open(encoding="utf-8") as f:
        report = json.load(f)

    artifacts = export_holdout_visualizations(
        report=report,
        output_dir=output_dir,
        checkpoints_dir=checkpoints_dir,
        train_results_dir=config.TRAIN_RESULTS_DIR,
    )

    logger.info("=" * 60)
    logger.info("Visualisations hold-out — %s", output_dir.resolve())
    for name, path in sorted(artifacts.items()):
        if path.suffix in (".png", ".csv"):
            logger.info("  %s", path.name)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
