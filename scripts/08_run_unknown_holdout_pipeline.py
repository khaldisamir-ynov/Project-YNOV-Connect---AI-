"""Pipeline hold-out : transforme LI-Medium comme LI-Small puis évalue les 3 GNN.

Usage :
    python scripts/08_run_unknown_holdout_pipeline.py
    python scripts/08_run_unknown_holdout_pipeline.py --skip-subgraph  # si CSV déjà prêt
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("unknown_holdout_pipeline")

PYTHON = sys.executable
SCRIPTS = Path(__file__).resolve().parent


def _run(cmd: list[str]) -> None:
    logger.info("> %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline hold-out unknown + évaluation.")
    p.add_argument("--skip-subgraph", action="store_true", help="Sous-graphe CSV déjà généré.")
    p.add_argument("--skip-rdf", action="store_true", help="A-Box déjà construite.")
    p.add_argument("--skip-pyg", action="store_true", help="Tenseurs PyG déjà présents.")
    p.add_argument("--skip-eval", action="store_true", help="Ne pas lancer 07_eval_holdout.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config.set_data_profile("unknown")
    logger.info("Profil actif : unknown → %s", config.ACTIVE_DATA_ROOT)
    logger.info("T-Box partagée : %s", config.ONTOLOGY_TTL)
    logger.info("Checkpoints train : %s", config.TRAIN_RESULTS_DIR)

    if not args.skip_subgraph:
        _run([PYTHON, str(SCRIPTS / "03_subgraph_sample.py"), "--data-profile", "unknown"])

    if not args.skip_rdf:
        _run([PYTHON, str(SCRIPTS / "02b_build_rdf_subgraph.py"), "--data-profile", "unknown"])
        _run([PYTHON, str(SCRIPTS / "02c_enrich_rdf.py"), "--data-profile", "unknown"])

    if not args.skip_pyg:
        _run([
            PYTHON,
            str(SCRIPTS / "04_build_pyg.py"),
            "--data-profile",
            "unknown",
            "--mode",
            "hetero",
            "--holdout-eval",
        ])
        _run([
            PYTHON,
            str(SCRIPTS / "04_build_pyg.py"),
            "--data-profile",
            "unknown",
            "--mode",
            "homo",
            "--holdout-eval",
        ])

    if not args.skip_eval:
        _run([PYTHON, str(SCRIPTS / "07_eval_holdout.py")])
        _run([PYTHON, str(SCRIPTS / "09_visualize_holdout.py")])

    logger.info("Pipeline hold-out terminé.")


if __name__ == "__main__":
    main()
