"""Sous-échantillonnage stratifié du dataset IBM AML.

Stratégie pragmatique pour la 1re itération :
    - On garde 100 % des transactions étiquetées `is_laundering == 1`
      (classe ultra-minoritaire, ~0.1 % du corpus).
    - On complète par un tirage aléatoire de transactions légitimes
      pour atteindre la taille cible (`SAMPLE_TARGET_SIZE`).

Limite connue : un tirage aléatoire casse la connexité du graphe — pour le
GNN on pourra à terme passer à un échantillonnage par sous-graphe ego /
fenêtre temporelle. Pour l'exercice de génération RDF c'est suffisant.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def _load_raw_csv(csv_path: Path) -> pd.DataFrame:
    """Charge le CSV brut en renommant les colonnes dupliquées `Account`."""
    df = pd.read_csv(
        csv_path,
        header=0,
        names=config.CANONICAL_COLUMNS,
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
    return df


def stratified_sample(
    csv_path: Path = config.RAW_CSV,
    target_size: int = config.SAMPLE_TARGET_SIZE,
    keep_all_laundering: bool = config.KEEP_ALL_LAUNDERING,
    random_seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Renvoie un échantillon stratifié du dataset AML."""
    logger.info("Chargement du CSV brut : %s", csv_path)
    df = _load_raw_csv(csv_path)
    n_total = len(df)
    n_fraud = int(df["is_laundering"].sum())
    logger.info(
        "Dataset brut : %s lignes | %s frauduleuses (%.4f %%)",
        f"{n_total:,}",
        f"{n_fraud:,}",
        100 * n_fraud / n_total,
    )

    fraud_df = df[df["is_laundering"] == 1]
    legit_df = df[df["is_laundering"] == 0]

    if keep_all_laundering:
        n_legit = max(target_size - len(fraud_df), 0)
    else:
        ratio = len(fraud_df) / n_total
        n_legit = int(target_size * (1 - ratio))
        fraud_df = fraud_df.sample(
            n=min(target_size - n_legit, len(fraud_df)),
            random_state=random_seed,
        )

    n_legit = min(n_legit, len(legit_df))
    legit_sample = legit_df.sample(n=n_legit, random_state=random_seed)

    sample = pd.concat([fraud_df, legit_sample], ignore_index=True)
    sample = sample.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    logger.info(
        "Échantillon final : %s lignes | %s frauduleuses (%.4f %%)",
        f"{len(sample):,}",
        f"{int(sample['is_laundering'].sum()):,}",
        100 * sample["is_laundering"].mean(),
    )
    return sample


def save_sample(df: pd.DataFrame, out_path: Path = config.SAMPLE_CSV) -> Path:
    """Persiste l'échantillon dans `data/csv/cleaned/`."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Échantillon écrit : %s", out_path)
    return out_path
