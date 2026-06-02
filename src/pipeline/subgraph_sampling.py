"""Sous-échantillonnage par sous-graphe ego (préalable au GNN).

Limite du sampling stratifié naïf (cf. `src/sampling.py`) : un tirage
aléatoire de transactions casse la connexité du graphe — la plupart des
comptes finissent avec un seul voisin, ce qui handicape l'apprentissage
GNN (le *message passing* n'a rien à agréger).

Stratégie corrective :
    1. Identifier les **comptes "graines"** = tous les comptes impliqués
       dans au moins une transaction frauduleuse.
    2. Étendre par voisinage à **K hops** (par défaut K=1) : on récupère
       toutes les transactions impliquant un compte graine OU un voisin
       de graine.
    3. Conserver 100 % des transactions ainsi collectées (donc 100 % des
       fraudes ET leur contexte topologique immédiat).
    4. Optionnel : ajouter quelques sous-graphes 100 % légitimes pour
       fournir un "fond" sain au modèle.

Le résultat est un sous-graphe **dense autour des fraudes**, beaucoup plus
exploitable par un GNN qu'un tirage aléatoire.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _account_keys(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Construit des clés stables `bank|account` pour les colonnes from / to."""
    from_key = df["from_bank"].astype(str) + "|" + df["from_account"].astype(str)
    to_key = df["to_bank"].astype(str) + "|" + df["to_account"].astype(str)
    return from_key, to_key


def _expand_one_hop(df: pd.DataFrame, seed_accounts: set[str]) -> set[str]:
    """Renvoie les nouveaux comptes atteints en un saut depuis seed_accounts."""
    from_key, to_key = _account_keys(df)
    touching_mask = from_key.isin(seed_accounts) | to_key.isin(seed_accounts)
    touched = df[touching_mask]
    expanded: set[str] = set()
    expanded.update(from_key[touching_mask].unique())
    expanded.update(to_key[touching_mask].unique())
    new_accounts = expanded - seed_accounts
    logger.info(
        "Expansion : +%s comptes (transactions touchées = %s)",
        f"{len(new_accounts):,}",
        f"{len(touched):,}",
    )
    return new_accounts


# --------------------------------------------------------------------------- #
# Sampler principal
# --------------------------------------------------------------------------- #
def ego_subgraph_sample(
    csv_path: Path = config.RAW_CSV,
    num_hops: int = config.EGO_NUM_HOPS,
    target_size: int = config.EGO_TARGET_SIZE,
    include_legit_background: bool = config.EGO_INCLUDE_LEGIT_BACKGROUND,
    random_seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Échantillonne un sous-graphe ego autour des comptes frauduleux.

    Args:
        csv_path: CSV brut.
        num_hops: profondeur d'expansion (1 = voisins directs des fraudes).
        target_size: plafond souhaité (les transactions ego sont gardées en
            entier ; le complément éventuel passe par sous-échantillonnage).
        include_legit_background: ajoute des transactions 100% extérieures.
        random_seed: reproductibilité.

    Returns:
        DataFrame trié chronologiquement, prêt à être sérialisé.
    """
    logger.info("Lecture du CSV brut : %s", csv_path)
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
    n_total = len(df)
    n_fraud = int(df["is_laundering"].sum())
    logger.info(
        "Brut : %s transactions | %s fraudes (%.4f %%)",
        f"{n_total:,}",
        f"{n_fraud:,}",
        100 * n_fraud / n_total,
    )

    from_key, to_key = _account_keys(df)

    # --- Étape 1 : comptes graines = tous ceux touchés par une fraude ---
    fraud_mask = df["is_laundering"] == 1
    seed_accounts: set[str] = set()
    seed_accounts.update(from_key[fraud_mask].unique())
    seed_accounts.update(to_key[fraud_mask].unique())
    logger.info("Comptes graines (touchés par une fraude) : %s", f"{len(seed_accounts):,}")

    # --- Étape 2 : expansion K hops ---
    accounts_in_subgraph = set(seed_accounts)
    for hop in range(1, num_hops + 1):
        logger.info("Hop %d / %d…", hop, num_hops)
        new_accounts = _expand_one_hop(df, accounts_in_subgraph)
        if not new_accounts:
            logger.info("Aucun nouveau compte au hop %d → arrêt.", hop)
            break
        accounts_in_subgraph |= new_accounts

    logger.info(
        "Comptes dans le sous-graphe ego : %s (départ : %s, expansion : +%s)",
        f"{len(accounts_in_subgraph):,}",
        f"{len(seed_accounts):,}",
        f"{len(accounts_in_subgraph) - len(seed_accounts):,}",
    )

    # --- Étape 3 : transactions impliquant uniquement des comptes du sous-graphe ---
    sub_mask = from_key.isin(accounts_in_subgraph) | to_key.isin(accounts_in_subgraph)
    ego_df = df[sub_mask].copy()
    logger.info(
        "Transactions du sous-graphe ego : %s | fraudes : %s",
        f"{len(ego_df):,}",
        f"{int(ego_df['is_laundering'].sum()):,}",
    )

    # --- Étape 4 : plafonnement éventuel (en priorité on garde les fraudes) ---
    if len(ego_df) > target_size:
        logger.warning(
            "Sous-graphe trop gros (%s > target %s). Tronquage en gardant 100%% des fraudes.",
            f"{len(ego_df):,}",
            f"{target_size:,}",
        )
        ego_fraud = ego_df[ego_df["is_laundering"] == 1]
        ego_legit = ego_df[ego_df["is_laundering"] == 0]
        n_legit_keep = max(target_size - len(ego_fraud), 0)
        ego_legit = ego_legit.sample(n=min(n_legit_keep, len(ego_legit)), random_state=random_seed)
        ego_df = pd.concat([ego_fraud, ego_legit], ignore_index=True)

    # --- Étape 5 (optionnelle) : bruit légitime "extérieur" ---
    if include_legit_background and len(ego_df) < target_size:
        budget = target_size - len(ego_df)
        outside_mask = ~sub_mask & (df["is_laundering"] == 0)
        outside_df = df[outside_mask].sample(
            n=min(budget, int(outside_mask.sum())),
            random_state=random_seed,
        )
        ego_df = pd.concat([ego_df, outside_df], ignore_index=True)
        logger.info("Ajout de %s transactions légitimes 'extérieures'.", f"{len(outside_df):,}")

    # --- Tri chronologique stable ---
    ego_df = ego_df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    logger.info(
        "Échantillon final : %s transactions | %s fraudes (%.4f %%)",
        f"{len(ego_df):,}",
        f"{int(ego_df['is_laundering'].sum()):,}",
        100 * ego_df["is_laundering"].mean(),
    )
    return ego_df


def save_subgraph_sample(df: pd.DataFrame, out_path: Path = config.SUBGRAPH_CSV) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Sous-graphe ego écrit : %s", out_path)
    return out_path


# --------------------------------------------------------------------------- #
# Diagnostic
# --------------------------------------------------------------------------- #
def connectivity_stats(df: pd.DataFrame) -> dict:
    """Statistiques de connexité utiles pour valider le sous-graphe."""
    from_key, to_key = _account_keys(df)
    accounts = pd.concat([from_key, to_key]).unique()
    out_deg = from_key.value_counts()
    in_deg = to_key.value_counts()
    total_deg = (out_deg.add(in_deg, fill_value=0)).astype(int)
    stats = {
        "n_transactions": int(len(df)),
        "n_accounts": int(len(accounts)),
        "avg_degree": float(total_deg.mean()),
        "median_degree": float(total_deg.median()),
        "pct_degree_one": float((total_deg == 1).mean() * 100),
        "max_degree": int(total_deg.max()),
    }
    return stats
