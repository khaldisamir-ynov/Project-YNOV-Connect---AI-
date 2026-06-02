"""Sous-échantillonnage par sous-graphe ego (préalable au GNN).

Pour les CSV très volumineux (LI-Medium), utiliser le mode *chunked* :
plusieurs passes sur le fichier sans charger les ~31M lignes en mémoire.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

# Types explicites : évite l'inférence float64 et les erreurs de consolidation pandas
_CSV_DTYPES: dict[str, str | type] = {
    "timestamp": str,
    "from_bank": str,
    "from_account": str,
    "to_bank": str,
    "to_account": str,
    "amount_received": np.float32,
    "receiving_currency": str,
    "amount_paid": np.float32,
    "payment_currency": str,
    "payment_format": str,
    "is_laundering": np.int8,
}


def _account_keys(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    from_key = df["from_bank"].astype(str) + "|" + df["from_account"].astype(str)
    to_key = df["to_bank"].astype(str) + "|" + df["to_account"].astype(str)
    return from_key, to_key


def _csv_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def should_use_chunked_read(csv_path: Path, chunked: bool | None) -> bool:
    if chunked is not None:
        return chunked
    return _csv_size_mb(csv_path) >= config.CSV_AUTO_CHUNK_MIN_MB


def _iter_csv_chunks(csv_path: Path, chunksize: int):
    return pd.read_csv(
        csv_path,
        header=0,
        names=config.CANONICAL_COLUMNS,
        dtype=_CSV_DTYPES,
        chunksize=chunksize,
        low_memory=True,
        engine="c",
    )


def _reservoir_update(
    rows_df: pd.DataFrame,
    reservoir: list[dict],
    seen: int,
    capacity: int,
    rng: np.random.Generator,
) -> int:
    """Met à jour un réservoir de taille `capacity` avec les lignes de `rows_df`."""
    if rows_df.empty or capacity <= 0:
        return seen

    records = rows_df.to_dict("records")
    for row in records:
        seen += 1
        if len(reservoir) < capacity:
            reservoir.append(row)
        else:
            j = int(rng.integers(0, seen))
            if j < capacity:
                reservoir[j] = row
    return seen


def _expand_accounts_one_hop_chunked(
    csv_path: Path,
    seed_accounts: set[str],
    chunksize: int,
) -> set[str]:
    """Étend le ensemble de comptes d'un hop en scannant le CSV par morceaux."""
    expanded = set(seed_accounts)
    for chunk in _iter_csv_chunks(csv_path, chunksize):
        from_key, to_key = _account_keys(chunk)
        touching = from_key.isin(seed_accounts) | to_key.isin(seed_accounts)
        if not touching.any():
            continue
        expanded.update(from_key[touching].unique())
        expanded.update(to_key[touching].unique())
    new = expanded - seed_accounts
    logger.info(
        "Expansion (chunked) : +%s comptes | total %s",
        f"{len(new):,}",
        f"{len(expanded):,}",
    )
    return expanded


def _finalize_ego_df(
    ego_df: pd.DataFrame,
    target_size: int,
    random_seed: int,
    include_legit_background: bool,
    outside_legit: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Tronque / complète vers `target_size` en gardant 100 %% des fraudes."""
    if len(ego_df) > target_size:
        logger.warning(
            "Sous-graphe trop gros (%s > %s). Tronquage (100 %% fraudes conservées).",
            f"{len(ego_df):,}",
            f"{target_size:,}",
        )
        ego_fraud = ego_df[ego_df["is_laundering"] == 1]
        ego_legit = ego_df[ego_df["is_laundering"] == 0]
        n_legit_keep = max(target_size - len(ego_fraud), 0)
        ego_legit = ego_legit.sample(
            n=min(n_legit_keep, len(ego_legit)),
            random_state=random_seed,
        )
        ego_df = pd.concat([ego_fraud, ego_legit], ignore_index=True)

    if include_legit_background and len(ego_df) < target_size and outside_legit is not None:
        budget = target_size - len(ego_df)
        if len(outside_legit) > 0:
            extra = outside_legit.sample(
                n=min(budget, len(outside_legit)),
                random_state=random_seed,
            )
            ego_df = pd.concat([ego_df, extra], ignore_index=True)
            logger.info("Ajout de %s transactions légitimes extérieures.", f"{len(extra):,}")

    if len(ego_df) < target_size and outside_legit is not None and len(outside_legit) > 0:
        budget = target_size - len(ego_df)
        extra = outside_legit.sample(
            n=min(budget, len(outside_legit)),
            random_state=random_seed,
        )
        ego_df = pd.concat([ego_df, extra], ignore_index=True)
        logger.info(
            "Complément légitime : +%s → total %s.",
            f"{len(extra):,}",
            f"{len(ego_df):,}",
        )

    ego_df = ego_df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    if len(ego_df) != target_size:
        logger.warning(
            "Taille finale %s ≠ cible %s.",
            f"{len(ego_df):,}",
            f"{target_size:,}",
        )

    logger.info(
        "Échantillon final : %s transactions | %s fraudes (%.4f %%)",
        f"{len(ego_df):,}",
        f"{int(ego_df['is_laundering'].sum()):,}",
        100 * ego_df["is_laundering"].mean(),
    )
    return ego_df


def _reservoir_sample_outside_legit(
    csv_path: Path,
    accounts_in_subgraph: set[str],
    n_sample: int,
    random_seed: int,
    chunksize: int,
) -> pd.DataFrame:
    """Échantillonne n lignes légitimes hors du sous-graphe (mémoire bornée)."""
    if n_sample <= 0:
        return pd.DataFrame(columns=config.CANONICAL_COLUMNS)

    rng = np.random.default_rng(random_seed)
    reservoir: list[dict] = []
    seen = 0

    for chunk in _iter_csv_chunks(csv_path, chunksize):
        from_key, to_key = _account_keys(chunk)
        outside = (
            (~from_key.isin(accounts_in_subgraph))
            & (~to_key.isin(accounts_in_subgraph))
            & (chunk["is_laundering"] == 0)
        )
        seen = _reservoir_update(chunk[outside], reservoir, seen, n_sample, rng)
        del chunk
        gc.collect()

    if not reservoir:
        return pd.DataFrame(columns=config.CANONICAL_COLUMNS)
    return pd.DataFrame(reservoir)


def ego_subgraph_sample_chunked(
    csv_path: Path,
    num_hops: int = config.EGO_NUM_HOPS,
    target_size: int = config.EGO_TARGET_SIZE,
    include_legit_background: bool = config.EGO_INCLUDE_LEGIT_BACKGROUND,
    random_seed: int = config.RANDOM_SEED,
    chunksize: int = config.CSV_CHUNKSIZE,
) -> pd.DataFrame:
    """Sous-graphe ego sans charger le CSV entier en mémoire."""
    logger.info(
        "Mode CHUNKED — %s (%.1f Mo) | chunksize=%s",
        csv_path.name,
        _csv_size_mb(csv_path),
        f"{chunksize:,}",
    )

    n_total = 0
    n_fraud = 0
    seed_accounts: set[str] = set()

    for i, chunk in enumerate(_iter_csv_chunks(csv_path, chunksize), start=1):
        n_total += len(chunk)
        fraud_mask = chunk["is_laundering"] == 1
        n_fraud += int(fraud_mask.sum())
        if fraud_mask.any():
            from_key, to_key = _account_keys(chunk)
            seed_accounts.update(from_key[fraud_mask].unique())
            seed_accounts.update(to_key[fraud_mask].unique())
        del chunk
        if i % 50 == 0:
            logger.info("Scan graines : chunk %d | %s lignes lues…", i, f"{n_total:,}")
            gc.collect()

    logger.info(
        "Corpus scanné : %s transactions | %s fraudes | %s comptes graines",
        f"{n_total:,}",
        f"{n_fraud:,}",
        f"{len(seed_accounts):,}",
    )
    if not seed_accounts:
        raise ValueError(f"Aucune fraude trouvée dans {csv_path}")

    accounts_in_subgraph = set(seed_accounts)
    for hop in range(1, num_hops + 1):
        logger.info("Hop %d / %d (chunked)…", hop, num_hops)
        accounts_in_subgraph = _expand_accounts_one_hop_chunked(
            csv_path, accounts_in_subgraph, chunksize
        )

    parts: list[pd.DataFrame] = []
    collected = 0
    for i, chunk in enumerate(_iter_csv_chunks(csv_path, chunksize), start=1):
        from_key, to_key = _account_keys(chunk)
        sub_mask = from_key.isin(accounts_in_subgraph) | to_key.isin(accounts_in_subgraph)
        if sub_mask.any():
            part = chunk[sub_mask]
            parts.append(part)
            collected += len(part)
        del chunk
        if i % 50 == 0:
            logger.info("Collecte ego : chunk %d | %s tx candidates…", i, f"{collected:,}")
            gc.collect()

    ego_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=config.CANONICAL_COLUMNS)
    del parts
    gc.collect()
    logger.info(
        "Transactions ego collectées : %s | fraudes : %s",
        f"{len(ego_df):,}",
        f"{int(ego_df['is_laundering'].sum()):,}" if len(ego_df) else "0",
    )

    need_outside = (
        (include_legit_background and len(ego_df) < target_size)
        or len(ego_df) < target_size
    )
    outside_legit = None
    if need_outside:
        budget = max(target_size - len(ego_df), target_size // 10)
        logger.info("Échantillonnage légitime hors ego (réservoir, max %s)…", f"{budget:,}")
        outside_legit = _reservoir_sample_outside_legit(
            csv_path,
            accounts_in_subgraph,
            n_sample=budget,
            random_seed=random_seed,
            chunksize=chunksize,
        )

    return _finalize_ego_df(
        ego_df,
        target_size=target_size,
        random_seed=random_seed,
        include_legit_background=include_legit_background,
        outside_legit=outside_legit,
    )


def presample_raw_csv(
    csv_path: Path,
    out_path: Path,
    target_size: int,
    random_seed: int = config.RANDOM_SEED,
    chunksize: int = config.CSV_CHUNKSIZE,
    keep_all_laundering: bool = True,
) -> Path:
    """Réduit un CSV brut (stratifié) avant l'ego — utile si le chunked reste trop lourd."""
    logger.info(
        "Présélection %s → %s (cible %s lignes, chunked)",
        csv_path.name,
        out_path.name,
        f"{target_size:,}",
    )
    fraud_path = out_path.with_name(out_path.stem + "_fraud_tmp.csv")
    legit_reservoir: list[dict] = []
    legit_seen = 0
    n_fraud = 0
    rng = np.random.default_rng(random_seed)
    legit_budget = max(target_size - 10_000, target_size // 2)
    fraud_header_written = False

    for i, chunk in enumerate(_iter_csv_chunks(csv_path, chunksize), start=1):
        fraud = chunk[chunk["is_laundering"] == 1]
        if len(fraud):
            fraud.to_csv(
                fraud_path,
                mode="a",
                header=not fraud_header_written,
                index=False,
            )
            fraud_header_written = True
            n_fraud += len(fraud)
        legit = chunk[chunk["is_laundering"] == 0]
        legit_seen = _reservoir_update(legit, legit_reservoir, legit_seen, legit_budget, rng)
        del chunk, fraud, legit
        if i % 50 == 0:
            logger.info("Présélection : chunk %d | fraudes=%s", i, f"{n_fraud:,}")
            gc.collect()

    fraud_df = (
        pd.read_csv(fraud_path, dtype=_CSV_DTYPES)
        if fraud_path.exists()
        else pd.DataFrame(columns=config.CANONICAL_COLUMNS)
    )
    if fraud_path.exists():
        fraud_path.unlink()

    legit_df = pd.DataFrame(legit_reservoir) if legit_reservoir else pd.DataFrame()
    n_legit_keep = max(target_size - len(fraud_df), 0) if keep_all_laundering else target_size
    if len(legit_df) > n_legit_keep:
        legit_df = legit_df.sample(n=n_legit_keep, random_state=random_seed)

    sample = pd.concat([fraud_df, legit_df], ignore_index=True)
    del fraud_df, legit_df, legit_reservoir
    gc.collect()
    sample = sample.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_path, index=False)
    del sample
    gc.collect()
    logger.info(
        "Présélection écrite : %s lignes (%s fraudes)",
        f"{len(sample):,}",
        f"{int(sample['is_laundering'].sum()):,}",
    )
    return out_path


def ego_subgraph_sample(
    csv_path: Path = config.RAW_CSV,
    num_hops: int = config.EGO_NUM_HOPS,
    target_size: int = config.EGO_TARGET_SIZE,
    include_legit_background: bool = config.EGO_INCLUDE_LEGIT_BACKGROUND,
    random_seed: int = config.RANDOM_SEED,
    chunked: bool | None = None,
    chunksize: int = config.CSV_CHUNKSIZE,
) -> pd.DataFrame:
    """Échantillonne un sous-graphe ego (auto-chunked si le fichier est volumineux)."""
    if should_use_chunked_read(csv_path, chunked):
        return ego_subgraph_sample_chunked(
            csv_path=csv_path,
            num_hops=num_hops,
            target_size=target_size,
            include_legit_background=include_legit_background,
            random_seed=random_seed,
            chunksize=chunksize,
        )

    logger.info("Mode mémoire — lecture complète de %s", csv_path)
    df = pd.read_csv(
        csv_path,
        header=0,
        names=config.CANONICAL_COLUMNS,
        dtype=_CSV_DTYPES,
    )
    n_total = len(df)
    n_fraud = int(df["is_laundering"].sum())
    logger.info(
        "Brut : %s transactions | %s fraudes (%.4f %%)",
        f"{n_total:,}",
        f"{n_fraud:,}",
        100 * n_fraud / n_total if n_total else 0,
    )

    from_key, to_key = _account_keys(df)
    fraud_mask = df["is_laundering"] == 1
    seed_accounts: set[str] = set()
    seed_accounts.update(from_key[fraud_mask].unique())
    seed_accounts.update(to_key[fraud_mask].unique())
    logger.info("Comptes graines : %s", f"{len(seed_accounts):,}")

    accounts_in_subgraph = set(seed_accounts)
    for hop in range(1, num_hops + 1):
        logger.info("Hop %d / %d…", hop, num_hops)
        from_k, to_k = _account_keys(df)
        touching = from_k.isin(accounts_in_subgraph) | to_k.isin(accounts_in_subgraph)
        expanded = set(from_k[touching].unique()) | set(to_k[touching].unique())
        new_accounts = expanded - accounts_in_subgraph
        if not new_accounts:
            break
        accounts_in_subgraph |= new_accounts
        logger.info("Expansion : +%s comptes", f"{len(new_accounts):,}")

    sub_mask = from_key.isin(accounts_in_subgraph) | to_key.isin(accounts_in_subgraph)
    ego_df = df[sub_mask].copy()

    outside_legit = None
    if len(ego_df) < target_size or include_legit_background:
        kept_idx = set(ego_df.index)
        outside_legit = df[(~df.index.isin(kept_idx)) & (df["is_laundering"] == 0)]

    return _finalize_ego_df(
        ego_df,
        target_size=target_size,
        random_seed=random_seed,
        include_legit_background=include_legit_background,
        outside_legit=outside_legit,
    )


def save_subgraph_sample(df: pd.DataFrame, out_path: Path = config.SUBGRAPH_CSV) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Sous-graphe ego écrit : %s", out_path)
    return out_path


def connectivity_stats(df: pd.DataFrame) -> dict:
    from_key, to_key = _account_keys(df)
    accounts = pd.concat([from_key, to_key]).unique()
    out_deg = from_key.value_counts()
    in_deg = to_key.value_counts()
    total_deg = (out_deg.add(in_deg, fill_value=0)).astype(int)
    return {
        "n_transactions": int(len(df)),
        "n_accounts": int(len(accounts)),
        "avg_degree": float(total_deg.mean()),
        "median_degree": float(total_deg.median()),
        "pct_degree_one": float((total_deg == 1).mean() * 100),
        "max_degree": int(total_deg.max()),
    }
