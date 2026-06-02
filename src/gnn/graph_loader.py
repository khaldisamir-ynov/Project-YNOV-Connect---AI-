"""Conversion du CSV échantillonné (sous-graphe ego) vers un objet PyG.

Modèle de graphe choisi pour cette première itération :
    **Graphe homogène orienté**
        - **Nœud** = Compte bancaire (identité = couple `bank|account`)
        - **Arête** = Transaction (orientée `from_account → to_account`)
        - **Label** = porté par l'arête (`is_laundering` 0/1)

Ce choix est volontairement simple (vs un `HeteroData` Account / Tx / Bank)
car la tâche cible — *classification d'arêtes* — s'exprime nativement sur
un graphe homogène, et la majorité des modèles PyG (GraphSAGE, GAT…)
travaillent ainsi.

Features :
    - **Nœud** : degré entrant (log), degré sortant (log), montant total
      envoyé (log), montant total reçu (log). 4 features numériques.
    - **Arête** : montant log-normalisé + one-hot du format de paiement
      + one-hot de la devise (top-N + "Other"). Dimension variable
      retournée dans `feature_info`.

Sortie : un `torch_geometric.data.Data` enrichi de masques train/val/test
**stratifiés sur le label d'arête**.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

from src import config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Métadonnées exportées avec le tenseur
# --------------------------------------------------------------------------- #
@dataclass
class FeatureInfo:
    """Décrit le contenu des features pour pouvoir interpréter le tenseur."""

    node_feature_names: list[str] = field(default_factory=list)
    edge_feature_names: list[str] = field(default_factory=list)
    payment_format_vocab: list[str] = field(default_factory=list)
    currency_vocab: list[str] = field(default_factory=list)
    n_nodes: int = 0
    n_edges: int = 0
    n_fraud_edges: int = 0


# --------------------------------------------------------------------------- #
# Encodage des features
# --------------------------------------------------------------------------- #
def _one_hot(series: pd.Series, vocab: list[str]) -> np.ndarray:
    """One-hot d'une série sur un vocabulaire fixé. Valeurs hors vocab → 'Other'."""
    series_clipped = series.where(series.isin(vocab), other="Other")
    # `Other` est toujours la dernière colonne si présent dans vocab
    cat = pd.Categorical(series_clipped, categories=vocab)
    return pd.get_dummies(cat).values.astype(np.float32)


def _top_n_vocab(series: pd.Series, n: int) -> list[str]:
    """Renvoie les n catégories les plus fréquentes + 'Other'."""
    top = series.value_counts().head(n).index.tolist()
    if "Other" not in top:
        top.append("Other")
    return top


# --------------------------------------------------------------------------- #
# Construction du graphe
# --------------------------------------------------------------------------- #
def build_pyg_data(
    df: pd.DataFrame,
    currency_top_n: int = 8,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_seed: int = config.RANDOM_SEED,
) -> tuple[Data, FeatureInfo]:
    """Construit un `Data` PyG à partir du DataFrame échantillonné."""
    logger.info("Construction du graphe PyG pour %s transactions…", f"{len(df):,}")

    # --- 1. Mapping comptes → indices entiers ---
    from_key = df["from_bank"].astype(str) + "|" + df["from_account"].astype(str)
    to_key = df["to_bank"].astype(str) + "|" + df["to_account"].astype(str)
    unique_accounts = pd.Index(pd.unique(np.concatenate([from_key.values, to_key.values])))
    node_id = pd.Series(np.arange(len(unique_accounts)), index=unique_accounts)
    src = node_id.loc[from_key.values].values
    dst = node_id.loc[to_key.values].values
    edge_index = torch.from_numpy(np.vstack([src, dst])).long()
    logger.info("Nœuds : %s | Arêtes : %s", f"{len(unique_accounts):,}", f"{len(df):,}")

    # --- 2. Features d'arêtes ---
    # 2a. Montant log-normalisé
    amount = df["amount_paid"].astype(np.float32).values
    amount_log = np.log1p(np.clip(amount, a_min=0, a_max=None))
    amount_log = (amount_log - amount_log.mean()) / (amount_log.std() + 1e-8)

    # 2b. One-hot du format de paiement
    payment_formats = df["payment_format"].astype(str)
    payment_format_vocab = sorted(payment_formats.unique().tolist())
    pf_onehot = _one_hot(payment_formats, payment_format_vocab)

    # 2c. One-hot de la devise (top-N + Other)
    currencies = df["payment_currency"].astype(str)
    currency_vocab = _top_n_vocab(currencies, n=currency_top_n)
    cur_onehot = _one_hot(currencies, currency_vocab)

    edge_attr = np.concatenate(
        [amount_log.reshape(-1, 1), pf_onehot, cur_onehot], axis=1
    ).astype(np.float32)
    edge_attr_t = torch.from_numpy(edge_attr)

    edge_feature_names = (
        ["amount_log_z"]
        + [f"pf_{v}" for v in payment_format_vocab]
        + [f"cur_{v}" for v in currency_vocab]
    )
    logger.info(
        "Features d'arêtes : %s (1 montant + %s formats + %s devises)",
        edge_attr.shape[1],
        len(payment_format_vocab),
        len(currency_vocab),
    )

    # --- 3. Features de nœuds (agrégats topologiques) ---
    out_count = pd.Series(src).value_counts().reindex(range(len(unique_accounts)), fill_value=0)
    in_count = pd.Series(dst).value_counts().reindex(range(len(unique_accounts)), fill_value=0)
    sent = pd.DataFrame({"node": src, "amt": amount}).groupby("node")["amt"].sum().reindex(
        range(len(unique_accounts)), fill_value=0.0
    )
    received = pd.DataFrame({"node": dst, "amt": amount}).groupby("node")["amt"].sum().reindex(
        range(len(unique_accounts)), fill_value=0.0
    )

    node_features = np.stack(
        [
            np.log1p(out_count.values.astype(np.float32)),
            np.log1p(in_count.values.astype(np.float32)),
            np.log1p(sent.values.astype(np.float32)),
            np.log1p(received.values.astype(np.float32)),
        ],
        axis=1,
    )
    node_features = (node_features - node_features.mean(axis=0)) / (
        node_features.std(axis=0) + 1e-8
    )
    x = torch.from_numpy(node_features.astype(np.float32))
    node_feature_names = ["log_out_deg", "log_in_deg", "log_amount_sent", "log_amount_received"]
    logger.info("Features de nœuds : %s", node_features.shape[1])

    # --- 4. Label d'arête ---
    y = torch.from_numpy(df["is_laundering"].astype(np.int64).values)
    n_fraud = int(y.sum())
    logger.info("Fraudes (arêtes positives) : %s / %s (%.2f %%)", n_fraud, len(y), 100 * n_fraud / len(y))

    # --- 5. Splits stratifiés (sur les arêtes) ---
    indices = np.arange(len(y))
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=val_size + test_size,
        stratify=y.numpy(),
        random_state=random_seed,
    )
    val_ratio = val_size / (val_size + test_size)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1 - val_ratio,
        stratify=y.numpy()[temp_idx],
        random_state=random_seed,
    )

    n_edges = len(y)
    train_mask = torch.zeros(n_edges, dtype=torch.bool)
    val_mask = torch.zeros(n_edges, dtype=torch.bool)
    test_mask = torch.zeros(n_edges, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    logger.info(
        "Splits stratifiés — train: %s | val: %s | test: %s",
        f"{int(train_mask.sum()):,}",
        f"{int(val_mask.sum()):,}",
        f"{int(test_mask.sum()):,}",
    )

    # --- 6. Assemblage final ---
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr_t,
        y=y,
        num_nodes=len(unique_accounts),
    )
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    info = FeatureInfo(
        node_feature_names=node_feature_names,
        edge_feature_names=edge_feature_names,
        payment_format_vocab=payment_format_vocab,
        currency_vocab=currency_vocab,
        n_nodes=int(data.num_nodes),
        n_edges=int(n_edges),
        n_fraud_edges=n_fraud,
    )
    return data, info


def save_pyg_data(data: Data, info: FeatureInfo, out_path: Path = config.PYG_DATA) -> Path:
    """Sérialise (Data + FeatureInfo) dans un unique fichier .pt."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"data": data, "info": info}, out_path)
    logger.info("Tenseur PyG écrit : %s (%s octets)", out_path, f"{out_path.stat().st_size:,}")
    return out_path


def load_pyg_data(path: Path = config.PYG_DATA) -> tuple[Data, FeatureInfo]:
    """Recharge le tenseur PyG sauvegardé."""
    bundle = torch.load(path, weights_only=False)
    return bundle["data"], bundle["info"]
