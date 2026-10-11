"""Scoring d'une transaction saisie (recherche ou injection dans le graphe)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from src import config
from src.app.hetero_inference import HeteroInferenceBundle, _load_hetero_model
from src.rdf.to_hetero_pyg import HeteroFeatureInfo, load_hetero_data

MATCH_COLUMNS = [
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_paid",
    "payment_currency",
    "payment_format",
]


@dataclass
class SingleTransactionResult:
    """Résultat pour une transaction analysée."""

    score: float
    prediction: int
    label: str
    threshold: float
    tx_index: int | None
    mode: str  # "existing" | "injected"
    from_account_in_graph: bool
    to_account_in_graph: bool
    true_label: int | None
    true_label_known: bool
    details: dict[str, Any]


def load_subgraph_dataframe() -> pd.DataFrame | None:
    path = config.SUBGRAPH_CSV
    if not path.exists():
        return None
    return pd.read_csv(path)


def find_transaction_index(
    df: pd.DataFrame,
    fields: dict[str, Any],
    *,
    tx_index: int | None = None,
) -> int | None:
    """Retrouve l'index d'une transaction dans le sous-graphe CSV."""
    if tx_index is not None:
        if 0 <= tx_index < len(df):
            return int(tx_index)
        return None

    work = df.copy()
    for col in MATCH_COLUMNS:
        if col not in work.columns or col not in fields:
            continue
        val = fields[col]
        if col == "amount_paid":
            work = work[np.isclose(work[col].astype(float), float(val), rtol=1e-4, atol=1e-2)]
        else:
            work = work[work[col].astype(str) == str(val)]

    if len(work) == 1:
        return int(work.index[0])
    if len(work) > 1:
        return int(work.index[0])
    return None


def build_account_key_index(
    data: HeteroData,
    subgraph_df: pd.DataFrame,
) -> dict[tuple[str, str], int]:
    """Associe (banque, compte) → index nœud account dans le graphe PyG."""
    sends = data["account", "sends_via", "transaction"].edge_index
    deposits = data["transaction", "deposits_to", "account"].edge_index

    from_acct_by_tx: dict[int, int] = {}
    to_acct_by_tx: dict[int, int] = {}
    for i in range(sends.size(1)):
        a, t = int(sends[0, i]), int(sends[1, i])
        from_acct_by_tx[t] = a
    for i in range(deposits.size(1)):
        t, a = int(deposits[0, i]), int(deposits[1, i])
        to_acct_by_tx[t] = a

    key_to_idx: dict[tuple[str, str], int] = {}
    n = min(data["transaction"].num_nodes, len(subgraph_df))
    for t in range(n):
        row = subgraph_df.iloc[t]
        fk = (str(row["from_bank"]), str(row["from_account"]))
        tk = (str(row["to_bank"]), str(row["to_account"]))
        if t in from_acct_by_tx:
            key_to_idx[fk] = from_acct_by_tx[t]
        if t in to_acct_by_tx:
            key_to_idx[tk] = to_acct_by_tx[t]
    return key_to_idx


def _parse_feature_names(feature_names: list[str]) -> dict[str, list[str]]:
    """Regroupe les colonnes one-hot par préfixe (ex. paymentCurrency_EUR)."""
    cats: dict[str, list[str]] = {}
    for name in feature_names:
        if name.endswith("_z"):
            continue
        if "_" in name:
            prefix = name.rsplit("_", 1)[0]
            cats.setdefault(prefix, []).append(name)
    return cats


def encode_transaction_features(
    fields: dict[str, Any],
    feature_names: list[str],
    reference_x: torch.Tensor,
) -> torch.Tensor:
    """Encode une transaction saisie dans le même espace que le graphe entraîné."""
    n_feat = len(feature_names)
    vec = np.zeros(n_feat, dtype=np.float32)
    name_to_i = {n: i for i, n in enumerate(feature_names)}

    ref = reference_x.cpu().numpy()
    amount = float(fields.get("amount_paid", 0.0))
    log_amt = np.log1p(max(amount, 0.0))

    if "amountPaid_z" in name_to_i:
        col = ref[:, name_to_i["amountPaid_z"]]
        mean, std = float(col.mean()), float(col.std()) + 1e-8
        vec[name_to_i["amountPaid_z"]] = (log_amt - mean) / std

    str_fields = {
        "paymentCurrency": str(fields.get("payment_currency", "")),
        "paymentFormat": str(fields.get("payment_format", "")),
        "timestamp": str(fields.get("timestamp", "")),
    }
    cat_groups = _parse_feature_names(feature_names)
    for prefix, value in str_fields.items():
        if prefix not in cat_groups:
            continue
        target = f"{prefix}_{value}"
        if target in name_to_i:
            vec[name_to_i[target]] = 1.0
        elif f"{prefix}_Other" in name_to_i:
            vec[name_to_i[f"{prefix}_Other"]] = 1.0
        else:
            for col_name in cat_groups[prefix]:
                if col_name.endswith("_Other") or col_name.split("_")[-1] == "Other":
                    vec[name_to_i[col_name]] = 1.0
                    break

    return torch.from_numpy(vec).unsqueeze(0)


def _append_transaction_edges(
    data: HeteroData,
    from_acct: int,
    to_acct: int,
    t_new: int,
) -> None:
    """Ajoute sends_via / deposits_to (+ arêtes inverses) pour une nouvelle transaction."""
    send = data["account", "sends_via", "transaction"].edge_index
    dep = data["transaction", "deposits_to", "account"].edge_index

    new_send = torch.tensor([[from_acct], [t_new]], dtype=torch.long)
    new_dep = torch.tensor([[t_new], [to_acct]], dtype=torch.long)

    data["account", "sends_via", "transaction"].edge_index = torch.cat([send, new_send], dim=1)
    data["transaction", "deposits_to", "account"].edge_index = torch.cat([dep, new_dep], dim=1)

    rev_send_type = ("transaction", "rev_sends_via", "account")
    rev_dep_type = ("account", "rev_deposits_to", "transaction")
    if rev_send_type in data.edge_types:
        rs = data[rev_send_type].edge_index
        data[rev_send_type].edge_index = torch.cat(
            [rs, torch.tensor([[t_new], [from_acct]], dtype=torch.long)], dim=1
        )
    if rev_dep_type in data.edge_types:
        rd = data[rev_dep_type].edge_index
        data[rev_dep_type].edge_index = torch.cat(
            [rd, torch.tensor([[to_acct], [t_new]], dtype=torch.long)], dim=1
        )


@torch.no_grad()
def score_injected_transaction(
    fields: dict[str, Any],
    *,
    checkpoint_path: Path,
    pyg_path: Path,
    info: HeteroFeatureInfo,
    threshold: float = 0.5,
    device: torch.device | None = None,
) -> SingleTransactionResult:
    """Injecte une transaction (comptes déjà dans le graphe) et recalcule le score GNN."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subgraph_df = load_subgraph_dataframe()
    if subgraph_df is None:
        raise FileNotFoundError(f"CSV sous-graphe introuvable : {config.SUBGRAPH_CSV}")

    data, _ = load_hetero_data(pyg_path)
    acct_map = build_account_key_index(data, subgraph_df)

    fk = (str(fields["from_bank"]), str(fields["from_account"]))
    tk = (str(fields["to_bank"]), str(fields["to_account"]))
    from_idx = acct_map.get(fk)
    to_idx = acct_map.get(tk)

    if from_idx is None or to_idx is None:
        return SingleTransactionResult(
            score=0.0,
            prediction=0,
            label="Indisponible",
            threshold=threshold,
            tx_index=None,
            mode="injected",
            from_account_in_graph=from_idx is not None,
            to_account_in_graph=to_idx is not None,
            true_label=None,
            true_label_known=False,
            details={
                "error": (
                    "Les comptes émetteur et/ou destinataire ne figurent pas dans le graphe ego. "
                    "Utilisez des comptes déjà présents dans le sous-graphe, ou saisissez une "
                    "transaction existante via l'index / les champs exacts."
                ),
            },
        )

    tx_names = info.node_feature_names.get("transaction", [])
    new_x = encode_transaction_features(
        fields, tx_names, data["transaction"].x
    )
    data = data.clone()
    t_new = data["transaction"].num_nodes
    data["transaction"].x = torch.cat([data["transaction"].x, new_x.to(data["transaction"].x.dtype)], dim=0)
    data["transaction"].y = torch.cat(
        [data["transaction"].y, torch.tensor([-1], dtype=data["transaction"].y.dtype)]
    )
    _append_transaction_edges(data, from_idx, to_idx, t_new)

    model, _ = _load_hetero_model(checkpoint_path, data, device)
    data = data.to(device)
    logits = model(data)
    score = float(torch.sigmoid(logits[-1]).item())
    pred = int(score >= threshold)

    return SingleTransactionResult(
        score=score,
        prediction=pred,
        label="Fraude suspecte" if pred else "Transaction légitime",
        threshold=threshold,
        tx_index=t_new,
        mode="injected",
        from_account_in_graph=True,
        to_account_in_graph=True,
        true_label=None,
        true_label_known=False,
        details={
            "message": "Transaction ajoutée au graphe en mémoire ; score recalculé par message passing.",
            "from_account_idx": from_idx,
            "to_account_idx": to_idx,
        },
    )


def score_from_bundle(
    bundle: HeteroInferenceBundle,
    fields: dict[str, Any],
    *,
    threshold: float = 0.5,
    tx_index: int | None = None,
    allow_inject: bool = True,
) -> SingleTransactionResult:
    """Score une transaction existante (recherche CSV) ou injectée si absente."""
    subgraph_df = load_subgraph_dataframe()

    idx = find_transaction_index(subgraph_df, fields, tx_index=tx_index) if subgraph_df is not None else tx_index

    if idx is not None and 0 <= idx < len(bundle.probs):
        score = float(bundle.probs[idx])
        pred = int(score >= threshold)
        y_true = int(bundle.y_true[idx])
        known = y_true in (0, 1)
        return SingleTransactionResult(
            score=score,
            prediction=pred,
            label="Fraude suspecte" if pred else "Transaction légitime",
            threshold=threshold,
            tx_index=idx,
            mode="existing",
            from_account_in_graph=True,
            to_account_in_graph=True,
            true_label=y_true if known else None,
            true_label_known=known,
            details={
                "message": "Transaction trouvée dans le sous-graphe ; score issu de l'inférence GNN globale.",
                "match": "exact",
            },
        )

    if not allow_inject:
        return SingleTransactionResult(
            score=0.0,
            prediction=0,
            label="Non trouvée",
            threshold=threshold,
            tx_index=None,
            mode="existing",
            from_account_in_graph=False,
            to_account_in_graph=False,
            true_label=None,
            true_label_known=False,
            details={
                "error": "Aucune transaction correspondante dans le graphe. Vérifiez les champs ou l'index.",
            },
        )

    return score_injected_transaction(
        fields,
        checkpoint_path=bundle.checkpoint_path,
        pyg_path=bundle.pyg_path,
        info=bundle.info,
        threshold=threshold,
    )


def sample_transaction_row(df: pd.DataFrame, *, fraud: bool | None = None) -> dict[str, Any]:
    """Exemple aléatoire pour préremplir le formulaire."""
    work = df
    if fraud is True and "is_laundering" in work.columns:
        work = work[work["is_laundering"] == 1]
    elif fraud is False and "is_laundering" in work.columns:
        work = work[work["is_laundering"] == 0]
    if work.empty:
        work = df
    row = work.sample(1, random_state=None).iloc[0]
    return {
        "tx_index": int(row.name) if hasattr(row, "name") else int(row.get("tx_index", 0)),
        "timestamp": str(row.get("timestamp", "")),
        "from_bank": str(row["from_bank"]),
        "from_account": str(row["from_account"]),
        "to_bank": str(row["to_bank"]),
        "to_account": str(row["to_account"]),
        "amount_received": float(row.get("amount_received", row.get("amount_paid", 0))),
        "receiving_currency": str(row.get("receiving_currency", "")),
        "amount_paid": float(row["amount_paid"]),
        "payment_currency": str(row["payment_currency"]),
        "payment_format": str(row["payment_format"]),
    }
