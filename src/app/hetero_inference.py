"""Inférence HeteroGNN et calcul de métriques pour l'application Streamlit."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from src import config
from src.gnn.homo.training import SplitMetrics
from src.gnn.hetero.models import HeteroTransactionGNN
from src.gnn.hetero.training import evaluate_hetero
from src.rdf.to_hetero_pyg import HeteroFeatureInfo, load_hetero_data

SPLIT_OPTIONS = ("train", "val", "test", "all")


@dataclass
class HeteroInferenceBundle:
    """Résultats d'une passe d'inférence sur le graphe hétérogène."""

    profile: str
    probs: np.ndarray
    y_true: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    checkpoint_path: Path
    pyg_path: Path
    info: HeteroFeatureInfo
    checkpoint_meta: dict[str, Any]
    predictions_df: pd.DataFrame
    metrics_at_training_threshold: dict[str, SplitMetrics]
    best_epoch: int | None


def _mask_for_split(
    split: str,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
) -> np.ndarray:
    if split == "train":
        return train_mask
    if split == "val":
        return val_mask
    if split == "test":
        return test_mask
    if split == "all":
        return np.ones(len(train_mask), dtype=bool)
    raise ValueError(f"Split inconnu : {split!r}")


def compute_split_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.5,
) -> SplitMetrics:
    """Métriques globales sur un sous-ensemble masqué (une « passe » d'évaluation)."""
    m = mask.astype(bool)
    if not m.any():
        return SplitMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)

    yt = y_true[m]
    ys = y_score[m]
    y_pred = (ys >= threshold).astype(int)

    pr_auc = float(average_precision_score(yt, ys)) if yt.sum() > 0 else 0.0
    return SplitMetrics(
        loss=0.0,
        pr_auc=pr_auc,
        precision=float(precision_score(yt, y_pred, zero_division=0)),
        recall=float(recall_score(yt, y_pred, zero_division=0)),
        f1=float(f1_score(yt, y_pred, zero_division=0)),
        n_samples=int(m.sum()),
        n_fraud=int(yt.sum()),
    )


def confusion_counts(
    y_true: np.ndarray,
    y_score: np.ndarray,
    mask: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, int]:
    m = mask.astype(bool)
    if not m.any():
        return {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    yt = y_true[m]
    yp = (y_score[m] >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def optimal_threshold_f1(
    y_true: np.ndarray,
    y_score: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, float]:
    """Seuil maximisant le F1 sur le masque fourni (typiquement validation)."""
    m = mask.astype(bool)
    if not m.any() or y_true[m].sum() == 0:
        return 0.5, 0.0
    yt = y_true[m]
    ys = y_score[m]
    prec, rec, thr = precision_recall_curve(yt, ys)
    f1s = 2 * prec * rec / (prec + rec + 1e-12)
    if len(thr) == 0:
        return 0.5, 0.0
    idx = int(np.argmax(f1s[:-1])) if len(f1s) > 1 else 0
    th = float(thr[idx]) if idx < len(thr) else 0.5
    return th, float(f1s[idx])


def _load_subgraph_transactions() -> pd.DataFrame | None:
    path = config.SUBGRAPH_CSV
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df.reset_index(drop=True)
    df["tx_index"] = np.arange(len(df))
    return df


def _build_predictions_dataframe(
    probs: np.ndarray,
    y_true: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    pred = (probs >= threshold).astype(int)
    split = np.full(len(probs), "unknown", dtype=object)
    split[train_mask] = "train"
    split[val_mask] = "val"
    split[test_mask] = "test"
    if train_mask.sum() == 0 and val_mask.sum() == 0 and test_mask.all():
        split[:] = "holdout"

    df = pd.DataFrame(
        {
            "tx_index": np.arange(len(probs)),
            "score_fraude": probs,
            "prediction": pred,
            "label_reel": y_true.astype(int),
            "correct": (pred == y_true).astype(int),
            "split": split,
            "alerte": pred.astype(int),
        }
    )

    tx_csv = _load_subgraph_transactions()
    if tx_csv is not None and len(tx_csv) == len(df):
        cols = [c for c in tx_csv.columns if c not in df.columns]
        df = pd.concat([df, tx_csv[cols]], axis=1)

    return df


def _load_hetero_model(
    checkpoint_path: Path,
    data,
    device: torch.device,
) -> tuple[HeteroTransactionGNN, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    in_channels = ckpt.get("in_channels") or {
        nt: data[nt].x.size(-1) for nt in data.node_types
    }
    metadata = ckpt.get("metadata", data.metadata())
    model = HeteroTransactionGNN(
        metadata=metadata,
        in_channels_dict=in_channels,
        hidden_channels=ckpt.get("hidden_dim", config.GNN_HIDDEN_DIM),
        num_layers=ckpt.get("num_layers", config.GNN_NUM_LAYERS),
        dropout=ckpt.get("dropout", config.GNN_DROPOUT),
        head_hidden=ckpt.get("edge_hidden", config.GNN_EDGE_HIDDEN_DIM),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def run_hetero_inference(
    *,
    profile: str = "train",
    checkpoint_path: Path | None = None,
    pyg_path: Path | None = None,
    device: torch.device | None = None,
    threshold: float = 0.5,
) -> HeteroInferenceBundle:
    """Charge graphe + checkpoint, infère sur toutes les transactions."""
    config.set_data_profile(profile)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = checkpoint_path or (config.RESULTS_DIR / "gnn_hetero_best.pt")
    if not ckpt_path.exists():
        train_ckpt = config.TRAIN_RESULTS_DIR / "gnn_hetero_best.pt"
        if train_ckpt.exists():
            ckpt_path = train_ckpt
    graph_path = pyg_path or config.PYG_HETERO_DATA

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint introuvable : {ckpt_path}\n"
            f"→ Entraînez avec `python scripts/05_train_gnn.py --model hetero` "
            f"(attendu : {config.TRAIN_RESULTS_DIR / 'gnn_hetero_best.pt'})."
        )
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Graphe hétérogène introuvable : {graph_path}\n"
            "→ Lancez `python scripts/04_build_pyg.py --mode hetero --data-profile {profile}`."
        )

    data, info = load_hetero_data(graph_path)
    model, ckpt = _load_hetero_model(ckpt_path, data, device)
    data = data.to(device)

    logits = model(data)
    probs = torch.sigmoid(logits).cpu().numpy()
    y_true = data["transaction"].y.cpu().numpy().astype(int)
    train_mask = data["transaction"].train_mask.cpu().numpy()
    val_mask = data["transaction"].val_mask.cpu().numpy()
    test_mask = data["transaction"].test_mask.cpu().numpy()

    criterion = nn.BCEWithLogitsLoss()
    metrics_at_training_threshold: dict[str, SplitMetrics] = {}
    for split in ("train", "val", "test"):
        mask_t = torch.from_numpy(_mask_for_split(split, train_mask, val_mask, test_mask))
        if mask_t.any():
            metrics_at_training_threshold[split] = evaluate_hetero(
                model, data, mask_t, criterion, device
            )
    if profile == "unknown" and test_mask.all():
        metrics_at_training_threshold["holdout"] = evaluate_hetero(
            model,
            data,
            torch.from_numpy(test_mask),
            criterion,
            device,
        )

    predictions_df = _build_predictions_dataframe(
        probs, y_true, train_mask, val_mask, test_mask, threshold
    )

    return HeteroInferenceBundle(
        profile=profile,
        probs=probs,
        y_true=y_true,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        checkpoint_path=ckpt_path,
        pyg_path=graph_path,
        info=info,
        checkpoint_meta=ckpt,
        predictions_df=predictions_df,
        metrics_at_training_threshold=metrics_at_training_threshold,
        best_epoch=ckpt.get("epoch"),
    )


def load_saved_training_metrics(profile: str = "train") -> dict[str, Any] | None:
    config.set_data_profile(profile)
    path = config.RESULTS_DIR / "gnn_hetero_metrics.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def metrics_to_display_dict(m: SplitMetrics) -> dict[str, Any]:
    return {
        "PR-AUC": round(m.pr_auc, 4),
        "Recall": round(m.recall, 4),
        "Precision": round(m.precision, 4),
        "F1": round(m.f1, 4),
        "Échantillons": m.n_samples,
        "Fraudes": m.n_fraud,
        "Prévalence": round(m.n_fraud / m.n_samples, 4) if m.n_samples else 0.0,
    }
