"""Boucle d'entraînement et métriques pour la classification d'arêtes GNN."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch_geometric.data import Data

from src import config
from src.gnn.homo.models import EdgeClassifierGNN
from src.gnn.graph_loader import load_pyg_data

logger = logging.getLogger(__name__)


@dataclass
class SplitMetrics:
    loss: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    n_samples: int
    n_fraud: int


@dataclass
class TrainingResult:
    model_type: str
    best_epoch: int
    train: SplitMetrics
    val: SplitMetrics
    test: SplitMetrics


def _compute_pos_weight(y: torch.Tensor, mask: torch.Tensor) -> float:
    labels = y[mask].float()
    n_pos = labels.sum().item()
    n_neg = (labels == 0).sum().item()
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


@torch.no_grad()
def evaluate(
    model: EdgeClassifierGNN,
    data: Data,
    mask: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
) -> SplitMetrics:
    model.eval()
    data = data.to(device)
    mask = mask.to(device)

    logits = model(data)
    labels = data.y.float()

    loss = criterion(logits[mask], labels[mask]).item()
    probs = torch.sigmoid(logits[mask]).cpu().numpy()
    y_true = labels[mask].cpu().numpy().astype(int)
    y_pred = (probs >= 0.5).astype(int)

    pr_auc = float(average_precision_score(y_true, probs)) if y_true.sum() > 0 else 0.0
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return SplitMetrics(
        loss=loss,
        pr_auc=pr_auc,
        precision=precision,
        recall=recall,
        f1=f1,
        n_samples=int(mask.sum()),
        n_fraud=int(y_true.sum()),
    )


def train_gnn(
    model_type: str = config.GNN_MODEL_TYPE,
    hidden_dim: int = config.GNN_HIDDEN_DIM,
    num_layers: int = config.GNN_NUM_LAYERS,
    dropout: float = config.GNN_DROPOUT,
    lr: float = config.GNN_LR,
    weight_decay: float = config.GNN_WEIGHT_DECAY,
    epochs: int = config.GNN_EPOCHS,
    patience: int = config.GNN_PATIENCE,
    edge_hidden: int = config.GNN_EDGE_HIDDEN_DIM,
    data_path: Path = config.PYG_DATA,
    checkpoint_path: Path | None = None,
    metrics_path: Path | None = None,
    history_path: Path | None = None,
    device: torch.device | None = None,
) -> TrainingResult:
    """Entraîne un GNN et sauvegarde le meilleur checkpoint (val PR-AUC)."""
    if checkpoint_path is None:
        checkpoint_path = config.RESULTS_DIR / f"gnn_{model_type}_best.pt"
    if metrics_path is None:
        metrics_path = config.RESULTS_DIR / f"gnn_{model_type}_metrics.json"
    if history_path is None:
        history_path = config.RESULTS_DIR / f"gnn_{model_type}_history.json"
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, info = load_pyg_data(data_path)
    data = data.to(device)

    model = EdgeClassifierGNN(
        in_channels=data.x.size(1),
        edge_dim=data.edge_attr.size(1),
        hidden_channels=hidden_dim,
        edge_hidden=edge_hidden,
        num_layers=num_layers,
        dropout=dropout,
        model_type=model_type,
    ).to(device)

    pos_weight = torch.tensor(
        [_compute_pos_weight(data.y, data.train_mask)],
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    logger.info(
        "Entraînement %s — device=%s | nœuds=%s | arêtes=%s | pos_weight=%.2f",
        model_type.upper(),
        device,
        f"{data.num_nodes:,}",
        f"{data.edge_index.size(1):,}",
        pos_weight.item(),
    )

    best_val_pr_auc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_state: dict | None = None
    epoch_history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask].float())
        loss.backward()
        optimizer.step()

        val_metrics = evaluate(model, data, data.val_mask, criterion, device)
        epoch_history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "val_pr_auc": val_metrics.pr_auc,
                "val_f1": val_metrics.f1,
                "val_recall": val_metrics.recall,
                "val_precision": val_metrics.precision,
            }
        )

        if val_metrics.pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_metrics.pr_auc
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 5 == 0 or epochs_without_improvement == 0:
            logger.info(
                "Epoch %3d | train_loss=%.4f | val PR-AUC=%.4f | val F1=%.4f | val Recall=%.4f",
                epoch,
                loss.item(),
                val_metrics.pr_auc,
                val_metrics.f1,
                val_metrics.recall,
            )

        if epochs_without_improvement >= patience:
            logger.info("Early stopping à l'epoch %d (meilleure : %d).", epoch, best_epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics = evaluate(model, data, data.train_mask, criterion, device)
    val_metrics = evaluate(model, data, data.val_mask, criterion, device)
    test_metrics = evaluate(model, data, data.test_mask, criterion, device)

    result = TrainingResult(
        model_type=model_type,
        best_epoch=best_epoch,
        train=train_metrics,
        val=val_metrics,
        test=test_metrics,
    )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": model_type,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "edge_hidden": edge_hidden,
            "feature_info": asdict(info) if hasattr(info, "__dataclass_fields__") else info,
            "best_epoch": best_epoch,
            "metrics": {
                "train": asdict(train_metrics),
                "val": asdict(val_metrics),
                "test": asdict(test_metrics),
            },
        },
        checkpoint_path,
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "model_type": result.model_type,
                "best_epoch": result.best_epoch,
                "train": asdict(result.train),
                "val": asdict(result.val),
                "test": asdict(result.test),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    history_path.write_text(
        json.dumps({"model_type": model_type, "epochs": epoch_history}, indent=2),
        encoding="utf-8",
    )

    logger.info("Checkpoint : %s", checkpoint_path)
    logger.info("Métriques   : %s", metrics_path)
    logger.info(
        "TEST — PR-AUC=%.4f | Recall=%.4f | F1=%.4f | Precision=%.4f",
        test_metrics.pr_auc,
        test_metrics.recall,
        test_metrics.f1,
        test_metrics.precision,
    )
    return result
