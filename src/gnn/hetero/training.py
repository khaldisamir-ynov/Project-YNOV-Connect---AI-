"""Entraînement du GNN hétérogène multi-relationnel."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch_geometric.data import HeteroData

from src import config
from src.gnn.homo.training import SplitMetrics, TrainingResult, _compute_pos_weight
from src.gnn.hetero.models import HeteroTransactionGNN
from src.rdf.to_hetero_pyg import load_hetero_data

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_hetero(
    model: HeteroTransactionGNN,
    data: HeteroData,
    mask: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
) -> SplitMetrics:
    model.eval()
    data = data.to(device)
    mask = mask.to(device)

    logits = model(data)
    labels = data["transaction"].y.float()

    loss = criterion(logits[mask], labels[mask]).item()
    probs = torch.sigmoid(logits[mask]).cpu().numpy()
    y_true = labels[mask].cpu().numpy().astype(int)
    y_pred = (probs >= 0.5).astype(int)

    pr_auc = float(average_precision_score(y_true, probs)) if y_true.sum() > 0 else 0.0
    return SplitMetrics(
        loss=loss,
        pr_auc=pr_auc,
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        n_samples=int(mask.sum()),
        n_fraud=int(y_true.sum()),
    )


def train_hetero_gnn(
    hidden_dim: int = config.GNN_HIDDEN_DIM,
    num_layers: int = config.GNN_NUM_LAYERS,
    dropout: float = config.GNN_DROPOUT,
    lr: float = config.GNN_LR,
    weight_decay: float = config.GNN_WEIGHT_DECAY,
    epochs: int = config.GNN_EPOCHS,
    patience: int = config.GNN_PATIENCE,
    head_hidden: int = config.GNN_EDGE_HIDDEN_DIM,
    data_path: Path = config.PYG_HETERO_DATA,
    checkpoint_path: Path | None = None,
    metrics_path: Path | None = None,
    history_path: Path | None = None,
    device: torch.device | None = None,
) -> TrainingResult:
    if checkpoint_path is None:
        checkpoint_path = config.RESULTS_DIR / "gnn_hetero_best.pt"
    if metrics_path is None:
        metrics_path = config.RESULTS_DIR / "gnn_hetero_metrics.json"
    if history_path is None:
        history_path = config.RESULTS_DIR / "gnn_hetero_history.json"
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, info = load_hetero_data(data_path)
    data = data.to(device)

    in_channels = {ntype: data[ntype].x.size(-1) for ntype in data.node_types}
    model = HeteroTransactionGNN(
        metadata=data.metadata(),
        in_channels_dict=in_channels,
        hidden_channels=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        head_hidden=head_hidden,
    ).to(device)

    train_mask = data["transaction"].train_mask
    pos_weight = torch.tensor(
        [_compute_pos_weight(data["transaction"].y, train_mask)],
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    logger.info(
        "HeteroGNN — device=%s | relations=%s | transacted_with=%s | same_bank_as=%s",
        device,
        len(data.edge_types),
        info.n_inferred_transacted_with,
        info.n_inferred_same_bank_as,
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
        loss = criterion(
            logits[train_mask],
            data["transaction"].y[train_mask].float(),
        )
        loss.backward()
        optimizer.step()

        val_metrics = evaluate_hetero(
            model, data, data["transaction"].val_mask, criterion, device
        )
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
                "Epoch %3d | train_loss=%.4f | val PR-AUC=%.4f | val F1=%.4f",
                epoch,
                loss.item(),
                val_metrics.pr_auc,
                val_metrics.f1,
            )

        if epochs_without_improvement >= patience:
            logger.info("Early stopping epoch %d (best=%d).", epoch, best_epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_m = evaluate_hetero(model, data, data["transaction"].train_mask, criterion, device)
    val_m = evaluate_hetero(model, data, data["transaction"].val_mask, criterion, device)
    test_m = evaluate_hetero(model, data, data["transaction"].test_mask, criterion, device)

    result = TrainingResult(
        model_type="hetero",
        best_epoch=best_epoch,
        train=train_m,
        val=val_m,
        test=test_m,
    )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": "hetero",
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "in_channels": in_channels,
            "metadata": data.metadata(),
            "best_epoch": best_epoch,
            "hetero_info": asdict(info),
            "metrics": {
                "train": asdict(train_m),
                "val": asdict(val_m),
                "test": asdict(test_m),
            },
        },
        checkpoint_path,
    )
    metrics_path.write_text(
        json.dumps(
            {
                "model_type": "hetero",
                "best_epoch": best_epoch,
                "edge_types": [list(et) for et in info.edge_types],
                "train": asdict(train_m),
                "val": asdict(val_m),
                "test": asdict(test_m),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    history_path.write_text(
        json.dumps({"model_type": "hetero", "epochs": epoch_history}, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "TEST — PR-AUC=%.4f | Recall=%.4f | F1=%.4f | Precision=%.4f",
        test_m.pr_auc,
        test_m.recall,
        test_m.f1,
        test_m.precision,
    )
    return result
