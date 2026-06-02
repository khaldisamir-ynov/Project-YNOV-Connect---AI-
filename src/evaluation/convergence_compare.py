"""Comparaison de convergence GraphSAGE vs HeteroGNN (métriques par epoch)."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.gnn.homo.training import train_gnn
from src.gnn.hetero.training import train_hetero_gnn
from src.evaluation.metrics_viz import set_plot_style

logger = logging.getLogger(__name__)

CONVERGENCE_JSON = config.RESULTS_DIR / "convergence_sage_vs_hetero.json"
CONVERGENCE_PNG = config.RESULTS_DIR / "convergence_sage_vs_hetero.png"
CONVERGENCE_GAP_PNG = config.RESULTS_DIR / "convergence_gap_sage_vs_hetero.png"


def run_aligned_training(
    epochs: int = 40,
    homo_first: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entraîne sage puis hetero (ou l'inverse) avec les mêmes hyperparamètres."""
    kwargs = dict(
        hidden_dim=config.GNN_HIDDEN_DIM,
        num_layers=config.GNN_NUM_LAYERS,
        dropout=config.GNN_DROPOUT,
        lr=config.GNN_LR,
        weight_decay=config.GNN_WEIGHT_DECAY,
        epochs=epochs,
        patience=epochs + 1,  # pas d'early stop : courbes complètes
    )

    order = [("sage", train_gnn), ("hetero", train_hetero_gnn)]
    if not homo_first:
        order = list(reversed(order))

    histories: dict[str, pd.DataFrame] = {}
    for name, train_fn in order:
        logger.info("Entraînement aligné : %s (%d epochs)", name, epochs)
        if name == "hetero":
            train_hetero_gnn(**kwargs)
            path = config.RESULTS_DIR / "gnn_hetero_history.json"
        else:
            train_gnn(model_type="sage", **kwargs)
            path = config.RESULTS_DIR / "gnn_sage_history.json"
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        histories[name] = pd.DataFrame(data["epochs"])
        histories[name]["model"] = name

    return histories["sage"], histories["hetero"]


def merge_convergence_histories(
    sage_hist: pd.DataFrame,
    hetero_hist: pd.DataFrame,
) -> pd.DataFrame:
    """Fusionne les historiques et calcule les écarts (hetero − sage)."""
    s = sage_hist.add_prefix("sage_").rename(columns={"sage_epoch": "epoch"})
    h = hetero_hist.add_prefix("hetero_").rename(columns={"hetero_epoch": "epoch"})
    merged = pd.merge(s, h, on="epoch", how="inner")
    merged["gap_val_pr_auc"] = merged["hetero_val_pr_auc"] - merged["sage_val_pr_auc"]
    merged["gap_val_f1"] = merged["hetero_val_f1"] - merged["sage_val_f1"]
    merged["gap_val_recall"] = merged["hetero_val_recall"] - merged["sage_val_recall"]
    merged["gap_val_precision"] = merged["hetero_val_precision"] - merged["sage_val_precision"]
    merged["gap_train_loss"] = merged["hetero_train_loss"] - merged["sage_train_loss"]
    merged["abs_gap_val_pr_auc"] = merged["gap_val_pr_auc"].abs()
    return merged


def analyze_divergence(
    merged: pd.DataFrame,
    pr_auc_threshold: float = 0.01,
    f1_threshold: float = 0.01,
) -> dict:
    """Identifie à partir de quelle epoch l'écart devient significatif."""
    out: dict = {}

    # Premier epoch où |gap PR-AUC| >= seuil
    sig_pr = merged[merged["abs_gap_val_pr_auc"] >= pr_auc_threshold]
    out["first_epoch_gap_pr_auc_0.01"] = int(sig_pr["epoch"].iloc[0]) if len(sig_pr) else None

    sig_f1 = merged[merged["gap_val_f1"].abs() >= f1_threshold]
    out["first_epoch_gap_f1_0.01"] = int(sig_f1["epoch"].iloc[0]) if len(sig_f1) else None

    # Epoch du max écart PR-AUC
    idx_max = merged["abs_gap_val_pr_auc"].idxmax()
    out["epoch_max_abs_gap_pr_auc"] = int(merged.loc[idx_max, "epoch"])
    out["max_abs_gap_pr_auc"] = float(merged.loc[idx_max, "abs_gap_val_pr_auc"])
    out["gap_at_max_epoch"] = float(merged.loc[idx_max, "gap_val_pr_auc"])

    # Dernière epoch
    last = merged.iloc[-1]
    out["final_epoch"] = int(last["epoch"])
    out["final_gap_val_pr_auc"] = float(last["gap_val_pr_auc"])
    out["final_sage_val_pr_auc"] = float(last["sage_val_pr_auc"])
    out["final_hetero_val_pr_auc"] = float(last["hetero_val_pr_auc"])
    out["final_gap_val_recall"] = float(last["gap_val_recall"])

    # Epoch où hetero dépasse sage en PR-AUC val (si jamais)
    hetero_wins = merged[merged["gap_val_pr_auc"] > 0]
    out["first_epoch_hetero_ahead_pr_auc"] = (
        int(hetero_wins["epoch"].iloc[0]) if len(hetero_wins) else None
    )
    hetero_loses = merged[merged["gap_val_pr_auc"] < 0]
    out["first_epoch_sage_ahead_pr_auc"] = (
        int(hetero_loses["epoch"].iloc[0]) if len(hetero_loses) else None
    )

    return out


def plot_convergence_comparison(merged: pd.DataFrame, out_path: Path = CONVERGENCE_PNG) -> Path:
    """Courbes superposées val PR-AUC, F1, train loss."""
    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(merged["epoch"], merged["sage_val_pr_auc"], "o-", label="GraphSAGE", color="#3498db", lw=2)
    axes[0].plot(merged["epoch"], merged["hetero_val_pr_auc"], "s-", label="HeteroGNN", color="#e74c3c", lw=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Val PR-AUC")
    axes[0].set_title("Convergence — PR-AUC (validation)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(merged["epoch"], merged["sage_val_f1"], "o-", label="GraphSAGE", color="#3498db", lw=2)
    axes[1].plot(merged["epoch"], merged["hetero_val_f1"], "s-", label="HeteroGNN", color="#e74c3c", lw=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val F1")
    axes[1].set_title("Convergence — F1 (validation)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(merged["epoch"], merged["sage_train_loss"], "o-", label="GraphSAGE", color="#3498db", lw=2)
    axes[2].plot(merged["epoch"], merged["hetero_train_loss"], "s-", label="HeteroGNN", color="#e74c3c", lw=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Train loss")
    axes[2].set_title("Loss d'entraînement")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("GraphSAGE (homogène) vs HeteroGNN (multi-relationnel)", fontsize=13, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_convergence_gaps(merged: pd.DataFrame, out_path: Path = CONVERGENCE_GAP_PNG) -> Path:
    """Écart hetero − sage par epoch."""
    set_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].axhline(0, color="gray", linestyle="--", lw=1)
    axes[0].axhline(0.01, color="orange", linestyle=":", alpha=0.7, label="Seuil ±0.01")
    axes[0].axhline(-0.01, color="orange", linestyle=":", alpha=0.7)
    axes[0].plot(merged["epoch"], merged["gap_val_pr_auc"], "o-", color="#9b59b6", lw=2)
    axes[0].set_ylabel("Δ Val PR-AUC (Hetero − SAGE)")
    axes[0].set_title("Écart de convergence — validation")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(merged["epoch"], merged["gap_val_recall"], "o-", color="#2ecc71", lw=2, label="Δ Recall")
    axes[1].plot(merged["epoch"], merged["gap_val_f1"], "s-", color="#f39c12", lw=2, label="Δ F1")
    axes[1].axhline(0, color="gray", linestyle="--", lw=1)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Δ métrique")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_convergence_study(
    epochs: int = 40,
    use_existing_history: bool = True,
    pr_auc_threshold: float = 0.01,
) -> dict:
    """Pipeline complet : entraînement aligné (optionnel), analyse, graphiques."""
    sage_path = config.RESULTS_DIR / "gnn_sage_history.json"
    hetero_path = config.RESULTS_DIR / "gnn_hetero_history.json"

    if not (use_existing_history and sage_path.exists() and hetero_path.exists()):
        sage_hist, hetero_hist = run_aligned_training(epochs=epochs)
    else:
        logger.info("Réutilisation des historiques existants.")
        with sage_path.open(encoding="utf-8") as f:
            sage_hist = pd.DataFrame(json.load(f)["epochs"])
        with hetero_path.open(encoding="utf-8") as f:
            hetero_hist = pd.DataFrame(json.load(f)["epochs"])

    merged = merge_convergence_histories(sage_hist, hetero_hist)
    analysis = analyze_divergence(merged, pr_auc_threshold=pr_auc_threshold)

    plot_convergence_comparison(merged)
    plot_convergence_gaps(merged)

    report = {
        "epochs_compared": int(len(merged)),
        "threshold_pr_auc": pr_auc_threshold,
        "analysis": analysis,
        "per_epoch": merged.to_dict(orient="records"),
    }
    CONVERGENCE_JSON.parent.mkdir(parents=True, exist_ok=True)
    CONVERGENCE_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Rapport convergence : %s", CONVERGENCE_JSON)
    return report
