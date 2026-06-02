"""Chargement des métriques GNN et fonctions de visualisation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    precision_recall_curve,
)

from src import config
from src.gnn.homo.models import EdgeClassifierGNN
from src.gnn.graph_loader import load_pyg_data
from src.gnn.hetero.models import HeteroTransactionGNN
from src.rdf.to_hetero_pyg import load_hetero_data

METRIC_NAMES = ["pr_auc", "recall", "precision", "f1", "loss"]
SPLIT_NAMES = ["train", "val", "test"]
SPLIT_LABELS = {"train": "Train", "val": "Validation", "test": "Test"}


def discover_metrics_files(results_dir: Path = config.RESULTS_DIR) -> list[Path]:
    """Liste les fichiers gnn_*_metrics.json disponibles."""
    return sorted(results_dir.glob("gnn_*_metrics.json"))


def load_metrics_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def metrics_to_dataframe(metrics_files: list[Path] | None = None) -> pd.DataFrame:
    """Convertit tous les JSON de métriques en un DataFrame long."""
    if metrics_files is None:
        metrics_files = discover_metrics_files()
    rows: list[dict] = []
    for path in metrics_files:
        data = load_metrics_json(path)
        model = data.get("model_type", path.stem.replace("gnn_", "").replace("_metrics", ""))
        for split in SPLIT_NAMES:
            if split not in data:
                continue
            m = data[split]
            rows.append(
                {
                    "model": model,
                    "split": split,
                    "split_label": SPLIT_LABELS[split],
                    "best_epoch": data.get("best_epoch"),
                    **{k: m[k] for k in METRIC_NAMES if k in m},
                    "n_samples": m.get("n_samples"),
                    "n_fraud": m.get("n_fraud"),
                }
            )
    return pd.DataFrame(rows)


def load_training_history(path: Path) -> pd.DataFrame | None:
    """Charge l'historique epoch par epoch si présent."""
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data.get("epochs", []))


def history_path_for_model(model_type: str) -> Path:
    return config.RESULTS_DIR / f"gnn_{model_type}_history.json"


def plot_metrics_bars(
    df: pd.DataFrame,
    metric: str = "pr_auc",
    split: str = "test",
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Barres comparatives des modèles pour une métrique et un split."""
    sub = df[(df["split"] == split) & (df[metric].notna())].copy()
    if sub.empty:
        raise ValueError(f"Aucune donnée pour metric={metric}, split={split}")
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=sub, x="model", y=metric, hue="model", legend=False, ax=ax, palette="viridis")
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_xlabel("Modèle")
    ax.set_title(title or f"{metric.replace('_', ' ').upper()} — jeu {SPLIT_LABELS[split]}")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2)
    return ax


def plot_all_metrics_grid(df: pd.DataFrame, split: str = "test") -> plt.Figure:
    """Grille 2×2 : PR-AUC, Recall, Precision, F1."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, metric in zip(axes.flat, ["pr_auc", "recall", "precision", "f1"]):
        try:
            plot_metrics_bars(df, metric=metric, split=split, ax=ax)
        except ValueError:
            ax.set_visible(False)
    fig.suptitle(f"Métriques par modèle — {SPLIT_LABELS[split]}", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_splits_comparison(df: pd.DataFrame, model: str = "hetero") -> plt.Figure:
    """Compare train / val / test pour un modèle."""
    sub = df[df["model"] == model]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, metric in zip(axes, ["pr_auc", "recall", "precision", "f1"]):
        s = sub[sub[metric].notna()]
        sns.barplot(data=s, x="split_label", y=metric, ax=ax, palette="muted")
        ax.set_title(metric.replace("_", " ").upper())
        ax.set_xlabel("")
        ax.set_ylim(0, max(1.05, s[metric].max() * 1.1))
    fig.suptitle(f"Évolution des métriques — modèle {model.upper()}", fontsize=13)
    fig.tight_layout()
    return fig


def plot_training_history(history_df: pd.DataFrame, model: str = "hetero") -> plt.Figure:
    """Courbes loss train et PR-AUC validation par epoch."""
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.plot(history_df["epoch"], history_df["train_loss"], color="#2ecc71", label="Train loss", linewidth=2)
    ax2.plot(history_df["epoch"], history_df["val_pr_auc"], color="#3498db", label="Val PR-AUC", linewidth=2)
    if "val_f1" in history_df.columns:
        ax2.plot(
            history_df["epoch"],
            history_df["val_f1"],
            color="#9b59b6",
            linestyle="--",
            label="Val F1",
            alpha=0.8,
        )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (train)", color="#2ecc71")
    ax2.set_ylabel("Val PR-AUC / F1", color="#3498db")
    ax1.set_title(f"Courbes d'entraînement — {model.upper()}")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    fig.tight_layout()
    return fig


@torch.no_grad()
def predict_homo(
    checkpoint_path: Path,
    data_path: Path = config.PYG_DATA,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scores et labels sur tout le graphe homogène (pour courbes PR)."""
    if device is None:
        device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    data, _ = load_pyg_data(data_path)
    data = data.to(device)
    model = EdgeClassifierGNN(
        in_channels=data.x.size(1),
        edge_dim=data.edge_attr.size(1),
        hidden_channels=ckpt.get("hidden_dim", config.GNN_HIDDEN_DIM),
        edge_hidden=ckpt.get("edge_hidden", config.GNN_EDGE_HIDDEN_DIM),
        num_layers=ckpt.get("num_layers", config.GNN_NUM_LAYERS),
        dropout=ckpt.get("dropout", config.GNN_DROPOUT),
        model_type=ckpt.get("model_type", "sage"),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logits = model(data)
    probs = torch.sigmoid(logits).cpu().numpy()
    y = data.y.cpu().numpy().astype(int)
    return y, probs, data.test_mask.cpu().numpy()


@torch.no_grad()
def predict_hetero(
    checkpoint_path: Path,
    data_path: Path = config.PYG_HETERO_DATA,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scores et labels sur le graphe hétérogène."""
    if device is None:
        device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    data, _ = load_hetero_data(data_path)
    data = data.to(device)
    in_channels = ckpt.get("in_channels") or {nt: data[nt].x.size(-1) for nt in data.node_types}
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
    logits = model(data)
    probs = torch.sigmoid(logits).cpu().numpy()
    y = data["transaction"].y.cpu().numpy().astype(int)
    test_mask = data["transaction"].test_mask.cpu().numpy()
    return y, probs, test_mask


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    ax: plt.Axes | None = None,
    label: str = "Modèle",
    baseline_rate: float | None = None,
) -> plt.Axes:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, linewidth=2, label=f"{label} (AP={pr_auc:.3f})")
    if baseline_rate is not None:
        ax.axhline(y=baseline_rate, color="gray", linestyle="--", label=f"Hasard (≈{baseline_rate:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Courbe Precision-Recall (jeu TEST)")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    return ax


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    ax: plt.Axes | None = None,
    title: str = "Matrice de confusion (TEST)",
) -> plt.Axes:
    y_pred = (y_score >= threshold).astype(int)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title(title)
    return ax


def plot_score_distribution(
    y_true: np.ndarray,
    y_score: np.ndarray,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    fraud = y_score[y_true == 1]
    legit = y_score[y_true == 0]
    ax.hist(legit, bins=50, alpha=0.6, label=f"Légitime (n={len(legit):,})", density=True)
    ax.hist(fraud, bins=50, alpha=0.6, label=f"Fraude (n={len(fraud):,})", density=True)
    ax.axvline(0.5, color="red", linestyle="--", label="Seuil 0.5")
    ax.set_xlabel("Score prédit (probabilité fraude)")
    ax.set_ylabel("Densité")
    ax.set_title("Distribution des scores — jeu TEST")
    ax.legend()
    return ax


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
    plt.rcParams["figure.dpi"] = 100


# --------------------------------------------------------------------------- #
# Comparaison d'architectures (homogène vs hétérogène)
# --------------------------------------------------------------------------- #
ARCHITECTURE_LABELS = {
    "hetero": "HeteroGNN\n(multi-relationnel)",
    "sage": "GraphSAGE\n(homogène)",
    "gat": "GAT\n(homogène)",
}

ARCHITECTURE_GROUPS = {
    "GNN homogène (legacy)": ["sage", "gat"],
    "GNN hétérogène (RDF enrichi)": ["hetero"],
}


def architecture_comparison_table(
    df: pd.DataFrame,
    split: str = "test",
    baseline_model: str = "sage",
    challenger_model: str = "hetero",
) -> pd.DataFrame:
    """Tableau comparatif avec écarts (challenger − baseline)."""
    sub = df[df["split"] == split].set_index("model")
    if baseline_model not in sub.index or challenger_model not in sub.index:
        return sub
    rows = []
    for metric in ["pr_auc", "recall", "precision", "f1"]:
        base = sub.loc[baseline_model, metric]
        chal = sub.loc[challenger_model, metric]
        rows.append(
            {
                "métrique": metric,
                baseline_model: base,
                challenger_model: chal,
                "Δ (hetero − sage)": chal - base,
            }
        )
    return pd.DataFrame(rows)


def plot_architecture_comparison(
    df: pd.DataFrame,
    split: str = "test",
    models: list[str] | None = None,
) -> plt.Figure:
    """Barres groupées : comparaison directe des architectures sur le TEST."""
    if models is None:
        models = [m for m in ["hetero", "sage", "gat"] if m in df["model"].values]
    sub = df[(df["split"] == split) & (df["model"].isin(models))].copy()
    sub["architecture"] = sub["model"].map(ARCHITECTURE_LABELS).fillna(sub["model"])

    melted = sub.melt(
        id_vars=["model", "architecture"],
        value_vars=["pr_auc", "recall", "precision", "f1"],
        var_name="métrique",
        value_name="valeur",
    )
    melted["métrique"] = melted["métrique"].str.replace("_", " ").str.upper()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=melted, x="métrique", y="valeur", hue="architecture", ax=ax, palette="Set2")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Comparaison des architectures GNN — {SPLIT_LABELS[split]}")
    ax.legend(title="Architecture", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def plot_architecture_radar(
    df: pd.DataFrame,
    split: str = "test",
    models: list[str] | None = None,
) -> plt.Figure:
    """Radar chart PR-AUC / Recall / Precision / F1 pour comparer visuellement."""
    if models is None:
        models = [m for m in ["hetero", "sage", "gat"] if m in df["model"].values]
    metrics = ["pr_auc", "recall", "precision", "f1"]
    labels = [m.replace("_", " ").upper() for m in metrics]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    colors = {"hetero": "#e74c3c", "sage": "#3498db", "gat": "#2ecc71"}

    for model in models:
        row = df[(df["model"] == model) & (df["split"] == split)]
        if row.empty:
            continue
        values = [float(row[m].iloc[0]) for m in metrics]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=ARCHITECTURE_LABELS.get(model, model), color=colors.get(model))
        ax.fill(angles, values, alpha=0.15, color=colors.get(model))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title(f"Profil des métriques — {SPLIT_LABELS[split]}", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    return fig


def plot_training_history_comparison(
    models: list[str],
    results_dir: Path = config.RESULTS_DIR,
) -> plt.Figure | None:
    """Superpose les courbes val PR-AUC de plusieurs modèles."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"hetero": "#e74c3c", "sage": "#3498db", "gat": "#2ecc71"}
    found = False
    for model in models:
        hist = load_training_history(results_dir / f"gnn_{model}_history.json")
        if hist is None or hist.empty:
            continue
        found = True
        ax.plot(
            hist["epoch"],
            hist["val_pr_auc"],
            label=ARCHITECTURE_LABELS.get(model, model),
            color=colors.get(model),
            linewidth=2,
        )
    if not found:
        plt.close(fig)
        return None
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val PR-AUC")
    ax.set_title("Convergence — comparaison des architectures")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
