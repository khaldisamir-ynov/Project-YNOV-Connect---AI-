"""Génère les graphiques de soutenance — phases entraînement et généralisation."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "docs" / "figures" / "soutenance"

# Métriques issues des notebooks metrics_evaluation.ipynb et holdout_metrics_evaluation.ipynb
TRAIN_TEST = {
    "phase": "Entraînement",
    "dataset": "LI-Small — jeu TEST (15 %)",
    "n_samples": 30_000,
    "n_fraud": 534,
    "fraud_rate": 1.78,
    "models": {
        "GraphSAGE": {"pr_auc": 0.119, "recall": 0.839, "precision": 0.038, "f1": 0.073},
        "GAT": {"pr_auc": 0.047, "recall": 0.625, "precision": 0.030, "f1": 0.056},
        "HeteroGNN": {"pr_auc": 0.430, "recall": 0.749, "precision": 0.216, "f1": 0.335},
    },
}

HOLDOUT = {
    "phase": "Généralisation",
    "dataset": "LI-Medium — hold-out inconnu",
    "n_samples": 200_000,
    "n_fraud": 16_041,
    "fraud_rate": 8.02,
    "models": {
        "GraphSAGE": {"pr_auc": 0.229, "recall": 0.781, "precision": 0.133, "f1": 0.228},
        "GAT": {"pr_auc": 0.150, "recall": 0.643, "precision": 0.114, "f1": 0.194},
        "HeteroGNN": {"pr_auc": 0.595, "recall": 0.699, "precision": 0.544, "f1": 0.612},
    },
}

COLORS = {
    "GraphSAGE": "#3498db",
    "GAT": "#95a5a6",
    "HeteroGNN": "#e74c3c",
}

METRICS = ["pr_auc", "recall", "precision", "f1"]
METRIC_LABELS = {
    "pr_auc": "PR-AUC",
    "recall": "Recall",
    "precision": "Précision",
    "f1": "F1-score",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.edgecolor": "#cccccc",
            "axes.labelcolor": "#2c3e50",
            "text.color": "#2c3e50",
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
        }
    )


def _format_metric(val: float) -> str:
    return f"{val * 100:.0f} %"


def _annotate_bars(ax, bars, values) -> None:
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            _format_metric(val),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold" if val == max(values) else "normal",
        )


def plot_phase_figure(phase_data: dict, output_path: Path, highlight_metric: str) -> None:
    """Graphique à 3 métriques (même gabarit pour les deux phases)."""
    _style()
    models = list(phase_data["models"].keys())
    metrics = METRICS
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6.5))

    for i, model in enumerate(models):
        vals = [phase_data["models"][model][m] for m in metrics]
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=model,
            color=COLORS[model],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        _annotate_bars(ax, bars, vals)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score (0 – 1)", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    phase = phase_data["phase"]
    dataset = phase_data["dataset"]
    phase_title = (
        f"Phase d'{phase.lower()}"
        if phase == "Entraînement"
        else f"Phase de {phase.lower()}"
    )
    ax.set_title(
        f"{phase_title} — {dataset}",
        pad=14,
    )

    subtitle = (
        f"{phase_data['n_samples']:,} transactions · "
        f"{phase_data['n_fraud']:,} fraudes ({phase_data['fraud_rate']:.2f} %)"
    )
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", fontsize=10, color="#555")

    # Encadré message clé (aligné sur la trame orale)
    if phase_data["phase"] == "Entraînement":
        sage_p = phase_data["models"]["GraphSAGE"]["precision"]
        hetero = phase_data["models"]["HeteroGNN"]
        msg = (
            f"Modèles standards : précision ≈ {sage_p * 100:.0f} % → "
            f"fausses alertes massives\n"
            f"HeteroGNN : PR-AUC {hetero['pr_auc']:.2f} · F1 {hetero['f1']:.2f} "
            f"(×{hetero['pr_auc'] / sage_p:.0f} meilleur classement)"
        )
    else:
        hetero = phase_data["models"]["HeteroGNN"]
        msg = (
            f"HeteroGNN : PR-AUC {hetero['pr_auc']:.2f} · "
            f"Recall {hetero['recall'] * 100:.0f} % · "
            f"Précision {hetero['precision'] * 100:.0f} % · F1 {hetero['f1']:.2f}\n"
            f"→ plus d'une alerte sur deux est une vraie fraude"
        )

    fig.text(
        0.5,
        0.01,
        msg,
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#2c3e50",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ecf0f1", edgecolor="#bdc3c7"),
    )

    # Surligner la métrique phare de la slide
    highlight_idx = metrics.index(highlight_metric)
    rect = mpatches.FancyBboxPatch(
        (highlight_idx - 0.42, 0),
        0.84,
        1.02,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=2,
        edgecolor="#e74c3c",
        facecolor="none",
        linestyle="--",
        transform=ax.get_xaxis_transform(),
        zorder=4,
    )
    ax.add_patch(rect)

    ax.legend(loc="upper right", framealpha=0.95, title="Modèle")

    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"OK {output_path}")


def plot_precision_comparison(output_path: Path) -> None:
    """Graphique complémentaire : précision entraînement vs généralisation (appuie les 2 slides)."""
    _style()
    models = ["GraphSAGE", "GAT", "HeteroGNN"]
    train_p = [TRAIN_TEST["models"][m]["precision"] * 100 for m in models]
    hold_p = [HOLDOUT["models"][m]["precision"] * 100 for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, train_p, width, label="Entraînement (LI-Small)", color="#5dade2")
    bars2 = ax.bar(x + width / 2, hold_p, width, label="Généralisation (LI-Medium)", color="#e74c3c")

    for bars, vals in ((bars1, train_p), (bars2, hold_p)):
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{val:.0f} %",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax.axhline(4, color="#7f8c8d", linestyle=":", linewidth=1.5, label="Seuil 4 % (standards, LI-Small)")
    ax.axhline(50, color="#27ae60", linestyle="--", linewidth=1.5, alpha=0.7, label="50 % (1 alerte / 2 fraudes)")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Précision (%)", fontsize=11)
    ax.set_ylim(0, 65)
    ax.set_title("Précision des alertes — entraînement vs généralisation", pad=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    fig.text(
        0.5,
        0.02,
        "Standards : < 4 % (LI-Small) · ~ 13 % (LI-Medium)  |  HeteroGNN : 54 % en conditions réelles",
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef9e7", edgecolor="#f4d03f"),
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"OK {output_path}")


def main() -> None:
    plot_phase_figure(
        TRAIN_TEST,
        OUTPUT_DIR / "slide12_phase_entrainement.png",
        highlight_metric="precision",
    )
    plot_phase_figure(
        HOLDOUT,
        OUTPUT_DIR / "slide13_phase_generalisation.png",
        highlight_metric="pr_auc",
    )
    plot_precision_comparison(OUTPUT_DIR / "slide12_13_precision_comparative.png")
    print(f"\nGraphiques disponibles dans : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
