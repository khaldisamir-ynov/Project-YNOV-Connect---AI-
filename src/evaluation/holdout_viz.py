"""Visualisations hold-out — mêmes sorties que la phase E (train) pour la soutenance."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config
from src.evaluation.metrics_viz import (
    ARCHITECTURE_LABELS,
    METRIC_NAMES,
    architecture_comparison_table,
    discover_metrics_files,
    load_metrics_json,
    load_training_history,
    metrics_to_dataframe,
    plot_all_metrics_grid,
    plot_architecture_comparison,
    plot_architecture_radar,
    plot_confusion_matrix,
    plot_metrics_bars,
    plot_precision_recall_curve,
    plot_score_distribution,
    plot_training_history,
    plot_training_history_comparison,
    predict_hetero,
    predict_homo,
    set_plot_style,
)

logger = logging.getLogger(__name__)

HOLDOUT_MODELS = ("hetero", "sage", "gat")
HOLDOUT_REPORT_NAME = "eval_holdout_metrics.json"


def holdout_report_path(output_dir: Path | None = None) -> Path:
    if output_dir is None:
        output_dir = config.RESULTS_DIR
    return output_dir / HOLDOUT_REPORT_NAME


def export_holdout_metrics_json(
    report: dict[str, Any],
    output_dir: Path | None = None,
) -> list[Path]:
    """Écrit `gnn_{model}_holdout_metrics.json` (format compatible notebook)."""
    if output_dir is None:
        output_dir = config.RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for model, payload in report.get("models", {}).items():
        if payload.get("status") != "ok":
            continue
        m = payload["holdout"]
        doc = {
            "model_type": model,
            "eval_type": "holdout_final",
            "best_epoch": payload.get("best_epoch_train"),
            "holdout": m,
        }
        path = output_dir / f"gnn_{model}_holdout_metrics.json"
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        paths.append(path)
    return paths


def holdout_metrics_dataframe(
    report: dict[str, Any] | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """DataFrame long (split=holdout) depuis le rapport ou les JSON par modèle."""
    if output_dir is None:
        output_dir = config.RESULTS_DIR
    if report is None:
        report_path = holdout_report_path(output_dir)
        if not report_path.exists():
            files = sorted(output_dir.glob("gnn_*_holdout_metrics.json"))
            if not files:
                raise FileNotFoundError(f"Aucune métrique hold-out dans {output_dir}")
            return metrics_to_dataframe(files)
        with report_path.open(encoding="utf-8") as f:
            report = json.load(f)

    rows: list[dict] = []
    for model, payload in report.get("models", {}).items():
        if payload.get("status") != "ok":
            continue
        m = payload["holdout"]
        rows.append(
            {
                "model": model,
                "split": "holdout",
                "split_label": "Hold-out (LI-Medium)",
                "best_epoch": payload.get("best_epoch_train"),
                **{k: m[k] for k in METRIC_NAMES if k in m},
                "n_samples": m.get("n_samples"),
                "n_fraud": m.get("n_fraud"),
            }
        )
    return pd.DataFrame(rows)


def train_test_metrics_dataframe(train_results_dir: Path | None = None) -> pd.DataFrame:
    """Métriques test interne LI-Small (pour comparaison train vs hold-out)."""
    if train_results_dir is None:
        train_results_dir = config.TRAIN_RESULTS_DIR
    files = discover_metrics_files(train_results_dir)
    if not files:
        return pd.DataFrame()
    return metrics_to_dataframe(files)


def plot_train_vs_holdout_comparison(
    df_train: pd.DataFrame,
    df_holdout: pd.DataFrame,
    metric: str = "pr_auc",
) -> plt.Figure:
    """Barres groupées : test LI-Small vs hold-out LI-Medium par modèle."""
    fig, ax = plt.subplots(figsize=(10, 5))
    rows = []
    for model in HOLDOUT_MODELS:
        tr = df_train[(df_train["model"] == model) & (df_train["split"] == "test")]
        ho = df_holdout[df_holdout["model"] == model]
        if tr.empty and ho.empty:
            continue
        if not tr.empty:
            rows.append({"model": model, "jeu": "Test LI-Small", "valeur": float(tr[metric].iloc[0])})
        if not ho.empty:
            rows.append({"model": model, "jeu": "Hold-out LI-Medium", "valeur": float(ho[metric].iloc[0])})
    if not rows:
        ax.set_visible(False)
        return fig
    plot_df = pd.DataFrame(rows)
    plot_df["architecture"] = plot_df["model"].map(ARCHITECTURE_LABELS).fillna(plot_df["model"])
    sns.barplot(data=plot_df, x="architecture", y="valeur", hue="jeu", ax=ax, palette="Set1")
    ax.set_ylim(0, max(1.05, plot_df["valeur"].max() * 1.1))
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_xlabel("Modèle")
    ax.set_title(f"Généralisation — {metric.replace('_', ' ').upper()} : train vs hold-out")
    ax.legend(title="Jeu d'évaluation")
    fig.tight_layout()
    return fig


def _save_fig(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure : %s", path)
    return path


def _checkpoint_paths(checkpoints_dir: Path) -> dict[str, Path]:
    return {m: checkpoints_dir / f"gnn_{m}_best.pt" for m in HOLDOUT_MODELS}


def export_holdout_visualizations(
    *,
    report: dict[str, Any] | None = None,
    output_dir: Path | None = None,
    checkpoints_dir: Path | None = None,
    train_results_dir: Path | None = None,
) -> dict[str, Path]:
    """Produit les graphiques et CSV hold-out (équivalent phase E du notebook train)."""
    config.set_data_profile("unknown")
    set_plot_style()

    if output_dir is None:
        output_dir = config.RESULTS_DIR
    if checkpoints_dir is None:
        checkpoints_dir = config.TRAIN_RESULTS_DIR
    if train_results_dir is None:
        train_results_dir = config.TRAIN_RESULTS_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    if report is None:
        report_path = holdout_report_path(output_dir)
        if not report_path.exists():
            raise FileNotFoundError(
                f"Rapport hold-out introuvable : {report_path}\n"
                "→ Lancez d'abord `python scripts/07_eval_holdout.py`."
            )
        with report_path.open(encoding="utf-8") as f:
            report = json.load(f)

    export_holdout_metrics_json(report, output_dir)
    df = holdout_metrics_dataframe(report)
    if df.empty:
        raise ValueError("Aucun modèle évalué avec succès dans le rapport hold-out.")

    artifacts: dict[str, Path] = {}

    csv_path = output_dir / "holdout_metrics_summary.csv"
    df.to_csv(csv_path, index=False)
    artifacts["summary_csv"] = csv_path

    artifacts["metrics_grid"] = _save_fig(
        plot_all_metrics_grid(df, split="holdout"),
        output_dir / "holdout_metrics_grid.png",
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_metrics_bars(
        df,
        metric="pr_auc",
        split="holdout",
        ax=ax,
        title="PR-AUC — comparaison des modèles (HOLD-OUT LI-Medium)",
    )
    baseline = df["n_fraud"].iloc[0] / df["n_samples"].iloc[0]
    ax.axhline(y=baseline, color="gray", linestyle="--", label=f"Hasard (≈{baseline:.3f})")
    ax.legend()
    artifacts["pr_auc_bars"] = _save_fig(fig, output_dir / "holdout_pr_auc_bars.png")

    artifacts["architecture_comparison"] = _save_fig(
        plot_architecture_comparison(df, split="holdout"),
        output_dir / "holdout_architecture_comparison.png",
    )
    artifacts["architecture_radar"] = _save_fig(
        plot_architecture_radar(df, split="holdout"),
        output_dir / "holdout_architecture_radar.png",
    )

    cmp_table = architecture_comparison_table(df, split="holdout")
    cmp_path = output_dir / "holdout_architecture_comparison_table.csv"
    cmp_table.to_csv(cmp_path, index=False)
    artifacts["architecture_table_csv"] = cmp_path

    checkpoints = _checkpoint_paths(checkpoints_dir)

    fig_pr, axes_pr = plt.subplots(1, 3, figsize=(16, 5))
    fig_cm, axes_cm = plt.subplots(1, 3, figsize=(14, 4))
    fig_sc, axes_sc = plt.subplots(1, 3, figsize=(16, 5))

    for ax_pr, ax_cm, ax_sc, model in zip(axes_pr, axes_cm, axes_sc, HOLDOUT_MODELS):
        ckpt = checkpoints.get(model)
        if ckpt is None or not ckpt.exists():
            for ax in (ax_pr, ax_cm, ax_sc):
                ax.set_visible(False)
            continue
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        baseline_rate = float(sub["n_fraud"].iloc[0] / sub["n_samples"].iloc[0])
        label = ARCHITECTURE_LABELS.get(model, model)

        if model == "hetero":
            y, probs, mask = predict_hetero(ckpt, config.PYG_HETERO_DATA)
        else:
            y, probs, mask = predict_homo(ckpt, config.PYG_DATA)

        eval_idx = mask.astype(bool) if mask.any() else np.ones(len(y), dtype=bool)
        y_eval = y[eval_idx]
        p_eval = probs[eval_idx]

        plot_precision_recall_curve(
            y_eval,
            p_eval,
            ax=ax_pr,
            label=label,
            baseline_rate=baseline_rate,
        )
        ax_pr.set_title(f"PR — {label}")

        plot_confusion_matrix(
            y_eval,
            p_eval,
            ax=ax_cm,
            title=f"Confusion — {label}",
        )
        plot_score_distribution(y_eval, p_eval, ax=ax_sc)
        ax_sc.set_title(f"Scores — {label}")

    for fig_multi, name in (
        (fig_pr, "holdout_pr_curves_all_models.png"),
        (fig_cm, "holdout_confusion_matrices_all_models.png"),
        (fig_sc, "holdout_score_distributions_all_models.png"),
    ):
        fig_multi.suptitle("Hold-out LI-Medium — 200 000 transactions", fontsize=12, y=1.02)
        fig_multi.tight_layout()
        artifacts[name.replace(".png", "")] = _save_fig(fig_multi, output_dir / name)

    primary = "hetero" if "hetero" in df["model"].values else str(df["model"].iloc[0])
    ckpt_p = checkpoints.get(primary)
    if ckpt_p and ckpt_p.exists():
        if primary == "hetero":
            y, probs, mask = predict_hetero(ckpt_p, config.PYG_HETERO_DATA)
        else:
            y, probs, mask = predict_homo(ckpt_p, config.PYG_DATA)
        eval_idx = mask.astype(bool) if mask.any() else np.ones(len(y), dtype=bool)
        y_eval = y[eval_idx]
        p_eval = probs[eval_idx]
        sub = df[df["model"] == primary]
        baseline_rate = float(sub["n_fraud"].iloc[0] / sub["n_samples"].iloc[0])

        fig = plt.figure(figsize=(8, 6))
        plot_precision_recall_curve(y_eval, p_eval, label=primary.upper(), baseline_rate=baseline_rate)
        plt.title(f"Courbe Precision-Recall — {primary.upper()} (HOLD-OUT)")
        artifacts[f"pr_curve_{primary}"] = _save_fig(fig, output_dir / f"holdout_pr_curve_{primary}.png")

        fig = plt.figure(figsize=(5, 4))
        plot_confusion_matrix(y_eval, p_eval, title=f"Matrice de confusion — {primary.upper()} (HOLD-OUT)")
        artifacts[f"confusion_{primary}"] = _save_fig(
            fig, output_dir / f"holdout_confusion_matrix_{primary}.png"
        )

        fig = plt.figure(figsize=(8, 5))
        plot_score_distribution(y_eval, p_eval)
        plt.title(f"Distribution des scores — {primary.upper()} (HOLD-OUT)")
        artifacts[f"scores_{primary}"] = _save_fig(
            fig, output_dir / f"holdout_score_distribution_{primary}.png"
        )

    df_train = train_test_metrics_dataframe(train_results_dir)
    if not df_train.empty:
        for metric in ("pr_auc", "f1", "recall", "precision"):
            artifacts[f"train_vs_holdout_{metric}"] = _save_fig(
                plot_train_vs_holdout_comparison(df_train, df, metric=metric),
                output_dir / f"holdout_train_vs_holdout_{metric}.png",
            )

    fig_hist = plot_training_history_comparison(
        ["hetero", "sage", "gat"],
        results_dir=train_results_dir,
    )
    if fig_hist is not None:
        fig_hist.axes[0].set_title(
            "Convergence à l'entraînement (LI-Small) — référence pour la soutenance"
        )
        artifacts["train_convergence_reference"] = _save_fig(
            fig_hist,
            output_dir / "holdout_train_convergence_reference.png",
        )

    for model in HOLDOUT_MODELS:
        hist_path = train_results_dir / f"gnn_{model}_history.json"
        hist_df = load_training_history(hist_path)
        if hist_df is None or hist_df.empty:
            continue
        artifacts[f"train_history_{model}"] = _save_fig(
            plot_training_history(hist_df, model=model),
            output_dir / f"holdout_train_history_{model}.png",
        )

    manifest = {
        "eval_type": "holdout_visualizations",
        "output_dir": str(output_dir.resolve()),
        "artifacts": {k: str(v.resolve()) for k, v in artifacts.items()},
        "models": report.get("models", {}),
    }
    manifest_path = output_dir / "holdout_viz_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    artifacts["manifest"] = manifest_path

    logger.info("%d fichiers générés dans %s", len(artifacts), output_dir)
    return artifacts
