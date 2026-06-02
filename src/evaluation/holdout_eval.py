"""Évaluation hold-out : checkpoints entraînés sur train → graphe unknown."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from src import config
from src.gnn.graph_loader import FeatureInfo, load_pyg_data
from src.gnn.homo.models import EdgeClassifierGNN
from src.gnn.homo.training import SplitMetrics, evaluate
from src.gnn.hetero.models import HeteroTransactionGNN
from src.gnn.hetero.training import evaluate_hetero
from src.rdf.to_hetero_pyg import load_hetero_data

logger = logging.getLogger(__name__)

HOLDOUT_MODELS = ("hetero", "sage", "gat")


def _checkpoint_path(model: str, checkpoints_dir: Path) -> Path:
    return checkpoints_dir / f"gnn_{model}_best.pt"


def _pyg_path(model: str) -> Path:
    if model == "hetero":
        return config.PYG_HETERO_DATA
    return config.PYG_DATA


def _load_homo_feature_info(checkpoint_path: Path) -> FeatureInfo | None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw = ckpt.get("feature_info")
    if raw is None:
        return None
    if isinstance(raw, FeatureInfo):
        return raw
    return FeatureInfo(**raw)


def _hetero_in_channels_match(ckpt: dict, data) -> None:
    expected = ckpt.get("in_channels")
    if not expected:
        return
    for ntype, dim in expected.items():
        actual = data[ntype].x.size(-1)
        if actual != dim:
            raise ValueError(
                f"Dimension incompatible pour '{ntype}' : checkpoint={dim}, graphe={actual}. "
                "Vérifiez que la T-Box et le pipeline RDF sont identiques."
            )


@torch.no_grad()
def evaluate_homo_holdout(
    model_type: str,
    checkpoint_path: Path,
    data_path: Path | None = None,
    device: torch.device | None = None,
) -> SplitMetrics:
    """Évalue un GNN homogène sur tout le graphe (masque test = 100 %)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if data_path is None:
        data_path = _pyg_path(model_type)

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
        model_type=ckpt.get("model_type", model_type),
    ).to(device)
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"Incompatibilité de dimensions pour {model_type} : reconstruire aml_graph.pt "
            "avec le vocabulaire du checkpoint train (pipeline --holdout-eval)."
        ) from exc

    criterion = nn.BCEWithLogitsLoss()
    eval_mask = data.test_mask if data.test_mask.any() else torch.ones(data.num_edges, dtype=torch.bool)
    return evaluate(model, data, eval_mask, criterion, device)


@torch.no_grad()
def evaluate_hetero_holdout(
    checkpoint_path: Path,
    data_path: Path | None = None,
    device: torch.device | None = None,
) -> SplitMetrics:
    """Évalue HeteroGNN sur toutes les transactions du graphe hold-out."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if data_path is None:
        data_path = config.PYG_HETERO_DATA

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    data, _ = load_hetero_data(data_path)
    _hetero_in_channels_match(ckpt, data)
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

    criterion = nn.BCEWithLogitsLoss()
    tx_mask = data["transaction"].test_mask
    if not tx_mask.any():
        tx_mask = torch.ones(data["transaction"].num_nodes, dtype=torch.bool, device=device)
    return evaluate_hetero(model, data, tx_mask, criterion, device)


def run_holdout_evaluation(
    *,
    checkpoints_dir: Path | None = None,
    output_dir: Path | None = None,
    models: tuple[str, ...] = HOLDOUT_MODELS,
    device: torch.device | None = None,
) -> dict:
    """Évalue hetero, sage et gat ; écrit `eval_holdout_metrics.json`."""
    if checkpoints_dir is None:
        checkpoints_dir = config.TRAIN_RESULTS_DIR
    if output_dir is None:
        output_dir = config.RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "eval_type": "holdout_final",
        "train_checkpoints_dir": str(checkpoints_dir.resolve()),
        "unknown_data_root": str(config.ACTIVE_DATA_ROOT.resolve()),
        "n_transactions_target": config.EGO_TARGET_SIZE,
        "models": {},
    }

    for model in models:
        ckpt_path = _checkpoint_path(model, checkpoints_dir)
        if not ckpt_path.exists():
            logger.warning("Checkpoint absent, modèle ignoré : %s", ckpt_path)
            report["models"][model] = {"status": "skipped", "reason": "checkpoint_missing"}
            continue

        pyg_path = _pyg_path(model)
        if not pyg_path.exists():
            raise FileNotFoundError(
                f"Graphe PyG introuvable pour {model} : {pyg_path}\n"
                "→ Lancez d'abord le pipeline unknown (scripts/08_run_unknown_holdout_pipeline.py)."
            )

        logger.info("Évaluation hold-out : %s", model.upper())
        if model == "hetero":
            metrics = evaluate_hetero_holdout(ckpt_path, pyg_path, device=device)
        else:
            metrics = evaluate_homo_holdout(model, ckpt_path, pyg_path, device=device)

        report["models"][model] = {
            "status": "ok",
            "checkpoint": str(ckpt_path.resolve()),
            "pyg_data": str(pyg_path.resolve()),
            "best_epoch_train": int(
                torch.load(ckpt_path, map_location="cpu", weights_only=False).get("best_epoch", -1)
            ),
            "holdout": asdict(metrics),
        }
        logger.info(
            "%s — PR-AUC=%.4f | Recall=%.4f | F1=%.4f | Precision=%.4f | n=%s (fraudes=%s)",
            model.upper(),
            metrics.pr_auc,
            metrics.recall,
            metrics.f1,
            metrics.precision,
            f"{metrics.n_samples:,}",
            metrics.n_fraud,
        )

    out_json = output_dir / "eval_holdout_metrics.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Rapport hold-out : %s", out_json)
    return report


def hetero_feature_names_from_train(
    checkpoints_dir: Path | None = None,
) -> dict[str, list[str]] | None:
    """Noms de colonnes features du HeteroGNN entraîné (pour aligner le hold-out)."""
    if checkpoints_dir is None:
        checkpoints_dir = config.TRAIN_RESULTS_DIR
    ckpt = torch.load(
        _checkpoint_path("hetero", checkpoints_dir),
        map_location="cpu",
        weights_only=False,
    )
    hi = ckpt.get("hetero_info") or {}
    names = hi.get("node_feature_names")
    return names if names else None


def homo_feature_info_from_train(model: str, checkpoints_dir: Path | None = None) -> FeatureInfo | None:
    """Vocabulaire homogène enregistré dans le checkpoint d'entraînement."""
    if checkpoints_dir is None:
        checkpoints_dir = config.TRAIN_RESULTS_DIR
    return _load_homo_feature_info(_checkpoint_path(model, checkpoints_dir))
