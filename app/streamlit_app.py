"""
Interface Streamlit — détection de fraude AML avec HeteroGNN.

Lancement (depuis la racine du projet) :
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.metrics import precision_recall_curve

from src import config
from src.app.hetero_inference import (
    HeteroInferenceBundle,
    SPLIT_OPTIONS,
    _mask_for_split,
    compute_split_metrics,
    confusion_counts,
    load_saved_training_metrics,
    metrics_to_display_dict,
    optimal_threshold_f1,
    run_hetero_inference,
)
from src.evaluation.metrics_viz import (
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_score_distribution,
    set_plot_style,
)

from app.single_transaction_ui import render_single_transaction_tab

st.set_page_config(
    page_title="Connect AI — HeteroGNN AML",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROFILE_LABELS = {
    "train": "LI-Small (entraînement)",
    "unknown": "LI-Medium (hold-out)",
}


@st.cache_resource(show_spinner="Chargement du modèle et du graphe…")
def cached_inference(
    profile: str,
    checkpoint_str: str,
    pyg_str: str,
) -> HeteroInferenceBundle:
    return run_hetero_inference(
        profile=profile,
        checkpoint_path=Path(checkpoint_str),
        pyg_path=Path(pyg_str),
    )


def _default_paths(profile: str) -> tuple[Path, Path]:
    """Graphe = profil actif ; checkpoint = toujours celui entraîné sur LI-Small."""
    config.set_data_profile(profile)
    pyg = config.PYG_HETERO_DATA
    ckpt = config.TRAIN_RESULTS_DIR / "gnn_hetero_best.pt"
    if profile == "train":
        local = config.RESULTS_DIR / "gnn_hetero_best.pt"
        if local.exists():
            ckpt = local
    return ckpt, pyg


def render_metrics_row(metrics: dict, cols: st.columns) -> None:
    labels = [
        ("PR-AUC", metrics.get("PR-AUC", "—")),
        ("Recall", metrics.get("Recall", "—")),
        ("Precision", metrics.get("Precision", "—")),
        ("F1", metrics.get("F1", "—")),
    ]
    for col, (name, val) in zip(cols, labels):
        col.metric(name, val if isinstance(val, str) else f"{val:.3f}")


def main() -> None:
    st.title("Détection AML — HeteroGNN")
    st.caption(
        "Saisissez une transaction pour obtenir **Fraude / Légitime** et le score du modèle. "
        "Le GNN utilise le graphe complet (comptes, banques, liens RDF)."
    )

    with st.sidebar:
        st.header("Configuration")
        profile = st.selectbox(
            "Jeu de données",
            options=["train", "unknown"],
            format_func=lambda p: PROFILE_LABELS[p],
        )
        ckpt_default, pyg_default = _default_paths(profile)
        ckpt_path = Path(
            st.text_input("Checkpoint", value=str(ckpt_default))
        )
        pyg_path = Path(st.text_input("Graphe PyG", value=str(pyg_default)))

        if not ckpt_path.exists():
            st.error(
                f"Checkpoint introuvable : {ckpt_path}\n\n"
                f"Entraînez d'abord le modèle : "
                f"`python scripts/05_train_gnn.py --model hetero` "
                f"(fichier attendu : `{config.TRAIN_RESULTS_DIR / 'gnn_hetero_best.pt'}`)."
            )
        elif profile == "unknown" and "train_data" not in str(ckpt_path).replace("\\", "/"):
            st.info(
                "Hold-out LI-Medium : le checkpoint doit rester celui entraîné sur **LI-Small** "
                f"(`data/train_data/results/gnn_hetero_best.pt`)."
            )
        if not pyg_path.exists():
            st.warning(
                f"Graphe absent : {pyg_path}. "
                f"Lancez `python scripts/04_build_pyg.py --mode hetero --data-profile {profile}`."
            )

        st.divider()
        threshold = st.slider("Seuil de décision", 0.0, 1.0, 0.5, 0.01)
        split_view = st.selectbox(
            "Jeu pour métriques & tableaux",
            options=list(SPLIT_OPTIONS),
            index=2 if profile == "train" else 3,
            format_func=lambda s: {
                "train": "Train (70 %)",
                "val": "Validation (15 %)",
                "test": "Test (15 %)",
                "all": "Tout le graphe",
            }[s],
        )
        top_n = st.number_input("Top N alertes à afficher", 10, 500, 50, 10)
        min_score = st.slider("Score minimum (filtre tableau)", 0.0, 1.0, 0.0, 0.01)

        run_btn = st.button("Lancer / actualiser l'inférence", type="primary", use_container_width=True)

    if not ckpt_path.exists() or not pyg_path.exists():
        st.info("Corrigez les chemins dans la barre latérale puis lancez l'inférence.")
        return

    if "bundle" not in st.session_state or run_btn:
        try:
            with st.spinner("Inférence en cours…"):
                st.session_state["bundle"] = cached_inference(
                    profile, str(ckpt_path), str(pyg_path)
                )
        except Exception as exc:
            st.error(f"Erreur : {exc}")
            return

    bundle: HeteroInferenceBundle = st.session_state["bundle"]
    mask = _mask_for_split(
        split_view, bundle.train_mask, bundle.val_mask, bundle.test_mask
    )

    live_metrics = compute_split_metrics(
        bundle.y_true, bundle.probs, mask, threshold=threshold
    )
    conf = confusion_counts(bundle.y_true, bundle.probs, mask, threshold=threshold)
    opt_th, opt_f1 = optimal_threshold_f1(bundle.y_true, bundle.probs, bundle.val_mask)

    # Mise à jour tableau prédictions avec seuil courant
    pred_df = bundle.predictions_df.copy()
    pred_df["prediction"] = (pred_df["score_fraude"] >= threshold).astype(int)
    pred_df["correct"] = (pred_df["prediction"] == pred_df["label_reel"]).astype(int)
    pred_df["alerte"] = pred_df["prediction"]

    tab_score, tab_overview, tab_metrics, tab_pr, tab_tx = st.tabs(
        [
            "Analyser une transaction",
            "Vue d'ensemble",
            "Métriques détaillées",
            "Courbe PR",
            "Transactions",
        ]
    )

    with tab_score:
        render_single_transaction_tab(bundle, threshold)

    with tab_overview:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("Modèle")
            st.write(f"**Profil** : {PROFILE_LABELS[profile]}")
            st.write(f"**Checkpoint** : `{bundle.checkpoint_path.name}`")
            if bundle.best_epoch is not None:
                st.write(f"**Meilleur epoch (train)** : {bundle.best_epoch}")
            st.write(f"**Device** : `{torch.device('cuda' if torch.cuda.is_available() else 'cpu')}`")
        with c2:
            st.subheader("Graphe")
            st.write(f"**Transactions** : {bundle.info.n_nodes.get('transaction', '—'):,}")
            st.write(f"**Comptes** : {bundle.info.n_nodes.get('account', '—'):,}")
            st.write(f"**Banques** : {bundle.info.n_nodes.get('bank', '—'):,}")
            st.write(f"**Fraudes (graphe)** : {bundle.info.n_fraud_transactions:,}")
        with c3:
            st.subheader(f"Métriques live — {split_view}")
            render_metrics_row(metrics_to_display_dict(live_metrics), st.columns(4))
            st.caption(
                f"Seuil **{threshold:.2f}** · Prévalence {live_metrics.n_fraud}/{live_metrics.n_samples} "
                f"({100 * live_metrics.n_fraud / max(live_metrics.n_samples, 1):.2f} %)"
            )

        st.divider()
        st.subheader("Matrice de confusion (seuil actuel)")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Vrais négatifs (TN)", conf["tn"])
        m2.metric("Faux positifs (FP)", conf["fp"])
        m3.metric("Faux négatifs (FN)", conf["fn"])
        m4.metric("Vrais positifs (TP)", conf["tp"])
        m5.metric("Seuil F1 optimal (val)", f"{opt_th:.3f}", help=f"F1 val = {opt_f1:.3f}")

        saved = load_saved_training_metrics("train" if profile == "train" else profile)
        if saved and profile == "train":
            st.subheader("Métriques enregistrées à l'entraînement (seuil 0,5)")
            rows = []
            for sp in ("train", "val", "test"):
                if sp in saved:
                    m = saved[sp]
                    rows.append(
                        {
                            "Split": sp,
                            "PR-AUC": round(m["pr_auc"], 4),
                            "Recall": round(m["recall"], 4),
                            "Precision": round(m["precision"], 4),
                            "F1": round(m["f1"], 4),
                        }
                    )
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab_metrics:
        st.subheader("Comparaison des splits (seuil courant)")
        rows = []
        for sp in SPLIT_OPTIONS:
            sp_mask = _mask_for_split(
                sp, bundle.train_mask, bundle.val_mask, bundle.test_mask
            )
            if not sp_mask.any():
                continue
            m = compute_split_metrics(bundle.y_true, bundle.probs, sp_mask, threshold)
            rows.append({"Split": sp, **metrics_to_display_dict(m)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.subheader("Impact du seuil sur le split sélectionné")
        thresholds = np.linspace(0.05, 0.95, 19)
        curve_rows = []
        for th in thresholds:
            m = compute_split_metrics(bundle.y_true, bundle.probs, mask, th)
            curve_rows.append(
                {
                    "seuil": th,
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1": m.f1,
                }
            )
        curve_df = pd.DataFrame(curve_rows)
        st.line_chart(curve_df.set_index("seuil")[["precision", "recall", "f1"]], height=280)

        st.info(
            "Les métriques **PR-AUC** ne dépendent pas du seuil (qualité du classement). "
            "Recall, precision et F1 **changent** quand vous bougez le curseur dans la barre latérale."
        )

    with tab_pr:
        set_plot_style()
        sub = mask.astype(bool)
        yt = bundle.y_true[sub]
        ys = bundle.probs[sub]
        if yt.sum() == 0:
            st.warning("Aucune fraude dans ce split — courbe PR indisponible.")
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            baseline = float(yt.mean())
            plot_precision_recall_curve(
                yt, ys, ax=ax, label="HeteroGNN", baseline_rate=baseline
            )
            # Point au seuil actuel
            y_pred = (ys >= threshold).astype(int)
            from sklearn.metrics import precision_score, recall_score

            p_at = precision_score(yt, y_pred, zero_division=0)
            r_at = recall_score(yt, y_pred, zero_division=0)
            ax.scatter([r_at], [p_at], s=120, c="crimson", zorder=5, label=f"Seuil {threshold:.2f}")
            ax.legend(loc="upper right")
            st.pyplot(fig)
            plt.close(fig)

            fig2, ax2 = plt.subplots(figsize=(8, 5))
            plot_score_distribution(yt, ys, ax=ax2)
            st.pyplot(fig2)
            plt.close(fig2)

            fig3, ax3 = plt.subplots(figsize=(5, 4))
            plot_confusion_matrix(
                yt, ys, threshold=threshold, ax=ax3, title=f"Confusion — {split_view}"
            )
            st.pyplot(fig3)
            plt.close(fig3)

    with tab_tx:
        st.subheader("Prédictions par transaction")
        view = pred_df[mask].copy()
        view = view[view["score_fraude"] >= min_score]
        view = view.sort_values("score_fraude", ascending=False)

        fraud_only = st.checkbox("Uniquement les fraudes réelles", value=False)
        alerts_only = st.checkbox("Uniquement les alertes (score ≥ seuil)", value=False)
        if fraud_only:
            view = view[view["label_reel"] == 1]
        if alerts_only:
            view = view[view["prediction"] == 1]

        display_cols = [
            c
            for c in [
                "tx_index",
                "score_fraude",
                "prediction",
                "label_reel",
                "correct",
                "split",
                "amount_paid",
                "payment_currency",
                "from_bank",
                "to_bank",
                "is_laundering",
            ]
            if c in view.columns
        ]
        st.dataframe(
            view[display_cols].head(int(top_n)),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Exporter CSV (split filtré)",
            view[display_cols].to_csv(index=False).encode("utf-8"),
            file_name=f"hetero_predictions_{profile}_{split_view}.csv",
            mime="text/csv",
        )

        st.caption(
            "Chaque ligne = une transaction du sous-graphe. "
            "`score_fraude` = probabilité après passage GNN sur le graphe complet. "
            "Métriques globales (precision/recall) = agrégat sur l'ensemble filtré, pas par ligne."
        )


if __name__ == "__main__":
    main()
