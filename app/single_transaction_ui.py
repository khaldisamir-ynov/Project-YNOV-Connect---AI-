"""Onglet Streamlit — saisie et scoring d'une transaction."""
from __future__ import annotations

import streamlit as st

from src.app.hetero_inference import HeteroInferenceBundle
from src.app.transaction_score import (
    load_subgraph_dataframe,
    sample_transaction_row,
    score_from_bundle,
)


def render_single_transaction_tab(
    bundle: HeteroInferenceBundle,
    threshold: float,
) -> None:
    st.subheader("Analyser une transaction")
    st.markdown(
        "Saisissez une transaction du sous-graphe (recherche par champs ou par **index**), "
        "ou une nouvelle transaction dont les **comptes existent déjà** dans le graphe. "
        "Le modèle utilise le **contexte réseau** complet (GNN + RDF)."
    )

    df = load_subgraph_dataframe()
    if df is None:
        st.error("CSV sous-graphe introuvable — impossible de faire correspondre les comptes.")
        return

    s1, s2 = st.columns(2)
    with s1:
        if st.button("Charger un exemple — fraude"):
            st.session_state["tx_form"] = sample_transaction_row(df, fraud=True)
            st.rerun()
    with s2:
        if st.button("Charger un exemple — légitime"):
            st.session_state["tx_form"] = sample_transaction_row(df, fraud=False)
            st.rerun()

    defaults = st.session_state.get("tx_form", sample_transaction_row(df, fraud=False))

    with st.form("transaction_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            from_bank = st.text_input("Banque émettrice", value=defaults["from_bank"])
            from_account = st.text_input("Compte émetteur", value=defaults["from_account"])
            to_bank = st.text_input("Banque destinataire", value=defaults["to_bank"])
        with c2:
            to_account = st.text_input("Compte destinataire", value=defaults["to_account"])
            amount_paid = st.number_input(
                "Montant payé",
                min_value=0.0,
                value=float(defaults["amount_paid"]),
                format="%.2f",
            )
            payment_currency = st.text_input(
                "Devise paiement", value=defaults["payment_currency"]
            )
        with c3:
            payment_format = st.text_input(
                "Format paiement", value=defaults["payment_format"]
            )
            timestamp = st.text_input("Horodatage", value=str(defaults.get("timestamp", "")))
            tx_index = st.number_input(
                "Index transaction (0 … N-1)",
                min_value=0,
                max_value=max(0, len(df) - 1),
                value=int(defaults.get("tx_index", 0)),
            )

        use_index_only = st.checkbox(
            "Rechercher uniquement par index",
            value=False,
            help="Décochez pour faire correspondre banques, comptes, montant et devise.",
        )
        submitted = st.form_submit_button("Analyser la transaction", type="primary")

    if not submitted:
        return

    fields = {
        "from_bank": from_bank.strip(),
        "from_account": from_account.strip(),
        "to_bank": to_bank.strip(),
        "to_account": to_account.strip(),
        "amount_paid": amount_paid,
        "payment_currency": payment_currency.strip(),
        "payment_format": payment_format.strip(),
        "timestamp": timestamp.strip(),
    }
    idx_arg = int(tx_index) if use_index_only else None

    with st.spinner("Scoring en cours…"):
        try:
            result = score_from_bundle(
                bundle,
                fields,
                threshold=threshold,
                tx_index=idx_arg,
                allow_inject=True,
            )
        except Exception as exc:
            st.error(f"Erreur : {exc}")
            return

    if "error" in result.details:
        st.error(result.details["error"])
        return

    verdict_color = "#e74c3c" if result.prediction == 1 else "#27ae60"
    st.markdown(
        f"""
        <div style="padding:1.2rem;border-radius:8px;border:2px solid {verdict_color};
        background:rgba(0,0,0,0.03);">
        <h2 style="margin:0;color:{verdict_color};">{result.label}</h2>
        <p style="margin:0.5rem 0 0 0;">Score fraude : <b>{result.score:.4f}</b>
        (seuil {result.threshold:.2f}) · Mode : <b>{result.mode}</b>
        {f"· Index {result.tx_index}" if result.tx_index is not None else ""}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Score (probabilité)", f"{result.score:.4f}")
    m2.metric("Prédiction", "Fraude" if result.prediction == 1 else "Légitime")
    m3.metric("Seuil", f"{result.threshold:.2f}")
    m4.metric(
        "Label réel (si connu)",
        "Fraude" if result.true_label == 1 else ("Légitime" if result.true_label == 0 else "—"),
    )

    if result.true_label_known and result.true_label is not None:
        if result.prediction == result.true_label:
            st.success("Prédiction correcte par rapport au label du jeu de données.")
        else:
            st.warning("Prédiction différente du label réel connu.")

    st.caption(result.details.get("message", ""))

    with st.expander("Détails techniques"):
        st.json(
            {
                "mode": result.mode,
                "tx_index": result.tx_index,
                "compte_émetteur_dans_graphe": result.from_account_in_graph,
                "compte_destinataire_dans_graphe": result.to_account_in_graph,
                **result.details,
            }
        )
