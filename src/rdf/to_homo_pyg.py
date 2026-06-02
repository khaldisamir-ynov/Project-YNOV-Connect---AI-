"""Conversion d'une A-Box RDF (Turtle) vers un DataFrame puis PyG.

Lit les triples conformes à la T-Box `aml.ttl` et reconstruit le tableau
de transactions attendu par `graph_loader.build_pyg_data`.

Topologie extraite :
    Account --sends_via--> Transaction --deposits_to--> Account

Chaque transaction devient une ligne avec les colonnes canoniques du projet.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from urllib.parse import quote, unquote

import pandas as pd
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF

from src import config
from src.gnn.graph_loader import FeatureInfo, build_pyg_data, load_pyg_data, save_pyg_data
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

AML = Namespace(config.AML_ONTOLOGY_NS)


def load_rdf_graph(ttl_path: Path = config.ABOX_SUBGRAPH_TTL) -> Graph:
    """Charge un fichier Turtle (A-Box) dans un graphe rdflib."""
    g = Graph()
    logger.info("Chargement RDF : %s", ttl_path)
    g.parse(ttl_path, format="turtle")
    logger.info("Triplets chargés : %s", f"{len(g):,}")
    return g


def _account_bank_map(g: Graph) -> dict[URIRef, str]:
    """Account URI → bank_id via aml:hosted_by."""
    mapping: dict[URIRef, str] = {}
    for acct, bank in g.subject_objects(AML.hosted_by):
        local = unquote(str(bank).split("#")[-1])
        if local.startswith("bank-"):
            mapping[acct] = local[len("bank-") :]
    return mapping


def _account_id_from_uri(acct_uri: URIRef, bank_id: str) -> str:
    """Extrait le numéro de compte depuis inst:acct-<bank>-<account>."""
    local = unquote(str(acct_uri).split("#")[-1])
    prefix = f"acct-{quote(str(bank_id), safe='')}-"
    if not local.startswith(prefix):
        prefix = f"acct-{bank_id}-"
    if not local.startswith(prefix):
        raise ValueError(f"URI compte incohérente : {local} (banque={bank_id})")
    return local[len(prefix) :]


def _tx_sort_key(tx_uri: URIRef) -> int:
    """Tri stable des transactions (inst:tx-<index>)."""
    local = unquote(str(tx_uri).split("#")[-1])
    m = re.match(r"tx-(\d+)$", local)
    return int(m.group(1)) if m else 0


def rdf_to_dataframe(g: Graph) -> pd.DataFrame:
    """Extrait les transactions de l'A-Box en DataFrame canonique."""
    acct_banks = _account_bank_map(g)
    rows: list[dict] = []

    transactions = list(g.subjects(RDF.type, AML.Transaction))
    transactions.sort(key=_tx_sort_key)
    logger.info("Transactions RDF trouvées : %s", f"{len(transactions):,}")

    for tx in transactions:
        src_accounts = list(g.subjects(AML.sends_via, tx))
        dst_accounts = list(g.objects(tx, AML.deposits_to))
        if len(src_accounts) != 1 or len(dst_accounts) != 1:
            logger.warning("Transaction ignorée (topologie invalide) : %s", tx)
            continue

        src, dst = src_accounts[0], dst_accounts[0]
        if src not in acct_banks or dst not in acct_banks:
            logger.warning("Compte sans hosted_by ignoré pour %s", tx)
            continue

        from_bank = acct_banks[src]
        to_bank = acct_banks[dst]
        from_account = _account_id_from_uri(src, from_bank)
        to_account = _account_id_from_uri(dst, to_bank)

        amount = g.value(tx, AML.amountPaid)
        currency = g.value(tx, AML.paymentCurrency)
        fmt = g.value(tx, AML.paymentFormat)
        ts = g.value(tx, AML.timestamp)
        fraud = g.value(tx, AML.isLaundering)

        rows.append(
            {
                "timestamp": str(ts) if ts is not None else "",
                "from_bank": from_bank,
                "from_account": from_account,
                "to_bank": to_bank,
                "to_account": to_account,
                "amount_received": float(amount) if amount is not None else 0.0,
                "receiving_currency": str(currency) if currency else "",
                "amount_paid": float(amount) if amount is not None else 0.0,
                "payment_currency": str(currency) if currency else "",
                "payment_format": str(fmt) if fmt else "",
                "is_laundering": int(fraud) if fraud is not None else 0,
            }
        )

    df = pd.DataFrame(rows)
    logger.info(
        "DataFrame extrait : %s lignes | %s fraudes",
        f"{len(df):,}",
        f"{int(df['is_laundering'].sum()):,}" if len(df) else "0",
    )
    return df


def build_pyg_from_rdf(
    ttl_path: Path = config.ABOX_SUBGRAPH_TTL,
    pyg_path: Path = config.PYG_DATA,
) -> tuple[Data, FeatureInfo]:
    """Pipeline complet : Turtle → DataFrame → PyG Data."""
    g = load_rdf_graph(ttl_path)
    df = rdf_to_dataframe(g)
    if df.empty:
        raise ValueError(f"Aucune transaction extraite de {ttl_path}")
    data, info = build_pyg_data(df)
    save_pyg_data(data, info, out_path=pyg_path)
    return data, info
