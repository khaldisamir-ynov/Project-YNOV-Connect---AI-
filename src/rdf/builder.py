"""Conversion du CSV échantillonné vers un graphe RDF (A-Box).

Le graphe produit est strictement conforme à la T-Box définie dans
`data/rdf/ontology_schema/aml.ttl` :

    Account  --sends_via-->     Transaction
    Transaction --deposits_to--> Account
    Account  --hosted_by-->     Bank

Chaque Transaction porte ses propriétés de données :
amountPaid, paymentCurrency, paymentFormat, timestamp, isLaundering.
"""
from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD
from tqdm import tqdm

from src import config

logger = logging.getLogger(__name__)

AML = Namespace(config.AML_ONTOLOGY_NS)
INST = Namespace(config.AML_INSTANCE_NS)


# --------------------------------------------------------------------------- #
# URI builders
# --------------------------------------------------------------------------- #
def _safe(value: str) -> str:
    """Encode un identifiant pour qu'il soit URI-safe."""
    return quote(str(value), safe="")


def bank_uri(bank_id: str) -> URIRef:
    return URIRef(f"{config.AML_INSTANCE_NS}bank-{_safe(bank_id)}")


def account_uri(bank_id: str, account_id: str) -> URIRef:
    # Un même numéro de compte peut exister dans plusieurs banques,
    # donc l'identité d'un Account = (banque, numéro).
    return URIRef(
        f"{config.AML_INSTANCE_NS}acct-{_safe(bank_id)}-{_safe(account_id)}"
    )


def transaction_uri(tx_id: int | str) -> URIRef:
    return URIRef(f"{config.AML_INSTANCE_NS}tx-{_safe(tx_id)}")


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def _bind_namespaces(g: Graph) -> None:
    g.bind("aml", AML)
    g.bind("inst", INST)
    g.bind("xsd", XSD)
    g.bind("rdfs", RDFS)
    g.bind("owl", OWL)


def build_graph(df: pd.DataFrame) -> Graph:
    """Construit le graphe RDF (A-Box) à partir du DataFrame échantillonné."""
    g = Graph()
    _bind_namespaces(g)

    seen_banks: set[str] = set()
    seen_accounts: set[tuple[str, str]] = set()

    logger.info("Construction de l'A-Box pour %s transactions…", f"{len(df):,}")

    for row in tqdm(df.itertuples(index=True, name="Tx"), total=len(df)):
        tx_idx = row.Index
        from_bank = str(row.from_bank)
        from_acc = str(row.from_account)
        to_bank = str(row.to_bank)
        to_acc = str(row.to_account)

        from_bank_uri = bank_uri(from_bank)
        to_bank_uri = bank_uri(to_bank)
        from_acc_uri = account_uri(from_bank, from_acc)
        to_acc_uri = account_uri(to_bank, to_acc)
        tx_uri = transaction_uri(tx_idx)

        # Banques (déclarées une seule fois)
        for bank_id, b_uri in ((from_bank, from_bank_uri), (to_bank, to_bank_uri)):
            if bank_id not in seen_banks:
                g.add((b_uri, RDF.type, AML.Bank))
                g.add((b_uri, RDFS.label, Literal(f"Bank {bank_id}")))
                g.add((b_uri, AML.bankCode, Literal(str(bank_id), datatype=XSD.string)))
                seen_banks.add(bank_id)

        # Comptes (déclarés une seule fois) + lien hosted_by
        for bank_id, acc_id, a_uri, b_uri in (
            (from_bank, from_acc, from_acc_uri, from_bank_uri),
            (to_bank, to_acc, to_acc_uri, to_bank_uri),
        ):
            key = (bank_id, acc_id)
            if key not in seen_accounts:
                g.add((a_uri, RDF.type, AML.Account))
                g.add((a_uri, AML.hosted_by, b_uri))
                g.add((a_uri, RDFS.label, Literal(f"Account {acc_id} @ {bank_id}")))
                seen_accounts.add(key)

        # Transaction + topologie
        g.add((tx_uri, RDF.type, AML.Transaction))
        g.add((from_acc_uri, AML.sends_via, tx_uri))
        g.add((tx_uri, AML.deposits_to, to_acc_uri))

        # Attributs de la transaction (Datatype Properties)
        g.add((tx_uri, AML.amountPaid, Literal(float(row.amount_paid), datatype=XSD.float)))
        g.add((tx_uri, AML.paymentCurrency, Literal(str(row.payment_currency), datatype=XSD.string)))
        g.add((tx_uri, AML.paymentFormat, Literal(str(row.payment_format), datatype=XSD.string)))
        g.add((tx_uri, AML.timestamp, Literal(str(row.timestamp), datatype=XSD.string)))
        g.add((tx_uri, AML.isLaundering, Literal(int(row.is_laundering), datatype=XSD.integer)))

    logger.info(
        "Graphe construit : %s triples | %s banques | %s comptes | %s transactions",
        f"{len(g):,}",
        f"{len(seen_banks):,}",
        f"{len(seen_accounts):,}",
        f"{len(df):,}",
    )
    return g


def serialize_graph(g: Graph, out_path: Path = config.ABOX_TTL) -> Path:
    """Sérialise le graphe en Turtle."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Sérialisation Turtle → %s", out_path)
    g.serialize(destination=out_path, format="turtle")
    logger.info("OK (%s octets)", f"{out_path.stat().st_size:,}")
    return out_path
