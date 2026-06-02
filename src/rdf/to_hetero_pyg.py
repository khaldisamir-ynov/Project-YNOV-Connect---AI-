"""Conversion d'une A-Box RDF enrichie vers un HeteroData PyG.

Préserve la sémantique multi-relationnelle du RDF :
    - Nœuds : account, transaction, bank
    - Arêtes typées : sends_via, deposits_to, hosted_by, transacted_with, same_bank_as
    - Labels : classification de nœuds sur `transaction` (isLaundering)

Chaque type de relation est traité par une couche distincte dans le GNN
(HeteroConv — équivalent pratique du R-GCN sur graphe RDF).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import torch
from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData

from src import config
from src.rdf.features import (
    NodeFeatureBundle,
    align_feature_bundle,
    extract_node_features,
    load_schema_properties,
    structural_account_features,
)

logger = logging.getLogger(__name__)

AML = Namespace(config.AML_ONTOLOGY_NS)


@dataclass
class HeteroFeatureInfo:
    """Métadonnées du graphe hétérogène."""

    node_types: list[str] = field(default_factory=list)
    edge_types: list[tuple[str, str, str]] = field(default_factory=list)
    node_feature_names: dict[str, list[str]] = field(default_factory=dict)
    n_nodes: dict[str, int] = field(default_factory=dict)
    n_fraud_transactions: int = 0
    n_inferred_transacted_with: int = 0
    n_inferred_same_bank_as: int = 0


def load_enriched_graph(ttl_path: Path = config.ABOX_SUBGRAPH_ENRICHED_TTL) -> Graph:
    g = Graph()
    g.parse(config.ONTOLOGY_TTL, format="turtle")
    g.parse(ttl_path, format="turtle")
    logger.info("Graphe enrichi chargé : %s triplets", f"{len(g):,}")
    return g


def _tx_sort_key(tx_uri: URIRef) -> int:
    local = unquote(str(tx_uri).split("#")[-1])
    m = re.match(r"tx-(\d+)$", local)
    return int(m.group(1)) if m else 0


def _collect_nodes(g: Graph) -> tuple[list[URIRef], list[URIRef], list[URIRef]]:
    accounts = sorted(g.subjects(RDF.type, AML.Account), key=str)
    transactions = sorted(g.subjects(RDF.type, AML.Transaction), key=_tx_sort_key)
    banks = sorted(g.subjects(RDF.type, AML.Bank), key=str)
    return list(accounts), list(transactions), list(banks)


def _edge_index_from_triples(
    g: Graph,
    predicate,
    src_uris: list[URIRef],
    dst_uris: list[URIRef],
) -> torch.Tensor:
    src_map = {u: i for i, u in enumerate(src_uris)}
    dst_map = {u: i for i, u in enumerate(dst_uris)}
    rows, cols = [], []
    for s, o in g.subject_objects(predicate):
        if s in src_map and o in dst_map:
            rows.append(src_map[s])
            cols.append(dst_map[o])
    if not rows:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([rows, cols], dtype=torch.long)


def _concat_features(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.size(0) != b.size(0):
        raise ValueError("Tailles de nœuds incohérentes pour concat features.")
    if a.size(1) == 0:
        return b
    if b.size(1) == 0:
        return a
    return torch.cat([a, b], dim=1)


def build_hetero_data(
    g: Graph,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_seed: int = config.RANDOM_SEED,
    holdout_eval: bool = False,
    reference_feature_names: dict[str, list[str]] | None = None,
) -> tuple[HeteroData, HeteroFeatureInfo]:
    """Construit HeteroData depuis le graphe RDF enrichi."""
    schema_by_class = load_schema_properties()
    account_uris, tx_uris, bank_uris = _collect_nodes(g)
    logger.info(
        "Nœuds RDF — accounts: %s | transactions: %s | banks: %s",
        f"{len(account_uris):,}",
        f"{len(tx_uris):,}",
        f"{len(bank_uris):,}",
    )

    acct_uri_to_idx = {u: i for i, u in enumerate(account_uris)}
    tx_uri_to_idx = {u: i for i, u in enumerate(tx_uris)}
    bank_uri_to_idx = {u: i for i, u in enumerate(bank_uris)}

    # --- Features dynamiques (T-Box) + structurelles ---
    tx_props = schema_by_class.get("Transaction", [])
    bank_props = schema_by_class.get("Bank", [])

    tx_bundle = extract_node_features(g, AML.Transaction, tx_uris, tx_props)
    bank_bundle = extract_node_features(g, AML.Bank, bank_uris, bank_props)
    acct_struct = structural_account_features(g, account_uris, acct_uri_to_idx)
    acct_owl = extract_node_features(g, AML.Account, account_uris, schema_by_class.get("Account", []))
    acct_combined = NodeFeatureBundle(
        x=_concat_features(acct_owl.x, acct_struct.x),
        feature_names=acct_owl.feature_names + acct_struct.feature_names,
    )

    if reference_feature_names:
        ref_acct = reference_feature_names.get("account", [])
        ref_tx = reference_feature_names.get("transaction", [])
        ref_bank = reference_feature_names.get("bank", [])
        if ref_acct:
            acct_combined = align_feature_bundle(acct_combined, ref_acct)
        if ref_tx:
            tx_bundle = align_feature_bundle(tx_bundle, ref_tx)
        if ref_bank:
            bank_bundle = align_feature_bundle(bank_bundle, ref_bank)
        logger.info(
            "Features alignées sur le train — account=%s | transaction=%s | bank=%s",
            acct_combined.x.size(1),
            tx_bundle.x.size(1),
            bank_bundle.x.size(1),
        )

    acct_x = acct_combined.x

    # --- Labels transaction ---
    labels = []
    for tx in tx_uris:
        val = g.value(tx, AML.isLaundering)
        labels.append(int(val) if val is not None else 0)
    y = torch.tensor(labels, dtype=torch.long)
    n_fraud = int(y.sum())

    # --- Masques (splits ou hold-out intégral) ---
    train_mask = torch.zeros(len(y), dtype=torch.bool)
    val_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)

    if holdout_eval:
        test_mask[:] = True
        logger.info("Mode hold-out : %s transactions en évaluation.", f"{len(y):,}")
    else:
        indices = np.arange(len(y))
        train_idx, temp_idx = train_test_split(
            indices,
            test_size=val_size + test_size,
            stratify=y.numpy(),
            random_state=random_seed,
        )
        val_ratio = val_size / (val_size + test_size)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=1 - val_ratio,
            stratify=y.numpy()[temp_idx],
            random_state=random_seed,
        )
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

    # --- Assemblage HeteroData ---
    data = HeteroData()
    data["account"].x = acct_x
    data["transaction"].x = tx_bundle.x
    data["bank"].x = bank_bundle.x
    data["transaction"].y = y
    data["transaction"].train_mask = train_mask
    data["transaction"].val_mask = val_mask
    data["transaction"].test_mask = test_mask

    # Arêtes assertées + inférées (chaque prédicat = type de relation distinct)
    data["account", "sends_via", "transaction"].edge_index = _edge_index_from_triples(
        g, AML.sends_via, account_uris, tx_uris
    )
    data["transaction", "deposits_to", "account"].edge_index = _edge_index_from_triples(
        g, AML.deposits_to, tx_uris, account_uris
    )
    data["account", "hosted_by", "bank"].edge_index = _edge_index_from_triples(
        g, AML.hosted_by, account_uris, bank_uris
    )
    data["account", "transacted_with", "account"].edge_index = _edge_index_from_triples(
        g, AML.transactedWith, account_uris, account_uris
    )
    data["account", "same_bank_as", "account"].edge_index = _edge_index_from_triples(
        g, AML.sameBankAs, account_uris, account_uris
    )

    # Arêtes inverses pour la propagation bidirectionnelle (HeteroConv)
    for src, rel, dst in list(data.edge_types):
        rev = (dst, f"rev_{rel}", src)
        data[rev].edge_index = data[src, rel, dst].edge_index.flip(0)

    n_tw = data["account", "transacted_with", "account"].edge_index.size(1)
    n_sb = data["account", "same_bank_as", "account"].edge_index.size(1)

    info = HeteroFeatureInfo(
        node_types=["account", "transaction", "bank"],
        edge_types=list(data.edge_types),
        node_feature_names={
            "account": list(acct_combined.feature_names),
            "transaction": list(tx_bundle.feature_names),
            "bank": list(bank_bundle.feature_names),
        },
        n_nodes={
            "account": len(account_uris),
            "transaction": len(tx_uris),
            "bank": len(bank_uris),
        },
        n_fraud_transactions=n_fraud,
        n_inferred_transacted_with=n_tw,
        n_inferred_same_bank_as=n_sb,
    )

    logger.info(
        "HeteroData — relations: %s | fraudes: %s | transacted_with: %s | same_bank_as: %s",
        len(data.edge_types),
        n_fraud,
        f"{n_tw:,}",
        f"{n_sb:,}",
    )
    return data, info


def save_hetero_data(
    data: HeteroData,
    info: HeteroFeatureInfo,
    out_path: Path = config.PYG_HETERO_DATA,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"data": data, "info": info}, out_path)
    logger.info("HeteroData écrit : %s (%s octets)", out_path, f"{out_path.stat().st_size:,}")
    return out_path


def load_hetero_data(path: Path = config.PYG_HETERO_DATA) -> tuple[HeteroData, HeteroFeatureInfo]:
    bundle = torch.load(path, weights_only=False)
    return bundle["data"], bundle["info"]


def build_hetero_pyg_from_rdf(
    ttl_path: Path = config.ABOX_SUBGRAPH_ENRICHED_TTL,
    pyg_path: Path = config.PYG_HETERO_DATA,
    holdout_eval: bool = False,
    reference_feature_names: dict[str, list[str]] | None = None,
) -> tuple[HeteroData, HeteroFeatureInfo]:
    g = load_enriched_graph(ttl_path)
    data, info = build_hetero_data(
        g,
        holdout_eval=holdout_eval,
        reference_feature_names=reference_feature_names,
    )
    save_hetero_data(data, info, pyg_path)
    return data, info
