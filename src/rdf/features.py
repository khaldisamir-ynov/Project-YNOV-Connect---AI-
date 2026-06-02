"""Extraction dynamique de features depuis la T-Box et l'A-Box RDF.

Parcourt les DatatypeProperty déclarées dans `aml.ttl` (domain + range XSD)
et construit automatiquement les tenseurs de features par type de nœud.
Ajouter une propriété à l'ontologie suffit pour l'inclure au pipeline GNN
(sans migration SQL ni refonte manuelle des colonnes CSV).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import torch
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD

from src import config

logger = logging.getLogger(__name__)

AML = Namespace(config.AML_ONTOLOGY_NS)

# Propriétés exclues des features (label cible ou identifiants)
EXCLUDE_FROM_FEATURES = {AML.isLaundering}

XSD_NUMERIC = {XSD.float, XSD.double, XSD.decimal, XSD.integer, XSD.int, XSD.long}
XSD_STRING = {XSD.string}


@dataclass
class PropertySchema:
    uri: URIRef
    name: str
    domain_class: URIRef
    range_type: URIRef
    is_numeric: bool
    is_string: bool


@dataclass
class NodeFeatureBundle:
    """Features encodées pour un type de nœud RDF."""

    x: torch.Tensor
    feature_names: list[str] = field(default_factory=list)
    categorical_vocabs: dict[str, list[str]] = field(default_factory=dict)


def load_schema_properties(schema_path: Path = config.ONTOLOGY_TTL) -> dict[str, list[PropertySchema]]:
    """Lit la T-Box et indexe les DatatypeProperty par classe de domaine."""
    g = Graph()
    g.parse(schema_path, format="turtle")

    class_local: dict[URIRef, str] = {}
    for cls in g.subjects(RDF.type, OWL.Class):
        local = _local_name(cls)
        if local:
            class_local[cls] = local

    by_class: dict[str, list[PropertySchema]] = {name: [] for name in class_local.values()}

    for prop in g.subjects(RDF.type, OWL.DatatypeProperty):
        if prop in EXCLUDE_FROM_FEATURES:
            continue
        domains = list(g.objects(prop, RDFS.domain))
        ranges = list(g.objects(prop, RDFS.range))
        if not domains or not ranges:
            continue
        domain_cls = domains[0]
        range_type = ranges[0]
        domain_name = class_local.get(domain_cls)
        if domain_name is None:
            continue
        local = _local_name(prop)
        by_class[domain_name].append(
            PropertySchema(
                uri=prop,
                name=local,
                domain_class=domain_cls,
                range_type=range_type,
                is_numeric=range_type in XSD_NUMERIC,
                is_string=range_type in XSD_STRING,
            )
        )

    for cls_name, props in by_class.items():
        logger.info("T-Box — %s : %s DatatypeProperty", cls_name, len(props))
    return by_class


def _local_name(uri: URIRef) -> str:
    text = unquote(str(uri))
    if "#" in text:
        return text.split("#")[-1]
    return text.rsplit("/", 1)[-1]


def _top_n_vocab(values: list[str], n: int) -> list[str]:
    from collections import Counter

    counts = Counter(values)
    top = [v for v, _ in counts.most_common(n)]
    if "Other" not in top:
        top.append("Other")
    return top


def _one_hot(value: str, vocab: list[str]) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=np.float32)
    key = value if value in vocab else "Other"
    if key in vocab:
        vec[vocab.index(key)] = 1.0
    elif "Other" in vocab:
        vec[vocab.index("Other")] = 1.0
    return vec


def extract_node_features(
    g: Graph,
    class_uri: URIRef,
    node_uris: list[URIRef],
    properties: list[PropertySchema],
    string_top_n: int = 8,
) -> NodeFeatureBundle:
    """Encode les DatatypeProperty pour une liste de nœuds du même type."""
    if not node_uris:
        return NodeFeatureBundle(x=torch.zeros(0, 1), feature_names=["bias"])

    # Collecte brute par propriété
    raw: dict[str, list] = {p.name: [] for p in properties}
    for uri in node_uris:
        for prop in properties:
            val = g.value(uri, prop.uri)
            raw[prop.name].append(val)

    columns: list[np.ndarray] = []
    names: list[str] = []
    vocabs: dict[str, list[str]] = {}

    for prop in properties:
        values = raw[prop.name]
        if prop.is_numeric:
            nums = np.array(
                [float(v) if v is not None else 0.0 for v in values],
                dtype=np.float32,
            )
            if prop.name in ("amountPaid",):
                nums = np.log1p(np.clip(nums, 0, None))
            mean, std = nums.mean(), nums.std() + 1e-8
            col = ((nums - mean) / std).reshape(-1, 1)
            columns.append(col)
            names.append(f"{prop.name}_z")
        elif prop.is_string:
            str_vals = [str(v) if v is not None else "" for v in values]
            vocab = _top_n_vocab(str_vals, string_top_n)
            vocabs[prop.name] = vocab
            mat = np.stack([_one_hot(v, vocab) for v in str_vals])
            columns.append(mat)
            names.extend([f"{prop.name}_{v}" for v in vocab])
        else:
            continue

    if not columns:
        # Fallback : constante pour permettre au GNN de projeter
        mat = np.ones((len(node_uris), 1), dtype=np.float32)
        return NodeFeatureBundle(x=torch.from_numpy(mat), feature_names=["bias"])

    x = np.concatenate(columns, axis=1).astype(np.float32)
    return NodeFeatureBundle(
        x=torch.from_numpy(x),
        feature_names=names,
        categorical_vocabs=vocabs,
    )


def align_feature_bundle(
    bundle: NodeFeatureBundle,
    target_names: list[str],
) -> NodeFeatureBundle:
    """Aligne les colonnes de features sur celles du modèle entraîné (hold-out)."""
    if not target_names:
        return bundle

    n = bundle.x.size(0)
    name_to_idx = {name: i for i, name in enumerate(bundle.feature_names)}
    parts: list[torch.Tensor] = []

    for name in target_names:
        if name in name_to_idx:
            idx = name_to_idx[name]
            parts.append(bundle.x[:, idx : idx + 1])
        elif name == "bias":
            parts.append(torch.ones((n, 1), dtype=bundle.x.dtype))
        else:
            parts.append(torch.zeros((n, 1), dtype=bundle.x.dtype))

    x = torch.cat(parts, dim=1) if parts else torch.zeros((n, 1), dtype=torch.float32)
    return NodeFeatureBundle(
        x=x,
        feature_names=list(target_names),
        categorical_vocabs=bundle.categorical_vocabs,
    )


def structural_account_features(
    g: Graph,
    account_uris: list[URIRef],
    uri_to_idx: dict[URIRef, int],
) -> NodeFeatureBundle:
    """Features topologiques calculées sur le graphe RDF (non déclarées en T-Box)."""
    n = len(account_uris)
    out_deg = np.zeros(n, dtype=np.float32)
    in_deg = np.zeros(n, dtype=np.float32)
    tx_with = np.zeros(n, dtype=np.float32)
    same_bank = np.zeros(n, dtype=np.float32)

    for acct, idx in uri_to_idx.items():
        out_deg[idx] = len(list(g.objects(acct, AML.sends_via)))
        in_deg[idx] = len(list(g.subjects(AML.deposits_to, acct)))
        tx_with[idx] = len(list(g.objects(acct, AML.transactedWith)))
        same_bank[idx] = len(list(g.objects(acct, AML.sameBankAs)))

    mat = np.stack(
        [
            np.log1p(out_deg),
            np.log1p(in_deg),
            np.log1p(tx_with),
            np.log1p(same_bank),
        ],
        axis=1,
    )
    mat = (mat - mat.mean(axis=0)) / (mat.std(axis=0) + 1e-8)
    return NodeFeatureBundle(
        x=torch.from_numpy(mat.astype(np.float32)),
        feature_names=["log_sends_via", "log_deposits_to", "log_transacted_with", "log_same_bank"],
    )
