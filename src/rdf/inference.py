"""Inférence sur l'A-Box RDF avant entraînement GNN.

Déduit de nouveaux liens logiques à partir de l'ontologie et des triplets
assertés, puis matérialise le graphe enrichi pour le GNN hétérogène.

Règles implémentées :
    1. transactedWith  — deux comptes liés par au moins une transaction.
    2. sameBankAs      — même banque (hosted_by) ET transactedWith (sparse).

Optionnel : expansion OWL-RL via owlrl si installé (axiomes SymmetricProperty…).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF

from src import config

logger = logging.getLogger(__name__)

AML = Namespace(config.AML_ONTOLOGY_NS)


def load_graph_with_schema(abox_path: Path, schema_path: Path = config.ONTOLOGY_TTL) -> Graph:
    """Charge T-Box + A-Box dans un graphe unique."""
    g = Graph()
    g.parse(schema_path, format="turtle")
    g.parse(abox_path, format="turtle")
    logger.info(
        "Graphe chargé (T-Box + A-Box) : %s triplets depuis %s",
        f"{len(g):,}",
        abox_path.name,
    )
    return g


def _account_bank_map(g: Graph) -> dict[URIRef, URIRef]:
    return {acct: bank for acct, bank in g.subject_objects(AML.hosted_by)}


def _transaction_endpoints(g: Graph) -> list[tuple[URIRef, URIRef]]:
    """Renvoie (compte_émetteur, compte_destinataire) pour chaque transaction."""
    pairs: list[tuple[URIRef, URIRef]] = []
    for tx in g.subjects(RDF.type, AML.Transaction):
        srcs = list(g.subjects(AML.sends_via, tx))
        dsts = list(g.objects(tx, AML.deposits_to))
        if len(srcs) == 1 and len(dsts) == 1:
            pairs.append((srcs[0], dsts[0]))
    return pairs


def infer_transacted_with(g: Graph) -> int:
    """Matérialise aml:transactedWith (symétrique) entre paires de comptes."""
    seen: set[tuple[URIRef, URIRef]] = set()
    added = 0
    for src, dst in _transaction_endpoints(g):
        if src == dst:
            continue
        key = (min(src, dst, key=str), max(src, dst, key=str))
        if key in seen:
            continue
        seen.add(key)
        g.add((src, AML.transactedWith, dst))
        g.add((dst, AML.transactedWith, src))
        added += 2
    logger.info("transactedWith : %s paires (%s triplets)", f"{len(seen):,}", f"{added:,}")
    return added


def infer_same_bank_as(g: Graph) -> int:
    """Matérialise aml:sameBankAs si même banque ET déjà transactedWith."""
    acct_bank = _account_bank_map(g)
    added = 0
    for a, b in g.subject_objects(AML.transactedWith):
        if a >= b:
            continue
        bank_a = acct_bank.get(a)
        bank_b = acct_bank.get(b)
        if bank_a is None or bank_b is None or bank_a != bank_b:
            continue
        if (a, AML.sameBankAs, b) in g:
            continue
        g.add((a, AML.sameBankAs, b))
        g.add((b, AML.sameBankAs, a))
        added += 2
    logger.info("sameBankAs : %s triplets ajoutés", f"{added:,}")
    return added


def run_owl_rl_expansion(g: Graph) -> int:
    """Expansion OWL-RL optionnelle (SymmetricProperty, domain/range…)."""
    try:
        from owlrl import DeductiveClosure, OWLRL_Semantics
    except ImportError:
        logger.warning("owlrl non installé — expansion OWL-RL ignorée.")
        return 0

    before = len(g)
    DeductiveClosure(OWLRL_Semantics).expand(g)
    added = len(g) - before
    logger.info("OWL-RL : +%s triplets inférés", f"{added:,}")
    return added


def enrich_rdf_graph(
    g: Graph,
    use_owlrl: bool = config.RDF_USE_OWLRL,
) -> Graph:
    """Applique toutes les règles d'inférence sur une copie du graphe."""
    enriched = Graph()
    for t in g:
        enriched.add(t)
    for prefix, ns in g.namespace_manager.namespaces():
        enriched.bind(prefix, ns)

    infer_transacted_with(enriched)
    infer_same_bank_as(enriched)
    if use_owlrl:
        run_owl_rl_expansion(enriched)

    logger.info("Graphe enrichi : %s triplets (+ %s)", f"{len(enriched):,}", f"{len(enriched) - len(g):,}")
    return enriched


def enrich_rdf_file(
    abox_path: Path = config.ABOX_SUBGRAPH_TTL,
    out_path: Path = config.ABOX_SUBGRAPH_ENRICHED_TTL,
    schema_path: Path = config.ONTOLOGY_TTL,
    use_owlrl: bool = config.RDF_USE_OWLRL,
) -> Path:
    """Pipeline fichier : charge, infère, sérialise."""
    g = load_graph_with_schema(abox_path, schema_path)
    enriched = enrich_rdf_graph(g, use_owlrl=use_owlrl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.serialize(destination=out_path, format="turtle")
    logger.info("A-Box enrichie écrite : %s (%s octets)", out_path, f"{out_path.stat().st_size:,}")
    return out_path
