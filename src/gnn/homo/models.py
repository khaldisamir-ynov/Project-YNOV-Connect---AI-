"""Modèles GNN pour la classification d'arêtes (transactions frauduleuses).

Architecture commune :
    1. Encodeur de nœuds (GraphSAGE ou GAT) → embeddings de comptes.
    2. Tête de classification d'arêtes :
       concat(source_emb, dest_emb, edge_attr) → MLP → logit fraude.

La tête exploite explicitement la topologie (via les embeddings de voisinage)
ET les attributs locaux de la transaction (montant, format, devise).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, SAGEConv


class EdgeClassifierGNN(nn.Module):
    """GNN homogène + classificateur d'arêtes."""

    def __init__(
        self,
        in_channels: int,
        edge_dim: int,
        hidden_channels: int = 64,
        edge_hidden: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        model_type: str = "sage",
    ) -> None:
        super().__init__()
        self.model_type = model_type.lower()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if self.model_type == "sage":
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_layers - 1):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        elif self.model_type == "gat":
            heads = 4
            out_per_head = hidden_channels // heads
            self.convs.append(GATConv(in_channels, out_per_head, heads=heads, concat=True))
            for _ in range(num_layers - 2):
                self.convs.append(
                    GATConv(hidden_channels, out_per_head, heads=heads, concat=True)
                )
            if num_layers > 1:
                self.convs.append(GATConv(hidden_channels, hidden_channels, heads=1, concat=False))
        else:
            raise ValueError(f"model_type inconnu : {model_type!r} (attendu 'sage' ou 'gat')")

        for _ in self.convs:
            self.norms.append(nn.LayerNorm(hidden_channels))

        edge_in = hidden_channels * 2 + edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, edge_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden, edge_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden // 2, 1),
        )

    def encode_nodes(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, data: Data) -> torch.Tensor:
        """Renvoie un logit par arête (shape [num_edges])."""
        h = self.encode_nodes(data.x, data.edge_index)
        src, dst = data.edge_index
        edge_repr = torch.cat([h[src], h[dst], data.edge_attr], dim=-1)
        return self.edge_mlp(edge_repr).squeeze(-1)
