"""GNN hétérogène multi-relationnel (HeteroConv) pour fraude AML.

Chaque prédicat RDF devient un type d'arête avec ses propres poids —
équivalent pratique du R-GCN : le modèle apprend des transformations
distinctes pour sends_via, transacted_with, same_bank_as, etc.

Tâche : classification de nœuds sur `transaction` (isLaundering).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv


class HeteroTransactionGNN(nn.Module):
    """Encodeur hétérogène + tête de classification sur les transactions."""

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        in_channels_dict: dict[str, int],
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        head_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        node_types, edge_types = metadata

        self.lin_dict = nn.ModuleDict(
            {
                ntype: nn.Linear(in_channels_dict[ntype], hidden_channels)
                for ntype in node_types
            }
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                conv_dict[edge_type] = SAGEConv(
                    (hidden_channels, hidden_channels),
                    hidden_channels,
                )
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
            self.norms.append(
                nn.ModuleDict({ntype: nn.LayerNorm(hidden_channels) for ntype in node_types})
            )

        # Concat embedding final + features brutes transaction
        head_in = hidden_channels + in_channels_dict["transaction"]
        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden // 2, 1),
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_raw = {ntype: data[ntype].x for ntype in data.node_types}
        x_dict = {
            ntype: F.relu(self.lin_dict[ntype](x_raw[ntype]))
            for ntype in data.node_types
        }

        edge_index_dict = {et: data[et].edge_index for et in data.edge_types}

        for conv, norm_dict in zip(self.convs, self.norms):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: norm_dict[k](F.relu(v)) for k, v in x_dict.items()}
            x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}

        # Skip : embedding GNN + features brutes transaction (T-Box dynamique)
        h_tx = torch.cat([x_dict["transaction"], x_raw["transaction"]], dim=-1)
        return self.head(h_tx).squeeze(-1)
