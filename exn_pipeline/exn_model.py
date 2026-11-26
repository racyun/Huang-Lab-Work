"""GNN that classifies the target (centre) cell of a 2-hop ego graph.

Same structure as models/cell_state_predictor.py (Stage 5), adapted to a binary
label:

    h0 = ELU(W_in x)                                  x: [m, in_dim] -> [m, H]
    h1 = LayerNorm(h0 + Dropout(ELU(GATv2(h0, A, e))))
    h2 = LayerNorm(h1 + Dropout(ELU(GATv2(h1, A, e))))  e: [edges, 5] edge features
    logit = MLP([h2[c] ; h0[c]])                      c = target cell, -> scalar

Two message-passing layers match the 2-hop subgraph, so information reaches the
target from exactly that far. The target's own input embedding h0[c] joins the
head directly, so a model that gains nothing from neighbours can still match an
own-features baseline. The output is a logit; train with BCEWithLogitsLoss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class ExnCellClassifier(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int = 5, hidden_dim: int = 64, heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.dropout = dropout
        self.embed = nn.Linear(in_dim, hidden_dim)
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False,
                               edge_dim=edge_dim, dropout=dropout, fill_value="mean")
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False,
                               edge_dim=edge_dim, dropout=dropout, fill_value="mean")
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ELU(),
                                  nn.Linear(hidden_dim, 1))

    def node_states(self, x, edge_index, edge_attr):
        h0 = F.elu(self.embed(x))
        h1 = self.norm1(h0 + F.dropout(F.elu(self.conv1(h0, edge_index, edge_attr)),
                                       p=self.dropout, training=self.training))
        h2 = self.norm2(h1 + F.dropout(F.elu(self.conv2(h1, edge_index, edge_attr)),
                                       p=self.dropout, training=self.training))
        return h0, h2

    def forward(self, data) -> torch.Tensor:
        h0, h2 = self.node_states(data.x, data.edge_index, data.edge_attr)
        c = data.center_index
        return self.head(torch.cat([h2[c], h0[c]], dim=1)).squeeze(-1)

    def forward_all(self, x, edge_index, edge_attr) -> torch.Tensor:
        """Logit for every node of a whole graph (used to check ego-graph exactness)."""
        h0, h2 = self.node_states(x, edge_index, edge_attr)
        return self.head(torch.cat([h2, h0], dim=1)).squeeze(-1)
