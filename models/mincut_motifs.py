#!/usr/bin/env python3
"""Soft-assignment GNN that partitions tissue graphs end to end.

The existing pipeline is two disconnected stages: an encoder trained with a
contrastive loss, then k-means on its output. Nothing ever told the encoder its
embeddings would be clustered, so it optimises one objective while the motifs
come from a different one applied afterwards.

This collapses them. One network maps cells to a soft assignment over clusters,
trained directly on partition quality:

    L = L_mincut + L_ortho

L_mincut is the relaxed normalised cut. Minimising it keeps edges INSIDE
clusters, which on a spatial graph means motifs come out spatially contiguous —
the same quantity validate_motifs.py measures after the fact as spatial
coherence, used here as the training signal instead.

L_ortho pushes the assignment matrix towards orthogonality. Without it the cut
term has a trivial optimum: put every cell softly in every cluster at
probability 1/c, cut nothing, and output nothing meaningful. Orthogonality
forces confident, near-one-hot assignments and comparably sized clusters.

Both come from PyG's dense_mincut_pool, the reference implementation of
Bianchi et al. (2020), rather than a reimplementation.

NUMBER OF CLUSTERS. `n_clusters` is a CEILING, not a target. The assignment
matrix has c columns but nothing requires every column to carry cells, so the
model can settle on fewer. Verified in the CytoCommunity paper: c "represents
the maximum number of TCNs to be detected" and the learned count "could be
smaller than c".

  Caveat worth keeping: the paper describes no explicit pruning step, the
  collapse is emergent, and L_ortho actively rewards BALANCED cluster sizes,
  which works against leaving columns empty. So c still influences the answer.
  Sweep it (10/20/30) and check the effective count is stable before trusting
  it. "Ceiling instead of exact k" is an improvement, not a free lunch.

DATE ADVERSARY. Replacing the encoder would otherwise throw away the v4 result,
which cut pure-batch excess from +0.289 to +0.168 by scrubbing imaging date
with a gradient-reversal adversary. That is ported here: the same
grad_reverse/ConditionAdversary act on node embeddings before assignment, so
the representation driving the partition is still date-scrubbed. Without it
this model would reintroduce the confound while fixing the cluster count.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, dense_mincut_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch

from .neighborhood_encoder import ConditionAdversary, grad_reverse  # noqa: F401


class SoftAssignGNN(nn.Module):
    """Cells -> soft cluster assignment.

    Message passing gives each cell a representation of its neighbourhood; a
    small head turns that into a distribution over at most `n_clusters`
    clusters. GATv2 and edge_dim match the existing encoder so the same graphs
    feed straight in.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, n_clusters: int = 20,
                 heads: int = 4, edge_dim: int = 4, dropout: float = 0.2,
                 n_layers: int = 3):
        super().__init__()
        self.n_clusters = n_clusters
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        d = in_dim
        for _ in range(n_layers):
            self.convs.append(
                GATv2Conv(d, hidden_dim // heads, heads=heads, edge_dim=edge_dim,
                          dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_dim))
            d = hidden_dim
        self.assign = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, n_clusters),
        )

    def embed(self, x, edge_index, edge_attr):
        h = x
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_new = norm(conv(h, edge_index, edge_attr))
            # residual once the width matches, as in the contrastive encoder
            h = h_new + h if i > 0 and h.shape == h_new.shape else h_new
            h = F.elu(h)
        return h

    def forward(self, x, edge_index, edge_attr, batch):
        """Returns (S, h, mask) with S the per-node soft assignment."""
        h = self.embed(x, edge_index, edge_attr)
        logits = self.assign(h)
        s_dense, mask = to_dense_batch(logits, batch)
        return s_dense, h, mask


def partition_losses(h, edge_index, batch, s_dense, mask):
    """MinCut + orthogonality for a batch of graphs.

    Graphs are densified per image. That is affordable here precisely because
    the unit is one image (hundreds of cells), not the whole 1.27M-cell
    dataset, which could never be densified.
    """
    h_dense, _ = to_dense_batch(h, batch)
    adj = to_dense_adj(edge_index, batch)
    _, _, mincut, ortho = dense_mincut_pool(h_dense, adj, s_dense, mask)
    return mincut, ortho


@torch.no_grad()
def effective_clusters(s_dense, mask, threshold: float = 0.01):
    """How many clusters actually carry cells — the number the model settled on.

    This is the quantity to watch: if it sits well below the ceiling and stays
    put as the ceiling rises, the model has found a count rather than filling
    whatever it was given.
    """
    hard = s_dense.argmax(-1)
    counts = torch.zeros(s_dense.size(-1), device=s_dense.device)
    for b in range(s_dense.size(0)):
        v = hard[b][mask[b]]
        counts += torch.bincount(v, minlength=s_dense.size(-1)).float()
    share = counts / counts.sum().clamp(min=1)
    return int((share >= threshold).sum()), share.cpu().numpy()
