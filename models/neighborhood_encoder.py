"""Stage 3 GNN encoder: GATv2 neighborhood encoder + SimCLR projection head.

Implements the architecture in docs/stage3_encoder_methodology.md §3, §5:
  - 3 GATv2 layers (edge-feature-conditioned attention), multi-head, ELU+dropout,
    a residual on the middle layer, mean readout -> a fixed-length neighborhood
    embedding ``h``.
  - a 2-layer MLP projection head -> ``z`` used ONLY for the contrastive loss
    (discarded for clustering; clustering uses ``h``).

The NT-Xent contrastive loss lives here too so the encoder + objective are in
one place.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


class _GradReverse(torch.autograd.Function):
    """Gradient Reversal Layer: identity forward, negated (x lambda) gradient back."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)


class ConditionAdversary(nn.Module):
    """Small MLP that predicts `condition` from the embedding, through a GRL.

    Forward multiplies the incoming gradient by -lambd (via the GRL) so the
    encoder is trained to make `condition` UNpredictable, while the adversary
    itself still learns to predict it as well as it can. Returns class logits.
    """

    def __init__(self, emb_dim: int, n_conditions: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_conditions),
        )

    def forward(self, h: torch.Tensor, lambd: float) -> torch.Tensor:
        return self.net(grad_reverse(h, lambd))


class NeighborhoodEncoder(nn.Module):
    """GATv2 encoder mapping a (batched) neighborhood subgraph -> embedding.

    forward() returns (h, z):
      h : [B, emb_dim]  neighborhood embedding   (KEEP — used for clustering)
      z : [B, emb_dim]  projected embedding       (loss only — discarded after)
    """

    def __init__(
        self,
        in_dim: int = 7,
        hidden_dim: int = 128,
        emb_dim: int = 64,
        heads: int = 4,
        edge_dim: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.dropout = dropout

        # Layer 1: in_dim -> hidden (concatenated heads => hidden*heads out).
        self.conv1 = GATv2Conv(in_dim, hidden_dim, heads=heads,
                               edge_dim=edge_dim, dropout=dropout)
        # Layer 2: hidden*heads -> hidden (heads concatenated again).
        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads,
                               edge_dim=edge_dim, dropout=dropout)
        # Layer 3: hidden*heads -> emb_dim, single head (clean embedding width).
        self.conv3 = GATv2Conv(hidden_dim * heads, emb_dim, heads=1,
                               edge_dim=edge_dim, dropout=dropout)

        # Residual projection so layer-2 input can be added to its output
        # (dims differ: hidden*heads -> hidden*heads is identity; we add the
        # pre-activation of conv2's input projected to match conv2 output).
        self.res_proj = nn.Linear(hidden_dim * heads, hidden_dim * heads)

        # SimCLR projection head (discarded for clustering).
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim),
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Run the GAT stack + readout -> [num_graphs, emb_dim] embeddings h."""
        h = F.elu(self.conv1(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)

        h2 = F.elu(self.conv2(h, edge_index, edge_attr))
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        h = h2 + self.res_proj(h)                       # residual (methodology §3e)

        h = self.conv3(h, edge_index, edge_attr)        # no activation on last layer
        return global_mean_pool(h, batch)               # mean readout (§4)

    def forward(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_attr = getattr(data, "edge_attr", None)
        h = self.encode(data.x, data.edge_index, edge_attr, data.batch)
        z = self.proj(h)
        return h, z


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """NT-Xent (normalized temperature-scaled cross-entropy), SimCLR-style.

    z1, z2 : [B, D] projected embeddings of the two augmented views (row i of z1
    and row i of z2 are the positive pair). Returns a scalar loss.
    """
    B = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)   # [2B, D], unit vectors
    sim = z @ z.t() / temperature                        # [2B, 2B] cosine sims

    # mask self-similarity so a view is never its own positive
    self_mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(self_mask, float("-inf"))

    # positive of row i is its sibling: i<->i+B
    targets = torch.arange(2 * B, device=z.device)
    targets = (targets + B) % (2 * B)
    return F.cross_entropy(sim, targets)
