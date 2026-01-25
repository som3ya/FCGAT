from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv
from torch_geometric.utils import dropout_edge, softmax

try:
    from torch_scatter import scatter_add
except Exception as e:  # pragma: no cover
    scatter_add = None


class Set2SetWithAttention(nn.Module):
    """Set2Set readout that returns per-node attention weights.

    This is a lightly modified version of the Set2Set idea for sets, where we
    also return the attention weights (alpha) for interpretability.

    - Input: node embeddings x [num_nodes, in_channels], batch [num_nodes]
    - Output: graph embeddings q_star [num_graphs, 2*in_channels]
              attention_steps: list of alpha vectors (length = processing_steps)
                where each alpha is [num_nodes] (softmax per graph).
    """

    def __init__(self, in_channels: int, processing_steps: int = 4, num_layers: int = 1):
        super().__init__()
        self.in_channels = int(in_channels)
        self.processing_steps = int(processing_steps)
        self.out_channels = 2 * self.in_channels
        self.lstm = nn.LSTM(self.out_channels, self.in_channels, num_layers=num_layers)

    def reset_parameters(self) -> None:
        self.lstm.reset_parameters()

    def forward(
        self,
        x: Tensor,
        batch: Tensor,
        batch_size: Optional[int] = None,
    ) -> Tuple[Tensor, List[Tensor]]:
        if scatter_add is None:  # pragma: no cover
            raise ImportError(
                "torch_scatter is required for Set2SetWithAttention. "
                "Install torch-geometric with the correct wheels for your torch/cuda version."
            )

        if x.dim() != 2:
            raise ValueError(f"Expected x to be 2D [num_nodes, channels], got {tuple(x.shape)}")
        if batch.dim() != 1 or batch.size(0) != x.size(0):
            raise ValueError("batch must be 1D and aligned with x")

        if batch_size is None:
            batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        h = (
            x.new_zeros((self.lstm.num_layers, batch_size, self.in_channels)),
            x.new_zeros((self.lstm.num_layers, batch_size, self.in_channels)),
        )
        q_star = x.new_zeros((batch_size, self.out_channels))

        attention_steps: List[Tensor] = []

        for _ in range(self.processing_steps):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.squeeze(0)  # [batch_size, in_channels]

            # e_i = x_i · q_{batch(i)}
            e = (x * q[batch]).sum(dim=-1)  # [num_nodes]

            # alpha = softmax per-graph
            a = softmax(e, batch, num_nodes=batch_size)  # [num_nodes]
            attention_steps.append(a)

            r = scatter_add(a.unsqueeze(-1) * x, batch, dim=0, dim_size=batch_size)  # [batch_size, in_channels]
            q_star = torch.cat([q, r], dim=-1)  # [batch_size, 2*in_channels]

        return q_star, attention_steps


@dataclass
class FCGATHParams:
    in_dim: int = 100
    hidden_dim: int = 64
    heads: int = 3
    set2set_steps: int = 4
    dropout_p: float = 0.5
    edge_dropout_p: float = 0.5


class FCGAT(nn.Module):
    """FCGAT model.

    Architecture (paper-aligned):
      - GATConv on reverse-FCG, no self loops
      - Add transformed initial node feature (W1 * x_init)
      - LeakyReLU
      - Set2Set readout (LSTM) returning attention weights
      - Dropout
      - Linear classifier

    The returned attention weights can be used to rank important functions.
    """

    def __init__(self, num_classes: int, hparams: Optional[FCGATHParams] = None):
        super().__init__()
        self.hparams = hparams or FCGATHParams()

        in_dim = self.hparams.in_dim
        hidden_dim = self.hparams.hidden_dim
        heads = self.hparams.heads

        out_dim = hidden_dim * heads

        self.gat = GATConv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            add_self_loops=False,
        )

        # W1 * x_init (paper) : map initial features to hidden space
        self.fc_init = nn.Linear(in_dim, out_dim, bias=False)

        self.act = nn.LeakyReLU()
        self.readout = Set2SetWithAttention(out_dim, processing_steps=self.hparams.set2set_steps)
        self.dropout = nn.Dropout(p=self.hparams.dropout_p)
        self.classifier = nn.Linear(2 * out_dim, num_classes)

    def forward(self, data) -> Tuple[Tensor, List[Tensor]]:
        """Forward pass.

        Args:
            data: torch_geometric.data.Data or Batch with fields x, edge_index, batch.

        Returns:
            logits: [batch_size, num_classes]
            attention_steps: list of per-node attention weights (len = set2set_steps)
        """
        x: Tensor = data.x
        edge_index: Tensor = data.edge_index
        batch: Tensor = data.batch

        x_init = x

        # Edge dropout
        edge_index_dp, _ = dropout_edge(edge_index, p=self.hparams.edge_dropout_p, training=self.training)

        x = self.gat(x, edge_index_dp)
        x = self.act(x)

        x = x + self.fc_init(x_init)
        x = self.act(x)

        graph_vec, attention_steps = self.readout(x, batch)

        graph_vec = self.dropout(graph_vec)
        logits = self.classifier(graph_vec)
        return logits, attention_steps

    @torch.no_grad()
    def explain_single(self, data, top_k: int = 20, step: Optional[int] = None):
        """Return a ranked list of (function_name, attention_weight) for a single graph.

        This expects `data` to represent a **single** graph (not a batch), and to
        have `func_name` aligned with nodes.
        """
        self.eval()
        # Make a pseudo-batch of size 1 if missing.
        if getattr(data, "batch", None) is None:
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)

        logits, attn_steps = self.forward(data)
        probs = F.softmax(logits, dim=-1).squeeze(0)

        if step is None:
            step = len(attn_steps) - 1
        step = int(step)

        weights = attn_steps[step].detach().cpu()  # [num_nodes]
        func_names = list(getattr(data, "func_name", [f"node{i}" for i in range(len(weights))]))

        pairs = list(zip(func_names, weights.tolist()))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return {
            "probs": probs.detach().cpu().tolist(),
            "top_functions": pairs[:top_k],
        }
