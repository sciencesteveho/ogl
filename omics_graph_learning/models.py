# sourcery skip: avoid-single-character-names-variables
#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] Add number of linear layers as a config param
# - [ ] Check number of optimal lin layers from gc-merge

"""GNN model architectures!"""

from typing import Any, Dict, List, Optional

import torch
from torch.nn import BatchNorm1d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sequential
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GPSConv
from torch_geometric.nn import GraphNorm
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.attention import PerformerAttention
import torch_geometric.transforms as T


def _initialize_lazy_linear_layers(
    in_size: int,
    out_size: int,
    linear_layers: int,
) -> nn.ModuleList:
    """Create linear layers for models w/ lazy initialization.

    Args:
        in_size: The input size of the linear layers.
        out_size: The output size of the final layer.
        gnn_layers: The number of linear layers.

    Returns:
        nn.ModuleList: The module list containing the linear layers.
    """
    layers = [nn.Linear(in_size, in_size) for _ in range(linear_layers - 1)]
    layers.append(nn.Linear(in_size, out_size))
    return nn.ModuleList(layers)


# Define/Instantiate GNN models
class GATv2(torch.nn.Module):
    """GATv2 model architecture.

    Args:
        in_size: Dimensions of the node features.
        embedding_size: Dimensions of the hidden layers.
        out_channels: The number of output channels.
        gnn_layers: The number of GATv2 layers.
        linear_layers: The number of linear layers.
        heads: The number of attention heads.
        dropout_rate: The dropout rate.
    """

    def __init__(
        self,
        in_size: int,
        embedding_size: int,
        out_channels: int,
        gnn_layers: int,
        linear_layers: int,
        heads: int,
        dropout_rate: Optional[float] = 0.5,
    ):
        """Initialize the model"""
        super().__init__()
        self.linear_layers = linear_layers
        self.dropout_rate = dropout_rate
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.convs.append(GATv2Conv(in_size, embedding_size, heads))

        # create graph convolution layers
        for _ in range(gnn_layers - 1):
            self.convs.append(GATv2Conv(heads * embedding_size, embedding_size, heads))
            self.batch_norms.append(GraphNorm(heads * embedding_size))

        # Create linear layers
        linear_sizes = (
            [heads * embedding_size]
            + [embedding_size] * (linear_layers - 1)
            + [out_channels]
        )
        self.linear_layers = self.create_linear_layers(linear_sizes)

    def create_linear_layers(self, sizes: List[int]) -> nn.ModuleList:
        """Create linear layers based on the given sizes.

        Args:
            sizes: A list of sizes for the linear layers.

        Returns:
            nn.ModuleList: The module list containing the linear layers.
        """
        return nn.ModuleList(
            [Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GATv2 model.

        Args:
            x: The input tensor.
            edge_index: The edge index tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))

        apply_dropout = isinstance(self.dropout_rate, float)
        for linear_layer in self.linear_layers[:-1]:
            x = F.relu(linear_layer(x))
            if apply_dropout:
                x = F.dropout(x, p=self.dropout_rate)

        return self.linear_layers[-1](x)


class GraphSAGE(torch.nn.Module):
    """GraphSAGE model architecture.

    Args:
        in_size: Dimensions of the node features.
        embedding_size: Dimensions of the hidden layers.
        out_channels: The number of output channels.
        gnn_layers: The number of GraphSAGE layers.
        linear_layers: The number of linear layers.
        dropout_rate: The dropout rate.

    Returns:
        torch.Tensor: The output tensor.
    """

    def __init__(
        self,
        in_size: int,
        embedding_size: int,
        out_channels: int,
        gnn_layers: int,
        linear_layers: int,
        dropout_rate: Optional[float] = 0.5,
    ):
        """Initialize the model"""
        super().__init__()
        self.dropout_rate = dropout_rate
        self.gnn_layers = gnn_layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.convs.append(SAGEConv(in_size, embedding_size, aggr="sum"))

        for _ in range(gnn_layers - 1):
            self.convs.append(SAGEConv(embedding_size, embedding_size, aggr="sum"))
            self.batch_norms.append(GraphNorm(embedding_size))

        # Create linear layers
        self.linears = _initialize_lazy_linear_layers(
            embedding_size, out_channels, linear_layers
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GraphSAGE model.

        Args:
            x: The input tensor.
            edge_index: The edge index tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))

        apply_dropout = isinstance(self.dropout_rate, float)
        for linear_layer in self.linears[:-1]:
            x = F.relu(linear_layer(x))
            if apply_dropout:
                x = F.dropout(x, p=self.dropout_rate)

        return self.linears[-1](x)


class GCN(torch.nn.Module):
    """GCN model architecture.

    Args:
        in_size: The input size of the GCN model.
        embedding_size: The size of the embedding dimension.
        out_channels: The number of output channels.
        gnn_layers: The number of GCN layers.

    Returns:
        torch.Tensor: The output tensor.
    """

    def __init__(
        self,
        in_size: int,
        embedding_size: int,
        out_channels: int,
        gnn_layers: int,
        linear_layers: int,
        dropout_rate: Optional[float] = 0.5,
    ):
        """Initialize the model"""
        super().__init__()
        self.gnn_layers = gnn_layers
        self.dropout_rate = dropout_rate
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.convs.append(GCNConv(in_size, embedding_size))

        for _ in range(gnn_layers - 1):
            self.convs.append(GCNConv(embedding_size, embedding_size))
            self.batch_norms.append(GraphNorm(embedding_size))

        # Create linear layers
        self.linears = _initialize_lazy_linear_layers(
            embedding_size, out_channels, linear_layers
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GCN model.

        Args:
            x: The input tensor.
            edge_index: The edge index tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))

        apply_dropout = isinstance(self.dropout_rate, float)
        for linear_layer in self.linears[:-1]:
            x = F.relu(linear_layer(x))
            if apply_dropout:
                x = F.dropout(x, p=self.dropout_rate)

        return self.linears[-1](x)


### baseline MLP
class MLP(torch.nn.Module):
    """MLP model architecture.

    Args:
        in_size: The input size of the MLP model.
        embedding_size: The size of the embedding dimension.
        out_channels: The number of output channels.

    Returns:
        torch.Tensor: The output tensor.
    """

    def __init__(
        self,
        in_size,
        embedding_size,
        out_channels,
    ):
        """Initialize the model"""
        super().__init__()
        self.lin1 = nn.Linear(in_size, embedding_size)
        self.lin2 = nn.Linear(embedding_size, embedding_size)
        self.lin3 = nn.Linear(embedding_size, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP model.

        Args:
            x: The input tensor.
            edge_index: The edge index tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        return x


class GPSTransformer(torch.nn.Module):
    """GPSTransformer model architecture.

    Args:
        in_size: The input size of the GPSTransformer model.
        embedding_size: The size of the embedding dimension.
        walk_length: The length of the random walk.
        channels: The number of channels.
        pe_dim: The dimension of positional encoding.
        gnn_layers: The number of layers.

    Returns:
        torch.Tensor: The output tensor.
    """

    def __init__(
        self,
        in_size,
        embedding_size,
        walk_length: int,
        channels: int,
        pe_dim: int,
        gnn_layers: int,
    ):
        """Initialize the model"""
        super().__init__()

        self.node_emb = nn.Linear(in_size, embedding_size - pe_dim)
        self.pe_lin = nn.Linear(walk_length, pe_dim)
        self.pe_norm = nn.BatchNorm1d(walk_length)

        self.convs = torch.nn.ModuleList()
        for _ in range(gnn_layers):
            gcnconv = GCNConv(embedding_size, embedding_size)
            conv = GPSConv(channels, gcnconv, heads=4, attn_kwargs={"dropout": 0.5})
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(channels, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, 1),
        )

        self.redraw_projection = RedrawProjection(self.convs, None)

    def forward(self, x, pe, edge_index, batch):
        """Forward pass of the GPSTransformer model.

        Args:
            x: The input tensor.
            pe: The positional encoding tensor.
            edge_index: The edge index tensor.
            batch: The batch tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x.squeeze(-1)), self.pe_lin(x_pe)), 1)

        for conv in self.convs:
            x = conv(x, edge_index, batch)
        return self.mlp(x)


class RedrawProjection:
    """RedrawProjection class.

    Args:
        model: The model to perform redraw projections on.
        redraw_interval: The interval at which to redraw projections.

    Returns:
        None
    """

    def __init__(self, model: torch.nn.Module, redraw_interval: Optional[int] = None):
        """Initialize the RedrawProjection object"""
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        """Redraw the projections"""
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module
                for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1
