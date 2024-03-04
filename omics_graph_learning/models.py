# sourcery skip: avoid-single-character-names-variables, upper-camel-case-classes
#! /usr/bin/env python
# -*- coding: utf-8 -*-
#

"""GNN model architectures for node regression. The models include four
different types of architectures:

(1) Classic message passing neural networks (MPNNs):
    Graph convolutional networks (GCN)
    GraphSAGE
    Principle neighborhood aggregation (PNA - MPNN w/ aggregators and scalers)

(2) Attention-based models:
    Graph attention networks V2 (GATv2)
    UniMPTransformer (GCN with UniMP transformer operator)

(3) Large scale models (transformers, or GNN with large number of layers):
    DeeperGCN
    Transformer based? To be implemented

(4) Basline:
    3-layer MLP
"""

from typing import Any, Callable, List, Optional

import torch
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GENConv
from torch_geometric.nn import GraphNorm
from torch_geometric.nn import LayerNorm
from torch_geometric.nn import PNAConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models import DeepGCNLayer


def _define_activation(activation: str) -> Callable:
    """Defines the activation function according to the given string"""
    activations = {"gelu": F.gelu, "leakyrelu": F.leaky_relu, "relu": F.relu}
    if activation in activations:
        return activations[activation]
    else:
        raise ValueError(
            "Invalid activation function. Supported: relu, leakyrelu, gelu"
        )


def _initialize_lazy_gnn_layers(
    in_size: int,
    embedding_size: int,
    gnn_layers: int,
    operator: Callable,
) -> nn.ModuleList:
    """Create gnn layers for models that support lazy initialization.

    Args:
        in_size (int): Size of the input tensor.
        embedding_size (int): Size of hidden dimensions.
        gnn_layers (int): Number of convolutional layers.
        operator (Callable): The convolutional operator to use.

    Returns:
        nn.ModuleList: Module list containing the convolutional layers.
    """
    convs = nn.ModuleList()
    norms = nn.ModuleList([GraphNorm(embedding_size) for _ in range(gnn_layers)])

    for i in range(gnn_layers):
        if operator == SAGEConv:
            convs.append(
                operator(
                    in_size if i == 0 else embedding_size, embedding_size, aggr="mean"
                )
            )
        else:
            convs.append(
                operator(in_size if i == 0 else embedding_size, embedding_size)
            )

    return convs, norms


def _initialize_lazy_linear_layers(
    in_size: int,
    out_size: int,
    linear_layers: int,
) -> nn.ModuleList:
    """Create linear layers for models that support lazy initialization.

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


def _get_linear_layer_sizes_attn_models(
    heads: int, embedding_size: int, linear_layers: int, out_channels: int
) -> List[int]:
    """Get the sizes of the linear layers for models w/ attention heads."""
    return (
        [heads * embedding_size]
        + [embedding_size] * (linear_layers - 1)
        + [out_channels]
    )


def _create_linear_layers_attn_models(sizes: List[int]) -> nn.ModuleList:
    """Create linear layers for models w/ attention heads.

    Args:
        sizes: A list of sizes for the linear layers.

    Returns:
        nn.ModuleList: The module list containing the linear layers.
    """
    return nn.ModuleList(
        [Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
    )


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
        activation: str,
        residual: bool = False,
        dropout_rate: Optional[float] = 0.5,
    ):
        """Initialize the model"""
        super().__init__()
        self.residual = residual
        self.dropout_rate = dropout_rate
        self.activation = _define_activation(activation)

        # Create graph convolution layers
        self.convs, self.norms = _initialize_lazy_gnn_layers(
            in_size=in_size,
            embedding_size=embedding_size,
            gnn_layers=gnn_layers,
            operator=GCNConv,
        )

        # Create linear layers
        self.linears = _initialize_lazy_linear_layers(
            in_size=embedding_size, out_size=out_channels, linear_layers=linear_layers
        )

        # Create linear projection for skip connection
        self.linear_projection = nn.Linear(in_size, embedding_size, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GCN model.

        Args:
            x: The input tensor.
            edge_index: The edge index tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        for conv, batch_norm in zip(self.convs, self.norms):
            if self.residual:
                if conv == self.convs[0]:
                    x = self.activation(
                        batch_norm(conv(x, edge_index))
                    ) + self.linear_projection(x)
                else:
                    x = self.activation(batch_norm(conv(x, edge_index))) + x
            else:
                x = self.activation(batch_norm(conv(x, edge_index)))

        apply_dropout = isinstance(self.dropout_rate, float)
        for linear_layer in self.linears[:-1]:
            x = self.activation(linear_layer(x))
            if apply_dropout:
                x = F.dropout(x, p=self.dropout_rate)

        return self.linears[-1](x)


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
        activation: str,
        residual: bool = False,
        dropout_rate: Optional[float] = 0.5,
    ):
        """Initialize the model"""
        super().__init__()
        self.embedding_size = embedding_size
        self.residual = residual
        self.dropout_rate = dropout_rate
        self.activation = _define_activation(activation)
        self.aggregator = "mean"

        # Create graph convolution layers
        self.convs, self.norms = _initialize_lazy_gnn_layers(
            in_size=in_size,
            embedding_size=embedding_size,
            gnn_layers=gnn_layers,
            operator=SAGEConv,
        )

        # Create linear layers
        self.linears = _initialize_lazy_linear_layers(
            embedding_size, out_channels, linear_layers
        )

        # Create linear projection for skip connection
        self.linear_projection = nn.Linear(in_size, embedding_size, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GraphSAGE model.

        Args:
            x: The input tensor.
            edge_index: The edge index tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        for conv, batch_norm in zip(self.convs, self.norms):
            if self.residual:
                if conv == self.convs[0]:
                    x = self.activation(
                        batch_norm(conv(x, edge_index))
                    ) + self.linear_projection(x)
                else:
                    x = self.activation(batch_norm(conv(x, edge_index))) + x
            else:
                x = self.activation(batch_norm(conv(x, edge_index)))

        apply_dropout = isinstance(self.dropout_rate, float)
        for linear_layer in self.linears[:-1]:
            x = self.activation(linear_layer(x))
            if apply_dropout:
                x = F.dropout(x, p=self.dropout_rate)

        return self.linears[-1](x)


class PNA(torch.nn.Module):
    """Principle neighborhood aggregation.

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
        activation: str,
        deg: torch.Tensor,
        residual: bool = False,
        dropout_rate: Optional[float] = 0.5,
    ):
        """Initialize the model"""
        super().__init__()
        self.embedding_size = embedding_size
        self.residual = residual
        self.dropout_rate = dropout_rate
        self.activation = _define_activation(activation)
        self.deg = deg
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]

        # create graph convolution layers
        self.convs.append(
            PNAConv(
                in_channels=in_size,
                out_channels=embedding_size,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                towers=2,  # double check this
                divide_input=False,
            )
        )
        self.norms.append(GraphNorm(embedding_size))
        for _ in range(gnn_layers - 1):
            self.convs.append(
                PNAConv(
                    in_channels=embedding_size,
                    out_channels=embedding_size,
                    aggregators=aggregators,
                    scalers=scalers,
                    deg=deg,
                    towers=2,
                    divide_input=False,
                )
            )
            self.norms.append(GraphNorm(embedding_size))

        # Create linear layers
        self.linear_layers = _initialize_lazy_linear_layers(
            in_size=embedding_size, out_size=out_channels, linear_layers=linear_layers
        )

        # Create linear projection for skip connection
        self.linear_projection = nn.Linear(in_size, embedding_size, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GATv2 model.

        Args:
            x: The input tensor.
            edge_index: The edge index tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        for conv, batch_norm in zip(self.convs, self.norms):
            if self.residual:
                if conv == self.convs[0]:
                    x = self.activation(
                        batch_norm(conv(x, edge_index))
                    ) + self.linear_projection(x)
                else:
                    x = self.activation(batch_norm(conv(x, edge_index))) + x
            else:
                x = self.activation(batch_norm(conv(x, edge_index)))

        apply_dropout = isinstance(self.dropout_rate, float)
        for linear_layer in self.linear_layers[:-1]:
            x = self.activation(linear_layer(x))
            if apply_dropout:
                x = F.dropout(x, p=self.dropout_rate)

        return self.linear_layers[-1](x)


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
        activation: str,
        heads: int,
        residual: bool = False,
        dropout_rate: Optional[float] = 0.5,
    ):
        """Initialize the model"""
        super().__init__()
        self.embedding_size = embedding_size
        self.residual = residual
        self.dropout_rate = dropout_rate
        self.activation = _define_activation(activation)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # create graph convolution layers
        for i in range(gnn_layers):
            in_channels = in_size if i == 0 else heads * embedding_size
            self.convs.append(GATv2Conv(in_channels, embedding_size, heads))
            self.norms.append(GraphNorm(heads * embedding_size))

        # Create linear layers
        linear_sizes = _get_linear_layer_sizes_attn_models(
            heads=heads,
            embedding_size=embedding_size,
            linear_layers=linear_layers,
            out_channels=out_channels,
        )
        self.linear_layers = _create_linear_layers_attn_models(linear_sizes)

        # Create linear projection for skip connection
        self.linear_projection = nn.Linear(in_size, embedding_size, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GATv2 model.

        Args:
            x: The input tensor.
            edge_index: The edge index tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        for conv, batch_norm in zip(self.convs, self.norms):
            if self.residual:
                if conv == self.convs[0]:
                    x = self.activation(
                        batch_norm(conv(x, edge_index))
                    ) + self.linear_projection(x)
                else:
                    x = self.activation(batch_norm(conv(x, edge_index))) + x
            else:
                x = self.activation(batch_norm(conv(x, edge_index)))

        apply_dropout = isinstance(self.dropout_rate, float)
        for linear_layer in self.linear_layers[:-1]:
            x = self.activation(linear_layer(x))
            if apply_dropout:
                x = F.dropout(x, p=self.dropout_rate)

        return self.linear_layers[-1](x)


# Define/Instantiate GNN models
class UniMPTransformer(torch.nn.Module):
    """Model utilizing the graph transformer operator from UniMP, but not the
    masked labelling task.

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
        activation: str,
        heads: int,
        residual: bool = False,
        dropout_rate: Optional[float] = 0.5,
    ):
        """Initialize the model"""
        super().__init__()
        self.embedding_size = embedding_size
        self.residual = residual
        self.dropout_rate = dropout_rate
        self.activation = _define_activation(activation)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # create graph convoluton layers
        for i in range(gnn_layers):
            in_channels = in_size if i == 0 else embedding_size
            concat = i != gnn_layers - 1
            self.convs.append(
                TransformerConv(
                    in_channels=in_channels,
                    out_channels=embedding_size,
                    heads=heads,
                    concat=concat,
                    beta=True,
                )
            )
            self.norms.append(GraphNorm(embedding_size * heads))

        # Create linear layers
        linear_sizes = _get_linear_layer_sizes_attn_models(
            heads=heads,
            embedding_size=embedding_size,
            linear_layers=linear_layers,
            out_channels=out_channels,
        )
        self.linear_layers = _create_linear_layers_attn_models(linear_sizes)

        # Create linear projection for skip connection
        self.linear_projection = nn.Linear(in_size, embedding_size, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GATv2 model.

        Args:
            x: The input tensor.
            edge_index: The edge index tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        for conv, batch_norm in zip(self.convs, self.norms):
            if self.residual:
                if conv == self.convs[0]:
                    x = self.activation(
                        batch_norm(conv(x, edge_index))
                    ) + self.linear_projection(x)
                else:
                    x = self.activation(batch_norm(conv(x, edge_index))) + x
            else:
                x = self.activation(batch_norm(conv(x, edge_index)))

        apply_dropout = isinstance(self.dropout_rate, float)
        for linear_layer in self.linear_layers[:-1]:
            x = self.activation(linear_layer(x))
            if apply_dropout:
                x = F.dropout(x, p=self.dropout_rate)

        return self.linear_layers[-1](x)


class DeeperGCN(torch.nn.Module):
    """DeeperGCN model architecture.

    Args:
        in_size: The input size of the DeeperGCN model.
        embedding_size: The size of the embedding dimension.
        out_channels: The number of output channels.
        gnn_layers: The number of GATv2 layers.
        linear_layers: The number of linear layers.
        activation: The activation function.
        dropout_rate: The dropout rate.
    """

    def __init__(
        self,
        in_size: int,
        embedding_size: int,
        out_channels: int,
        gnn_layers: int,
        linear_layers: int,
        activation: str,
        dropout_rate: Optional[float] = 0.5,
    ):
        """Initialize the model"""
        super().__init__()
        self.activation = _define_activation(activation)
        self.dropout_rate = dropout_rate
        self.layers = nn.ModuleList()
        self.node_encoder = Linear(in_size, embedding_size)

        layer_act = self._define_activation_nonfunctional(activation)

        for i in range(gnn_layers):
            conv = GENConv(
                in_channels=embedding_size,
                out_channels=embedding_size,
                aggr="softmax",
                t=1.0,
                learn_t=True,
                num_layers=2,
                norm="layer",
            )
            norm = LayerNorm(embedding_size)
            act = layer_act(inplace=True)
            layer = DeepGCNLayer(
                conv=conv,
                norm=norm,
                act=act,
                block="res+",
                dropout=0.1,
                ckpt_grad=i % 3,
            )
            self.layers.append(layer)

        # Create linear layers
        self.linears = _initialize_lazy_linear_layers(
            embedding_size, out_channels, linear_layers
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass of the DeeperGCN model.

        Args:
            x: The input tensor.
            edge_index: The edge index tensor.

        Returns:
            torch.Tensor: The output tensor."""
        x = self.node_encoder(x)
        for layer in self.layers:
            x = layer(x, edge_index)

        apply_dropout = isinstance(self.dropout_rate, float)
        for linear_layer in self.linears[:-1]:
            x = self.activation(linear_layer(x))
            if apply_dropout:
                x = F.dropout(x, p=self.dropout_rate)

        return self.linears[-1](x)

    @staticmethod
    def _define_activation_nonfunctional(activation: str) -> Callable:
        """Defines the activation function according to the given string"""
        activations = {
            "gelu": torch.nn.GELU,
            "leakyrelu": torch.nn.LeakyReLU,
            "relu": torch.nn.ReLU,
        }
        if activation in activations:
            return activations[activation]
        else:
            raise ValueError(
                "Invalid activation function. Supported: relu, leakyrelu, gelu"
            )


class MLP(torch.nn.Module):
    """Simple three layer MLP model architecture.

    Args:
        in_size: The input size of the MLP model.
        embedding_size: The size of the embedding dimension.
        out_channels: The number of output channels.
        activation: String specifying the activation function between linear
        layers.

    Returns:
        torch.Tensor: The output tensor.
    """

    def __init__(
        self,
        in_size: int,
        embedding_size: int,
        out_channels: int,
        activation: str,
    ):
        """Initialize the model"""
        super().__init__()
        self.linears = nn.ModuleList(
            [
                nn.Linear(in_size, embedding_size),
                nn.Linear(embedding_size, embedding_size),
                nn.Linear(embedding_size, out_channels),
            ]
        )
        self.activation = _define_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP model.

        Args:
            x: The input tensor.
            edge_index: The edge index tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))

        return self.linears[-1](x)


# class GPSTransformer(torch.nn.Module):
#     """GPSTransformer model architecture.

#     Args:
#         in_size: The input size of the GPSTransformer model.
#         embedding_size: The size of the embedding dimension.
#         walk_length: The length of the random walk.
#         channels: The number of channels.
#         pe_dim: The dimension of positional encoding.
#         gnn_layers: The number of layers.

#     Returns:
#         torch.Tensor: The output tensor.
#     """

#     def __init__(
#         self,
#         in_size,
#         embedding_size,
#         walk_length: int,
#         channels: int,
#         pe_dim: int,
#         gnn_layers: int,
#     ):
#         """Initialize the model"""
#         super().__init__()

#         self.node_emb = nn.Linear(in_size, embedding_size - pe_dim)
#         self.pe_lin = nn.Linear(walk_length, pe_dim)
#         self.pe_norm = nn.BatchNorm1d(walk_length)

#         self.convs = torch.nn.ModuleList()
#         for _ in range(gnn_layers):
#             gcnconv = GCNConv(embedding_size, embedding_size)
#             conv = GPSConv(channels, gcnconv, heads=4, attn_kwargs={"dropout": 0.5})
#             self.convs.append(conv)

#         self.mlp = Sequential(
#             Linear(channels, channels // 2),
#             ReLU(),
#             Linear(channels // 2, channels // 4),
#             ReLU(),
#             Linear(channels // 4, 1),
#         )

#         self.redraw_projection = RedrawProjection(self.convs, None)

#     def forward(self, x, pe, edge_index, batch):
#         """Forward pass of the GPSTransformer model.

#         Args:
#             x: The input tensor.
#             pe: The positional encoding tensor.
#             edge_index: The edge index tensor.
#             batch: The batch tensor.

#         Returns:
#             torch.Tensor: The output tensor.
#         """
#         x_pe = self.pe_norm(pe)
#         x = torch.cat((self.node_emb(x.squeeze(-1)), self.pe_lin(x_pe)), 1)

#         for conv in self.convs:
#             x = conv(x, edge_index, batch)
#         return self.mlp(x)


# class RedrawProjection:
#     """RedrawProjection class.

#     Args:
#         model: The model to perform redraw projections on.
#         redraw_interval: The interval at which to redraw projections.

#     Returns:
#         None
#     """

#     def __init__(self, model: torch.nn.Module, redraw_interval: Optional[int] = None):
#         """Initialize the RedrawProjection object"""
#         self.model = model
#         self.redraw_interval = redraw_interval
#         self.num_last_redraw = 0

#     def redraw_projections(self):
#         """Redraw the projections"""
#         if not self.model.training or self.redraw_interval is None:
#             return
#         if self.num_last_redraw >= self.redraw_interval:
#             fast_attentions = [
#                 module
#                 for module in self.model.modules()
#                 if isinstance(module, PerformerAttention)
#             ]
#             for fast_attention in fast_attentions:
#                 fast_attention.redraw_projection_matrix()
#             self.num_last_redraw = 0
#             return
#         self.num_last_redraw += 1
