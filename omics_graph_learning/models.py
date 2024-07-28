# sourcery skip: avoid-single-character-names-variables, no-complex-if-expressions, upper-camel-case-classes
#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""GNN model architectures for node regression. We adopt a base class for the
GNN architecture space. Specific GNN models inherit from the baseclass to
allow flexible and module GNN design. Model args specific for each convolutional
operator are hardcoded into the class definition.

The models include four different classes of architectures:

(1) Classic message passing neural networks (MPNNs): 
    Graph convolutional networks (GCN)
    GraphSAGE
    Principle neighborhood aggregation (PNA - MPNN w/ aggregators and scalers)

(2) Attention-based models:
    Graph attention networks V2 (GATv2)
    UniMPTransformer (GCN with UniMP transformer operator)

(3) Large scale models (GNN with large number of layers):
    DeeperGCN

(4) Baseline: 3-layer MLP
"""


from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv  # type: ignore
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GENConv
from torch_geometric.nn import GraphNorm
from torch_geometric.nn import LayerNorm
from torch_geometric.nn import PNAConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models import DeepGCNLayer  # type: ignore


def define_activation(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Defines the activation function according to the given string"""
    activations: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "gelu": F.gelu,
        "leakyrelu": F.leaky_relu,
        "relu": F.relu,
    }
    try:
        return activations[activation]
    except KeyError as error:
        raise ValueError(
            "Invalid activation function. Supported: relu, leakyrelu, gelu"
        ) from error


class MLP(nn.Module):
    """Simple three layer MLP model architecture.

    Args:
        in_size: The input size of the MLP model.
        embedding_size: The size of the embedding dimension.
        out_channels: The number of output channels.
        activation: String specifying the activation function between linear
        layers. Defaults to `relu`.

    Returns:
        torch.Tensor: The output tensor.
    """

    def __init__(
        self,
        in_size: int,
        embedding_size: int,
        out_channels: int,
        activation: str = "relu",
    ):
        """Initialize the MLP model."""
        super().__init__()
        self.linears = nn.ModuleList(
            [
                nn.Linear(in_size, embedding_size),
                nn.Linear(embedding_size, embedding_size),
                nn.Linear(embedding_size, out_channels),
            ]
        )
        self.activation = define_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP model.

        Args:
            x: The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))

        return self.linears[-1](x)


class ModularGNN(nn.Module):
    """Highly modular and flexible GNN model architecture.

    Attributes:
        activation: The activation function to use, from `relu`, `leakyrelu`, `gelu`.
        dropout_rate: The dropout rate.
        num_tasks: The number of output tasks. When set to 1, a single output head
            is used as a shared MLP. When set to greater than 1, multiple output heads are used.
        skip_connection: The type of skip connection.
        positional_encoding: Boolean indicating if positional encoding is used.
        positional_encoding_dim: The number of dimensions for positional encoding.
        convs: The graph convolutional layers.
        norms: The normalization layers.
        linears: The linear layers.
        task_head: The task specific heads.
        linear_projection: The linear projection for skip connections.
    """

    def __init__(
        self,
        activation: str,
        in_size: int,
        embedding_size: int,
        out_channels: int,
        gnn_layers: int,
        shared_mlp_layers: int,
        gnn_operator_config: Dict[str, Any],
        heads: Optional[int] = None,
        dropout_rate: Optional[float] = None,
        skip_connection: Optional[str] = None,
        task_specific_mlp: bool = False,
    ):
        """Initialize the model"""
        super().__init__()
        self.dropout_rate = dropout_rate
        self.skip_connection = skip_connection
        self.shared_mlp_layers = shared_mlp_layers
        self.task_specific_mlp = task_specific_mlp
        self.activation = define_activation(activation)

        # add attention heads if specified in config
        self.heads = gnn_operator_config.get("heads")

        # Initialize GNN layers and batch normalization
        self.convs = self._create_gnn_layers(
            in_size=in_size,
            embedding_size=embedding_size,
            layers=gnn_layers,
            config=gnn_operator_config,
        )

        # Initialize normalization layers
        self.norms = self._get_normalization_layers(
            layers=gnn_layers, embedding_size=embedding_size, heads=self.heads
        )

        # Initialize linear layers (task-specific MLP)
        self.linears = self._get_linear_layers(
            in_size=embedding_size,
            layers=shared_mlp_layers,
            heads=self.heads,
        )

        # Initialize task head(s)
        if self.task_specific_mlp:
            self.task_head = self._task_specific_head(
                in_size=embedding_size, out_size=out_channels
            )
        else:
            self.task_head = self._general_task_head(
                in_size=embedding_size, out_size=out_channels
            )

        # Create linear projection for skip connections
        self.linear_projection = None
        if self.skip_connection:
            self.linear_projection = self._residuals(
                in_size=in_size,
                embedding_size=(
                    self.heads * embedding_size if self.heads else embedding_size
                ),
            )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the neural network.

        Args:
            x: The input tensor.
            edge_index: The edge index tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        h1 = x  # save input for skip connections

        # graph convolutions with normalization and optional skip connections.
        print(f"Input shape: {x.shape}")
        for i, (conv, batch_norm) in enumerate(zip(self.convs, self.norms)):
            if self.linear_projection:
                if isinstance(self.linear_projection, nn.Linear):
                    residual = (
                        self.linear_projection(h1) if conv == self.convs[0] else x
                    )
                    x = self.activation(batch_norm(conv(x, edge_index))) + residual
                elif isinstance(self.linear_projection, nn.ModuleList):
                    residual = self.linear_projection[i](
                        h1 if conv == self.convs[0] else x
                    )
                    x = self.activation(batch_norm(conv(x, edge_index))) + residual
            else:
                x = self.activation(batch_norm(conv(x, edge_index)))
            print(f"After conv {i+1} shape: {x.shape}")

        # shared linear layers
        for linear_layer in self.linears:
            x = self.activation(linear_layer(x))
            print(f"After linear layer shape: {x.shape}")
            if isinstance(self.dropout_rate, float):
                assert self.dropout_rate is not None
                x = F.dropout(x, p=self.dropout_rate)

        # task specific head(s)
        return (
            self.task_specific_forward(node_mask, x)
            if self.task_specific_mlp
            else self.task_head(x)
        )

    def task_specific_forward(
        self, node_mask: Optional[torch.Tensor], x: torch.Tensor
    ) -> torch.Tensor:
        """Use task specific MLPs for each output head."""
        if node_mask is None:
            raise ValueError("Node mask must be provided for task specific MLPs.")
        if not isinstance(self.task_head, nn.ModuleList):
            raise ValueError(
                "Task specific MLPs must be used for task specific forward."
            )
        node_specific_x = x[node_mask]
        print(f"Node specific x shape after masking: {node_specific_x.shape}")
        print(f"Shape of first task head: {self.task_head[0].weight.shape}")
        print(f"Shape of second task head: {self.task_head[1].weight.shape}")

        # ensure dimensions match
        node_specific_x = F.relu(self.task_head[0](node_specific_x))
        print(f"Node specific x shape after task_head[0]: {node_specific_x.shape}")

        node_specific_out = self.task_head[1](node_specific_x)
        print(f"Node specific out shape after task_head[1]: {node_specific_out.shape}")

        # create the output tensor and fill the masked values
        out = torch.zeros(x.size(0), 1, device=x.device)
        out[node_mask] = node_specific_out

        print(f"Final output shape: {out.shape}")
        return out

    def _get_linear_layers(
        self,
        in_size: int,
        layers: int,
        heads: Optional[int] = None,
    ) -> nn.ModuleList:
        """Create linear layers for the model along with optional linear
        projection for residual connections.

        Args:
            in_size: The input size of the linear layers.
            out_size: The output size of the final layer.
            gnn_layers: The number of linear layers.

        Returns:
            nn.ModuleList: The module list containing the linear layers.
        """
        if self.task_specific_mlp:
            layers -= 1
        return self._linear_module(
            self._linear_layer_dimensions(in_size, layers, heads)
        )

    def _task_specific_head(
        self,
        in_size: int,
        out_size: int,
    ) -> Union[nn.ModuleList, nn.Linear]:
        """Create task specific MLP for each output head."""
        if self.shared_mlp_layers == 1:
            in_size = in_size * self.heads if self.heads else in_size
            node_specific_linear = nn.Linear(in_size, in_size)
        elif self.shared_mlp_layers == 2:
            adjusted_in_size = in_size * self.heads if self.heads else in_size
            node_specific_linear = nn.Linear(adjusted_in_size, in_size)
        else:
            node_specific_linear = nn.Linear(in_size, in_size)

        node_specific_task_head = nn.Linear(in_size, out_size)
        return nn.ModuleList([node_specific_linear, node_specific_task_head])

    def _general_task_head(
        self,
        in_size: int,
        out_size: int,
    ) -> nn.Linear:
        """Create a general task head for the model."""
        if self.shared_mlp_layers == 1:
            in_size = in_size * self.heads if self.heads else in_size
        return nn.Linear(in_size, out_size)

    def _residuals(
        self, in_size: int, embedding_size: int
    ) -> Union[nn.ModuleList, nn.Linear]:
        """Create skip connection linear projections"""
        if self.skip_connection == "shared_source":
            return nn.Linear(in_size, embedding_size, bias=False)
        elif self.skip_connection == "distinct_source":
            return nn.ModuleList(
                [
                    nn.Linear(
                        in_size if i == 0 else embedding_size,
                        embedding_size,
                        bias=False,
                    )
                    for i in range(len(self.convs))
                ]
            )
        else:
            raise ValueError(
                "Invalid skip connection type: must be `shared_source` or `distinct_source`."
            )

    @staticmethod
    def _create_gnn_layers(
        in_size: int, embedding_size: int, layers: int, config: dict
    ) -> nn.ModuleList:
        """Create graph convolutional layers and normalization layers for GNN
        models.

        Args:
            in_size (int): Size of the input tensor.
            embedding_size (int): Size of hidden dimensions.
            layers (int): Number of convolutional layers.
            config (dict): Configuration for operator specific params.

        Returns:
            nn.ModuleList: Module list containing the convolutional layers.
        """
        convs = nn.ModuleList()

        operator = config.pop("operator")
        base_config = {
            "out_channels": embedding_size,
            **config,
        }

        for i in range(layers):
            layer_config = {
                "in_channels": (
                    in_size
                    if i == 0
                    else (
                        config["heads"] * embedding_size
                        if "heads" in config
                        else embedding_size
                    )
                ),
            } | base_config
            convs.append(operator(**layer_config))
        return convs

    @staticmethod
    def _get_normalization_layers(
        layers: int,
        embedding_size: int,
        heads: Optional[int] = None,
    ) -> nn.ModuleList:
        """Create GraphNorm normalization layers for the model."""
        embedding_size = embedding_size * heads if heads else embedding_size
        return nn.ModuleList([GraphNorm(embedding_size) for _ in range(layers)])

    @staticmethod
    def _linear_layer_dimensions(
        embedding_size: int,
        linear_layers: int,
        heads: Optional[int] = None,
    ) -> List[int]:
        """Get the sizes of the linear layers for models w/ attention heads."""
        if heads:
            return [heads * embedding_size] + [embedding_size] * (linear_layers - 1)
        return [embedding_size] * linear_layers

    @staticmethod
    def _linear_module(sizes: List[int]) -> nn.ModuleList:
        """Create a linear layer module.

        Args:
            sizes: A list of sizes for the linear layers.
        """
        return nn.ModuleList(
            [Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        )


class GCN(ModularGNN):
    """Classic Graph Convolutional Network (GCN) model architecture."""

    def __init__(self, *args: Any, **kwargs: Any):
        gcn_config = {
            "operator": GCNConv,
        }
        kwargs["gnn_operator_config"] = {
            **gcn_config,
            **kwargs.get("gnn_operator_config", {}),
        }
        super().__init__(*args, **kwargs)


class GraphSAGE(ModularGNN):
    """GraphSAGE model architecture."""

    def __init__(self, *args: Any, **kwargs: Any):
        sage_config = {
            "operator": SAGEConv,
            "aggr": "mean",
        }
        kwargs["gnn_operator_config"] = {
            **sage_config,
            **kwargs.get("gnn_operator_config", {}),
        }
        super().__init__(*args, **kwargs)


class PNA(ModularGNN):
    """GCN architecture with multiple aggregators (Principle Neighborhood
    Aggregation).
    """

    def __init__(self, deg: torch.Tensor, *args: Any, **kwargs: Any):
        pna_config = {
            "operator": PNAConv,
            "aggregators": ["mean", "min", "max", "std"],
            "scalers": ["identity", "amplification", "attenuation"],
            "deg": deg,
            "towers": 2,
            "divide_input": False,
        }
        kwargs["gnn_operator_config"] = {
            **pna_config,
            **kwargs.get("gnn_operator_config", {}),
        }
        super().__init__(*args, **kwargs)


class GATv2(ModularGNN):
    """Graph Attention Network (GAT) model architecture using the V2 operator
    using dynamic attention.

    The dimensions should look like:
        # create graph convolution layers
        for i in range(gnn_layers):
            in_channels = in_size if i == 0 else heads * embedding_size
            self.convs.append(GATv2Conv(in_channels, embedding_size, heads))
            self.norms.append(GraphNorm(heads * embedding_size))
    """

    def __init__(self, heads: int, *args: Any, **kwargs: Any):
        gat_config = {
            "operator": GATv2Conv,
            "heads": heads,
        }
        kwargs["gnn_operator_config"] = {
            **gat_config,
            **kwargs.get("gnn_operator_config", {}),
        }
        super().__init__(*args, **kwargs)


class UniMPTransformer(ModularGNN):
    """UniMP Transformer model architecture.

    The dimensions should be exactly the same as the GATv2 model, especially as
    we do not pass an argument for concat. It will default to `true` and provide
    and output size of `heads * embedding_size`.
    """

    def __init__(self, heads: int, *args: Any, **kwargs: Any):
        transformer_config = {
            "operator": TransformerConv,
            "heads": heads,
        }
        kwargs["gnn_operator_config"] = {
            **transformer_config,
            **kwargs.get("gnn_operator_config", {}),
        }
        super().__init__(*args, **kwargs)


class DeeperGCN(nn.Module):
    """DeeperGCN model architecture.Does not inherit from the ModularGNN class
    as it uses the DeepGCNLayer as well as a different initial layer and
    unshared params and activations. Additionally, skip connections are not
    coded into the forward pass because they're inherent to the architecture and
    handled by the DeepGCNLayer class.

    Args:
        in_size: The input size of the DeeperGCN model.
        embedding_size: The size of the embedding dimension.
        out_channels: The number of output channels.
        gnn_layers: The number of convolutional layers.
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
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.convs = nn.ModuleList()
        self.linears = nn.ModuleList()

        # Create deeper layers
        self.convs.append(
            self.get_deepergcn_layers(
                in_channels=in_size, out_channels=embedding_size, layer_number=1
            )
        )
        for layer_number in range(2, gnn_layers + 1):
            self.convs.append(
                self.get_deepergcn_layers(
                    in_channels=embedding_size,
                    out_channels=embedding_size,
                    layer_number=layer_number,
                )
            )

        # Create linear layers
        self.linears.append(Linear(embedding_size, embedding_size))
        for _ in range(linear_layers - 2):
            self.linears.append(Linear(embedding_size, embedding_size))
        self.linears.append(Linear(embedding_size, out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network.

        Args:
            x: The input tensor.
            edge_index: The edge index tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        print(f"Input shape: {x.shape}")
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            print(f"After conv {i+1} shape: {x.shape}")

        for i, linear_layer in enumerate(self.linears[:-1]):
            x = self.nonfunctional_activation(self.activation)()(linear_layer(x))
            if isinstance(self.dropout_rate, float):
                assert self.dropout_rate is not None
                x = F.dropout(x, p=self.dropout_rate)
            print(f"After linear {i+1} shape: {x.shape}")

        x = self.linears[-1](x)
        print(f"Final output shape: {x.shape}")
        return x

    def get_deepergcn_layers(
        self,
        in_channels: int,
        out_channels: int,
        layer_number: int,
    ) -> DeepGCNLayer:
        """Create DeeperGCN layers"""
        conv = GENConv(
            in_channels=in_channels,
            out_channels=out_channels,
            aggr="softmax",
            t=1.0,
            learn_t=True,
            num_layers=2,
            norm="layer",
        )
        norm = LayerNorm(out_channels, affine=True)
        act = self.nonfunctional_activation(self.activation)()
        return DeepGCNLayer(
            conv=conv,
            norm=norm,
            act=act,
            block="res+" if layer_number > 1 else "plain",
            dropout=0.1,
            ckpt_grad=1 if layer_number == 1 else (layer_number % 3),
        )

    @staticmethod
    def nonfunctional_activation(activation: str) -> Callable:
        """Defines the activation function according to the given string"""
        activations = {
            "gelu": torch.nn.GELU,
            "leakyrelu": torch.nn.LeakyReLU,
            "relu": torch.nn.ReLU,
        }
        try:
            return activations[activation]
        except ValueError as error:
            raise ValueError(
                "Invalid activation function. Supported: relu, leakyrelu, gelu"
            ) from error
