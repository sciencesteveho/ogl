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

from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

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


class TaskSpecificMLPs(nn.Module):
    """Class construct to manage task specific MLPs for each output head.

    We use a ModuleDict to store the task specific MLPs, so that we only make
    MLPs as needed to save on memory. The MLPs themselves are 2 layer MLPs with
    bottleneck layers and L1 regularization. The bottleneck layers help to
    compress the task-specific representation and L1 regularization helps to
    reduce overfitting while promoting sparsity and feature importance.

    Attributes:
        in_size: The input size of the task specific MLP. This will likely
        be the embedding size of the model.
        out_size: The output size of the task specific MLP. This will likely be
        1 for single node regression.
        activation: The activation function to use.
        task_specific_mlps: The task specific MLPs.
        adjusted_in_size: The adjusted input size for the task specific
        bottleneck_factor: The factor to reduce the input size for the
        bottleneck
        l1_lambda: The lambda value for L1 regularization.

    Methods
    --------
    get_mlp(key: str) -> nn.Sequential:
        Get the task specific MLP for the given index. If the MLP does not
        exist, it will be created as needed to manage memory.
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        activation: str,
        heads: Optional[int] = None,
        bottleneck_factor: float = 0.5,
        l1_lambda: float = 0.0001,
    ) -> None:
        """Initialize the task specific MLP storage class"""
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.activation = activation
        self.bottleneck_factor = bottleneck_factor
        self.l1_lambda = l1_lambda

        self.task_specific_mlps = nn.ModuleDict()
        self.adjusted_in_size = in_size * heads if heads else in_size

    def get_mlp(self, key: str) -> nn.Sequential:
        """Get the task specific MLP for the given index. If the MLP does not
        exist, create it.
        """
        if key not in self.task_specific_mlps:
            self.task_specific_mlps[key] = self.construct_task_specific_mlp()
        return self.task_specific_mlps[key]

    def forward(
        self, x: torch.Tensor, keys: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the task specific MLPs to the input tensor."""
        outputs = []
        l1_reg = torch.tensor(0.0, device=x.device, requires_grad=False)

        for key, x_i in zip(keys, x):
            mlp = self.get_mlp(key)
            x_i = x_i.to(mlp[0].weight.device)  # ensure device compatibility
            output = mlp(x_i.unsqueeze(0))
            outputs.append(output)

            # L1 regularization
            l1_reg += sum(p.abs().sum() for p in mlp.parameters())

        combined_output = torch.cat(outputs)
        return combined_output, self.l1_lambda * l1_reg

    def construct_task_specific_mlp(
        self,
    ) -> nn.Module:
        """Create task specific MLP for each output head. We consider the
        shared-MLP layers as the input, thus the function returns 2 linear
        layers with activation per output head and uses a bottleneck layer.
        """
        bottleneck_size = max(
            int(self.adjusted_in_size * self.bottleneck_factor), self.out_size
        )

        return nn.Sequential(
            nn.Linear(self.adjusted_in_size, bottleneck_size),
            self.get_activation_module(self.activation),
            nn.Linear(bottleneck_size, self.out_size),
        )

    @staticmethod
    def get_activation_module(activation: str) -> nn.Module:
        """Returns the PyTorch module for the given activation function."""
        activations = {
            "gelu": nn.GELU(),
            "leakyrelu": nn.LeakyReLU(),
            "relu": nn.ReLU(),
        }
        try:
            return activations[activation]
        except KeyError as error:
            raise ValueError(
                "Invalid activation function. Supported: relu, leakyrelu, gelu"
            ) from error


class MLP(nn.Module):
    """Simple three layer MLP model architecture.

    Attributes:
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
        self.activation = get_activation_function(activation)

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
        linear_projection: The linear projection for residual connections.
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
        # heads: Optional[int] = None,
        dropout_rate: Optional[float] = None,
        residual: Optional[str] = None,
        task_specific_mlp: bool = False,
    ):
        """Initialize the model"""
        super().__init__()
        self.embedding_size = embedding_size
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
        self.residual = residual
        self.shared_mlp_layers = shared_mlp_layers
        self.task_specific_mlp = task_specific_mlp
        self.activation_name = activation
        self.activation = get_activation_function(activation)

        # add attention heads if specified in config
        self.heads = gnn_operator_config.get("heads")

        # for debugging
        print(
            f"ModularGNN initialized with in_size: {in_size}, embedding_size: {embedding_size}, out_channels: {out_channels}, heads: {self.heads}"
        )

        # initialize GNN layers and batch normalization
        self.convs = self._create_gnn_layers(
            in_size=in_size,
            embedding_size=embedding_size,
            layers=gnn_layers,
            config=gnn_operator_config,
        )

        # initialize normalization layers
        self.norms = self._get_normalization_layers(
            layers=gnn_layers, embedding_size=embedding_size, heads=self.heads
        )

        # initialize linear layers (task-specific MLP)
        self.linears = self._get_linear_layers(
            in_size=embedding_size,
            layers=shared_mlp_layers,
            heads=self.heads,
        )

        # initialize task head(s)
        if self.task_specific_mlp:
            self.task_specific_mlps = TaskSpecificMLPs(
                in_size=in_size,
                out_size=out_channels,
                activation=self.activation_name,
                heads=self.heads,
            )
        else:
            self.task_head = self._general_task_head(
                in_size=embedding_size, out_size=out_channels
            )

        # create linear projection for residual connections
        self.linear_projection = None
        if self.residual:
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
        regression_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the neural network.

        Args:
            x: The input tensor.
            edge_index: The edge index tensor.
            regression_mask: Boolean mask of shape (num_nodes,) indicating which
            nodes should be regressed. True for nodes to be regressed, False
            otherwise.

        Returns:
            torch.Tensor: The output tensor.
        """
        # for debugging
        print(f"Input tensor shape in ModularGNN forward: {x.shape}")

        h1 = x  # save input for residual connections

        # graph convolutions with normalization and optional residual connections.
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

            # check for NaN values
            if torch.isnan(x).any():
                print(f"Warning: NaN detected in layer {i}")

        # for debugging
        # after the convolutional layers
        print(f"Shape after convolutions: {x.shape}")

        # shared linear layers
        for linear_layer in self.linears:
            x = self.activation(linear_layer(x))
            if isinstance(self.dropout_rate, float):
                assert self.dropout_rate is not None
                x = F.dropout(x, p=self.dropout_rate)

        # for debugging
        # After the shared linear layers
        print(f"Shape after shared linear layers: {x.shape}")

        # task-specific MLPs or general task head
        if self.task_specific_mlp:
            return self.task_specific_forward(x=x, regression_mask=regression_mask)
        else:
            return self.general_task_forward(
                x=x, regression_mask=regression_mask
            ), torch.tensor(0.0, device=x.device)

    def task_specific_forward(
        self,
        x: torch.Tensor,
        regression_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use task specific MLPs for each output head. Because the number of
        targets changes for the train/test/val splits, MLPs are dynamically
        created for each split during the forward pass. One MLP is built per
        regression target.
        """
        # check that task head is None, as it should be dynamically created
        if not isinstance(self.task_specific_mlps, nn.Module):
            raise ValueError("Task specific MLPs must be a ModuleDict.")

        # check regression mask fidelity
        ensure_mask_fidelity(x, regression_mask)

        # get indices for node regressions
        regression_indices = torch.where(regression_mask)[0]

        # keys for task specific MLPs
        keys = [str(idx.item()) for idx in regression_indices]

        # apply task specific MLPs to regression targets
        out, l1_reg = self.task_specific_mlps(x[regression_indices], keys)

        # for debugging
        print(f"Output shape from task_specific_forward: {out.shape}")

        return (
            output_tensor(x=x, out=out, regression_indices=regression_indices),
            l1_reg,
        )

    def general_task_forward(
        self, x: torch.Tensor, regression_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass using a general task head, which is a single linear
        layer to finish the MLP.
        """
        # check that task head is a linear layer
        if not isinstance(self.task_head, nn.Linear):
            raise ValueError("General task head must be a linear layer.")

        # check regression mask fidelity
        ensure_mask_fidelity(x, regression_mask)

        # process only regression nodes
        regression_indices = torch.where(regression_mask)[0]
        out = self.task_head(x[regression_indices])

        return output_tensor(x=x, out=out, regression_indices=regression_indices)

    def _general_task_head(
        self,
        in_size: int,
        out_size: int,
    ) -> nn.Linear:
        """Create a general task head for the model, a single linear layer that
        regresses to the output size.
        """
        if self.shared_mlp_layers == 1:
            in_size = in_size * self.heads if self.heads else in_size
        return nn.Linear(in_size, out_size)

    def _residuals(
        self, in_size: int, embedding_size: int
    ) -> Union[nn.ModuleList, nn.Linear]:
        """Create residual connection linear projections. The skip connection
        tpye can be specificed as shared_source (from the input) or
        distinct_source (from the output of each layer).
        """
        if self.residual == "shared_source":
            return nn.Linear(in_size, embedding_size, bias=False)
        elif self.residual == "distinct_source":
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
        if self.task_specific_mlp > 1:
            layers -= 1
        return self._linear_module(
            self._linear_layer_dimensions(in_size, layers, heads)
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


class DeeperGCN(nn.Module):
    """DeeperGCN model architecture.Does not inherit from the ModularGNN class
    as it uses the DeepGCNLayer as well as a different initial layer and
    unshared params and activations. residual connections are not coded into the
    forward pass because they're inherent to the architecture and handled by the
    DeepGCNLayer class.

    Args:
        in_size: The input size of the DeeperGCN model.
        embedding_size: The size of the embedding dimension.
        out_channels: The number of output channels.
        gnn_layers: The number of convolutional layers.
        linear_layers: The number of linear layers.
        activation: The activation function.
        dropout_rate: The dropout rate.
        task_specific_mlp: Boolean indicating if task specific MLPs are used.
        num_targets: The number of output targets.
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
        task_specific_mlp: bool = False,
    ):
        """Initialize the model"""
        super().__init__()
        self.dropout_rate = dropout_rate
        self.task_specific_mlp = task_specific_mlp
        self.activation_name = activation

        self.convs = nn.ModuleList()
        self.linears = nn.ModuleList()
        self.activation = get_activation_function(activation)
        self.layer_act = self.nonfunctional_activation(activation)

        # create deeper layers
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

        # create linear layers
        self.linears.append(Linear(embedding_size, embedding_size))
        for _ in range(linear_layers - 2):
            self.linears.append(Linear(embedding_size, embedding_size))

        # task-specific MLP or general task head
        if task_specific_mlp:
            self.task_specific_mlps = TaskSpecificMLPs(
                in_size=embedding_size,
                out_size=out_channels,
                activation=self.activation_name,
            )
        else:
            self.task_head = Linear(embedding_size, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        regression_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the neural network.

        Args:
            x: The input tensor.
            edge_index: The edge index tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # graph convolutions
        for conv in self.convs:
            x = conv(x, edge_index)

        # shared linear layers
        for linear_layer in self.linears:
            x = self.activation(linear_layer(x))
            if isinstance(self.dropout_rate, float):
                assert self.dropout_rate is not None
                x = F.dropout(x, p=self.dropout_rate)

        # apply task head(s)
        if self.task_specific_mlp:
            return self.task_specific_forward(x=x, regression_mask=regression_mask)
        else:
            return self.general_task_forward(
                x=x, regression_mask=regression_mask
            ), torch.tensor(0.0, device=x.device)

    def task_specific_forward(
        self,
        x: torch.Tensor,
        regression_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass using task specific MLPs for each output head."""
        # check regression mask fidelity
        ensure_mask_fidelity(x, regression_mask)

        # get indices for node regressions
        regression_indices = torch.where(regression_mask)[0]

        # keys for task specific MLPs
        keys = [str(idx.item()) for idx in regression_indices]

        # apply task specific MLPs to regression targets
        out, l1_reg = self.task_specific_mlps(x[regression_indices], keys)

        return (
            output_tensor(x=x, out=out, regression_indices=regression_indices),
            l1_reg,
        )

    def general_task_forward(
        self, x: torch.Tensor, regression_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass using a general task head"""
        # check regression mask fidelity
        ensure_mask_fidelity(x, regression_mask)

        # process only regression nodes
        regression_indices = torch.where(regression_mask)[0]
        out = self.task_head(x[regression_indices])

        return output_tensor(x=x, out=out, regression_indices=regression_indices)

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
        act = self.layer_act()
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

    @staticmethod
    def produce_output_tensor(
        x: torch.Tensor, out: torch.Tensor, regression_indices: torch.Tensor
    ) -> torch.Tensor:
        """Produce the output tensor for the model by create a tensor of"""
        full_out = torch.zeros(x.size(0), device=x.device)
        full_out[regression_indices] = out.squeeze()
        return full_out


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


def get_activation_function(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
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


def output_tensor(
    x: torch.Tensor, out: torch.Tensor, regression_indices: torch.Tensor
) -> torch.Tensor:
    """Produce the output tensor for the model by create a tensor of the
    same shape, but only filling in values for the regression indices.
    """
    full_out = torch.zeros(x.size(0), device=x.device)
    full_out[regression_indices] = out.squeeze()
    return full_out


def ensure_mask_fidelity(x: torch.Tensor, regression_mask: torch.Tensor) -> None:
    """Run a series of checks to ensure that the regression mask is valid."""
    if regression_mask.sum() == 0:
        raise ValueError(
            "Regression mask is empty. No targets specified for regression."
        )

    if regression_mask.dtype != torch.bool:
        raise TypeError("Regression mask must be a boolean tensor.")

    if regression_mask.shape != (x.shape[0],):
        raise ValueError(
            f"Regression mask shape {regression_mask.shape} does not match input shape {x.shape[0]}"
        )
