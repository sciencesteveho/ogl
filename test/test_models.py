#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Test the flexible GNN module and submodules. Ensure that dimensions are
consistent, operators are proper, and layers are correctly implemented."""


from typing import Any, Dict, List, Type, Union

from omics_graph_learning.models import DeeperGCN
from omics_graph_learning.models import GATv2
from omics_graph_learning.models import GCN
from omics_graph_learning.models import GraphSAGE
from omics_graph_learning.models import ModularGNN
from omics_graph_learning.models import PNA
from omics_graph_learning.models import UniMPTransformer
import pytest
import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.data import Data  # type: ignore


@pytest.fixture
def graph_data() -> Data:
    """Create a simple PyG Data object of specified size for testing."""
    num_nodes = 100
    num_features = 10
    num_edges = 500

    data = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    return Data(x=data, edge_index=edge_index)


@pytest.fixture(params=[True, False])
def task_specific_mlp(request) -> bool:
    """Fixture for testing task-specific MLP implementation."""
    return request.param


@pytest.fixture(params=[None, "shared_source", "distinct_source"])
def skip_connection(request) -> str:
    """Fixture for testing residual connections"""
    return request.param


@pytest.fixture(params=[1, 2, 3])
def layers(request) -> int:
    """Fixture for testing differing # of layers."""
    return request.param


@pytest.fixture(params=["relu", "leakyrelu", "gelu"])
def activation(request) -> str:
    """Fixture for testing different activation functions."""
    return request.param


@pytest.fixture
def model_params(
    layers: int,
    activation: str,
    task_specific_mlp: bool,
    skip_connection: Union[str, None],
) -> Dict[str, Any]:
    """Define simple/ default GNN model params."""
    return {
        "in_size": 10,  # commensurate with num_features
        "embedding_size": 128,
        "out_channels": 1,
        "gnn_layers": layers,
        "shared_mlp_layers": layers,
        "activation": activation,
        "task_specific_mlp": task_specific_mlp,
        "skip_connection": skip_connection,
    }


@pytest.fixture(
    params=[
        (GCN, None),
        (GraphSAGE, None),
        (GATv2, 1),
        (GATv2, 2),
        (GATv2, 4),
        (UniMPTransformer, 1),
        (UniMPTransformer, 2),
        (UniMPTransformer, 4),
        (PNA, None),
    ]
)
def model(
    request: pytest.FixtureRequest,
    model_params: Dict[str, Any],
) -> ModularGNN:
    """Fixture for defining GNN models."""
    model_class, heads = request.param
    base_params = model_params.copy()

    if model_class == PNA:
        deg = torch.ones(10)  # dummy in-degree histogram tensor
        base_params["deg"] = deg
    elif model_class in [GATv2, UniMPTransformer]:
        base_params["heads"] = heads

    return model_class(**base_params)


@pytest.fixture
def deepergcn(model_params: Dict[str, Any]) -> DeeperGCN:
    """Fixture for defining DeeperGCN model."""
    return DeeperGCN(
        in_size=model_params["in_size"],
        embedding_size=model_params["embedding_size"],
        out_channels=model_params["out_channels"],
        gnn_layers=model_params["gnn_layers"],
        linear_layers=model_params["shared_mlp_layers"],
        activation=model_params["activation"],
        dropout_rate=model_params.get("dropout_rate", 0.5),
    )


def test_model_output(model: Union[ModularGNN, DeeperGCN], graph_data: Data) -> None:
    """Test that the size and shape of the forward pass is correct and check
    that a gradient can be computed.

    Raises:
        AssertionError: If the model output doesn't meet the expected criteria.
    """
    # test with possible node mask for task-specific MLP models
    if isinstance(model, ModularGNN):
        node_mask = (
            torch.randint(0, 2, (graph_data.num_nodes,)).bool()
            if model.task_specific_mlp
            else None
        )
        out = model(graph_data.x, graph_data.edge_index, node_mask)
    else:  # DeeperGCN does not require node mask
        out = model(graph_data.x, graph_data.edge_index)

    assert out.numel() > 0, "Output tensor is empty"
    assert not torch.isnan(out).any(), "Output contains NaN values"

    # check gradients
    out.sum().backward()
    for param in model.parameters():
        assert param.grad is not None, "Gradients are not properly computed"


def test_modular_model_structure(
    model: ModularGNN, model_params: Dict[str, Any]
) -> None:
    """Test the basic structure of the model: num_norms should match num_layers,
    task heads should be specified, number of conv layers, etc.

    Raises:
        AssertionError: If the model structure doesn't meet the expected from
        the config.
    """
    # check normalization layers
    assert (
        len(model.norms) == model_params["gnn_layers"]
    ), f"Incorrect number of normalization layers. \
        Expected {model_params['gnn_layers']}, got {len(model.norms)}"

    # check expected linear layers
    expected_linear_layers = max(
        0,
        model_params["shared_mlp_layers"]
        - (2 if model_params["task_specific_mlp"] else 1),
    )
    expected_task_layers = 2 if model_params["task_specific_mlp"] else 1

    actual_linear_layers = len(model.linears)
    actual_task_layers = (
        len(model.task_head) if isinstance(model.task_head, nn.ModuleList) else 1
    )

    assert (
        actual_linear_layers == expected_linear_layers
    ), f"Incorrect number of shared linear layers. \
        Expected {expected_linear_layers}, got {actual_linear_layers}"
    assert (
        actual_task_layers == expected_task_layers
    ), f"Incorrect number of task-specific layers. \
        Expected {expected_task_layers}, got {actual_task_layers}"

    # check convolutional layers
    assert (
        len(model.convs) == model_params["gnn_layers"]
    ), f"Incorrect number of convolutional layers. \
        Expected {model_params['gnn_layers']}, got {len(model.convs)}"

    # check residual connections
    if model_params["skip_connection"] is not None:
        assert (
            model.linear_projection is not None
        ), "Skip connection not properly implemented"
        if model_params["skip_connection"] == "shared_source":
            assert isinstance(
                model.linear_projection, torch.nn.Linear
            ), f"Incorrect skip connection type. \
                Expected torch.nn.Linear, got {type(model.linear_projection)}"
        elif model_params["skip_connection"] == "distinct_source":
            assert isinstance(
                model.linear_projection, torch.nn.ModuleList
            ), f"Incorrect skip connection type. \
                Expected torch.nn.ModuleList, got {type(model.linear_projection)}"

    # check if the first convolutional input dimension is correct
    expected_in_channels = model_params["in_size"]
    actual_in_channels = model.convs[0].in_channels
    assert (
        actual_in_channels == expected_in_channels
    ), f"Incorrect input dimension for first convolutional layer. \
        Expected {expected_in_channels}, got {actual_in_channels}"

    # check if the first convolutional output dimension is correct
    expected_out_channels = model_params["embedding_size"]
    actual_out_channels = model.convs[0].out_channels
    assert (
        actual_out_channels == expected_out_channels
    ), f"Incorrect output dimension for first convolutional layer. \
        Expected {expected_out_channels}, got {actual_out_channels}"


@pytest.mark.parametrize("model_class", [GCN, GraphSAGE])
def test_basic_models(
    model_class: Type[ModularGNN], graph_data: Data, model_params: Dict[str, Any]
) -> None:
    """Test basic GNN models (GCN, GraphSAGE) that have minimal customization.

    Raises:
        AssertionError: If the model doesn't pass the output or structure tests.
    """
    model = model_class(**model_params)
    test_model_output(model, graph_data)
    test_modular_model_structure(model, model_params)


@pytest.fixture(params=[1, 2, 4])
def heads(request) -> int:
    """Fixture for testing number of attention heads."""
    return request.param


@pytest.mark.parametrize("model_class", [GATv2, UniMPTransformer])
def test_attention_models(
    model_class: Type[ModularGNN],
    graph_data: Data,
    model_params: Dict[str, Any],
    heads: int,
) -> None:
    """Test GNN models with attention mechanisms (GATv2, UniMPTransformer) which
    requires additional size calculations.

    Raises:
        AssertionError: If the model doesn't pass the output or structure tests.
    """
    model_params["gnn_operator_config"] = {"heads": heads}
    model = model_class(heads=heads, **model_params)
    test_model_output(model, graph_data)
    test_modular_model_structure(model, model_params)


def test_pna(graph_data: Data, model_params: Dict[str, Any]) -> None:
    """Test the PNA (Principal Neighbourhood Aggregation) model which requires
    in-degree histogram tensor and other parameters.

    Raises:
        AssertionError: If the model doesn't pass the output or structure tests.
    """
    deg: Tensor = torch.ones(10)  # dummy in-degree histogram tensor
    model = PNA(deg=deg, **model_params)
    test_model_output(model, graph_data)
    test_modular_model_structure(model, model_params)


# def test_deeper_gcn_structure(model: DeeperGCN, model_params: Dict[str, Any]) -> None:
#     """Test the structure of the DeeperGCN model.

#     Raises:
#         AssertionError: If the model structure doesn't meet the expected from
#         the config.
#     """
#     assert (
#         len(model.linears) == model_params["shared_mlp_layers"]
#     ), f"Incorrect number of linear layers in DeeperGCN. Expected {model_params['shared_mlp_layers']}, got {len(model.linears)}"
#     assert (
#         len(model.convs) == model_params["gnn_layers"]
#     ), f"Incorrect number of convolutional layers in DeeperGCN. Expected {model_params['gnn_layers']}, got {len(model.convs)}"

#     # Check if the first convolutional layer has the correct input dimension
#     assert (
#         model.convs[0].conv.in_channels == model_params["in_size"]
#     ), f"Incorrect input dimension for first convolutional layer in DeeperGCN. Expected {model_params['in_size']}, got {model.convs[0].conv.in_channels}"

#     # Check if all convolutional layers have the correct output dimension
#     for conv in model.convs:
#         assert (
#             conv.conv.out_channels == model_params["embedding_size"]
#         ), f"Incorrect output dimension for convolutional layer in DeeperGCN. Expected {model_params['embedding_size']}, got {conv.conv.out_channels}"

#     # Check if the last linear layer has the correct output dimension
#     assert (
#         model.linears[-1].out_features == model_params["out_channels"]
#     ), f"Incorrect output dimension for last linear layer in DeeperGCN. Expected {model_params['out_channels']}, got {model.linears[-1].out_features}"


# def test_deeper_gcn(graph_data: Data, model_params: Dict[str, Any]) -> None:
#     """Test the DeeperGCN model.

#     Raises:
#         AssertionError: If the model doesn't pass the output or structure tests.
#     """
#     print("\nTesting DeeperGCN")
#     model = DeeperGCN(
#         in_size=model_params["in_size"],
#         embedding_size=model_params["embedding_size"],
#         out_channels=model_params["out_channels"],
#         gnn_layers=model_params["gnn_layers"],
#         linear_layers=model_params["shared_mlp_layers"],
#         activation=model_params["activation"],
#     )

#     # Print model parameters
#     print("Model parameters:")
#     for name, param in model.named_parameters():
#         print(f"{name}: {param.shape}")

#     test_model_output(model, graph_data)
#     test_deeper_gcn_structure(model, model_params)
