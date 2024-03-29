#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Test the flexible GNN module"""


from omics_graph_learning.models import GATv2
from omics_graph_learning.models import GCN
from omics_graph_learning.models import GPSTransformer
from omics_graph_learning.models import GraphSAGE
from omics_graph_learning.models import MLP
import pytest
import torch


@pytest.fixture
def input_data():
    # Define input data for testing
    # Modify this based on the input requirements of your models
    return {
        "x": torch.randn(10, 64),  # Example input features
        "edge_index": torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        "batch": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "pe": torch.randn(10, 5),  # Example positional encoding
    }


def test_graphsage(input_data):
    # Test default configuration
    model = GraphSAGE(in_size=64, embedding_size=128, out_channels=32, num_layers=2)
    output = model(input_data["x"], input_data["edge_index"])
    assert output.shape == (10, 32)

    # Test with different number of linear layers
    model = GraphSAGE(
        in_size=64, embedding_size=128, out_channels=32, num_layers=2, lin_layers=4
    )
    output = model(input_data["x"], input_data["edge_index"])
    assert output.shape == (10, 32)


def test_gcn(input_data):
    # Test default configuration
    model = GCN(in_size=64, embedding_size=128, out_channels=32, num_layers=2)
    output = model(input_data["x"], input_data["edge_index"])
    assert output.shape == (10, 32)


def test_gatv2(input_data):
    # Test default configuration
    model = GATv2(
        in_size=64, embedding_size=128, out_channels=32, num_layers=2, heads=2
    )
    output = model(input_data["x"], input_data["edge_index"])
    assert output.shape == (10, 32)


def test_gpstransformer(input_data):
    # Test default configuration
    model = GPSTransformer(
        in_size=64,
        embedding_size=128,
        walk_length=10,
        channels=32,
        pe_dim=5,
        num_layers=2,
    )
    output = model(
        input_data["x"], input_data["pe"], input_data["edge_index"], input_data["batch"]
    )
    assert output.shape == (10, 1)


def test_mlp(input_data):
    # Test default configuration
    model = MLP(in_size=64, embedding_size=128, out_channels=32)
    output = model(input_data["x"], input_data["edge_index"])
    assert output.shape == (10, 32)

    # Test with different input size
    model = MLP(in_size=128, embedding_size=128, out_channels=32)
    output = model(input_data["x"], input_data["edge_index"])
    assert output.shape == (10, 32)


# Add more tests as needed for other models and functionalities
