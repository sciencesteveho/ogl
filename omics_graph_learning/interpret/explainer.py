#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Implementation of GNNxplainer.

In contrast to perturbations, GNNExplainer learns a generalizable strategy on
graph embeddings and applies them to potential subgraphs. The explanations focus
on subgraph structures rather than specific features as with perturbations.
"""

from typing import List

import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch_geometric.data import Data  # type: ignore
from torch_geometric.explain import Explainer  # type: ignore
from torch_geometric.explain import GNNExplainer  # type: ignore
from torch_geometric.explain.config import ExplanationType  # type: ignore
from torch_geometric.utils import k_hop_subgraph  # type: ignore


class MyModelWrapper(nn.Module):
    """Wrap base_model so that GNNExplainer can call it easily."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x, edge_index, **kwargs):
        """Use a dummy mask."""
        mask = torch.ones(edge_index.size(1), dtype=torch.float, device=x.device)
        if "edge_mask" in kwargs:
            mask = kwargs["edge_mask"]
        assert (
            mask.shape[0] == edge_index.shape[1]
        ), "Mask size does not match number of edges."
        return self.base_model(x, edge_index, mask=mask)


def build_explainer(model: nn.Module, epochs: int = 30, lr: float = 0.003) -> Explainer:
    """Build a GNNExplainer instance for explaining a trained GNN model."""
    wrapped_model = MyModelWrapper(model)

    return Explainer(
        model=wrapped_model,
        algorithm=GNNExplainer(epochs=epochs, lr=lr),
        explanation_type=ExplanationType.model,
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="regression",
            task_level="node",
            return_type="raw",
        ),
    )


def generate_explanations(explainer: Explainer, data: Data, index: int) -> Data:
    """Generate explanations for target nodes in batches."""
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=index,
        num_hops=3,
        edge_index=data.edge_index,
        relabel_nodes=True,
    )

    subgraph = Data(
        x=data.x[subset],
        edge_index=edge_index,
        y=data.y[subset],
    )

    subgraph.orig_nodes = subset
    subgraph_target = mapping[0].item()
    device = next(explainer.model.parameters()).device
    subgraph = subgraph.to(device)

    # debugging
    print(f"Target Index: {index}")
    print(f"Subgraph Size: {subgraph.num_nodes}")
    print(f"Edge Index Shape: {subgraph.edge_index.shape}")
    print(f"Subgraph Target (Local Index): {subgraph_target}")

    try:
        explanation = explainer(
            x=subgraph.x,
            edge_index=subgraph.edge_index,
            index=subgraph_target,
        )
    except AssertionError as ae:
        print(f"AssertionError for target index {index}: {ae}")
        return None
    except Exception as e:
        print(f"Error for target index {index}: {e}")
        return None

    explanation.orig_nodes = subset.cpu()  # (Tensor of shape [# subgraph nodes])
    explanation.global_target_node = index

    return explanation
