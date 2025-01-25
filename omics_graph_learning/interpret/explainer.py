#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Implementation of PGExplainer.

In contrast to perturbations, PGExplainer learns a generalizable strategy on
graph embeddings and applies them to potential subgraphs. The explanations focus
on subgraph structures rather than specific features as with perturbations.
"""

from typing import List

import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch_geometric.data import Data  # type: ignore
from torch_geometric.explain import Explainer  # type: ignore
from torch_geometric.explain import PGExplainer  # type: ignore
from torch_geometric.explain.config import ExplanationType  # type: ignore
from torch_geometric.explain.config import MaskType  # type: ignore
from torch_geometric.explain.config import ModelMode  # type: ignore
from torch_geometric.loader import DataLoader  # type: ignore
from torch_geometric.loader import NeighborLoader  # type: ignore


def build_explainer(model: nn.Module, epochs: int = 30, lr: float = 0.003) -> Explainer:
    """Build a PGExplainer instance for explaning dataset phenomena."""
    return Explainer(
        model=model,
        algorithm=PGExplainer(epochs=epochs, lr=lr),
        explanation_type=ExplanationType.phenomenon,
        model_mode=ModelMode.regression,
        node_mask_type=MaskType.attributes,
        edge_mask_type=MaskType.object,
    )


def train_explainer(
    model: nn.Module,
    sampled_indices: torch.Tensor,
    epochs: int = 30,
    lr: float = 0.003,
    batch_size: int = 32,
) -> None:
    """Train the explainer on a sampled subset of nodes."""
    # initialize the explainer
    explainer = build_explainer(model=model, epochs=epochs, lr=lr)

    sampled_dataset = torch.utils.data.TensorDataset(sampled_indices)
    loader: DataLoader = DataLoader(
        sampled_dataset, batch_size=batch_size, shuffle=True
    )

    # train explainer
    for epoch in range(epochs):
        trainloss = 0.0
        print(f"Epoch {epoch+1}/{epochs} started.")
        for batch in loader:
            loss = explainer.algorithm.train(
                epoch=epoch,
                model=model,
                x=batch.x,
                edge_index=batch.edge_index,
                y=batch.y,
                batch=batch.batch,
            )
            trainloss += loss
        print(f"Train loss: {trainloss:.4f}")
    print("Training completed.")


def generate_explanations(
    explainer: Explainer,
    data: Data,
    target_indices: torch.Tensor,
    batch_size: int = 32,
) -> List[torch.Tensor]:
    """Generate explanations for target nodes in batches."""
    explainer.algorithm.eval()
    all_explanations = []
    loader: DataLoader = DataLoader(target_indices, batch_size=batch_size)
    for batch in loader:
        explanations = explainer(
            x=data.x,
            edge_index=data.edge_index,
            node_idx=batch,
        )
        all_explanations.extend(explanations)
    return all_explanations
