#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Build GNN models using the ModularGNN class from models.py."""


from typing import Any, Dict

import torch
import torch.nn as nn
import torch_geometric  # type: ignore
from torch_geometric.utils import degree  # type: ignore

from models import DeeperGCN
from models import GATv2
from models import GCN
from models import GraphSAGE
from models import MLP
from models import PNA
from models import UniMPTransformer


class GNNArchitectureBuilder:
    """Builder class to construct GNN models."""

    def __init__(self) -> None:
        """Instantiate the GNNArchitectureBuilder class."""
        self.model_constructors = {
            "GCN": GCN,
            "GraphSAGE": GraphSAGE,
            "PNA": PNA,
            "GAT": GATv2,
            "UniMPTransformer": UniMPTransformer,
            "DeeperGCN": DeeperGCN,
            "MLP": MLP,
        }

    def build(self, model: str, **kwargs) -> nn.Module:
        """Construct GNN."""
        self._validate_model(model)
        builder_method = getattr(self, f"_build_{model.lower()}")
        return builder_method(**kwargs)

    def _validate_model(self, model: str) -> None:
        """Ensure the model type is valid."""
        if model not in self.model_constructors:
            raise ValueError(
                f"Invalid model type: {model}. Choose from {list(self.model_constructors.keys())}"
            )

    def _build_deepergcn(self, **kwargs) -> nn.Module:
        """DeeperGCN builder, not part of ModularGNN."""
        deepergcn_kwargs = {
            "in_size": kwargs["in_size"],
            "embedding_size": kwargs["embedding_size"],
            "out_channels": kwargs["out_channels"],
            "gnn_layers": kwargs["gnn_layers"],
            "linear_layers": kwargs["shared_mlp_layers"],
            "activation": kwargs["activation"],
            "dropout_rate": kwargs["dropout_rate"],
        }
        return self.model_constructors["DeeperGCN"](**deepergcn_kwargs)

    def _build_mlp(self, **kwargs) -> nn.Module:
        """MLP builder, not part of ModularGNN."""
        mlp_kwargs = {
            "in_size": kwargs["in_size"],
            "embedding_size": kwargs["embedding_size"],
            "out_channels": kwargs["out_channels"],
            "activation": kwargs["activation"],
        }
        return self.model_constructors["MLP"](**mlp_kwargs)

    def _build_modular_gnn(self, model: str, **kwargs) -> nn.Module:
        """Build GNN models that inherit from ModularGNN."""
        gnn_kwargs = {
            "activation": kwargs["activation"],
            "in_size": kwargs["in_size"],
            "embedding_size": kwargs["embedding_size"],
            "out_channels": kwargs["out_channels"],
            "gnn_layers": kwargs["gnn_layers"],
            "shared_mlp_layers": kwargs["shared_mlp_layers"],
            "dropout_rate": kwargs["dropout_rate"],
            "residual": kwargs["residual"],
            "attention_task_head": kwargs["attention_task_head"],
            "gnn_operator_config": {},
        }

        # handle attention-based models
        if model in {"GAT", "UniMPTransformer"}:
            self.attention_args(model, gnn_kwargs, kwargs["heads"])

        # handle extra args for principle neighborhood aggregation
        elif model == "PNA":
            self.pna_args(gnn_kwargs, kwargs["train_dataset"])

        return self.model_constructors[model](**gnn_kwargs)

    def attention_args(
        self, model: str, gnn_kwargs: Dict[str, Any], heads: int
    ) -> None:
        """Set heads for attention-based models."""
        if not heads:
            raise ValueError(
                f"Attention-based model {model} requires the 'heads' parameter to be set."
            )
        gnn_kwargs["heads"] = heads
        gnn_kwargs["gnn_operator_config"] = {"heads": heads}

    def pna_args(
        self, gnn_kwargs: Dict[str, Any], train_dataset: torch_geometric.data.DataLoader
    ) -> None:
        """Serialize the in-degree histogram for PNA."""
        if not train_dataset:
            raise ValueError(
                "PNA requires `train_dataset` to compute the in-degree histogram."
            )
        gnn_kwargs["deg"] = self._compute_pna_histogram_tensor(train_dataset)

    def _build_gcn(self, **kwargs) -> nn.Module:
        """Build GCN"""
        return self._build_modular_gnn("GCN", **kwargs)

    def _build_graphsage(self, **kwargs) -> nn.Module:
        """Build GraphSAGE"""
        return self._build_modular_gnn("GraphSAGE", **kwargs)

    def _build_pna(self, **kwargs) -> nn.Module:
        """Build PNA"""
        return self._build_modular_gnn("PNA", **kwargs)

    def _build_gat(self, **kwargs) -> nn.Module:
        """Build GATv2"""
        return self._build_modular_gnn("GAT", **kwargs)

    def _build_unimptransformer(self, **kwargs) -> nn.Module:
        """Build UniMPTransformer"""
        return self._build_modular_gnn("UniMPTransformer", **kwargs)

    @staticmethod
    def _compute_pna_histogram_tensor(
        train_dataset: torch_geometric.data.DataLoader,
    ) -> torch.Tensor:
        """Computes the maximum in-degree in the training data and the in-degree
        histogram tensor for principle neighborhood aggregation (PNA).

        Adapted from pytorch geometric's PNA example:
        https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pna.py
        """
        # compute the maximum in-degree in the training data
        max_degree = -1
        for data in train_dataset:
            computed_degree = degree(
                data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long
            )
            max_degree = max(max_degree, int(computed_degree.max()))

        # compute the in-degree histogram tensor
        deg = torch.zeros(max_degree + 1, dtype=torch.long)
        for data in train_dataset:
            computed_degree = degree(
                data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long
            )
            deg += torch.bincount(computed_degree, minlength=deg.numel())
        return deg


def build_gnn_architecture(**kwargs) -> nn.Module:
    """Pass kwargs to the builder, with a quick check to ensure PNA models pass
    the dataloader for extra calculations.
    """
    if kwargs.get("model") == "PNA" and "train_dataset" not in kwargs:
        raise ValueError("PNA requires the `train_dataset` parameter to be set.")
    return GNNArchitectureBuilder().build(**kwargs)
