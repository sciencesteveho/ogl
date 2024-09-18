#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Custom TensorBoard logger to automatically log useful metrics such as model
weights and gradient histograms."""


from pathlib import Path
from typing import Any, Dict

import torch
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
import torch_geometric  # type: ignore
from torch_geometric.utils import subgraph  # type: ignore


class TensorBoardLogger:
    """Class to handle logging of metrics, hyperparameters, and model weights to
    TensorBoard.

    Methods
    --------
    log_hyperparameters:
        Log hyperparameters.
    log_metrics:
        Log a scalar metric.
    log_gradients:
        Log gradients of model parameters.
    log_learning_rate:
        Log learning rate.
    log_weights:
        Log model weights.
    log_model_graph:
        Log a graph of the GNN architecture using a sampled subgraph from the
        data.
    close:
        Close TensorBoard writer.

    Examples:
    --------
    # Call the logger and pass it to the GNN trainer. Instantiation will
    automatically create a log directory.
    >>> tb_logger = TensorBoardLogger(log_dir="logs")
    >>> trainer = GNNTrainer(
            model=model,
            device=device,
            data=data,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
            tb_logger=tb_logger,
        )
    """

    num_toy_nodes = 100
    num_toy_edges = 200

    def __init__(self, log_dir: Path) -> None:
        """Initialize TensorBoard writer."""
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def log_hyperparameters(self, hparams: Dict[str, Any]) -> None:
        """Log hyperparameters to TensorBoard."""
        self.writer.add_hparams(hparams, {})

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics to TensorBoard."""
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)

    def log_gradients(self, model: torch.nn.Module, step: int) -> None:
        """Log gradients of model parameters to TensorBoard."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f"Gradients/{name}", param.grad, step)

    def log_learning_rate(self, optimizer: Optimizer, step: int) -> None:
        """Log learning rate to TensorBoard."""
        for idx, param_group in enumerate(optimizer.param_groups):
            lr = param_group.get("lr", None)
            if lr is not None:
                self.writer.add_scalar(f"Learning_Rate/group_{idx}", lr, step)

    def log_weights(self, model: torch.nn.Module, step: int) -> None:
        """Log model weights to TensorBoard."""
        for name, param in model.named_parameters():
            self.writer.add_histogram(f"Weights/{name}", param, step)

    def log_model_graph(
        self,
        model: torch.nn.Module,
        data: torch_geometric.data.Data,
        device: torch.device,
        sample_size: int = 100,
    ) -> None:
        """Log model graph to TensorBoard using a sampled subgraph."""
        # sample a connected subgraph
        try:
            # ensure the graph is connected; if not, sample from the largest component
            if data.num_nodes < sample_size:
                sample_size = data.num_nodes

            # randomly sample a seed node and extract a subgraph
            seed_nodes = torch.randint(0, data.num_nodes, (1,)).squeeze()
            sub_nodes = torch_geometric.utils.k_hop_subgraph(
                node_idx=seed_nodes,
                num_hops=10,
                edge_index=data.edge_index,
                num_nodes=data.num_nodes,
                flow="source_to_target",
            )[0]

            if sub_nodes.numel() > sample_size:
                sub_nodes = sub_nodes[torch.randperm(sub_nodes.numel())[:sample_size]]

            # get subgraph
            sub_edge_index, _ = subgraph(
                subset=sub_nodes,
                edge_index=data.edge_index,
                relabel_nodes=True,
                num_nodes=data.num_nodes,
            )

            # get subgraph features and mask
            sub_x = data.x[sub_nodes]
            sub_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            sub_mask[sub_nodes] = True

            # move to gpu
            sub_x = sub_x.to(device)
            sub_edge_index = sub_edge_index.to(device)
            sub_mask = sub_mask.to(device)

            # log the graph!
            self.writer.add_graph(model, (sub_x, sub_edge_index, sub_mask))
            print(
                f"Logged subgraph with {sub_nodes.numel()}"
                f"nodes and {sub_edge_index.size(1)} edges."
            )

        except Exception as e:
            print(f"Failed to log model graph: {e}")
            raise e

    def close(self) -> None:
        """Close TensorBoard writer."""
        self.writer.close()
