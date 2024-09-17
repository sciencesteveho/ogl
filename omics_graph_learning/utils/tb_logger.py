#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Custom TensorBoard logger to automatically log useful metrics such as model
weights and gradient histograms."""


from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter
import torch_geometric  # type: ignore


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
        Log a graph of the GNN architecture.
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
    ) -> None:
        """Log model graph to TensorBoard using a toy input."""
        device = next(model.parameters()).device

        # get toy input tensors
        input_x, edge_index, mask = self.get_toy_input(data)

        # ensure dummy inputs are on the correct device
        for tensor in (input_x, edge_index, mask):
            tensor.to(device)

        self.writer.add_graph(model, (input_x, edge_index, mask))

    def get_toy_input(
        self, data: torch_geometric.data.Data
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create toy input tensors for model graph logging."""
        toy_x = torch.zeros((self.num_toy_nodes, data.x.shape[1]))
        toy_edge_index = torch.randint(0, self.num_toy_nodes, (2, self.num_toy_edges))
        toy_mask = torch.ones(self.num_toy_nodes, dtype=torch.bool)
        return toy_x, toy_edge_index, toy_mask

    def close(self) -> None:
        """Close TensorBoard writer."""
        self.writer.close()
