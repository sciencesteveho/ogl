#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Custom TensorBoard logger to automatically log useful metrics such as model
weights and gradient histograms."""


from pathlib import Path
from typing import Any, Dict

import torch
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter


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

    @torch.no_grad()
    def log_model_graph(
        self,
        model: torch.nn.Module,
        device: torch.device,
        feature_size: int = 39,
        num_dummy_nodes: int = 2,
    ) -> None:
        """Log the model architecture to TensorBoard using toy inputs."""
        try:
            print("Logging model graph.")
            model.eval()

            # create toy node features
            dummy_x = torch.randn(num_dummy_nodes, feature_size).to(device)
            print(f"Dummy node features (x) shape: {dummy_x.shape}")

            # create a simple edge index
            dummy_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).to(
                device
            )
            print(f"Dummy edge_index shape: {dummy_edge_index.shape}")

            # create toy regression mask
            dummy_regression_mask = torch.tensor([True, False], dtype=torch.bool).to(
                device
            )
            print(f"Dummy regression_mask: {dummy_regression_mask}")

            self.writer.add_graph(
                model, (dummy_x, dummy_edge_index, dummy_regression_mask)
            )
            print("Successfully logged the model graph to TensorBoard.")

        except Exception as e:
            print(f"Failed to log model graph: {e}")
            raise e

    def close(self) -> None:
        """Close TensorBoard writer."""
        self.writer.close()
