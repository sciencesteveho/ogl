#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Custom TensorBoard logger to automatically log useful metrics such as model
weights and gradient histograms."""


from collections import defaultdict
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

    # Log hyperparameters and metrics
    >>> tb_logger.log_gradients(model, current_step)
    >>> tb_logger.log_weights(model, current_step)
    >>> tb_logger.log_learning_rate(optimizer, current_step)
    """

    def __init__(self, log_dir: Path) -> None:
        """Initialize TensorBoard writer."""
        self.log_dir = log_dir

        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def log_hyperparameters(self, hparams: Dict[str, Any]) -> None:
        """Log hyperparameters to TensorBoard."""
        self.writer.add_hparams(hparams, {})

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics to TensorBoard."""
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)

    def log_learning_rate(self, optimizer: Optimizer, step: int) -> None:
        """Log learning rate to TensorBoard."""
        for idx, param_group in enumerate(optimizer.param_groups):
            lr = param_group.get("lr", None)
            if lr is not None:
                self.writer.add_scalar(f"Learning_Rate/group_{idx}", lr, step)

    def log_gradient_norms(self, model: torch.nn.Module, step: int) -> None:
        """Log L2 norm of model gradients to TensorBoard."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad, p=2)
                self.writer.add_scalar(f"Gradient_Norms/{name}", grad_norm, step)

    def log_aggregate_module_metrics(self, model: torch.nn.Module, step: int) -> None:
        """Log aggregate metrics grouped by module type to TensorBoard."""
        weight_stats = defaultdict(list)
        bias_stats = defaultdict(list)
        grad_weight_stats = defaultdict(list)
        grad_bias_stats = defaultdict(list)

        for module_name, module in model.named_modules():
            if module_name == "":
                continue  # skip root module

            module_type = type(module).__name__
            for param_name, param in module.named_parameters(recurse=False):
                # separate weights and biases
                if "bias" in param_name:
                    bias_stats[module_type].append(param.data.view(-1))
                    if param.grad is not None:
                        grad_bias_stats[module_type].append(param.grad.view(-1))
                else:
                    weight_stats[module_type].append(param.data.view(-1))
                    if param.grad is not None:
                        grad_weight_stats[module_type].append(param.grad.view(-1))

        # aggregate and log metrics
        for module_type, params in weight_stats.items():
            params_concat = torch.cat(params)
            self.writer.add_scalar(
                f"Aggregate/{module_type}/mean_weight", params_concat.mean(), step
            )
            self.writer.add_scalar(
                f"Aggregate/{module_type}/std_weight", params_concat.std(), step
            )

        for module_type, params in bias_stats.items():
            params_concat = torch.cat(params)
            self.writer.add_scalar(
                f"Aggregate/{module_type}/mean_bias", params_concat.mean(), step
            )
            self.writer.add_scalar(
                f"Aggregate/{module_type}/std_bias", params_concat.std(), step
            )

        for module_type, grads in grad_weight_stats.items():
            grads_concat = torch.cat(grads)
            self.writer.add_scalar(
                f"Aggregate/{module_type}/mean_grad_weight", grads_concat.mean(), step
            )
            self.writer.add_scalar(
                f"Aggregate/{module_type}/std_grad_weight", grads_concat.std(), step
            )

        for module_type, grads in grad_bias_stats.items():
            grads_concat = torch.cat(grads)
            self.writer.add_scalar(
                f"Aggregate/{module_type}/mean_grad_bias", grads_concat.mean(), step
            )
            self.writer.add_scalar(
                f"Aggregate/{module_type}/std_grad_bias", grads_concat.std(), step
            )

    def log_summary_statistics(self, model: torch.nn.Module, step: int) -> None:
        """Log summary statistics of model parameters and gradients to TensorBoard."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_scalar(f"Summary/{name}/mean", param.mean(), step)
                self.writer.add_scalar(f"Summary/{name}/std", param.std(), step)
                self.writer.add_scalar(f"Summary/{name}/min", param.min(), step)
                self.writer.add_scalar(f"Summary/{name}/max", param.max(), step)
                self.writer.add_scalar(f"Summary/{name}/median", param.median(), step)
                self.writer.add_scalar(
                    f"Summary/{name}/sparsity", (param == 0).float().mean(), step
                )

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
