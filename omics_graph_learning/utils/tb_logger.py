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
from torch_geometric.utils import coalesce  # type: ignore
from torch_geometric.utils import degree  # type: ignore
from torch_geometric.utils import subgraph  # type: ignore
from torch_geometric.utils import to_undirected  # type: ignore


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
        # quick check data integrity
        assert (
            data.edge_index.max() < data.num_nodes
        ), "Edge index contains out-of-range node indices."
        assert data.edge_index.min() >= 0, "Edge index contains negative node indices."

        try:
            print(
                f"Starting log_model_graph with sample_size={sample_size} and num_nodes={data.num_nodes}"
            )

            if data.num_nodes < sample_size:
                print(f"Adjusting sample_size to match num_nodes: {data.num_nodes}")
                sample_size = data.num_nodes

            # ensure edge_index is undirected
            data.edge_index = to_undirected(data.edge_index)
            data.edge_index, _ = coalesce(
                data.edge_index, None, data.num_nodes, data.num_nodes
            )
            print(
                f"Edge_index is undirected and coalesced with shape: {data.edge_index.shape}"
            )

            # compute degrees
            degrees = degree(data.edge_index[0], num_nodes=data.num_nodes)

            # select a high-degree node as seed
            high_degree_node = degrees.argmax().item()
            seed_nodes = torch.tensor([high_degree_node])
            print(f"High-degree seed node selected: {seed_nodes.item()}")

            # check degree of seed node
            seed_degree = degrees[seed_nodes].item()
            print(f"Seed node {seed_nodes.item()} has degree: {seed_degree}")
            if seed_degree == 0:
                raise ValueError(f"Seed node {seed_nodes.item()} is isolated.")

            # extract k-hop subgraph
            sub_nodes, sub_edge_index, subsets, _ = (
                torch_geometric.utils.k_hop_subgraph(
                    node_idx=seed_nodes,
                    num_hops=5,
                    edge_index=data.edge_index,
                    num_nodes=data.num_nodes,
                    flow="source_to_target",
                )
            )

            # debugging the contents of `subsets`
            print(f"Number of subsets: {len(subsets)}")
            if len(subsets) == 0:
                raise ValueError(
                    f"Subsets are empty for seed node {seed_nodes.item()}."
                )

            for i, subset in enumerate(subsets):
                print(f"Subset {i}: {subset}")

            print(
                f"Subgraph extracted: {sub_nodes.numel()} nodes and {sub_edge_index.size(1)} edges."
            )

            if sub_nodes.numel() > sample_size:
                print("Subgraph has more nodes than sample_size. Reducing nodes.")
                sub_nodes = sub_nodes[torch.randperm(sub_nodes.numel())[:sample_size]]

            print(f"Final sub_nodes count: {sub_nodes.numel()}")

            # get subgraph
            sub_edge_index, _ = subgraph(
                subset=sub_nodes,
                edge_index=data.edge_index,
                relabel_nodes=True,
                num_nodes=data.num_nodes,
            )

            print(
                f"Subgraph edge index: {sub_edge_index.size(1)} edges after relabeling."
            )

            # get subgraph features and mask
            sub_x = data.x[sub_nodes]
            sub_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            sub_mask[sub_nodes] = True

            # move to GPU
            sub_x = sub_x.to(device)
            sub_edge_index = sub_edge_index.to(device)
            sub_mask = sub_mask.to(device)

            print(
                f"Subgraph moved to device: {device}. Features shape: {sub_x.shape}, Edge index shape: {sub_edge_index.shape}"
            )

            # log the graph!
            self.writer.add_graph(model, (sub_x, sub_edge_index, sub_mask))
            print(
                f"Successfully logged subgraph with {sub_nodes.numel()} nodes "
                f"and {sub_edge_index.size(1)} edges."
            )

        except Exception as e:
            print(f"Failed to log model graph: {e}")
            raise e

    def close(self) -> None:
        """Close TensorBoard writer."""
        self.writer.close()
