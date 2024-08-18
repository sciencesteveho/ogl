#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Data set-up, training loops, evaluation, and automated plotting for GNNs in
the OGL pipeline."""


import argparse
import json
import logging
import math
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import torch_geometric  # type: ignore
from torch_geometric.loader import NeighborLoader  # type: ignore
from torch_geometric.utils import k_hop_subgraph  # type: ignore
from tqdm import tqdm  # type: ignore

from config_handlers import ExperimentConfig
from constants import EARLY_STOP_PATIENCE
from gnn_architecture_builder import build_gnn_architecture
from graph_to_pytorch import GraphToPytorch
from ogl import parse_pipeline_arguments
from perturbation import PerturbationConfig
from schedulers import OptimizerSchedulerHandler
from utils import _set_matplotlib_publication_parameters
from utils import _tensor_out_to_array
from utils import dir_check_make
from utils import plot_predicted_versus_expected
from utils import plot_training_losses
from utils import PyGDataChecker
from utils import setup_logging


class TensorBoardLogger:
    """Log hyperparameters, metrics, and model graph to TensorBoard."""

    def __init__(self, log_dir: Path) -> None:
        """Instantiate TensorBoard writer."""
        self.writer = SummaryWriter(log_dir)

    def log_hyperparameters(self, hparams: Dict[str, Any]) -> None:
        """Log hyperparameters to TensorBoard."""
        self.writer.add_hparams(hparams, {})

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics to TensorBoard."""
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)

    def log_model_graph(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, int],
        edge_index_shape: int,
    ) -> None:
        """Log model graph to TensorBoard."""

        # create a dummy input
        dummy_x = torch.zeros(input_shape, device=model.device)
        dummy_edge_index = torch.zeros(
            edge_index_shape, dtype=torch.long, device=model.device
        )
        dummy_mask = torch.ones(input_shape[0], dtype=torch.bool, device=model.device)

        self.writer.add_graph(model, (dummy_x, dummy_edge_index, dummy_mask, "train"))

    def close(self) -> None:
        """Close TensorBoard writer."""
        self.writer.close()


class GNNTrainer:
    """Class to handle GNN training and evaluation.

    Methods
    --------
    train:
        Train GNN model on graph data via minibatching
    evaluate:
        Evaluate GNN model on validation or test set
    inference_all_neighbors:
        Evaluate GNN model on test set using all neighbors
    training_loop:
        Execute training loop for GNN models

    Examples:
    --------
    >>> trainer = GNNTrainer(
            model=model,
            device=device,
            data=data,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger
        )

    >>> trainer.training_loop(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=epochs,
            writer=writer,
            model_dir=model_dir,
            args=args
        )
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        data: torch_geometric.data.Data,
        optimizer: Optimizer,
        scheduler: Union[LRScheduler, ReduceLROnPlateau],
        logger: logging.Logger,
        tb_logger: Optional[TensorBoardLogger] = None,
    ) -> None:
        """Instantiate model trainer."""
        self.model = model
        self.device = device
        self.data = data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger

        if tb_logger:
            self.tb = tb_logger

    def train(
        self,
        train_loader: torch_geometric.data.DataLoader,
        epoch: int,
        subset_batches: Optional[int] = None,
    ) -> float:
        """Train GNN model on graph data"""
        self.model.train()
        pbar = tqdm(total=len(train_loader))
        pbar.set_description(
            f"\nTraining Model: {self.model.__class__.__name__} epoch: {epoch:04d} - "
        )

        total_loss = float(0)
        total_examples = 0
        for batch_idx, data in enumerate(train_loader):
            if subset_batches and batch_idx >= subset_batches:
                break

            self.optimizer.zero_grad()
            data = data.to(self.device)

            out = self.model(
                x=data.x,
                edge_index=data.edge_index,
                regression_mask=data.train_mask_loss,
            )

            mse_loss = F.mse_loss(
                out[data.train_mask_loss].squeeze(),
                data.y[data.train_mask_loss].squeeze(),
            )
            mse_loss.backward()

            # check for NaN gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN gradient detected in {name}")

            self.optimizer.step()

            # step warmup schedulers
            if isinstance(self.scheduler, LRScheduler) and not isinstance(
                self.scheduler, ReduceLROnPlateau
            ):
                self.scheduler.step()

            total_loss += float(mse_loss) * int(data.train_mask_loss.sum())
            total_examples += int(data.train_mask_loss.sum())

            pbar.update(1)

        pbar.close()
        return total_loss / total_examples

    @torch.no_grad()
    def evaluate(
        self,
        data_loader: torch_geometric.data.DataLoader,
        epoch: int,
        mask: str,
        subset_batches: Optional[int] = None,
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """Base function for model evaluation or inference."""
        self.model.eval()
        pbar = tqdm(total=len(data_loader))
        pbar.set_description(
            f"\nEvaluating Model: {self.model.__class__.__name__} epoch: {epoch:04d}"
        )

        outs, labels = [], []

        for batch_idx, data in enumerate(data_loader):
            if subset_batches and batch_idx >= subset_batches:
                break

            data = data.to(self.device)
            regression_mask = getattr(data, f"{mask}_mask_loss")

            # forward pass
            out = self.model(
                x=data.x,
                edge_index=data.edge_index,
                regression_mask=regression_mask,
            )

            outs.append(out[regression_mask].squeeze().cpu())
            labels.append(data.y[regression_mask].squeeze().cpu())

            pbar.update(1)
        pbar.close()

        # calculate RMSE
        predictions = torch.cat(outs)
        targets = torch.cat(labels)
        mse = F.mse_loss(predictions, targets)
        rmse = torch.sqrt(mse)

        return rmse.item(), predictions, targets

    @staticmethod
    @torch.no_grad()
    def inference_all_neighbors(
        model: torch.nn.Module,
        device: torch.device,
        data: torch_geometric.data.Data,
        data_loader: torch_geometric.data.DataLoader,
        mask: torch.Tensor,
        split: str = "test",
        epoch: Optional[int] = None,
    ) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Use model for inference or to evaluate on validation set"""
        model.eval()
        predictions = []
        node_indices = []

        description = "\nEvaluating"
        if epoch is not None:
            description += f" epoch: {epoch:04d}"

        for batch in tqdm(data_loader, desc=description):
            batch = batch.to(device)
            regression_mask = getattr(batch, f"{split}_mask_loss")

            # forward pass
            out = model(
                x=batch.x,
                edge_index=batch.edge_index,
                regression_mask=regression_mask,
            )

            # filter for input nodes
            input_nodes_mask = batch.input_id < batch.num_input_nodes
            predictions.append(out[input_nodes_mask].squeeze().cpu())
            node_indices.extend(batch.input_id[input_nodes_mask].cpu().tolist())

        # concatenate predictions and sort by node index
        all_preds = torch.cat(predictions)
        original_indices = torch.tensor(node_indices)
        sorted_indices = torch.argsort(torch.tensor(node_indices))
        all_preds = all_preds[sorted_indices]
        original_indices = original_indices[sorted_indices]

        # get predictions
        labels = data.y[mask].squeeze()
        mse = F.mse_loss(all_preds, labels)
        rmse = torch.sqrt(mse)

        return rmse.item(), all_preds, labels, original_indices

    def train_model(
        self,
        train_loader: torch_geometric.data.DataLoader,
        val_loader: torch_geometric.data.DataLoader,
        test_loader: torch_geometric.data.DataLoader,
        epochs: int,
        model_dir: Path,
        args: argparse.Namespace,
    ) -> Tuple[torch.nn.Module, float]:
        """Execute training loop for GNN models.

        The loop will train and evaluate the model on the validation set with
        minibatching. Afterward, the model is evaluated on the test set
        utilizing all neighbors.
        """
        if self.tb:
            self.tb.log_hyperparameters(vars(args))

        best_validation = float(0)
        stop_counter = 0  # set up early stopping counter
        for epoch in range(epochs + 1):
            loss = self.train(train_loader=train_loader, epoch=epoch)
            self.logger.info(f"\nEpoch: {epoch:03d}, Train: {loss}")

            val_rmse, _, _ = self.evaluate(
                data_loader=val_loader, epoch=epoch, mask="val"
            )
            self.logger.info(f"\nEpoch: {epoch:03d}, Validation: {val_rmse:.4f}")

            test_rmse, _, _, _ = self.inference_all_neighbors(
                model=self.model,
                device=self.device,
                data=self.data,
                data_loader=test_loader,
                mask=self.data.test_mask_loss,
                epoch=epoch,
            )
            self.logger.info(f"\nEpoch: {epoch:03d}, Test: {test_rmse:.4f}")

            # log metrics to tensorboard
            metrics = {
                "Training loss": loss,
                "Validation RMSE": val_rmse,
                "Test RMSE": test_rmse,
            }
            if self.tb:
                self.tb.log_metrics(metrics, epoch)

            # step scheduler if normal plateau scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_rmse)

            if args.early_stop:
                if epoch == 0 or val_rmse < best_validation:
                    stop_counter = 0
                    best_validation = val_rmse
                    torch.save(
                        self.model.state_dict(),
                        model_dir
                        / f"{args.model}_{epoch}_mse_{best_validation:.4f}.pt",
                    )
                elif best_validation < val_rmse:
                    stop_counter += 1
                if stop_counter == EARLY_STOP_PATIENCE:
                    self.logger.info("***********Early stopping!")
                    break

        return self.model, best_validation


def setup_device(args: argparse.Namespace) -> torch.device:
    """Check for GPU and set device accordingly."""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        return torch.device(f"cuda:{args.device}")
    return torch.device("cpu")


def prep_loader(
    data: torch_geometric.data.Data,
    mask: str,
    batch_size: int,
    shuffle: bool = False,
    layers: int = 2,
    avg_connectivity: bool = True,
) -> torch_geometric.data.DataLoader:
    """Loads data into NeighborLoader for GNN training. Returns a DataLoader
    with randomly sampled neighbors, either by 10 neighbors * layers or by using
    the average connectivity in the graph."""
    if avg_connectivity:
        num_neighbors = [data.avg_edges] * layers
    else:
        num_neighbors = [10] * layers
    return NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        input_nodes=getattr(data, mask),
        shuffle=shuffle,
    )


def parse_arguments() -> argparse.Namespace:
    """Parse args for training GNN"""
    parser = parse_pipeline_arguments()
    parser.add_argument("--split_name", type=str, required=True)
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed to use (default: 42)"
    )
    parser.add_argument(
        "--device", type=int, default=0, help="which gpu to use if any (default: 0)"
    )
    return parser.parse_args()


def plot_loss_and_performance(
    device: torch.device,
    data: torch_geometric.data.Data,
    data_loader: torch_geometric.data.DataLoader,
    model_dir: Path,
    model: Optional[torch.nn.Module],
    tb_logger: TensorBoardLogger,
    best_validation: Optional[float] = None,
) -> None:
    """Plot training losses and performance"""
    # set params for plotting
    _set_matplotlib_publication_parameters()

    # plot either final model or best validation model
    # if best_validation:
    #     models_dir = model_dir / "models"
    #     best_checkpoint = models_dir.glob(f"*{best_validation}")
    #     checkpoint = torch.load(best_checkpoint, map_location=device)
    #     model.load_state_dict(checkpoint, strict=False)
    #     if model is not None:
    #         model.to(device)
    # elif model:
    # model.to(device)

    # get predictions
    if model is None:
        raise ValueError("Model is None. Cannot plot performance")

    rmse, outs, labels, original_indices = GNNTrainer.inference_all_neighbors(
        model=model,
        device=device,
        data=data,
        data_loader=data_loader,
        split="test",
        mask=data.test_mask_loss,
    )

    # convert output tensors to numpy arrays
    predictions_median = _tensor_out_to_array(outs, 0)
    labels_median = _tensor_out_to_array(labels, 0)

    # plot training losses
    loss = plot_training_losses(
        log=model_dir / "logs" / "training_log.txt",
    )
    tb_logger.writer.add_figure("Training loss", loss)

    # plot performance
    performance = plot_predicted_versus_expected(
        predicted=predictions_median,
        expected=labels_median,
        rmse=rmse,
    )
    tb_logger.writer.add_figure("Model performance", performance)

    # map node indices back to original idx
    # predictions_dict = {idx.item(): pred.item() for idx, pred in zip(original_indices, outs)}
    # full_predictions = torch.zeros(data.num_nodes)
    # for idx, pred in predictions_dict.items():
    #     full_predictions[idx] = pred


def _dump_metadata_json(
    args: argparse.Namespace, experiment_config: ExperimentConfig, run_dir: Path
) -> None:
    """Dump metadata json to run directory."""
    metadata = vars(args)
    metadata["experiment_name"] = experiment_config.experiment_name
    metadata["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(run_dir / "metadata.json", "w") as output:
        json.dump(metadata, output, indent=4)


def _experiment_setup(
    args: argparse.Namespace, experiment_config: ExperimentConfig
) -> Tuple[Path, logging.Logger]:
    """Load experiment configuration from YAML file."""

    # prepare directories and set up logging
    model_base_dir = (
        experiment_config.root_dir / "models" / experiment_config.experiment_name
    )
    run_dir = model_base_dir / f"run_{int(time.time())}"

    for folder in ["logs", "plots"]:
        dir_check_make(run_dir / folder)

    logger = setup_logging(log_file=str(run_dir / "logs" / "training_log.txt"))

    # dump metadata to run directory
    _dump_metadata_json(
        args=args,
        experiment_config=experiment_config,
        run_dir=run_dir,
    )

    # log experiment information
    logger.info("Experiment setup initialized.")
    logger.info(f"Experiment configuration: {experiment_config}")
    logger.info(f"Model directory: {run_dir}")
    return run_dir, logger


def prepare_pertubation_config(
    args: argparse.Namespace,
) -> Optional[PerturbationConfig]:
    """Set up perturbation config, if applicable."""
    if args.node_perturbation or args.edge_perturbation:
        perturbations = {
            "node_perturbation": args.node_perturbation or None,
            "edge_perturbation": args.edge_perturbation or None,
            "total_random_edges": args.total_random_edges or None,
        }
        return PerturbationConfig(**perturbations)
    return None


def main() -> None:
    """Main function to train GNN on graph data!"""
    args = parse_arguments()
    experiment_config = ExperimentConfig.from_yaml(args.experiment_yaml)

    model_dir, logger = _experiment_setup(
        args=args, experiment_config=experiment_config
    )

    # check for GPU
    device = setup_device(args)

    # get graph data
    data = GraphToPytorch(
        experiment_config=experiment_config,
        split_name=args.split_name,
        regression_target=args.target,
        positional_encoding=args.positional_encoding,
        perturbation_config=prepare_pertubation_config(args),
    ).make_data_object()

    # check data integreity
    PyGDataChecker.check_pyg_data(data)

    # set up tensorboard logger
    tb_logger = TensorBoardLogger(log_dir=model_dir / "tensorboard")

    # set up data loaders
    mask_suffix = "_loss" if args.gene_only_loader else ""
    train_loader = prep_loader(
        data=data,
        mask=f"train_mask{mask_suffix}",
        batch_size=args.batch_size,
        shuffle=True,
        layers=args.gnn_layers,
    )
    val_loader = prep_loader(
        data=data,
        mask=f"val_mask{mask_suffix}",
        batch_size=args.batch_size,
        shuffle=False,
        layers=args.gnn_layers,
    )
    test_loader = NeighborLoader(
        data,
        num_neighbors=[-1],  # all neighbors at each hop
        batch_size=args.batch_size,
        input_nodes=getattr(data, f"test_mask{mask_suffix}"),
        shuffle=False,
    )

    # CHOOSE YOUR WEAPON
    num_targets = len(data.y[0]) if len(data.y.shape) > 1 else 1

    model = build_gnn_architecture(
        model=args.model,
        activation=args.activation,
        in_size=data.x.shape[1],
        embedding_size=args.dimensions,
        out_channels=1,
        gnn_layers=args.gnn_layers,
        shared_mlp_layers=args.linear_layers,
        heads=args.heads,
        dropout_rate=args.dropout or None,
        skip_connection=args.residual,
        attention_task_head=args.attention_task_head,
        num_targets=num_targets,
        train_dataset=train_loader if args.model == "PNA" else None,
    ).to(device)

    # log model graph to tensorboard
    tb_logger.log_model_graph(
        model=model,
        input_shape=data.x.shape,
        edge_index_shape=data.edge_index.shape,
    )

    # set up optimizer & scheduler
    total_steps, warmup_steps = OptimizerSchedulerHandler.calculate_training_steps(
        train_loader=train_loader,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    optimizer = OptimizerSchedulerHandler.set_optimizer(
        optimizer_type=args.optimizer,
        learning_rate=args.learning_rate,
        model_params=model.parameters(),
    )
    scheduler = OptimizerSchedulerHandler.set_scheduler(
        scheduler_type=args.scheduler,
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        training_steps=total_steps,
    )

    # start model training and initialize tensorboard utilities
    epochs = args.epochs
    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=4, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            str(model_dir / "tensorboard")
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )
    prof.start()

    # initialize trainer
    trainer = GNNTrainer(
        model=model,
        device=device,
        data=data,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
        tb_logger=tb_logger,
    )

    logger.info(f"Training for {epochs} epochs")
    model, final_val_rmse = trainer.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=epochs,
        model_dir=model_dir,
        args=args,
    )

    # close out tensorboard utilities and save final model
    prof.stop()
    torch.save(
        model.state_dict(),
        model_dir / f"{args.model}_final_mse_{final_val_rmse:.3f}.pt",
    )

    # generate loss and prediction plots
    plot_loss_and_performance(
        model=model,
        device=device,
        data=data,
        data_loader=test_loader,
        model_dir=model_dir,
        tb_logger=tb_logger,
    )

    tb_logger.writer.close()


if __name__ == "__main__":
    main()
