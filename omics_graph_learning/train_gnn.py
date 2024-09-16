#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Data set-up, training loops, evaluation, and automated plotting for GNNs in
the OGL pipeline."""


import argparse
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats  # type: ignore
from scipy.stats import pearsonr  # type: ignore
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import torch_geometric  # type: ignore
from torch_geometric.loader import NeighborLoader  # type: ignore
from tqdm import tqdm  # type: ignore

from architecture_builder import build_gnn_architecture
from omics_graph_learning.config_handlers import ExperimentConfig
from omics_graph_learning.graph_to_pytorch import GraphToPytorch
from omics_graph_learning.perturbation import PerturbationConfig
from omics_graph_learning.schedulers import OptimizerSchedulerHandler
from omics_graph_learning.utils.arg_parser import OGLCLIParser
from omics_graph_learning.utils.common import \
    _set_matplotlib_publication_parameters
from omics_graph_learning.utils.common import dir_check_make
from omics_graph_learning.utils.common import plot_predicted_versus_expected
from omics_graph_learning.utils.common import plot_training_losses
from omics_graph_learning.utils.common import PyGDataChecker
from omics_graph_learning.utils.common import setup_logging
from omics_graph_learning.utils.common import tensor_out_to_array
from omics_graph_learning.utils.constants import EARLY_STOP_PATIENCE
from omics_graph_learning.utils.constants import RANDOM_SEEDS


class TensorBoardLogger:
    """Log hyperparameters, metrics, and model graph to TensorBoard."""

    def __init__(self, log_dir: Path) -> None:
        """Initialize TensorBoard writer."""
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
        device = next(model.parameters()).device

        # create a dummy input
        dummy_x = torch.zeros(input_shape, device=device)
        dummy_edge_index = torch.zeros(
            edge_index_shape, dtype=torch.long, device=device
        )
        dummy_mask = torch.ones(input_shape[0], dtype=torch.bool, device=device)
        self.writer.add_graph(model, (dummy_x, dummy_edge_index, dummy_mask))

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
        """Initialize model trainer."""
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
            f"\nTraining {self.model.__class__.__name__} model @ epoch: {epoch:04d} - "
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
    ) -> Tuple[float, torch.Tensor, torch.Tensor, float]:
        """Base function for model evaluation or inference."""
        self.model.eval()
        pbar = tqdm(total=len(data_loader))
        pbar.set_description(
            f"\nEvaluating {self.model.__class__.__name__} model @ epoch: {epoch:04d}"
        )

        outs, labels = [], []

        for batch_idx, data in enumerate(data_loader):
            if subset_batches and batch_idx >= subset_batches:
                break

            data = data.to(self.device)
            regression_mask = getattr(data, f"{mask}_mask_loss")

            # skip batch if no nodes to evaluate
            if regression_mask.sum() == 0:
                pbar.update(1)
                continue

            # forward pass
            out = self.model(
                x=data.x,
                edge_index=data.edge_index,
                regression_mask=regression_mask,
            )

            # ensure output is 1D
            out_masked = out[regression_mask].squeeze()
            if out_masked.dim() == 0:
                out_masked = out_masked.unsqueeze(0)
            outs.append(out_masked.cpu())

            # ensure labels are 1D
            labels_masked = data.y[regression_mask].squeeze()
            if labels_masked.dim() == 0:
                labels_masked = labels_masked.unsqueeze(0)
            labels.append(labels_masked.cpu())

            pbar.update(1)
        pbar.close()

        # calculate RMSE
        predictions = torch.cat(outs)
        targets = torch.cat(labels)
        mse = F.mse_loss(predictions, targets)
        rmse = torch.sqrt(mse)

        # calculate Pearson correlation
        r, _ = pearsonr(predictions, targets)

        return rmse.item(), predictions, targets, r

    def train_model(
        self,
        train_loader: torch_geometric.data.DataLoader,
        val_loader: torch_geometric.data.DataLoader,
        test_loader: torch_geometric.data.DataLoader,
        epochs: int,
        model_dir: Path,
        args: argparse.Namespace,
    ) -> Tuple[torch.nn.Module, float, bool]:
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

            # train
            loss = self.train(train_loader=train_loader, epoch=epoch)
            self.logger.info(f"\nEpoch: {epoch:03d}, Train loss: {loss}")

            # validation
            val_rmse, _, _, r = self.evaluate(
                data_loader=val_loader, epoch=epoch, mask="val"
            )
            self.logger.info(
                f"\nEpoch: {epoch:03d}, "
                f"Validation RMSE: {val_rmse:.4f}, "
                f"Validation Pearson's R: {r:.4f}",
            )

            # test
            test_rmse, _, _, r = self.evaluate(
                data_loader=test_loader, epoch=epoch, mask="test"
            )
            self.logger.info(
                f"\nEpoch: {epoch:03d}, "
                f"Test RMSE: {test_rmse:.4f}"
                f"Test Pearson's R: {r:.4f}"
            )

            # log metrics to tensorboard
            metrics = {
                "Training loss": loss,
                "Validation RMSE": val_rmse,
                "Validation Pearson's R": r,
                "Test RMSE": test_rmse,
                "Test Pearson's R": r,
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
                        model_dir / f"{args.model}_best_model.pt",
                    )
                elif best_validation < val_rmse:
                    stop_counter += 1
                if stop_counter == EARLY_STOP_PATIENCE:
                    self.logger.info("***********Early stopping!")
                    break

        return self.model, best_validation, stop_counter == EARLY_STOP_PATIENCE

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


def get_seed(run_number: int) -> int:
    """Get seed for reproducibility. We run models for 3 runs, each of which
    uses a different seed.
    """
    return RANDOM_SEEDS[run_number + 1]


def setup_device(args: argparse.Namespace) -> Tuple[torch.device, int]:
    """Check for GPU and set device accordingly."""
    seed = get_seed(args.run_number)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        return torch.device(f"cuda:{args.device}"), seed
    return torch.device("cpu"), seed


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


def boostrap_evaluation(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    n_bootstraps: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Perform bootstrap evaluation by resampling with replacement."""
    n_samples = len(predictions)
    bootstrap_rmses: List[float] = []

    for _ in range(n_bootstraps):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        bootstrap_pred = predictions[indices]
        bootstrap_labels = labels[indices]
        mse = F.mse_loss(bootstrap_pred, bootstrap_labels)
        rmse = torch.sqrt(mse)
        bootstrap_rmses.append(rmse.item())

    mean_rmse = np.mean(bootstrap_rmses)
    ci_lower, ci_upper = stats.t.interval(
        confidence,
        len(bootstrap_rmses) - 1,
        loc=mean_rmse,
        scale=stats.sem(bootstrap_rmses),
    )

    return mean_rmse, ci_lower, ci_upper


def load_final_model(
    model_name: str,
    model: torch.nn.Module,
    device: torch.device,
    model_dir: Path,
) -> torch.nn.Module:
    """Load final model from checkpoint."""
    checkpoint = torch.load(
        model_dir / f"{model_name}_best_model.pt", map_location=device
    )
    model.load_state_dict(checkpoint, strict=False)
    return model


def make_model_plots(
    outs: torch.Tensor,
    labels: torch.Tensor,
    rmse: float,
    model_dir: Path,
    tb_logger: TensorBoardLogger,
) -> None:
    """Plot training losses and performance"""
    # set params for plotting
    _set_matplotlib_publication_parameters()

    # convert to numpy arrays
    predictions_median = tensor_out_to_array(outs)
    labels_median = tensor_out_to_array(labels)

    # plot loss
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


def post_model_evaluation(
    model_name: str,
    model: torch.nn.Module,
    device: torch.device,
    data: torch_geometric.data.Data,
    data_loader: torch_geometric.data.DataLoader,
    tb_logger: TensorBoardLogger,
    logger: logging.Logger,
    run_dir: Path,
    early_stop: bool,
) -> None:
    """Plot training losses and performance"""

    # load early stopping model, otherwise, use the final model
    if early_stop:
        model = load_final_model(
            model_name=model_name,
            model=model,
            device=device,
            model_dir=run_dir,
        )

    if model is not None:
        model.to(device)
    else:
        raise ValueError("Model is None. Cannot plot performance")

    rmse, outs, labels, _ = GNNTrainer.inference_all_neighbors(
        model=model,
        device=device,
        data=data,
        data_loader=data_loader,
        split="test",
        mask=data.test_mask_loss,
    )

    # bootstrap evaluation
    mean_rmse, ci_lower, ci_upper = boostrap_evaluation(
        predictions=outs,
        labels=labels,
    )

    logger.info(f"Final, all neighbor test RMSE: {rmse:.4f}")
    logger.info(f"Bootstrap RMSE: {mean_rmse:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")

    metrics = {
        "Final test RMSE": rmse,
        "Bootstrap RMSE": mean_rmse,
        "CI lower": ci_lower,
        "CI upper": ci_upper,
    }

    tb_logger.log_metrics(metrics, epoch=0)
    with open(run_dir / "eval_metrics.json", "w") as output:
        json.dump(metrics, output, indent=4)

    # make model plots
    make_model_plots(
        outs=outs,
        labels=labels,
        rmse=rmse,
        model_dir=run_dir,
        tb_logger=tb_logger,
    )

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
    """Prepare directories and set up logging for experiment."""
    # set run_number
    run_dir = (
        experiment_config.root_dir
        / "models"
        / experiment_config.experiment_name
        / f"run_{args.run_number}"
    )

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


def parse_training_args() -> argparse.Namespace:
    """Parse training arguments."""
    parser = OGLCLIParser()
    parser.add_gnn_training_args()
    return parser.parse_args()


def main() -> None:
    """Main function to train GNN on graph data!"""
    args = parse_training_args()
    experiment_config = ExperimentConfig.from_yaml(args.experiment_yaml)
    model_dir, logger = _experiment_setup(
        args=args, experiment_config=experiment_config
    )

    # check for GPU
    device, seed = setup_device(args)
    logger.info(f"Using device: {device}")
    logger.info(f"Random seed set to: {seed}")

    # get graph data
    data = GraphToPytorch(
        experiment_config=experiment_config,
        split_name=args.split_name,
        regression_target=args.target,
        positional_encoding=args.positional_encoding,
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
    # tb_logger.log_model_graph(
    #     model=model,
    #     input_shape=data.x.shape,
    #     edge_index_shape=data.edge_index.shape,
    # )

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
    model, _, early_stop = trainer.train_model(
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
        model_dir / f"{args.model}_final_model.pt",
    )

    # generate loss and prediction plots for best model
    post_model_evaluation(
        model_name=args.model,
        model=model,
        device=device,
        data=data,
        data_loader=test_loader,
        model_dir=model_dir,
        tb_logger=tb_logger,
        early_stop=early_stop,
    )

    tb_logger.writer.close()


if __name__ == "__main__":
    main()
