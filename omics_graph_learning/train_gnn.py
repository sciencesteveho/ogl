#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Data set-up, training loops, evaluation, and automated plotting for GNNs in
the OGL pipeline."""


import argparse
import logging
import math
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import torch_geometric  # type: ignore
from torch_geometric.loader import NeighborLoader  # type: ignore
from torch_geometric.utils import k_hop_subgraph  # type: ignore
from tqdm import tqdm  # type: ignore

from config_handlers import ExperimentConfig
from gnn_architecture_builder import build_gnn_architecture
from graph_to_pytorch import GraphToPytorch
from ogl import parse_pipeline_arguments
from perturbation import PerturbationConfig
from utils import _set_matplotlib_publication_parameters
from utils import _tensor_out_to_array
from utils import dir_check_make
from utils import plot_predicted_versus_expected
from utils import plot_training_losses
from utils import setup_logging


def setup_device(args: argparse.Namespace) -> torch.device:
    """Check for GPU and set device accordingly."""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        return torch.device(f"cuda:{args.device}")
    return torch.device("cpu")


def calculate_training_steps(
    train_loader: torch_geometric.data.DataLoader,
    batch_size: int,
    epochs: int,
    warmup_ratio: float = 0.1,
) -> int:
    """Calculate the total number of projected training steps to provide a
    percentage of warm-up steps."""
    steps_per_epoch = math.ceil(len(train_loader.dataset) / batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    return total_steps, warmup_steps


def _set_optimizer(
    optimizer_type: str, learning_rate: float, model_params: Iterator[Parameter]
) -> Optimizer:
    """Choose optimizer."""
    # set gradient descent optimizer
    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(
            model_params,
            lr=learning_rate,
        )
    elif optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            model_params,
            lr=learning_rate,
        )
    return optimizer


def _set_scheduler(
    scheduler_type: str,
    optimizer: Optimizer,
    warmup_steps: int,
    training_steps: int,
) -> LRScheduler:
    """Set learning rate scheduler"""
    if scheduler_type == "plateau":
        return ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-5
        )
    if scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=training_steps,
        )
    elif scheduler_type == "linear_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=training_steps,
        )
    return scheduler


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LRScheduler:
    """
    Adapted from HuggingFace:
    https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py

    Create a scheduler with a learning rate that decreases following the values
    of the cosine function between the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Returns:
        an instance of LambdaLR.
    """

    def lr_lambda(current_step: int) -> float:
        """Compute the learning rate lambda value based on current step."""
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LRScheduler:
    """Creates a learning rate scheduler with linear warmup where the learning
    rate increases linearly from 0 to the initial optimizer learning rate over
    `num_warmup_steps`. After completing the warmup, the learning rate will
    decrease from the initial optimizer learning rate to 0 linearly over the
    remaining steps until `num_training_steps`.
    """

    def lr_lambda(current_step: int) -> float:
        """Compute the learning rate lambda value based on current step."""
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            1.0
            - (
                float(current_step - num_warmup_steps)
                / float(max(1, num_training_steps - num_warmup_steps))
            ),
        )

    return LambdaLR(optimizer, lr_lambda)


def train(
    model: torch.nn.Module,
    device: torch.cuda.device,
    optimizer: Optimizer,
    train_loader: torch_geometric.data.DataLoader,
    epoch: int,
    subset_batches: int = None,
) -> float:
    """Train GNN model on graph data"""
    model.train()
    pbar = tqdm(total=len(train_loader))
    pbar.set_description(f"Training epoch: {epoch:04d}")

    total_loss = total_examples = 0
    for batch_idx, data in enumerate(train_loader):
        # break out of the loop if subset_batches is set and reached.
        if subset_batches and batch_idx >= subset_batches:
            break

        optimizer.zero_grad()
        data = data.to(device)

        if hasattr(model, "task_specific_mlp") and model.task_specific_mlp:
            out = model(data.x, data.edge_index, data.train_mask_loss)
        else:
            out = model(data.x, data.edge_index)

        loss = F.mse_loss(out[data.train_mask_loss], data.y[data.train_mask_loss])
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * int(data.train_mask_loss.sum())
        total_examples += int(data.train_mask_loss.sum())

        pbar.update(1)

    pbar.close()
    return total_loss / total_examples


def _evaluate_model(
    model: torch.nn.Module,
    device: torch.cuda.device,
    data_loader: torch_geometric.data.DataLoader,
    epoch: int,
    mask: str,
    subset_batches: int = None,
    collect_outputs: bool = False,
) -> Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Base function for model evaluation or inference."""
    model.eval()
    pbar = tqdm(total=len(data_loader))
    pbar.set_description(f"Evaluating epoch: {epoch:04d}")
    mse, outs, labels = [], [], []

    for batch_idx, data in enumerate(data_loader):
        # break out of the loop if subset_batches is set and reached.
        if subset_batches and batch_idx >= subset_batches:
            break

        data = data.to(device)

        # task specific or general mlp forward pass
        if model.task_specific_mlp:
            if mask == "val":
                out = model(data.x, data.edge_index, data.val_mask_loss)
            elif mask == "test":
                out = model(data.x, data.edge_index, data.test_mask_loss)
            else:
                raise ValueError("Invalid mask type. Use 'val' or 'test'.")
        else:
            out = model(data.x, data.edge_index)

        # use proper mask for loss calculation
        if mask == "val":
            idx_mask = data.val_mask_loss
        elif mask == "test":
            idx_mask = data.test_mask_loss
        else:
            raise ValueError("Invalid mask type. Use 'val' or 'test'.")

        mse.append(F.mse_loss(out[idx_mask], data.y[idx_mask]).cpu())

        if collect_outputs:
            outs.extend(out[idx_mask])
            labels.extend(data.y[idx_mask])

        pbar.update(1)

    pbar.close()
    loss = torch.stack(mse)
    rmse = math.sqrt(float(loss.mean()))

    return (
        rmse,
        (torch.stack(outs) if collect_outputs else None),
        (torch.stack(labels) if collect_outputs else None),
    )


@torch.no_grad()
def test(
    model: torch.nn.Module,
    device: torch.cuda.device,
    data_loader: torch_geometric.data.DataLoader,
    epoch: int,
    mask: str,
    subset_batches: int = None,
) -> float:
    """Evaluate GNN model on validation or test set."""
    rmse, _, _ = _evaluate_model(
        model=model,
        device=device,
        data_loader=data_loader,
        epoch=epoch,
        mask=mask,
        subset_batches=subset_batches,
        collect_outputs=False,
    )
    return rmse


@torch.no_grad()
def inference(
    model: torch.nn.Module,
    device: torch.cuda.device,
    data_loader: torch_geometric.data.DataLoader,
    epoch: int,
    subset_batches: int = None,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """Use model for inference on the test set"""
    rmse, outs, labels = _evaluate_model(
        model,
        device,
        data_loader,
        epoch,
        mask="test",
        subset_batches=subset_batches,
        collect_outputs=True,
    )
    return rmse, outs, labels


@torch.no_grad()
def inference_all_neighbors(
    model: torch.nn.Module,
    device: torch.cuda.device,
    data: torch_geometric.data.Data,
    epoch: int,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """Use model for inference or to evaluate on validation set"""
    model.eval()
    loader = NeighborLoader(data, num_neighbors=[-1], batch_size=1, shuffle=False)
    predictions = {}
    for batch in tqdm(loader, desc=f"Evaluating epoch: {epoch:04d}"):
        batch = batch.to(device)

        # forward pass
        if model.task_specific_mlp:
            out = model(batch.x, batch.edge_index, batch.test_mask_loss)
        else:
            out = model(batch.x, batch.edge_index)

        node_idx = batch.batch.item()
        if node_idx in predictions:
            predictions[node_idx].append(out.squeeze())
        else:
            predictions[node_idx] = [out.squeeze()]

    all_preds = torch.tensor(
        [torch.mean(torch.stack(preds), dim=0) for preds in predictions.values()]
    )
    outs = all_preds[data.test_mask_loss]
    labels = data.y[data.test_mask_loss]
    mse = F.mse_loss(outs, labels)
    return math.sqrt(float(mse.cpu())), outs, labels


def get_max_hop_subgraph(
    node: int,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> Tuple[torch.Tensor, int, int]:
    """Finds the maximum hop subgraph for a given node."""
    hop = 0
    node_idx = torch.tensor([node])
    prev_size = 0

    while True:
        node_idx, _, mapping, _ = k_hop_subgraph(
            node, hop + 1, edge_index, relabel_nodes=True, num_nodes=num_nodes
        )
        if len(node_idx) == prev_size:  # no new nodes added
            break
        prev_size = len(node_idx)
        hop += 1

    return node_idx, mapping[0], hop


@torch.no_grad()
def test_all_neighbors(
    model: torch.nn.Module,
    device: torch.cuda.device,
    data: torch_geometric.data.Data,
    epoch: int,
    mask: str,
    batch_size: int = 32,
) -> Tuple[float, float, float, Dict[int, int]]:
    """Modified testing function. Uses neighborloader with num_neighbors=-1 and
    batch_size=1 to ensure that the entire neighborhood is loaded for each node
    during evaluation.  Stores each prediction in a dictionary of key, list and
    takes a mean of all predictions for each node to get the final
    prediction."""
    model.eval()
    if mask == "val":
        eval_mask = data.val_mask
        loss_mask = data.val_mask_loss
    elif mask == "test":
        eval_mask = data.test_mask
        loss_mask = data.test_mask_loss
    else:
        raise ValueError("Invalid mask type. Use 'val' or 'test'.")

    predictions = {}
    hops_used = {}
    nodes_of_interest = torch.where(eval_mask)[0].tolist()

    for i in tqdm(
        range(0, len(nodes_of_interest), batch_size),
        desc=f"Evaluating epoch: {epoch:04d}",
    ):
        batch_nodes = nodes_of_interest[i : i + batch_size]
        batch_subgraphs = []
        batch_mappings = []

        for node in batch_nodes:
            node_idx, mapping, hops = get_max_hop_subgraph(
                node, data.edge_index, data.num_nodes
            )
            batch_subgraphs.append(node_idx)
            batch_mappings.append(mapping)
            hops_used[node] = hops

        # combine subgraphs
        combined_nodes = torch.cat(batch_subgraphs).unique()
        combined_subgraph, combined_edge_index, combined_mapping, _ = k_hop_subgraph(
            combined_nodes,
            1,
            data.edge_index,
            relabel_nodes=True,
            num_nodes=data.num_nodes,
        )

        # prepare the subgraph data
        sub_x = data.x[combined_subgraph].to(device)
        sub_edge_index = combined_edge_index.to(device)

        # forward pass
        if model.task_specific_mlp:
            sub_mask = torch.zeros(
                len(combined_subgraph), dtype=torch.bool, device=device
            )
            sub_mask[combined_mapping[torch.tensor(batch_nodes)]] = True
            out = model(sub_x, sub_edge_index, sub_mask)
        else:
            out = model(sub_x, sub_edge_index)

        # Store predictions for nodes of interest
        for j, node in enumerate(batch_nodes):
            node_in_subgraph = combined_mapping[batch_mappings[j]]
            predictions[node] = out[node_in_subgraph].cpu()

    # convert predictions to tensor
    nodes = torch.tensor(list(predictions.keys()))
    preds = torch.stack(list(predictions.values()))

    # sort predictions to match the order of nodes in the original graph
    sorted_indices = torch.argsort(nodes)
    all_preds = preds[sorted_indices]

    # calculate MSE only on nodes in loss_mask
    mse = torch.nn.functional.mse_loss(all_preds[loss_mask], data.y[loss_mask])

    # calculate average and max hops used
    avg_hops = sum(hops_used.values()) / len(hops_used)
    max_hops = max(hops_used.values())

    return torch.sqrt(mse).item(), avg_hops, max_hops, hops_used


def prep_loader(
    data: torch_geometric.data.Data,
    mask: torch.Tensor,
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


def construct_save_string(components: List[str], args: argparse.Namespace) -> str:
    """Adds args to specify save string for model and logs"""
    if args.residual:
        components.append("residual")
    if args.heads:
        components.append(f"heads{args.heads}")
    if args.dropout > 0.0:
        components.append(f"dropout{args.dropout}")
    if args.randomize_node_feats:
        components.append("randomnodefeats")
    if args.zero_nodes:
        components.append("zeronodefeats")
    if args.total_random_edges:
        components.append(f"totalrandomedges{args.total_random_edges}")
    if args.gene_only_loader:
        components.append("geneonlyloader")
    return "_".join(components)


def plot_loss_and_performance(
    device: torch.cuda.device,
    data_loader: torch_geometric.data.DataLoader,
    model_dir: Path,
    savestr: str,
    model: Optional[torch.nn.Module],
    best_validation: Optional[float] = None,
) -> None:
    """Plot training losses and performance"""
    # set params for plotting
    _set_matplotlib_publication_parameters()

    # plot either final model or best validation model
    if best_validation:
        models_dir = model_dir / "models"
        best_checkpoint = models_dir.glob(f"*{best_validation}")
        checkpoint = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        model.to(device)
        savestr += "_best_"
    elif model:
        model.to(device)
        savestr += "_final_"

    # get predictions
    rmse, outs, labels = inference_all_neighbors(
        model=model,
        device=device,
        data_loader=data_loader,
        epoch=0,
    )

    predictions_median = _tensor_out_to_array(outs, 0)
    labels_median = _tensor_out_to_array(labels, 0)

    # plot training losses
    plot_training_losses(
        outfile=model_dir / "plots" / f"{savestr}_loss.png",
        savestr=savestr,
        log=model_dir / "logs" / "training_log.txt",
    )

    # plot performance
    plot_predicted_versus_expected(
        outfile=model_dir / "plots" / f"{savestr}_performance.png",
        savestr=savestr,
        predicted=predictions_median,
        expected=labels_median,
        rmse=rmse,
    )


def training_loop(
    model: torch.nn.Module,
    device: torch.cuda.device,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    train_loader: torch_geometric.data.DataLoader,
    val_loader: torch_geometric.data.DataLoader,
    test_loader: torch_geometric.data.DataLoader,
    epochs: int,
    writer: SummaryWriter,
    model_dir: Path,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Tuple[torch.nn.Module, float]:
    """Execute training loop for GNN models.

    The loop will train and evaluate the model on the validation set with
    minibatching. Afterward, the model is evaluated on the test set utilizing
    all neighbors.
    """
    best_validation = stop_counter = 0  # set up early stopping variable
    for epoch in range(epochs + 1):
        loss = train(
            model=model,
            device=device,
            optimizer=optimizer,
            train_loader=train_loader,
            epoch=epoch,
        )
        writer.add_scalar("Training loss", loss, epoch)
        logger.info(f"Epoch: {epoch:03d}, Train: {loss}")

        val_acc = test(
            model=model,
            device=device,
            data=val_loader.dataset,
            epoch=epoch,
            mask="val",
        )
        logger.info(f"Epoch: {epoch:03d}, Validation: {val_acc:.4f}")
        writer.add_scalar("Validation RMSE", val_acc, epoch)

        test_acc = test_all_neighbors(
            model=model,
            device=device,
            data=test_loader.dataset,
            epoch=epoch,
            mask="test",
            num_hops=args.gnn_layers,
            batch_size=args.batch_size,
        )
        logger.info(f"Epoch: {epoch:03d}, Test: {test_acc:.4f}")
        writer.add_scalar("Test RMSE", test_acc, epoch)

        scheduler.step(val_acc)
        if args.early_stop:
            if epoch == 0 or val_acc < best_validation:
                stop_counter = 0
                best_validation = val_acc
                model_path = (
                    model_dir / f"{args.model}_{epoch}_mse_{best_validation}.pt"
                )
                torch.save(model.state_dict(), model_path)
            elif best_validation < val_acc:
                stop_counter += 1
            if stop_counter == 20:
                logger.info("***********Early stopping!")
                break

    return model, val_acc


def _experiment_setup(
    args: argparse.Namespace, experiment_config: ExperimentConfig
) -> Tuple[str, Path, logging.Logger]:
    """Load experiment configuration from YAML file."""
    savestr = construct_save_string(
        [
            experiment_config.experiment_name,
            f"{args.model}",
            f"target-{args.target}",
            f"gnnlayers{args.gnn_layers}",
            f"linlayers{args.linear_layers}",
            f"{args.activation}",
            f"dim{args.dimensions}",
            f"batch{args.batch_size}",
            f"{args.optimizer}",
            f"{args.split_name}",
        ],
        args,
    )
    logger.info(f"Save string: {savestr}")
    model_dir = experiment_config.root_dir / "models" / f"{savestr}"

    # make directories and set up training log
    for folder in ["logs", "plots"]:
        dir_check_make(model_dir / folder)

    logger = setup_logging(log_file=str(model_dir / "logs" / "training_log.txt"))

    # Log experiment information
    logger.info("Experiment setup initialized.")
    logger.info(f"Experiment configuration: {experiment_config}")
    logger.info(f"Model directory: {model_dir}")
    return savestr, model_dir, logger


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
    # Parse training settings
    args = parse_arguments()
    experiment_config = ExperimentConfig.from_yaml(args.experiment_yaml)

    savestr, model_dir, logger = _experiment_setup(
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

    # set up data loaders
    mask_suffix = "_loss" if args.gene_only_loader else ""
    train_loader = prep_loader(
        data=data,
        mask=f"train_mask{mask_suffix}",
        batch_size=args.batch_size,
        shuffle=True,
        layers=args.gnn_layers,
    )
    test_loader = prep_loader(
        data=data,
        mask=f"test_mask{mask_suffix}",
        batch_size=args.batch_size,
        shuffle=False,
        layers=args.gnn_layers,
    )
    val_loader = prep_loader(
        data=data,
        mask=f"val_mask{mask_suffix}",
        batch_size=args.batch_size,
        shuffle=False,
        layers=args.gnn_layers,
    )

    # CHOOSE YOUR WEAPON
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
        task_specific_mlp=args.task_specific_mlp,
        train_dataset=train_loader if args.model == "PNA" else None,
    ).to(device)

    # set up optimizer & scheduler
    total_steps, warmup_steps = calculate_training_steps(
        train_loader=train_loader,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    optimizer = _set_optimizer(
        optimizer_type=args.optimizer,
        learning_rate=args.learning_rate,
        model_params=model.parameters(),
    )
    scheduler = _set_scheduler(
        scheduler_type=args.scheduler,
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        training_steps=total_steps,
    )

    # start model training and initialize tensorboard utilities
    writer = SummaryWriter(model_dir / "logs")
    epochs = args.epochs
    prof = torch.profiler.profile(
        # activities=[
        #     torch.profiler.ProfilerActivity.CPU,
        #     torch.profiler.ProfilerActivity.CUDA,
        # ],
        schedule=torch.profiler.schedule(wait=1, warmup=4, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(model_dir / "logs"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )

    prof.start()
    logger.info(f"Training for {epochs} epochs")
    model, final_val_acc = training_loop(
        model=model,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=epochs,
        writer=writer,
        model_dir=model_dir,
        args=args,
        logger=logger,
    )

    # close out tensorboard utilities
    writer.flush()
    prof.stop()

    # Save final model
    model_path = model_dir / f"{args.model}_final_mse_{final_val_acc}.pt"
    torch.save(model.state_dict(), model_path)

    # generate loss and prediction plots
    plot_loss_and_performance(
        model=model,
        device=device,
        data_loader=test_loader,
        model_dir=model_dir,
        savestr=savestr,
    )


if __name__ == "__main__":
    main()
