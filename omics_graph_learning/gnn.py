#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# from torch_geometric.explain import Explainer
# from torch_geometric.explain import GNNExplainer

"""Code to train GNNs on the graph data!"""

import argparse
import logging
import math
import pathlib
from typing import Any, Dict, Iterator, List, Optional

import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import torch_geometric
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import degree
from tqdm import tqdm

from graph_to_pytorch import graph_to_pytorch
from models import DeeperGCN
from models import GATv2
from models import GCN
from models import GraphSAGE
from models import MLP
from models import PNA
from models import UniMPTransformer
import utils


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LRScheduler:
    """
    Implementation by Huggingface:
    https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py

    Create a schedule with a learning rate that decreases following the values
    of the cosine function between the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just
            decrease from the max value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def _compute_pna_histogram_tensor(
    train_dataset: torch_geometric.data.DataLoader,
) -> torch.Tensor:
    """Adapted from pytorch geometric's PNA example"""
    # Compute the maximum in-degree in the training data.
    max_degree = -1
    for data in train_dataset:
        computed_degree = degree(
            data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long
        )
        max_degree = max(max_degree, int(computed_degree.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_dataset:
        computed_degree = degree(
            data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long
        )
        deg += torch.bincount(computed_degree, minlength=deg.numel())
    return deg


def create_model(
    model_type: str,
    in_size: int,
    embedding_size: int,
    out_channels: int,
    gnn_layers: int,
    linear_layers: int,
    activation: str,
    residual: bool = False,
    dropout_rate: float = None,
    heads: int = None,
    train_dataset: torch_geometric.data.DataLoader = None,
) -> torch.nn.Module:  # sourcery skip: dict-assign-update-to-union
    """Create GNN model based on model type and parameters"""
    model_constructors = {
        "GCN": GCN,
        "GraphSAGE": GraphSAGE,
        "PNA": PNA,
        "GAT": GATv2,
        "UniMPTransformer": UniMPTransformer,
        "DeeperGCN": DeeperGCN,
        "MLP": MLP,
    }
    if model_type not in model_constructors:
        raise ValueError(f"Invalid model type: {model_type}")
    kwargs = {
        "in_size": in_size,
        "embedding_size": embedding_size,
        "out_channels": out_channels,
        "gnn_layers": gnn_layers,
        "linear_layers": linear_layers,
        "activation": activation,
        "dropout_rate": dropout_rate,
        "residual": residual,
    }
    if model_type in {"GAT", "UniMPTransformer"}:
        kwargs["heads"] = heads
    elif model_type == "PNA":
        kwargs["deg"] = _compute_pna_histogram_tensor(train_dataset)
    elif model_type == "DeeperGCN":
        del kwargs["residual"]
    return model_constructors[model_type](**kwargs)


def train(
    model: torch.nn.Module,
    device: torch.cuda.device,
    optimizer,
    train_loader: torch_geometric.data.DataLoader,
    epoch: int,
    gps=False,
):
    """Train GNN model on graph data"""
    model.train()
    pbar = tqdm(total=len(train_loader))
    pbar.set_description(f"Training epoch: {epoch:04d}")

    total_loss = total_examples = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)

        if gps:
            model.redraw_projection.redraw_projections()
            out = model(data.x, data.pe, data.edge_index, data.batch)
        else:
            out = model(data.x, data.edge_index)

        loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())

        pbar.update(1)

    pbar.close()
    return total_loss / total_examples


@torch.no_grad()
def test(
    model: torch.nn.Module,
    device: torch.cuda.device,
    data_loader: torch_geometric.data.DataLoader,
    epoch: int,
    mask: torch.Tensor,
    gps: bool = False,
):
    """Test GNN model on test set"""
    model.eval()
    pbar = tqdm(total=len(data_loader))
    pbar.set_description(f"Evaluating epoch: {epoch:04d}")

    mse = []
    for data in data_loader:
        data = data.to(device)
        if gps:
            out = model(data.x, data.pe, data.edge_index, data.batch)
        else:
            out = model(data.x, data.edge_index)

        if mask == "val":
            idx_mask = data.val_mask
        elif mask == "test":
            idx_mask = data.test_mask
        mse.append(F.mse_loss(out[idx_mask], data.y[idx_mask]).cpu())
        loss = torch.stack(mse)
        pbar.update(1)

    pbar.close()
    return math.sqrt(float(loss.mean()))


@torch.no_grad()
def inference(
    model: torch.nn.Module,
    device: torch.cuda.device,
    data_loader: torch_geometric.data.DataLoader,
    epoch: int,
    gps: bool = False,
):
    """Use model for inference or to evaluate on validation set"""
    model.eval()
    pbar = tqdm(total=len(data_loader))
    pbar.set_description(f"Evaluating epoch: {epoch:04d}")

    mse, outs, labels = [], [], []
    for data in data_loader:
        data = data.to(device)
        if gps:
            out = model(data.x, data.pe, data.edge_index, data.batch)
        else:
            out = model(data.x, data.edge_index)

        outs.extend(out[data.test_mask])
        labels.extend(data.y[data.test_mask])
        mse.append(F.mse_loss(out[data.test_mask], data.y[data.test_mask]).cpu())

        loss = torch.stack(mse)

        pbar.update(1)
    pbar.close()
    return math.sqrt(float(loss.mean())), outs, labels


def get_loader(
    data: torch_geometric.data.Data,
    mask: torch.Tensor,
    batch_size: int,
    shuffle: bool = False,
) -> torch_geometric.data.DataLoader:
    """Loads data into NeighborLoader for GNN training"""
    return NeighborLoader(
        data,
        num_neighbors=[5] * 5 + [3],
        batch_size=batch_size,
        input_nodes=getattr(data, mask),
        shuffle=shuffle,
    )


def setup_device(args: argparse.Namespace) -> torch.device:
    """Check for GPU and set device accordingly"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        return torch.device(f"cuda:{args.device}")
    return torch.device("cpu")


def save_model(
    model: torch.nn.Module,
    directory: pathlib.PosixPath,
    filename: str,
) -> None:
    """Save model state"""
    model_path = directory / filename
    torch.save(model.state_dict(), model_path)


def _set_optimizer(args: argparse.Namespace, model_params: Iterator[Parameter]):
    """Choose optimizer"""
    # set gradient descent optimizer
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model_params,
            lr=args.learning_rate,
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-5
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            model_params,
            lr=args.learning_rate,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=5,
            num_training_steps=args.epochs,
        )
    return optimizer, scheduler


def parse_arguments() -> argparse.Namespace:
    """Parse args for training GNN"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_config",
        type=str,
        help="Path to .yaml file with experimental conditions",
    )
    parser.add_argument("--model", type=str, default="GCN")
    parser.add_argument("--target", type=str, default="expression_median_only")
    parser.add_argument("--gnn_layers", type=int, default="2")
    parser.add_argument("--linear_layers", type=int, default="2")
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "leaky_relu", "gelu"],
        help="Activation function to use. Options: relu, leaky_relu, gelu (default: relu).",
    )
    parser.add_argument("--dimensions", type=int, default="256")
    parser.add_argument("--epochs", type=int, default="100")
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed to use (default: 42)"
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help="Which optimizer to use for learning. Options: AdamW or Adam (default: AdamW)",
    )
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--heads", type=int, required=False)
    parser.add_argument(
        "--device", type=int, default=0, help="which gpu to use if any (default: 0)"
    )
    parser.add_argument("--split_name", type=str)
    parser.add_argument("--total_random_edges", type=int, required=False, default=None)
    parser.add_argument("--graph_type", type=str, default="full")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--zero_nodes", action="store_true")
    parser.add_argument("--randomize_node_feats", action="store_true")
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--randomize_edges", action="store_true")
    return parser.parse_args()


def construct_save_string(base_str: List[str], args: argparse.Namespace) -> str:
    """Adds args to specify save string for model and logs"""
    components = [base_str]
    if args.residual:
        components.append("_residual")
    if args.heads:
        components.append(f"_heads{args.heads}")
    if args.dropout > 0.0:
        components.append(f"_dropout{args.dropout}")
    if args.randomize_node_feats:
        components.append("_randomnodefeats")
    if args.zero_nodes:
        components.append("_zeronodefeats")
    if args.total_random_edges:
        components.append(f"_totalrandomedges{args.total_random_edges}")
    return "_".join(components)


def _plot_loss_and_performance(
    device: torch.cuda.device,
    data_loader: torch_geometric.data.DataLoader,
    model_dir: pathlib.PosixPath,
    savestr: str,
    model: Optional[torch.nn.Module],
    best_validation: Optional[float] = None,
) -> None:
    """Plot training losses and performance"""
    # set params for plotting
    utils._set_matplotlib_publication_parameters()

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
    rmse, outs, labels = inference(
        model=model,
        device=device,
        data_loader=data_loader,
        epoch=0,
    )

    predictions_median = utils._tensor_out_to_array(outs, 0)
    labels_median = utils._tensor_out_to_array(labels, 0)

    # plot training losses
    utils.plot_training_losses(
        outfile=model_dir / "plots" / f"{savestr}_loss.png",
        savestr=savestr,
        log=model_dir / "log" / "training_log.txt",
    )

    # plot performance
    utils.plot_predicted_versus_expected(
        outfile=model_dir / "plots" / f"{savestr}_performance.png",
        savestr=savestr,
        predicted=predictions_median,
        expected=labels_median,
        rmse=rmse,
    )


def main() -> None:
    """Main function to train GNN on graph data!"""
    # Parse training settings
    args = parse_arguments()
    params = utils.parse_yaml(args.experiment_config)

    # set up helper variables
    working_directory = pathlib.Path(params["working_directory"])
    root_dir = working_directory / {params["experiment_name"]}
    savestr = construct_save_string(
        f"{params['experiment_name']}\
            _{args.model}\
            _target-{args.target}\
            _gnnlayers{args.gnn_layers}\
            _linlayers{args.linear_layers}\
            _{args.activation}\
            _dim{args.dimensions}\
            _batch{args.batch_size}\
            _{args.optimizer},\
            _{args.split_name}",
        args,
    )
    model_dir = working_directory / "models" / f"{savestr}"

    # make directories and set up training log
    for folder in ["logs", "plots"]:
        utils.dir_check_make(model_dir / folder)

    logging.basicConfig(
        filename=model_dir / "log" / "training_log.txt",
        level=logging.DEBUG,
    )

    # check for GPU
    device = setup_device(args)

    # get graph data
    data = graph_to_pytorch(
        experiment_name=params["experiment_name"],
        graph_type=args.graph_type,
        root_dir=root_dir,
        split_name=args.split_name,
        regression_target=args.target,
        test_chrs=params["training_targets"]["test_chrs"],
        val_chrs=params["training_targets"]["val_chrs"],
        randomize_feats=args.randomize_node_feats,
        zero_node_feats=args.zero_nodes,
        randomize_edges=args.randomize_edges,
        total_random_edges=(
            args.total_random_edges if args.randomize_edges > 0 else None
        ),
    )

    # temporary - to check number of edges for randomization tests
    print(f"Number of edges: {data.num_edges}")

    # set up data loaders
    train_loader = get_loader(
        data=data, mask="train_mask", batch_size=args.batch_size, shuffle=True
    )
    test_loader = get_loader(data=data, mask="test_mask", batch_size=args.batch_size)
    val_loader = get_loader(data=data, mask="val_mask", batch_size=args.batch_size)

    # CHOOSE YOUR WEAPON
    model = create_model(
        model=args.model,
        in_size=data.x.shape[1],
        embedding_size=args.dimensions,
        out_channels=1,
        gnn_layers=args.gnn_layers,
        linear_layers=args.linear_layers,
        activation=args.activation,
        dropout_rate=args.dropout_rate,
        heads=args.heads,
        train_dataset=args.model == "PNA",
    ).to(device)

    # set up optimizer & scheduler
    optimizer, scheduler = _set_optimizer(args=args, model_params=model.parameters())

    # start model training and initialize tensorboard utilities
    writer = SummaryWriter(model_dir / "log")
    epochs = args.epochs
    best_validation = stop_counter = 0
    prof = torch.profiler.profile(
        # activities=[
        #     torch.profiler.ProfilerActivity.CPU,
        #     torch.profiler.ProfilerActivity.CUDA,
        # ],
        schedule=torch.profiler.schedule(wait=1, warmup=4, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(model_dir / "log"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    )

    prof.start()
    print(f"Training for {epochs} epochs")
    for epoch in range(epochs + 1):
        loss = train(
            model=model,
            device=device,
            optimizer=optimizer,
            train_loader=train_loader,
            epoch=epoch,
            gps=args.model == "GPS",
        )
        print(f"Epoch: {epoch:03d}, Train: {loss}")
        writer.add_scalar("Training loss", loss, epoch)
        logging.info(f"Epoch: {epoch:03d}, Train: {loss}")

        val_acc = test(
            model=model,
            device=device,
            data_loader=val_loader,
            epoch=epoch,
            mask="val",
            gps=args.model == "GPS",
        )
        print(f"Epoch: {epoch:03d}, Validation: {val_acc:.4f}")
        logging.info(f"Epoch: {epoch:03d}, Validation: {val_acc:.4f}")
        writer.add_scalar("Validation RMSE", val_acc, epoch)

        test_acc = test(
            model=model,
            device=device,
            data_loader=test_loader,
            epoch=epoch,
            mask="test",
            gps=args.model == "GPS",
        )
        print(f"Epoch: {epoch:03d}, Test: {test_acc:.4f}")
        logging.info(f"Epoch: {epoch:03d}, Test: {test_acc:.4f}")
        writer.add_scalar("Test RMSE", test_acc, epoch)

        scheduler.step(val_acc)
        if args.early_stop:
            if epoch == 0 or val_acc < best_validation:
                stop_counter = 0
                best_validation = val_acc
                model_path = (
                    model_dir
                    / "models"
                    / f"{args.model}_{epoch}_mse_{best_validation}.pt"
                )
                torch.save(model.state_dict(), model_path)
            elif best_validation < val_acc:
                stop_counter += 1
            if stop_counter == 15:
                print("***********Early stopping!")
                final_acc = val_acc
                break

    # close out tensorboard utilities
    writer.flush()
    prof.stop()

    # Save final model
    save_model(
        model=model,
        directory=model_dir / "models",
        filename=f"{args.model}_final_mse_{val_acc}.pt",
    )

    # plot final model
    _plot_loss_and_performance(
        model=model,
        device=device,
        data_loader=test_loader,
        model_dir=model_dir,
        savestr=savestr,
    )

    # # GNN Explainer!
    # if args.model != "MLP":
    #     with open(
    #         "/ocean/projects/bio210019p/stevesho/data/preprocess/graphs/explainer_node_ids.pkl",
    #         "rb",
    #     ) as file:
    #         ids = pickle.load(file)
    #     explain_path = "/ocean/projects/bio210019p/stevesho/data/preprocess/explainer"
    #     explainer = Explainer(
    #         model=model,
    #         algorithm=GNNExplainer(epochs=200),
    #         explanation_type="model",
    #         node_mask_type="attributes",
    #         edge_mask_type="object",
    #         model_config=dict(mode="regression", task_level="node", return_type="raw"),
    #     )

    #     data = data.to(device)
    #     for index in random.sample(ids, 5):
    #         explanation = explainer(data.x, data.edge_index, index=index)

    #         print(f"Generated explanations in {explanation.available_explanations}")

    #         path = f"{explain_path}/feature_importance_{savestr}_{best_validation}.png"
    #         explanation.visualize_feature_importance(path, top_k=10)
    #         print(f"Feature importance plot has been saved to '{path}'")

    #         path = f"{explain_path}/subgraph_{savestr}_{best_validation}.pdf"
    #         explanation.visualize_graph(path)
    #         print(f"Subgraph visualization plot has been saved to '{path}'")


if __name__ == "__main__":
    main()
