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
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import torch_geometric
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import degree
from tqdm import tqdm

from graph_to_pytorch import graph_to_pytorch
from models import GATv2
from models import GCN
from models import GraphSAGE
from models import MLP
from models import PNA
import utils


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
    train_dataset,
    dropout_rate: float = None,
    heads: int = None,
) -> torch.nn.Module:  # sourcery skip: dict-assign-update-to-union
    """Create GNN model based on model type and parameters"""
    model_constructors = {
        "GCN": GCN,
        "GraphSAGE": GraphSAGE,
        "PNA": PNA,
        "GAT": GATv2,
        "UniMPTransformer": UniMPTransformer,
        "DeeperGCN": DeeperGCN,
        "Graphormer": Graphormer,
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
    }
    if model_type == "GAT":
        kwargs["heads"] = heads
    elif model_type == "GPS":
        kwargs.update({"walk_length": 20, "channels": embedding_size, "pe_dim": 8})
        del kwargs["out_channels"]
        del kwargs["linear_layers"]
        del kwargs["dropout_rate"]
    elif model_type == "PNA":
        kwargs["deg"] = _compute_pna_histogram_tensor(train_dataset)
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


def parse_arguments() -> argparse.Namespace:
    """Parse args for training GNN"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_config",
        type=str,
        help="Path to .yaml file with experimental conditions",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="GCN",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="expression_median_only",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default="2",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default="600",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default="50",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed to use (default: 42)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="which gpu to use if any (default: 0)",
    )
    parser.add_argument(
        "--graph_type",
        type=str,
        default="full",
    )
    parser.add_argument(
        "--zero_nodes",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--randomize_node_feats",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--early_stop",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--randomize_edges",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--total_random_edges",
        type=int,
        default="0",
    )
    return parser.parse_args()


def construct_save_string(base_str: List[str], args: argparse.Namespace) -> str:
    """Adds args to specify save string for model and logs"""
    components = [base_str]
    if args.dropout > 0.0:
        components.append(f"dropout_{args.dropout}")
    if args.randomize_node_feats:
        components.append("random_node_feats")
    if args.randomize_edges:
        components.append("randomize_edges")
    if args.total_random_edges > 0:
        components.append(f"totalrandomedges_{args.total_random_edges}")
    return "_".join(components)


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


def main() -> None:
    """_summary_"""
    # Parse training settings
    args = parse_arguments()
    params = utils.parse_yaml(args.experiment_config)

    # set up helper variables
    working_directory = pathlib.Path(params["working_directory"])
    root_dir = working_directory / {params["experiment_name"]}
    savestr = construct_save_string(
        f"{params['experiment_name']}_{args.model}_{args.layers}_{args.dimensions}_{args.learning_rate}_batch{args.batch_size}_{args.graph_type}_idx",
        args,
    )

    # make directories and set up training log
    utils.dir_check_make(working_directory / "models/logs")
    utils.dir_check_make(working_directory / f"models/{savestr}")

    logging.basicConfig(
        filename=working_directory / f"models/logs/{savestr}.log",
        level=logging.DEBUG,
    )

    # check for GPU
    device = setup_device(args)

    data = graph_to_pytorch(
        experiment_name=params["experiment_name"],
        graph_type=args.graph_type,
        root_dir=root_dir,
        regression_target=args.target,
        test_chrs=params["training_targets"]["test_chrs"],
        val_chrs=params["training_targets"]["val_chrs"],
        randomize_feats=args.randomize_node_feats,
        zero_node_feats=args.zero_nodes,
        randomize_edges=args.randomize_edges,
        total_random_edges=args.total_random_edges,
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
    heads = 2 if args.model == "GAT" else None
    dropout_rate = args.dropout if args.dropout > 0.0 else None
    model = create_model(
        args.model,
        in_size=data.x.shape[1],
        embedding_size=args.dimensions,
        out_channels=1,
        gnn_layers=args.gnn_layers,
        linear_layers=args.linear_layers,
        heads=heads,
        dropout_rate=dropout_rate,
        train_dataset=args.model == "PNA",
    ).to(device)

    # set gradient descent optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    # set scheduler to reduce learning rate on plateau
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-5
    )

    epochs = args.epochs
    print(f"Training for {epochs} epochs")
    best_validation = stop_counter = 0
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

        scheduler.step(val_acc)
        if args.early_stop:
            if epoch == 0 or val_acc < best_validation:
                stop_counter = 0
                best_validation = val_acc
                model_path = (
                    working_directory
                    / f"models/{savestr}/{savestr}_early_epoch_{epoch}_mse_{best_validation}.pt"
                )
                torch.save(model.state_dict(), model_path)
            elif best_validation < val_acc:
                stop_counter += 1
            if stop_counter == 15:
                print("***********Early stopping!")
                break

    save_model(
        model,
        working_directory / f"models/{savestr}",
        f"{savestr}_mse_{best_validation}.pt",
    )

    # set params for plotting(())
    utils._set_matplotlib_publication_parameters()

    # calculate and plot spearmann rho, predictions vs. labels
    # first, load checkpoints
    checkpoint = torch.load(
        f"{working_directory}/models/{savestr}/{savestr}_mse_{best_validation}.pt",
        map_location=torch.device("cuda:0"),
    )
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)

    # get predictions
    rmse, outs, labels = inference(
        model=model,
        device=device,
        data_loader=test_loader,
        epoch=0,
        gps=args.model == "GPS",
    )

    predictions_median = utils._tensor_out_to_array(outs, 0)
    labels_median = utils._tensor_out_to_array(labels, 0)

    experiment_name = params["experiment_name"]
    if args.randomize_node_feats == "true":
        experiment_name = f"{experiment_name}_random_node_feats"
    if args.zero_nodes == "true":
        experiment_name = f"{experiment_name}_zero_node_feats"
    if args.randomize_edges == "true":
        experiment_name = f"{experiment_name}_randomize_edges"
    if args.total_random_edges > 0:
        experiment_name = (
            f"{experiment_name}_totalrandomedges_{args.total_random_edges}"
        )

    # plot performance
    utils.plot_predicted_versus_expected(
        expected=labels_median,
        predicted=predictions_median,
        experiment_name=experiment_name,
        model=args.model,
        layers=args.layers,
        width=args.dimensions,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        outdir=f"{working_directory}/models/plots",
        rmse=rmse,
    )

    # plot training losses
    utils.plot_training_losses(
        log=f"{working_directory}/models/logs/{savestr}.log",
        experiment_name=experiment_name,
        model=args.model,
        layers=args.layers,
        width=args.dimensions,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        outdir=f"{working_directory}/models/plots",
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
