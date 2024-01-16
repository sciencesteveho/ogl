#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] implement tensorboard
# - [ ] add dropout as arg

"""Code to train GNNs on the graph data!"""

import argparse
import math

import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from graph_to_pytorch import graph_to_pytorch
from models import GATv2
from models import GCN
from models import GPSTransformer
from models import GraphSAGE
from models import MLP
from utils import DataVizUtils
from utils import GeneralUtils


def create_model(
    model_type,
    in_size,
    embedding_size,
    out_channels,
    num_layers,
    heads=None,
):
    if model_type == "GraphSAGE":
        return GraphSAGE(
            in_size=in_size,
            embedding_size=embedding_size,
            out_channels=out_channels,
            num_layers=num_layers,
        )
    elif model_type == "GCN":
        return GCN(
            in_size=in_size,
            embedding_size=embedding_size,
            out_channels=out_channels,
            num_layers=num_layers,
        )
    elif model_type == "GAT":
        return GATv2(
            in_size=in_size,
            embedding_size=embedding_size,
            out_channels=out_channels,
            num_layers=num_layers,
            heads=heads,
        )
    elif model_type == "MLP":
        return MLP(
            in_size=in_size, embedding_size=embedding_size, out_channels=out_channels
        )
    elif model_type == "GPS":
        return GPSTransformer(
            in_size=in_size,
            embedding_size=embedding_size,
            walk_length=20,
            channels=embedding_size,
            pe_dim=8,
            num_layers=num_layers,
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")


@torch.no_grad()
def inference(model, device, data_loader, epoch, gps=False):
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

    # loss = torch.stack(mse)
    pbar.close()
    return math.sqrt(float(loss.mean())), outs, labels


def main() -> None:
    """_summary_"""
    # Parse training settings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_config",
        type=str,
        help="Path to .yaml file with experimental conditions",
        default="/ocean/projects/bio210019p/stevesho/data/preprocess/genomic_graph_mutagenesis/configs/ablation_experiments/regulatory_only_all_loops_test_8_9_val_7_13_mediantpm.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="GAT",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default="2",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default="256",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default="25",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed to use (default: 42)",
    )
    parser.add_argument(
        "--loader",
        type=str,
        default="neighbor",
        help="'neighbor' or 'random' node loader (default: 'random')",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
    )
    parser.add_argument(
        "--idx",
        type=str,
        default="true",
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
        type=str,
        default="false",
    )
    parser.add_argument(
        "--randomize_node_feats",
        type=str,
        default="false",
    )
    parser.add_argument(
        "--early_stop",
        type=str,
        default="true",
    )
    parser.add_argument(
        "--expression_only",
        type=str,
        default="true",
    )
    parser.add_argument(
        "--randomize_edges",
        type=str,
        default="true",
    )
    parser.add_argument(
        "--total_random_edges",
        type=int,
        default="10000000",
    )
    args = parser.parse_args()

    params = GeneralUtils.parse_yaml(args.experiment_config)

    # set up helper variables
    working_directory = params["working_directory"]
    print(f"Working directory: {working_directory}")
    root_dir = f"{working_directory}/{params['experiment_name']}"

    # check for GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")

    data = graph_to_pytorch(
        experiment_name=params["experiment_name"],
        graph_type=args.graph_type,
        root_dir=root_dir,
        targets_types=params["training_targets"]["targets_types"],
        test_chrs=params["training_targets"]["test_chrs"],
        val_chrs=params["training_targets"]["val_chrs"],
        randomize_feats=args.randomize_node_feats,
        zero_node_feats=args.zero_nodes,
        randomize_edges=args.randomize_edges,
        total_random_edges=args.total_random_edges,
    )

    # CHOOSE YOUR WEAPON
    model = create_model(
        args.model,
        in_size=data.x.shape[1],
        embedding_size=args.dimensions,
        out_channels=1,
        num_layers=args.layers,
        heads=2 if args.model == "GAT" else None,
    ).to(device)

    # set params for plotting
    DataVizUtils._set_matplotlib_publication_parameters()

    # calculate and plot spearmann rho, predictions vs. labels
    # first, load checkpoints
    checkpoint_file = "regulatory_only_all_loops_test_8_9_val_7_13_mediantpm_GAT_2_128_0.0001_batch64_neighbor_full_idx_dropout_expression_only_randomize_edges_totalrandomedges_100000000_early_epoch_21_mse_1.7035869706785296.pt"
    checkpoint_dir = "/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/models/regulatory_only_all_loops_test_8_9_val_7_13_mediantpm_GAT_2_128_0.0001_batch64_neighbor_full_idx_dropout_expression_only_randomize_edges_totalrandomedges_100000000"
    checkpoint = torch.load(
        f"{checkpoint_dir}/{checkpoint_file}",
        map_location=torch.device("cuda:" + str(0)),
    )
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)

    data = graph_to_pytorch(
        experiment_name=params["experiment_name"],
        graph_type=args.graph_type,
        root_dir=root_dir,
        targets_types=params["training_targets"]["targets_types"],
        test_chrs=params["training_targets"]["test_chrs"],
        val_chrs=params["training_targets"]["val_chrs"],
        randomize_feats=args.randomize_node_feats,
        zero_node_feats=args.zero_nodes,
        randomize_edges=args.randomize_edges,
        total_random_edges=args.total_random_edges,
    )

    for percentile in [None, 10, 25, 50, 75, 90]:
        test_loader = NeighborLoader(
            data,
            num_neighbors=[5, 5, 5, 5, 5, 3],
            batch_size=args.batch_size,
            input_nodes=data.test_mask,
            percentile_cutoff=percentile,
        )

        # get predictions
        rmse, outs, labels = inference(
            model=model,
            device=device,
            data_loader=test_loader,
            epoch=0,
        )

        predictions_median = GeneralUtils._tensor_out_to_array(outs, 0)
        labels_median = GeneralUtils._tensor_out_to_array(labels, 0)

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
        experiment_name = f"{experiment_name}_cutoff_{percentile}"

        # plot performance
        DataVizUtils.plot_predicted_versus_expected(
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


if __name__ == "__main__":
    main()
