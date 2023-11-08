#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] first TODO
#   - [ ] nested TODO
#
# Load model
# Load data w/ modification
# loader = NeighborLoader(new_data, input_nodes=new_node_ids, ...)
# out = model(new_data.x, new_data.edge_index)
# compare co-essential perturbations to random perturbation


"""_summary_ of project"""

import csv
import math
import os
import pickle
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from gnn import GATv2
from graph_to_pytorch import graph_to_pytorch
from utils import filtered_genes_from_bed
from utils import TISSUES_early_testing


def _tensor_out_to_array(tensor, idx):
    return np.stack([x[idx].cpu().numpy() for x in tensor], axis=0)


def _load_GAT_model_for_inference(
    in_size, 
    embedding_size,
    num_layers,
    checkpoint,
    map_location
):
    """_summary_ of function"""
    model = GATv2(
        in_size=in_size,
        embedding_size=embedding_size,
        out_channels=1,
        num_layers=num_layers,
        heads=2,
    ).to_device()
    
    checkpoint = torch.load(checkpoint, map_location=map_location)
    model.load_state_dict(checkpoint, strict=False)
    model.to_device()

    return model


def _load_graph_data(
    experiment_name,
    graph_type,
    root_dir,
    targets_types,
    test_chrs,
    val_chrs,
    randomize_feats,
    zero_node_feats,
):
    return graph_to_pytorch(
        experiment_name=experiment_name,
        graph_type=graph_type,
        root_dir=root_dir,
        targets_types=targets_types,
        test_chrs=test_chrs,
        val_chrs=val_chrs,
        randomize_feats=randomize_feats,
        zero_node_feats=zero_node_feats,
    )


@torch.no_grad()
def inference(model, device, data_loader, epoch):
    model.eval()

    pbar = tqdm(total=len(data_loader))
    pbar.set_description(f"Evaluating epoch: {epoch:04d}")

    mse, outs, labels = [], [], []
    for data in data_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)

        # calculate loss
        outs.extend(out[data.test_mask])
        labels.extend(data.y[data.test_mask])
        mse.append(F.mse_loss(out[data.test_mask], data.y[data.test_mask]).cpu())
        loss = torch.stack(mse)

        pbar.update(1)

    pbar.close()
    # print(spearman(torch.stack(outs), torch.stack(labels)))
    return math.sqrt(float(loss.mean())), outs, labels


def main(
    mode: str,
    graph: str,
    graph_idxs: str,
    need_baseline: bool = False,
    feat_perturbation: bool = False,
    coessentiality: bool = False,
) -> None:
    """Main function"""

    def _perturb_loader(data):
        test_loader = NeighborLoader(
            data,
            num_neighbors=[5, 5, 5, 5, 5, 3],
            batch_size=1024,
            input_nodes=data.test_mask,
        )

    # check for device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        device = torch.device("cuda:" + str(0))
        map_location = torch.device("cuda:" + str(0))
    else:
        device = torch.device("cpu")
        map_location = torch.device("cpu")

    # only using to check size for model init
    # data = graph_to_pytorch(
    #     root_dir="/ocean/projects/bio210019p/stevesho/data/preprocess",
    #     graph_type="full",
    # )
    # data.x.shape[1]  # 41

    # prepare stuff
    graph = "/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/regulatory_only_all_loops_test_8_9_val_7_13_mediantpm/graphs/regulatory_only_all_loops_test_8_9_val_7_13_mediantpm_full_graph_scaled.pkl"
    graph_idxs = "/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/regulatory_only_all_loops_test_8_9_val_7_13_mediantpm/graphs/regulatory_only_all_loops_test_8_9_val_7_13_mediantpm_full_graph_idxs.pkl"

    # open graph
    with open(graph, "rb") as file:
        graph = pickle.load(file)

    # open idxs
    with open(graph_idxs, "rb") as file:
        graph_idxs = pickle.load(file)

    # initialize model model
    model = GraphSAGE(
        in_size=41,
        embedding_size=250,
        out_channels=2,
        num_layers=2,
    ).to(device)

    # load checkpoint
    checkpoint_file = "/ocean/projects/bio210019p/stevesho/data/preprocess/GraphSAGE_2_250_5e-05_batch1024_neighbor_idx_early_epoch_52_mse_0.8029395813403142.pt"
    checkpoint = torch.load(checkpoint_file, map_location=map_location)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)

    if need_baseline:
        # get baseline expression
        baseline_data = graph_to_pytorch(
            root_dir="/ocean/projects/bio210019p/stevesho/data/preprocess",
            graph_type="full",
            only_expression_no_fold=True,
        )
        loader = NeighborLoader(
            data=baseline_data,
            num_neighbors=[5, 5, 5, 5, 5, 3],
            batch_size=1024,
            input_nodes=baseline_data.test_mask,
        )
        rmse, outs, labels = inference(
            model=model,
            device=device,
            data_loader=loader,
            epoch=0,
        )

        predictions_median = _tensor_out_to_array(outs, 0)
        # predictions_fold = _tensor_out_to_array(outs, 1)
        labels_median = _tensor_out_to_array(labels, 0)
        # labels_fold = _tensor_out_to_array(labels, 1)

        with open("base_predictions_median.pkl", "wb") as f:
            pickle.dump(predictions_median, f)

        # with open("predictions_fold.pkl", "wb") as f:
        #     pickle.dump(predictions_fold, f)

        with open("base_labels_median.pkl", "wb") as f:
            pickle.dump(labels_median, f)

        with open("labels_fold.pkl", "wb") as f:
            pickle.dump(labels_fold, f)

    # prepare feature perturbation data
    if feat_perturbation:
        perturbed_data = graph_to_pytorch(
            root_dir="/ocean/projects/bio210019p/stevesho/data/preprocess",
            graph_type="full",
            node_perturbation="h3k27ac",
        )
        loader = NeighborLoader(
            data=perturbed_data,
            num_neighbors=[5, 5, 5, 5, 5, 3],
            batch_size=1024,
            input_nodes=perturbed_data.test_mask,
        )
        rmse, outs, labels = inference(
            model=model,
            device=device,
            data_loader=loader,
            epoch=0,
        )
        labels = _tensor_out_to_array(labels, 0)
        h3k27ac_perturbed = _tensor_out_to_array(outs, 0)
        with open("h3k27ac_perturbed_expression.pkl", "wb") as f:
            pickle.dump(h3k27ac_perturbed, f)

        with open("h3k27ac_labels.pkl", "wb") as f:
            pickle.dump(labels, f)

        perturbed_data = graph_to_pytorch(
            root_dir="/ocean/projects/bio210019p/stevesho/data/preprocess",
            graph_type="full",
            node_perturbation="h3k4me3",
        )
        loader = NeighborLoader(
            data=perturbed_data,
            num_neighbors=[5, 5, 5, 5, 5, 3],
            batch_size=1024,
            input_nodes=perturbed_data.test_mask,
        )
        rmse, outs, labels = inference(
            model=model,
            device=device,
            data_loader=loader,
            epoch=0,
        )
        labels = _tensor_out_to_array(labels, 0)
        h3k4me3_perturbed = _tensor_out_to_array(outs, 0)
        with open("h3k4me3_perturbed_expression.pkl", "wb") as f:
            pickle.dump(h3k4me3_perturbed, f)

        with open("h3k4me3_labels.pkl", "wb") as f:
            pickle.dump(labels, f)

if __name__ == "__main__":
    main()
    # main(
    #     model="/ocean/projects/bio210019p/shared/model_checkpoint.pt",
    #     graph="/ocean/projects/bio210019p/stevesho/data/preprocess/graphs/scaled/all_tissue_full_graph_scaled.pkl",
    #     graph_idxs="/ocean/projects/bio210019p/stevesho/data/preprocess/graphs/all_tissue_full_graph_idxs.pkl",
    # )

# pos_idxs, neg_idxs = {}, {}
# for tissue in TISSUES_early_testing:
#     for tup in positive_pairs:
#         pos_idxs[graph_idxs[f"{tup[0]}_{tissue}"]] = graph_idxs[f"{tup[1]}_{tissue}"]
#     pos_idxs[]
#     pos_idxs.extend(
#         (graph_idxs[f"{tup[0]}_{tissue}"], graph_idxs[f"{tup[1]}_{tissue}"])
#         for tup in positive_pairs
#     )
#     neg_idxs.extend(
#         (graph_idxs[f"{tup[0]}_{tissue}"], graph_idxs[f"{tup[1]}_{tissue}"])
#         for tup in negative_pairs
#     )
