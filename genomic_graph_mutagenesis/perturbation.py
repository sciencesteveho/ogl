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

import argparse
import csv
import math
import pickle
import random
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from gnn import GATv2
from graph_to_pytorch import graph_to_pytorch
from utils import parse_yaml


def _tensor_out_to_array(tensor, idx):
    return np.stack([x[idx].cpu().numpy() for x in tensor], axis=0)


def _load_GAT_model_for_inference(
    in_size, 
    embedding_size,
    num_layers,
    checkpoint,
    map_location,
    device,
):
    """_summary_ of function"""
    model = GATv2(
        in_size=in_size,
        embedding_size=embedding_size,
        out_channels=1,
        num_layers=num_layers,
        heads=2,
    ).to(device)
    
    checkpoint = torch.load(checkpoint, map_location=map_location)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)

    return model


def _device_check():
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        return torch.device("cuda:" + str(0)), torch.device("cuda:" + str(0))
    else:
        return torch.device("cpu"), torch.device("cpu")
    

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


def main() -> None:
    """Main function"""

    # prepare stuff
    graph = "/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/curated/graphs/curated_full_graph_scaled.pkl"
    graph_idxs = "/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/curated/graphs/curated_full_graph_idxs.pkl"
    checkpoint_file = "/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/models/curated_GAT_2_500_0.0001_batch32_neighbor_full_idx_dropout_scaled_expression_only/curated_GAT_2_500_0.0001_batch32_neighbor_full_idx_dropout_scaled_expression_only_mse_1.8369761025042963.pt"
    savedir='/ocean/projects/bio210019p/stevesho/data/preprocess/pickles'
    savestr='curated'

    # parse yaml for params, used to load data
    config = '/ocean/projects/bio210019p/stevesho/data/preprocess/genomic_graph_mutagenesis/configs/ablation_experiments/curated.yaml'
    # parser = argparse.ArgumentParser()
    params = parse_yaml(config)
    
    # open graph
    with open(graph, "rb") as file:
        graph = pickle.load(file)

    # open idxs
    with open(graph_idxs, "rb") as file:
        graph_idxs = pickle.load(file)
        
    # check for device
    device, map_location = _device_check()

    # initialize model and load checkpoint weights
    model = _load_GAT_model_for_inference(
        in_size=41,
        embedding_size=500,
        num_layers=2,
        checkpoint=checkpoint_file,
        map_location=map_location,
        device=device,
    )
    
    working_directory = params["working_directory"]
    root_dir = f"{working_directory}/{params['experiment_name']}"

    # load data
    data = graph_to_pytorch(
        experiment_name=params["experiment_name"],
        graph_type='full',
        root_dir=root_dir,
        targets_types=params["training_targets"]["targets_types"],
        test_chrs=params["training_targets"]["test_chrs"],
        val_chrs=params["training_targets"]["val_chrs"],
    )
    
    # get test data cuz we here
    batch_size=32
    test_loader = NeighborLoader(
        data,
        num_neighbors=[5, 5, 5, 5, 5, 3],
        batch_size=batch_size,
        input_nodes=data.test_mask,
    )
    _, outs, labels = inference(
        model=model,
        device=device,
        data_loader=test_loader,
    )
    
    predictions_median = _tensor_out_to_array(outs, 0)
    labels_median = _tensor_out_to_array(labels, 0)
    
    with open(f'{savedir}/{savestr}_median_predictions.pkl', 'wb') as file:
        pickle.dump(predictions_median, file)
    
    with open(f'{savedir}/{savestr}_median_labels.pkl', 'wb') as file:
        pickle.dump(labels_median, file)


    # perform feature perturbations
    # remove h3k27ac
    perturbed_data = graph_to_pytorch(
        experiment_name=params["experiment_name"],
        graph_type='full',
        root_dir=root_dir,
        targets_types=params["training_targets"]["targets_types"],
        test_chrs=params["training_targets"]["test_chrs"],
        val_chrs=params["training_targets"]["val_chrs"],
        node_perturbation="h3k27ac",
    )
    loader = NeighborLoader(
        data=perturbed_data,
        num_neighbors=[5, 5, 5, 5, 5, 3],
        batch_size=batch_size,
        input_nodes=perturbed_data.test_mask,
    )
    _, outs, labels = inference(
        model=model,
        device=device,
        data_loader=loader,
        epoch=0,
    )
    labels = _tensor_out_to_array(labels, 0)
    h3k27ac_perturbed = _tensor_out_to_array(outs, 0)
    
    with open(f"{savedir}/{savestr}_h3k27ac_perturbed_expression.pkl", "wb") as f:
        pickle.dump(h3k27ac_perturbed, f)

    with open(f"{savedir}/{savestr}_h3k27ac_labels.pkl", "wb") as f:
        pickle.dump(labels, f)

    # remove h3k4me3
    perturbed_data = graph_to_pytorch(
        experiment_name=params["experiment_name"],
        graph_type='full',
        root_dir=root_dir,
        targets_types=params["training_targets"]["targets_types"],
        test_chrs=params["training_targets"]["test_chrs"],
        val_chrs=params["training_targets"]["val_chrs"],
        node_perturbation="h3k4me3",
    )
    loader = NeighborLoader(
        data=perturbed_data,
        num_neighbors=[5, 5, 5, 5, 5, 3],
        batch_size=batch_size,
        input_nodes=perturbed_data.test_mask,
    )
    _, outs, labels = inference(
        model=model,
        device=device,
        data_loader=loader,
        epoch=0,
    )
    
    labels = _tensor_out_to_array(labels, 0)
    h3k4me3_perturbed = _tensor_out_to_array(outs, 0)
    
    with open(f"{savedir}/{savestr}_h3k4me3_perturbed_expression.pkl", "wb") as f:
        pickle.dump(h3k4me3_perturbed, f)

    with open(f"{savedir}/{savestr}_h3k4me3_labels.pkl", "wb") as f:
        pickle.dump(labels, f)

    
if __name__ == "__main__":
    main()
