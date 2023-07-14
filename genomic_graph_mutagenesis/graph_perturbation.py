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
import os
import pickle
import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import RandomNodeLoader
from tqdm import tqdm

from gnn import GAT
from gnn import GCN
from gnn import GraphSAGE
from graph_to_pytorch import graph_to_pytorch
from utils import TISSUES


def _get_idxs_for_coessential_pairs(
    coessential_pos: str,
    coessential_neg: str,
    graph_idxs: Dict[str, str],
) -> List[Tuple[int, int]]:
    """_summary_ of function"""

    def _dict_init(first_pairs, second_pairs):
        keys = []
        for tissue in TISSUES:
            for pair in first_pairs:
                try:
                    keys.append(graph_idxs[f"{pair[0]}_{tissue}"])
                except KeyError:
                    pass
        return set(keys)

    pos_pairs = [
        (line[0], line[1])
        for line in csv.reader(open(coessential_pos, newline=""), delimiter="\t")
    ]

    neg_pairs = [
        (line[0], line[1])
        for line in csv.reader(open(coessential_neg, newline=""), delimiter="\t")
    ]

    pos_keys = _dict_init(pos_pairs, neg_pairs)
    pos_coessn_idxs = {key: [] for key in pos_keys}  # init dict
    for tissue in TISSUES:
        for tup in pos_pairs:
            if graph_idxs[f"{tup[1]}_{tissue}"] < graph_idxs[f"{tup[0]}_{tissue}"]:
                try:
                    pos_coessn_idxs[graph_idxs[f"{tup[0]}_{tissue}"]].append(
                        graph_idxs[f"{tup[1]}_{tissue}"]
                    )
                except KeyError:
                    pass
    return pos_coessn_idxs


def _random_gene_pairs(
    coessential_idxs: Dict[str, str],
    graph_idxs: Dict[str, str],
) -> List[Tuple[int, int]]:
    """_summary_ of function"""
    random_pool = list(graph_idxs.values())
    for key in coessential_idxs.keys():
        num_elements = len(coessential_idxs[key])
        coessential_idxs[key] = random.sample((random_pool), num_elements)

    return coessential_idxs


def _remove_node_features():
    """_summary_ of function"""


@torch.no_grad()
def test(model, device, data_loader, epoch):
    model.eval()

    pbar = tqdm(total=len(data_loader))
    pbar.set_description(f"Evaluating epoch: {epoch:04d}")

    mse = []
    for data in data_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)

    pbar.close()
    return out


def main(
    mode: str,
    graph: str,
    graph_idxs: str,
    feat_perturbation: bool = False,
    coessentiality: bool = False,
) -> None:
    """Main function"""

    def _perturb_loader(data):
        return RandomNodeLoader(
            data=data,
            num_parts=250,
            shuffle=False,
            num_workers=5,
        )

    # check for device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        device = torch.device("cuda:" + str(0))
        map_location = torch.device("cuda:" + str(0))
    else:
        device = torch.device("cpu")
        map_location = torch.device("cpu")

    # open graph
    with open(graph, "rb") as file:
        graph = pickle.load(file)

    # open idxs
    with open(graph_idxs, "rb") as file:
        graph_idxs = pickle.load(file)

    # prepare IDXs for different perturbations
    coessential_idxs = _get_idxs_for_coessential_pairs(
        coessential_pos="/ocean/projects/bio210019p/stevesho/data/preprocess/comparisons/coessential_gencode_named_pos.txt",
        coessential_neg="/ocean/projects/bio210019p/stevesho/data/preprocess/comparisons/coessential_gencode_named_neg.txt",
        graph_idxs=graph_idxs,
    )

    random_co_idxs = _random_gene_pairs(
        coessential_idxs=coessential_idxs,
        graph_idxs=graph_idxs,
    )

    test_genes = random.sample(list(coessential_idxs.keys()), 100)
    test_random = random.sample(list(random_co_idxs.keys()), 100)

    # initialize model model
    model = GraphSAGE(
        in_size=data.x.shape[1],
        embedding_size=300,
        out_channels=4,
        layers=2,
    ).to(device)

    # load checkpoint
    checkpoint = torch.load(checkpoint_file, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)

    # prepare feature perturbation data
    if feat_perturbation:
        perturbed_data = graph_to_pytorch(
            root_dir="/ocean/projects/bio210019p/stevesho/data/preprocess",
            graph_type="full",
            node_perturbation="h3k27ac",
        )
        loader = _perturb_loader(perturbed_data)
        inference = test(
            model=model,
            device=device,
            data_loader=loader,
            epoch=0,
        )

        perturbed_data = graph_to_pytorch(
            root_dir="/ocean/projects/bio210019p/stevesho/data/preprocess",
            graph_type="full",
            node_perturbation="h3k4me1",
        )
        loader = _perturb_loader(perturbed_data)
        inference = test(
            model=model,
            device=device,
            data_loader=loader,
            epoch=0,
        )

    # coessentiality
    # get baseline expression
    if coessentiality:
        baselines = []
        changes = []
        for gene in coessential_idxs.keys():
            changed = []
            baseline_data = graph_to_pytorch(
                root_dir="/ocean/projects/bio210019p/stevesho/data/preprocess",
                graph_type="full",
                single_gene=gene,
            )
            loader = _perturb_loader(baseline_data)
            # baseline_measure
            for co_gene in coessential_idxs[gene]:
                perturbed_graph = graph_to_pytorch(
                    root_dir="/ocean/projects/bio210019p/stevesho/data/preprocess",
                    graph_type="full",
                    single_gene=gene,
                    node_remove_edges=co_gene,
                )


if __name__ == "__main__":
    main(
        model="/ocean/projects/bio210019p/shared/model_checkpoint.pt",
        graph="/ocean/projects/bio210019p/stevesho/data/preprocess/graphs/scaled/all_tissue_full_graph_scaled.pkl",
        graph_idxs="/ocean/projects/bio210019p/stevesho/data/preprocess/graphs/all_tissue_full_graph_idxs.pkl",
    )

model = "/ocean/projects/bio210019p/shared/model_checkpoint.pt"
graph = "/ocean/projects/bio210019p/stevesho/data/preprocess/graphs/scaled/all_tissue_full_graph_scaled.pkl"
graph_idxs = "/ocean/projects/bio210019p/stevesho/data/preprocess/graphs/all_tissue_full_graph_idxs.pkl"

# pos_idxs, neg_idxs = {}, {}
# for tissue in TISSUES:
#     for tup in pos_pairs:
#         pos_idxs[graph_idxs[f"{tup[0]}_{tissue}"]] = graph_idxs[f"{tup[1]}_{tissue}"]
#     pos_idxs[]
#     pos_idxs.extend(
#         (graph_idxs[f"{tup[0]}_{tissue}"], graph_idxs[f"{tup[1]}_{tissue}"])
#         for tup in pos_pairs
#     )
#     neg_idxs.extend(
#         (graph_idxs[f"{tup[0]}_{tissue}"], graph_idxs[f"{tup[1]}_{tissue}"])
#         for tup in neg_pairs
#     )
