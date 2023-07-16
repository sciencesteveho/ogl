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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from gnn import GATv2
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
    graph = "/ocean/projects/bio210019p/stevesho/data/preprocess/graphs/scaled/all_tissue_full_graph_scaled.pkl"
    graph_idxs = "/ocean/projects/bio210019p/stevesho/data/preprocess/graphs/all_tissue_full_graph_idxs.pkl"

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

    def _tensor_to_numpy(tensor):
        return tensor.cpu().detach().numpy()

    if need_baseline:
        # get baseline expression
        baseline_data = graph_to_pytorch(
            root_dir="/ocean/projects/bio210019p/stevesho/data/preprocess",
            graph_type="full",
        )
        loader = NeighborLoader(
            data=baseline_data,
            num_neighbors=[5, 5, 5, 5, 5, 3],
            batch_size=1024,
            input_nodes=baseline_data.test_mask,
        )
        rmse, outs, labels = test(
            model=model,
            device=device,
            data_loader=loader,
            epoch=0,
        )
        with open("baseline_expression_pred.pkl", "wb") as f:
            pickle.dump(outs, f)

        with open("baseline_expression_labels.pkl", "wb") as f:
            pickle.dump(labels, f)

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
        with open("h3k27ac_expression.pkl", "wb") as f:
            pickle.dump(inference, f)

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
        with open("h3k4me1_expression.pkl", "wb") as f:
            pickle.dump(inference, f)

    # coessentiality
    # get baseline expression
    if coessentiality:
        baselines = {}
        for gene in coessential_idxs.keys():
            baselines[gene] = []
            baseline_data = graph_to_pytorch(
                root_dir="/ocean/projects/bio210019p/stevesho/data/preprocess",
                graph_type="full",
                single_gene=gene,
            )
            loader = _perturb_loader(baseline_data)
            baseline = test(
                model=model,
                device=device,
                data_loader=loader,
                epoch=0,
            )
            baselines.append(baseline)
            # baseline_measure
            for co_gene in coessential_idxs[gene]:
                perturbed_graph = graph_to_pytorch(
                    root_dir="/ocean/projects/bio210019p/stevesho/data/preprocess",
                    graph_type="full",
                    single_gene=gene,
                    node_remove_edges=co_gene,
                )
                loader = _perturb_loader(perturbed_graph)
                inference = test(
                    model=model,
                    device=device,
                    data_loader=loader,
                    epoch=0,
                )
                baselines[gene].append(inference)

        with open("coessentiality.pkl", "wb") as f:
            pickle.dump(baselines, f)


if __name__ == "__main__":
    main()
    # main(
    #     model="/ocean/projects/bio210019p/shared/model_checkpoint.pt",
    #     graph="/ocean/projects/bio210019p/stevesho/data/preprocess/graphs/scaled/all_tissue_full_graph_scaled.pkl",
    #     graph_idxs="/ocean/projects/bio210019p/stevesho/data/preprocess/graphs/all_tissue_full_graph_idxs.pkl",
    # )

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
