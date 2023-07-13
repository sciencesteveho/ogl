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
from tqdm import tqdm

from gnn import GAT
from gnn import GCN
from gnn import GraphSAGE
from utils import TISSUES


def _edge_perturbation():
    """_summary_ of function"""
    pass


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
    pass


@torch.no_grad()
def test(model, device, test_loader, epoch):
    model.eval()

    pbar = tqdm(total=len(test_loader))
    pbar.set_description(f"Evaluating epoch: {epoch:04d}")

    val_mse, test_mse = []
    for data in test_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)

        # get indices to mask -1 values
        val_indices = data.y[data.val_mask] != -1
        masked_prediction_val = out[data.val_mask][val_indices]
        masked_labels_val = data.y[data.val_mask][val_indices]

        test_indices = data.y[data.test_mask] != -1
        masked_prediction_test = out[data.test_mask][test_indices]
        masked_labels_test = data.y[data.test_mask][test_indices]

        # calculate loss
        val_acc = F.mse_loss(masked_prediction_val, masked_labels_val)
        test_acc = F.mse_loss(masked_prediction_test, masked_labels_test)
    pbar.close()
    return float(torch.cat(val_mse, dim=0).mean()), float(
        torch.cat(test_mse, dim=0).mean()
    )


def main(
    mode: str,
    graph: str,
    graph_idxs: str,
) -> None:
    """Main function"""
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

    coessential_idxs = _get_idxs_for_coessential_pairs(
        coessential_pos="/ocean/projects/bio210019p/stevesho/data/preprocess/comparisons/coessential_gencode_named_pos.txt",
        coessential_neg="/ocean/projects/bio210019p/stevesho/data/preprocess/comparisons/coessential_gencode_named_neg.txt",
        graph_idxs=graph_idxs,
    )

    for key in coessential_idxs.keys():
        testers = random.sample(coessential_idxs[key], 100)
        
    for key in testers:
        eval_key = key
        for subkey in coessential_idxs[key]:
            
        subset = torch.tensor([coessential_idxs[key]].extend(eval_key))
        sub_graph = subgraph(subset, graph.edge_index)

    random_co_idxs = _random_gene_pairs(
        coessential_idxs=coessential_idxs,
        graph_idxs=graph_idxs,
    )

    # # load model
    # model = GraphSAGE(
    #     in_size=data.x.shape[1],
    #     embedding_size=600,
    #     out_channels=4,
    #     layers=args.layers,
    # ).to(device)

    # checkpoint = torch.load(checkpoint_file, map_location=map_location)
    # model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    # model.to(device)

    # ### evaluate on test set
    # test_loader = GraphDataLoader(dataset=testset, batch_size=2, collate_fn=collate_dgl)

    # truth_vals = [sample[1] for sample in testset]
    # test_vals, pred_vals, rmse = eval_model_test(device, model, test_loader, truth_vals)

    # save_pickle("truth.pt", test_vals)
    # save_pickle("pred.pt", pred_vals)


if __name__ == "__main__":
    main(
        model="/ocean/projects/bio210019p/shared/model_checkpoint.pt",
        graph="/ocean/projects/bio210019p/stevesho/data/preprocess/graphs/scaled/all_tissue_full_graph_scaled.pkl",
        graph_idxs="/ocean/projects/bio210019p/stevesho/data/preprocess/graphs/all_tissue_full_graph_idxs.pkl",
    )


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