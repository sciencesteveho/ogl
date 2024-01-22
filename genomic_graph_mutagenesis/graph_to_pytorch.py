#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ]
#

"""Convert graphs from np tensors to pytorch geometric Data objects.

Graphs are padded with zeros to ensure that all graphs have the same number of
numbers, saved as a pytorch geometric Data object, and a mask is applied to only
consider the nodes that pass the TPM filter.
"""

import csv
import pickle
import random
from typing import Dict, List

import numpy as np
import torch
from torch_geometric.data import Data


def _get_mask_idxs(
    index: str,
    split: Dict[str, List[str]],
    percentile_cutoff: int = None,
) -> np.ndarray:
    """_summary_

    Args:
        index (str): _description_
        split (_type_): _description_

    Returns:
        np.ndarray: _description_
    """
    # load graph indexes
    with open(index, "rb") as f:
        graph_index = pickle.load(f)

    def get_tensor_for_genes(gene_list):
        return torch.tensor(
            [graph_index[gene] for gene in gene_list if gene in graph_index.keys()],
            dtype=torch.long,
        )

    all_genes = split["train"] + split["test"] + split["validation"]

    if percentile_cutoff:
        with open(
            f"/ocean/projects/bio210019p/stevesho/data/preprocess/graph_processing/regulatory_only_all_loops_test_8_9_val_7_13_mediantpm/graphs/test_split_cutoff_{percentile_cutoff}.pkl",
            "rb",
        ) as f:
            test_genes = pickle.load(f)
        test_genes = list(test_genes.keys())

        return (
            graph_index,
            get_tensor_for_genes(split["train"]),
            torch.tensor(
                [gene for gene in test_genes if gene in graph_index.values()],
                dtype=torch.long,
            ),
            get_tensor_for_genes(split["validation"]),
            get_tensor_for_genes(all_genes),
        )
    else:
        return (
            graph_index,
            get_tensor_for_genes(split["train"]),
            get_tensor_for_genes(split["test"]),
            get_tensor_for_genes(split["validation"]),
            get_tensor_for_genes(all_genes),
        )


def _get_target_values_for_mask(
    targets: str,
) -> np.ndarray:
    """_summary_

    Args:
        targets (str): _description_

    Returns:
        np.ndarray: _description_
    """
    # load graph indexes
    with open(targets, "rb") as f:
        graph_targets = pickle.load(f)

    all_dict = {}
    for split in ["train", "test", "validation"]:
        all_dict |= graph_targets[split]

    return all_dict


def _get_masked_tensor(
    num_nodes: int,
    fill_value: int = -1,
) -> torch.Tensor:
    """_summary_

    Args:
        num_nodes (int): _description_
    """
    return torch.full((num_nodes,), fill_value, dtype=torch.float)


def graph_to_pytorch(
    experiment_name: str,
    graph_type: str,
    root_dir: str,
    targets_types: str,
    randomize_feats: str = "false",
    zero_node_feats: str = "false",
    node_perturbation: str = None,
    node_remove_edges: List[str] = None,
    single_gene: str = None,
    randomize_edges: str = "false",
    total_random_edges: int = 0,
    scaled: bool = False,
    remove_node: str = None,
    percentile_cutoff: int = None,
):
    """_summary_

    Args:
        root_dir (str): _description_
        graph_type (str): _description_

    Returns:
        _type_: _description_
    """
    graph_dir = f"{root_dir}/graphs"
    graph = f"{graph_dir}/{experiment_name}_{graph_type}_graph_scaled.pkl"
    index = f"{graph_dir}/{experiment_name}_{graph_type}_graph_idxs.pkl"

    with open(f"{graph_dir}/training_targets_split.pkl", "rb") as f:
        split = pickle.load(f)

    with open(graph, "rb") as file:
        graph_data = pickle.load(file)

    # convert np arrays to torch tensors
    # delete edges if perturbing
    if node_remove_edges:
        edge_index = torch.tensor(
            np.array(
                [
                    np.delete(graph_data["edge_index"][0], node_remove_edges),
                    np.delete(graph_data["edge_index"][1], node_remove_edges),
                ]
            ),
            dtype=torch.long,
        )
    elif randomize_edges == "true":
        total_range = max(
            np.ptp(graph_data["edge_index"][0]), np.ptp(graph_data["edge_index"][1])
        )
        if total_random_edges != 0:
            total_edges = total_random_edges
        else:
            total_edges = len(graph_data["edge_index"][0])
        edge_index = torch.tensor(
            np.array(
                [
                    np.random.randint(0, total_range, total_edges),
                    np.random.randint(0, total_range, total_edges),
                ]
            ),
            dtype=torch.long,
        )
    else:
        edge_index = torch.tensor(graph_data["edge_index"], dtype=torch.long)

    # add optional node perturbation
    if node_perturbation == "h3k27ac":
        graph_data["node_feat"][:, 9] = 0
        x = torch.tensor(graph_data["node_feat"], dtype=torch.float)
    elif node_perturbation == "h3k4me3":
        graph_data["node_feat"][:, 14] = 0
        x = torch.tensor(graph_data["node_feat"], dtype=torch.float)
    if not node_perturbation:
        if zero_node_feats == "true":
            x = torch.zeros(graph_data["node_feat"].shape, dtype=torch.float)
        elif randomize_feats == "true":
            x = torch.rand(graph_data["node_feat"].shape, dtype=torch.float)
        else:
            x = torch.tensor(graph_data["node_feat"], dtype=torch.float)

    # get mask indexes
    if percentile_cutoff:
        graph_index, train, test, val, all_idx = _get_mask_idxs(
            index=index, split=split, percentile_cutoff=percentile_cutoff
        )
    else:
        graph_index, train, test, val, all_idx = _get_mask_idxs(
            index=index, split=split
        )

    # get individual if querying for single gene
    if single_gene:
        gene_idx = torch.tensor([single_gene], dtype=torch.long)
        gene_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        gene_mask[gene_idx] = True

    # set up pytorch geometric Data object
    data = Data(x=x, edge_index=edge_index)

    # create masks
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[train] = True

    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask[test] = True

    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask[val] = True

    all_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    all_mask[all_idx] = True

    # get target values. shape should be [num_nodes, 4]
    if scaled:
        targets = f"{graph_dir}/targets_scaled.pkl"
    else:
        targets = f"{graph_dir}/targets.pkl"
    target_values = _get_target_values_for_mask(targets=targets)

    # handling of targets_types, to only put proper targets into mask
    target_indices = {
        "expression_median_only": [0],
        "expression_median_and_foldchange": [0, 1],
        "difference_from_average": [2],
        "protein_targets": [0, 1, 2, 3],
    }

    if targets_types not in target_indices:
        raise ValueError("Invalid targets_types provided.")

    remapped = {
        graph_index[target]: target_values[target]
        for target in target_values
        if target in graph_index
    }

    y_tensors = [
        _get_masked_tensor(data.num_nodes)
        for _ in range(len(target_indices[targets_types]))
    ]

    for idx, tensor in enumerate(y_tensors):
        for key, values in remapped.items():
            tensor[key] = values[target_indices[targets_types][idx]]

    y = torch.stack(y_tensors).view(len(y_tensors), -1)

    # add mask and target values to data object
    data.train_mask = train_mask
    data.test_mask = gene_mask if single_gene else test_mask
    data.val_mask = val_mask
    data.y = y.T
    data.all_mask = all_mask

    return data
