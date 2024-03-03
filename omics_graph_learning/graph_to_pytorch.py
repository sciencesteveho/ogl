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

import pathlib
import pickle
from typing import Dict, List, Optional

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data

NODE_FEATURE_IDXS = {
    "atac": 4,
    "cnv": 5,
    "cpg": 6,
    "ctcf": 7,
    "dnase": 8,
    "h3k27ac": 9,
    "h3k27me3": 10,
    "h3k36me3": 11,
    "h3k4me1": 12,
    "h3k4me2": 13,
    "h3k4me3": 14,
    "h3k79me2": 15,
    "h3k9ac": 16,
    "h3k9me3": 17,
}


def _get_mask_idxs(
    index: str,
    split: Dict[str, List[str]],
    percentile_cutoff: int = None,
    cutoff_file: str = None,
) -> np.ndarray:
    """Get the mask indexes for train, test, and validation sets.

    Args:
        index (str): The path to the index file.
        split (Dict[str, List[str]]): The split dictionary containing train,
        test, and validation gene lists.
        percentile_cutoff (int, optional): The percentile cutoff for test genes.
        Default is None.
        cutoff_file (str, optional): The path to the cutoff file. Default is
        None.

    Returns:
        np.ndarray: The mask indexes for train, test, validation, and all genes.
    """

    def get_tensor_for_genes(gene_list: List[str]) -> torch.Tensor:
        return torch.tensor(
            [graph_index[gene] for gene in gene_list if gene in graph_index.keys()],
            dtype=torch.long,
        )

    # load graph indexes
    with open(index, "rb") as f:
        graph_index = pickle.load(f)

    all_genes = split["train"] + split["test"] + split["validation"]
    train_tensor = get_tensor_for_genes(split["train"])
    validation_tensor = get_tensor_for_genes(split["validation"])
    all_genes_tensor = get_tensor_for_genes(all_genes)

    if percentile_cutoff:
        with open(
            f"{cutoff_file}_{percentile_cutoff}.pkl",
            "rb",
        ) as f:
            test_genes = pickle.load(f)
        test_genes = list(test_genes.keys())
        test_tensor = torch.tensor(
            [graph_index.get(gene, -1) for gene in test_genes],
            dtype=torch.long,
        )
    else:
        test_tensor = get_tensor_for_genes(split["test"])

    return (
        graph_index,
        train_tensor,
        test_tensor,
        validation_tensor,
        all_genes_tensor,
    )


def _get_target_values_for_mask(
    targets: str,
) -> Dict[str, np.ndarray]:
    """Returns a collapsed dictionary of the targets for each split"""
    # load graph indexes
    with open(targets, "rb") as f:
        graph_targets = pickle.load(f)

    return {
        k: v
        for split in ["train", "test", "validation"]
        for k, v in graph_targets[split].items()
    }


def _get_masked_tensor(
    num_nodes: int,
    fill_value: int = -1,
) -> torch.Tensor:
    """Returns a tensor of size (num_nodes) with fill_value (default -1)"""
    return torch.full((num_nodes,), fill_value, dtype=torch.float)


def create_edge_index(
    graph_data: dict,
    node_remove_edges: Optional[List[str]],
    randomize_edges: bool,
    total_random_edges: int,
) -> torch.Tensor:
    """Create the edge index tesnors for the graph, perturbing if necessary
    according to args.

    Args:
        graph_data (dict): The graph data dictionary. node_remove_edges
        (Optional[List[str]]): A list of node names to remove edges from.
        Default is None.
        randomize_edges (bool): Flag indicating whether to randomize the edges.
        Default is False.
        total_random_edges (int): The total number of random edges to generate.
        Default is 0.

    Returns:
        torch.Tensor: the edge index tensors
    """
    if node_remove_edges:
        return torch.tensor(
            [
                np.delete(graph_data["edge_index"][0], node_remove_edges),
                np.delete(graph_data["edge_index"][1], node_remove_edges),
            ],
            dtype=torch.long,
        )
    elif randomize_edges:
        total_range = max(
            np.ptp(graph_data["edge_index"][0]), np.ptp(graph_data["edge_index"][1])
        )
        total_edges = total_random_edges or len(graph_data["edge_index"][0])
        return torch.tensor(
            np.random.randint(0, total_range, (2, total_edges)),
            dtype=torch.long,
        )
    else:
        return torch.tensor(graph_data["edge_index"], dtype=torch.long)


def create_node_tensors(
    graph_data: dict,
    node_perturbation: Optional[str],
    zero_node_feats: bool,
    randomize_feats: bool,
) -> torch.Tensor:
    """Create the node tensors for the graph, perturbing if necessary according
    to input args

    Args:
        graph_data (dict): The graph data dictionary.
        node_perturbation (Optional[str]): The type of node perturbation to
        apply. Default is None.
        zero_node_feats (bool): Flag indicating whether to zero out the node
        features. Default is False.
        randomize_feats (bool): Flag indicating whether to randomize the node
        features. Default is False.

    Returns torch.Tensor
    """
    if node_perturbation in NODE_FEATURE_IDXS:
        graph_data["node_feat"][:, NODE_FEATURE_IDXS[node_perturbation]] = 0
    elif zero_node_feats:
        return torch.zeros(graph_data["node_feat"].shape, dtype=torch.float)
    elif randomize_feats:
        return torch.rand(graph_data["node_feat"].shape, dtype=torch.float)
    return torch.tensor(graph_data["node_feat"], dtype=torch.float)


def create_mask(num_nodes: int, indices: List[int]) -> torch.Tensor:
    """Create a boolean mask tensor, so that only gene nodes are regressed

    Args:
        num_nodes (int): The number of nodes.
        indices (List[int]): The indices to set to True in the mask.

    Returns:
        torch.Tensor: The boolean mask tensor.
    """
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[indices] = True
    return mask


def _get_target_indices(regression_target):
    target_indices = {
        "expression_median_only": [0],
        "expression_median_and_foldchange": [0, 1],
        "difference_from_average": [2],
        "foldchange_from_average": [3],
        "protein_targets": [0, 1, 2, 3, 4, 5],
    }
    if regression_target not in target_indices:
        raise ValueError("Invalid regression_target provided.")
    return target_indices[regression_target]


def _load_data_object_prerequisites(
    experiment_name: str,
    graph_type: str,
    root_dir: str,
    split_name: str,
    scaled: bool = False,
):
    """Load specific files needed to make the PyG Data object"""
    graph_dir = pathlib.Path(root_dir) / "graphs" / split_name

    # get training split
    with open(f"{graph_dir}/training_targets_split.pkl", "rb") as f:
        split = pickle.load(f)

    # get training targets
    if scaled:
        targets = f"{graph_dir}/training_targets.pkl"
    else:
        targets = f"{graph_dir}/training_targets_scaled.pkl"

    # load the graph!
    with open(
        f"{graph_dir}/{experiment_name}_{graph_type}_{split_name}_graph_scaled.pkl",
        "rb",
    ) as file:
        graph_data = pickle.load(file)

    return (
        f"{root_dir}/graphs/{experiment_name}_{graph_type}_idxs.pkl",
        split,
        graph_data,
        targets,
    )


def graph_to_pytorch(
    experiment_name: str,
    graph_type: str,
    root_dir: str,
    split_name: str,
    regression_target: str,
    randomize_feats: bool = False,
    zero_node_feats: bool = False,
    node_perturbation: Optional[str] = None,
    node_remove_edges: Optional[List[str]] = None,
    single_gene: Optional[str] = None,
    randomize_edges: bool = False,
    total_random_edges: int = 0,
    scaled: bool = False,
    remove_node: Optional[str] = None,
    percentile_cutoff: Optional[int] = None,
) -> torch_geometric.data.Data:
    """_summary_

    Args:
        root_dir (str): _description_
        graph_type (str): _description_

    Returns:
        _type_: _description_
    """

    index, split, graph_data, targets = _load_data_object_prerequisites(
        experiment_name=experiment_name,
        graph_type=graph_type,
        root_dir=root_dir,
        split_name=split_name,
        scaled=scaled,
    )

    # convert np arrays to torch tensors
    # delete edges if perturbing
    edge_index = create_edge_index(
        graph_data, node_remove_edges, randomize_edges, total_random_edges
    )

    # get nodes, perturb if according to args
    node_tensors = create_node_tensors(
        graph_data, node_perturbation, zero_node_feats, randomize_feats
    )

    # get mask indexes
    if percentile_cutoff:
        graph_index, train, test, val, all_idx = _get_mask_idxs(
            index=index, split=split, percentile_cutoff=percentile_cutoff
        )
    else:
        graph_index, train, test, val, all_idx = _get_mask_idxs(
            index=index, split=split
        )

    # set up pytorch geometric Data object
    data = Data(x=node_tensors, edge_index=edge_index)

    # create masks
    train_mask = create_mask(data.num_nodes, train)
    test_mask = create_mask(data.num_nodes, test)
    val_mask = create_mask(data.num_nodes, val)
    all_mask = create_mask(data.num_nodes, all_idx)

    if single_gene:
        gene_idx = torch.tensor([single_gene], dtype=torch.long)
        gene_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        gene_mask[gene_idx] = True
        test_mask = gene_mask

    # get target values. shape should be [num_nodes, 6]
    target_values = _get_target_values_for_mask(targets=targets)

    # handling of regression_target, to only put proper targets into mask
    indices = _get_target_indices(regression_target)

    remapped = {
        graph_index[target]: target_values[target]
        for target in target_values
        if target in graph_index
    }

    y_tensors = [_get_masked_tensor(data.num_nodes) for _ in indices]

    for idx, tensor in enumerate(y_tensors):
        for key, values in remapped.items():
            tensor[key] = values[indices[idx]]

    y_vals = torch.stack(y_tensors).view(len(y_tensors), -1)

    # add mask and target values to data object
    data.train_mask = train_mask
    data.test_mask = gene_mask if single_gene else test_mask
    data.val_mask = val_mask
    data.y = y_vals.T
    data.all_mask = all_mask

    return data
