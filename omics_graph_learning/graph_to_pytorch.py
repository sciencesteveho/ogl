#! /usr/bin/env python
# -*- coding: utf-8 -*-
#

"""Convert graphs from np tensors to pytorch geometric Data objects.

Graphs are padded with zeros to ensure that all graphs have the same number of
numbers, saved as a pytorch geometric Data object, and a mask is applied to only
consider the nodes that pass the TPM filter.
"""

from dataclasses import dataclass
import pickle
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx  # type: ignore
import numpy as np
import torch
import torch_geometric  # type: ignore
from torch_geometric.data import Data  # type: ignore

from config_handlers import ExperimentConfig
from constants import NODE_FEAT_IDXS


@dataclass
class GraphConfig:
    randomize_feats: bool = False
    zero_node_feats: bool = False
    node_perturbation: Optional[str] = None
    node_remove_edges: Optional[List[str]] = None
    single_gene: Optional[str] = None
    randomize_edges: bool = False
    total_random_edges: int = 0
    scaled: bool = False
    positional_encoding: bool = False
    remove_node: Optional[str] = None
    percentile_cutoff: Optional[int] = None


def _assign_nodes_to_split(
    graph_data: Dict[str, Any],
    test_chrs: List[str],
    val_chrs: List[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign nodes to train, test, and validation sets.

    Args:
        graph (nx.Graph): The graph object.
        test_chrs (List[str]): The list of test chromosomes.
        val_chrs (List[str]): The list of validation chromosomes.

    Returns:
        Dict[str, List[str]]: The split dictionary containing train, test, and
        validation gene lists.
    """
    coordinates = graph_data["coordinates"]
    train, test, validation = [], [], []
    for node in range(graph_data["num_nodes"]):
        if coordinates[node]["chr"] in test_chrs:
            train.append(node)
        elif coordinates[node]["chr"] in val_chrs:
            test.append(node)
        else:
            validation.append(node)
    return (
        torch.tensor(train, dtype=torch.float),
        torch.tensor(test, dtype=torch.float),
        torch.tensor(validation, dtype=torch.float),
    )


def _get_mask_idxs(
    graph_index: Dict[str, int],
    split: Dict[str, List[str]],
    percentile_cutoff: Optional[int] = None,
    cutoff_file: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
            [graph_index[gene] for gene in gene_list if gene in graph_index],
            dtype=torch.long,
        )

    # get masks used for calculating loss, so these masks only contain gene
    # nodes
    all_genes = split["train"] + split["test"] + split["validation"]
    train_mask_loss = get_tensor_for_genes(split["train"])
    val_mask_loss = get_tensor_for_genes(split["validation"])
    all_genes_mask = get_tensor_for_genes(all_genes)

    if percentile_cutoff:
        with open(
            f"{cutoff_file}_{percentile_cutoff}.pkl",
            "rb",
        ) as f:
            test_genes = pickle.load(f)
        test_genes = list(test_genes.keys())
        test_mask_loss = torch.tensor(
            [graph_index.get(gene, -1) for gene in test_genes],
            dtype=torch.long,
        )
    else:
        test_mask_loss = get_tensor_for_genes(split["test"])

    return (
        train_mask_loss,
        test_mask_loss,
        val_mask_loss,
        all_genes_mask,
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
                np.delete(graph_data["edge_index"][0], np.array(node_remove_edges)),
                np.delete(graph_data["edge_index"][1], np.array(node_remove_edges)),
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
    positional_encoding: bool = False,
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
    if node_perturbation in NODE_FEAT_IDXS:
        graph_data["node_feat"][:, NODE_FEAT_IDXS[node_perturbation]] = 0
    elif zero_node_feats:
        return torch.zeros(graph_data["node_feat"].shape, dtype=torch.float)
    elif randomize_feats:
        return torch.rand(graph_data["node_feat"].shape, dtype=torch.float)
    elif positional_encoding:
        return torch.tensor(
            np.concatenate(
                (graph_data["node_feat"], graph_data["node_positional_encoding"]),
                axis=1,
            ),
            dtype=torch.float,
        )
    return torch.tensor(graph_data["node_feat"], dtype=torch.float)


def create_mask(num_nodes: int, indices: torch.Tensor) -> torch.Tensor:
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
        "rna_seq": [0],
    }
    if regression_target not in target_indices:
        raise ValueError("Invalid regression_target provided.")
    return target_indices[regression_target]


def _load_data_object_prerequisites(
    experiment_config: ExperimentConfig,
    graph_type: str,
    split_name: str,
    scaled: bool = False,
):
    """Load specific files needed to make the PyG Data object"""
    graph_dir = experiment_config.graph_dir
    experiment_name = experiment_config.experiment_name
    index_file = f"{experiment_config.working_directory}/graphs/{experiment_name}_{graph_type}_graph_idxs.pkl"

    # get training split
    with open(graph_dir / "training_targets_split.pkl", "rb") as f:
        split = pickle.load(f)

    # get training targets
    if scaled:
        targets = graph_dir / "training_targets_scaled.pkl"
    else:
        targets = graph_dir / "training_targets.pkl"

    # load the graph!
    with open(
        graph_dir / f"{experiment_name}_{graph_type}_{split_name}_graph_scaled.pkl",
        "rb",
    ) as file:
        graph_data = pickle.load(file)

    # load index
    with open(index_file, "rb") as f:
        index = pickle.load(f)

    return (
        index,
        split,
        graph_data,
        targets,
    )


def graph_to_pytorch(
    experiment_config: ExperimentConfig,
    graph_type: str,
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
    positional_encoding: bool = False,
    remove_node: Optional[str] = None,
    percentile_cutoff: Optional[int] = None,
) -> torch_geometric.data.Data:
    """_summary_

    Args:
        working_dir (str): _description_
        graph_type (str): _description_

    Returns:
        _type_: _description_
    """

    # load necessary files
    graph_index, split, graph_data, targets = _load_data_object_prerequisites(
        experiment_config=experiment_config,
        graph_type=graph_type,
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
        graph_data,
        node_perturbation,
        zero_node_feats,
        randomize_feats,
        positional_encoding,
    )

    # get mask indexes for train_loss, test_loss, val_loss, and all_loss
    if percentile_cutoff:
        train_loss_idxs, test_loss_idxs, val_loss_idxs, all_loss_idxs = _get_mask_idxs(
            graph_index=graph_index,
            split=split,
            percentile_cutoff=percentile_cutoff,
        )
    else:
        train_loss_idxs, test_loss_idxs, val_loss_idxs, all_loss_idxs = _get_mask_idxs(
            graph_index=graph_index,
            split=split,
        )

    # get indexes used for convolutions, so these masks contain all nodes
    train_idxs, test_idxs, val_idxs = _assign_nodes_to_split(
        graph_data=graph_data,
        test_chrs=experiment_config.test_chrs,
        val_chrs=experiment_config.val_chrs,
    )

    # set up pytorch geometric Data object
    data = Data(x=node_tensors.contiguous(), edge_index=edge_index.contiguous())

    # create masks
    train_mask_loss = create_mask(data.num_nodes, train_loss_idxs)
    test_mask_loss = create_mask(data.num_nodes, test_loss_idxs)
    val_mask_loss = create_mask(data.num_nodes, val_loss_idxs)
    all_mask_loss = create_mask(data.num_nodes, all_loss_idxs)
    train_mask = create_mask(data.num_nodes, train_idxs)
    test_mask = create_mask(data.num_nodes, test_idxs)
    val_mask = create_mask(data.num_nodes, val_idxs)

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
    data.train_mask = train_mask.contiguous()
    data.test_mask = test_mask.contiguous()
    data.val_mask = val_mask.contiguous()

    data.train_mask_loss = train_mask_loss.contiguous()
    data.test_mask_loss = (
        gene_mask.contiguous() if single_gene else test_mask_loss.contiguous()
    )
    data.val_mask = val_mask_loss.contiguous()
    data.all_mask_loss = all_mask_loss.contiguous()
    data.y = y_vals.T.contiguous()

    return data
