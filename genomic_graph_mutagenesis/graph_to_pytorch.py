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

# from dataset_split import _genes_train_test_val_split
# from dataset_split import _GenomeDataUtils.genes_from_gff
# from utils import TISSUES


# def filter_genes(
#     root_dir,
#     tissues,
# ):
#     """Filters and only keeps targets that pass the TPM filter of >.1 TPM across
#     20% of samples

#     Args:
#         tissues (Dict[Tuple[str, str]]): _description_
#         targets (Dict[str, Dict[str, np.ndarray]]): _description_

#     Returns:
#         Dict[str, Dict[str, np.ndarray]]: _description_
#     """

#     def filtered_genes(tpm_filtered_genes: str) -> List[str]:
#         with open(tpm_filtered_genes, newline="") as file:
#             return [f"{line[3]}_{tissue}" for line in csv.reader(file, delimiter="\t")]

#     for idx, tissue in enumerate(tissues):
#         if idx == 0:
#             genes = filtered_genes(f"{root_dir}/{tissue}/tpm_filtered_genes.bed")
#         else:
#             update_genes = filtered_genes(f"{root_dir}/{tissue}/tpm_filtered_genes.bed")
#             genes += update_genes

#     return set(genes)


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
    test_chrs: List[str],
    val_chrs: List[str],
    randomize_feats: str = "false",
    zero_node_feats: str = "false",
    node_perturbation: str = None,
    node_remove_edges: List[str] = None,
    gene_gtf: str = "/ocean/projects/bio210019p/stevesho/data/preprocess/shared_data/local/gencode_v26_genes_only_with_GTEx_targets.bed",
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

    # filtered_genes = filter_genes(root_dir=root_dir, tissues=TISSUES)
    # filtered_split = dict.fromkeys(["train", "test", "validation"])
    # for data_split in ["train", "test", "validation"]:
    #     filtered_split[data_split] = [
    #         x for x in split[data_split] if x in filtered_genes
    #     ]

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
    if node_perturbation == "h3k4me3":
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

    # change the key in one dict to the value of another dict, which has its key as the index
    remapped = {}
    for target in target_values:
        if target in graph_index:
            remapped[graph_index[target]] = target_values[target]

    first, second, third, fourth, fifth = (
        _get_masked_tensor(data.num_nodes),
        _get_masked_tensor(data.num_nodes),
        _get_masked_tensor(data.num_nodes),
        _get_masked_tensor(data.num_nodes),
        _get_masked_tensor(data.num_nodes),
    )

    if targets_types == "expression_median_only":
        for idx in [0]:
            for key, values in remapped.items():
                first[key] = values[idx]
        y = first.view(1, -1)
    elif targets_types == "expression_median_and_foldchange":
        for idx in [0, 1]:
            for key, value in remapped.items():
                if idx == 0:
                    first[key] = value[idx]
                elif idx == 1:
                    second[key] = value[idx]
        y = torch.cat(
            (first.view(1, -1), second.view(1, -1)),
            dim=0,
        )
    elif targets_types == "difference_from_average":
        for idx in [2]:
            for key, values in remapped.items():
                third[key] = values[idx]
            y = third.view(1, -1)
    elif targets_types == "protein_targets":
        pass
    else:
        raise ValueError(
            "targets_types must be one of the following: expression_median_only, expression_median_and_foldchange, difference_from_average, protein_targets"
        )

    # if protein_targets:
    #     first, second, third, fourth = (
    #         _get_masked_tensor(data.num_nodes),
    #         _get_masked_tensor(data.num_nodes),
    #         _get_masked_tensor(data.num_nodes),
    #         _get_masked_tensor(data.num_nodes),
    #     )
    #     for idx in [0, 1, 2, 3]:
    #         for key, value in remapped.items():
    #             if idx == 0:
    #                 first[key] = value[idx]
    #             elif idx == 1:
    #                 second[key] = value[idx]
    #             elif idx == 2:
    #                 third[key] = value[idx]
    #             elif idx == 3:
    #                 fourth[key] = value[idx]
    #         y = torch.cat(
    #             (
    #                 first.view(1, -1),
    #                 second.view(1, -1),
    #                 third.view(1, -1),
    #                 fourth.view(1, -1),
    #             ),
    #             dim=0,
    #         )

    # add mask and target values to data object
    data.train_mask = train_mask
    if single_gene:
        data.test_mask = gene_mask
    else:
        data.test_mask = test_mask
    data.val_mask = val_mask
    data.y = y.T
    data.all_mask = all_mask

    return data
