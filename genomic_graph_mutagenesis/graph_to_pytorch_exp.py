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
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from dataset_split import _chr_split_train_test_val
from dataset_split import _genes_from_gff
from utils import TISSUES


def filter_genes(
    root_dir,
    tissues,
):
    """Filters and only keeps targets that pass the TPM filter of >1 TPM across
    20% of samples

    Args:
        tissues (Dict[Tuple[str, str]]): _description_
        targets (Dict[str, Dict[str, np.ndarray]]): _description_

    Returns:
        Dict[str, Dict[str, np.ndarray]]: _description_
    """

    def filtered_genes(tpm_filtered_genes: str) -> List[str]:
        with open(tpm_filtered_genes, newline="") as file:
            return [f"{line[3]}_{tissue}" for line in csv.reader(file, delimiter="\t")]

    for idx, tissue in enumerate(tissues):
        if idx == 0:
            genes = filtered_genes(f"{root_dir}/{tissue}/tpm_filtered_genes.bed")
        else:
            update_genes = filtered_genes(f"{root_dir}/{tissue}/tpm_filtered_genes.bed")
            genes += update_genes

    return set(genes)


def _get_mask_idxs(index: str, split: Dict[str, List[str]]) -> np.ndarray:
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

    return (
        graph_index,
        torch.tensor(
            [
                graph_index[gene]
                for gene in split["train"]
                if gene in graph_index.keys()
            ],
            dtype=torch.long,
        ),
        torch.tensor(
            [graph_index[gene] for gene in split["test"] if gene in graph_index.keys()],
            dtype=torch.long,
        ),
        torch.tensor(
            [
                graph_index[gene]
                for gene in split["validation"]
                if gene in graph_index.keys()
            ],
            dtype=torch.long,
        ),
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
        all_dict.update(graph_targets[split])

    return all_dict


def _get_masked_tensor(num_nodes: int):
    """_summary_

    Args:
        num_nodes (int): _description_

    Returns:
        _type_: _description_
    """
    tensor = torch.zeros(num_nodes, dtype=torch.float)
    tensor[tensor == 0] = -1
    return tensor


def graph_to_pytorch(
    root_dir: str,
    graph_type: str,
):
    """_summary_

    Args:
        root_dir (str): _description_
        graph_type (str): _description_

    Returns:
        _type_: _description_
    """
    graph_dir = f"{root_dir}/graphs"
    graph = f"{graph_dir}/scaled/all_tissue_{graph_type}_graph_scaled.pkl"
    index = f"{graph_dir}/all_tissue_{graph_type}_graph_idxs.pkl"
    targets = f"{graph_dir}/training_targets_exp.pkl"

    gene_gtf = (
        f"{root_dir}/shared_data/local/gencode_v26_genes_only_with_GTEx_targets.bed"
    )
    test_chrs = ["chr8", "chr9"]
    val_chrs = ["chr7", "chr13"]

    split = _chr_split_train_test_val(
        genes=_genes_from_gff(gene_gtf),
        test_chrs=test_chrs,
        val_chrs=val_chrs,
        tissue_append=True,
    )

    filtered_genes = filter_genes(root_dir=root_dir, tissues=TISSUES)
    filtered_split = dict.fromkeys(["train", "test", "validation"])
    for data_split in ["train", "test", "validation"]:
        filtered_split[data_split] = [
            x for x in split[data_split] if x in filtered_genes
        ]

    with open(graph, "rb") as file:
        graph_data = pickle.load(file)

    # convert np arrays to torch tensors
    edge_index = torch.tensor(graph_data["edge_index"], dtype=torch.long)
    x = torch.tensor(graph_data["node_feat"], dtype=torch.float)

    # get mask indexes
    graph_index, train, test, val = _get_mask_idxs(index=index, split=filtered_split)

    # set up pytorch geometric Data object
    data = Data(x=x, edge_index=edge_index)

    # create masks
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[train] = True

    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask[test] = True

    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask[val] = True

    # get target values. shape should be [num_nodes, 4]
    target_values = _get_target_values_for_mask(targets=targets)

    first, second = (
        _get_masked_tensor(data.num_nodes),
        _get_masked_tensor(data.num_nodes),
    )

    # change the key in one dict to the value of another dict, which has its key as the index
    remapped = {}
    for target in target_values:
        if target in graph_index:
            remapped[graph_index[target]] = target_values[target]

    for idx in [0, 1, 2, 3]:
        for key, value in remapped.items():
            if idx == 0:
                first[key] = value[idx]
            elif idx == 1:
                second[key] = value[idx]

    y = torch.cat(
        (first.view(1, -1), second.view(1, -1)),
        dim=0,
    )

    # add mask and target values to data object
    data.train_mask = train_mask
    data.test_mask = test_mask
    data.val_mask = val_mask
    data.y = y.T

    return data
