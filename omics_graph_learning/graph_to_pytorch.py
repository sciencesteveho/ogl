#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Convert graphs from np tensors to pytorch geometric Data objects.

Graphs are partitioned into train, test, and validation sets based on the
provided split. Data are prepared as pytorch tensors and stored in a pytorch
geometric Data object. Two masks are produced:

    "split" masks are used for the dataloaders, ensuring that all nodes for the
    loader are referenced
    "split_loss" masks are a subset of the "split" masks containing only genes
    that passed the preceding filter steps for loss calculation (i.e. loss is
    only calculated over these gene nodes)
"""


from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data  # type: ignore

from config_handlers import ExperimentConfig
from constants import NodePerturbation
from constants import TARGET_FILE
from constants import TARGET_FILE_SCALED
from constants import TRAINING_SPLIT_FILE
from perturbation import get_node_perturbation
from perturbation import perturb_edge_index
from perturbation import perturb_node_features
from perturbation import PerturbationConfig


class GraphToPytorch:
    """Class to handle the conversion of graphs stored as np.arrays to tensors
    and ultimately to Pytorch Geometric Data objects.

    Uses class PerturbationConfig to handle potential graph perturbations.
    """

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        split_name: str,
        regression_target: str,
        positional_encoding: bool = False,
        scaled_targets: bool = False,
        perturbation_config: Optional[PerturbationConfig] = None,
    ) -> None:
        """Instantiate the GraphToPytorch object."""
        self.experiment_config = experiment_config
        self.split_name = split_name
        self.regression_target = regression_target
        self.positional_encoding = positional_encoding
        self.scaled_targets = scaled_targets
        self.perturbation_config = perturbation_config

        self.graph_indexes, self.split, self.graph_data, self.targets = (
            self._load_data_object_prerequisites(
                experiment_config=experiment_config,
                graph_type=experiment_config.graph_type,
                split_name=split_name,
                scaled_targets=scaled_targets,
            )
        )

    def create_edge_index(self) -> torch.Tensor:
        """Create the edge index tensor for the graph."""
        edge_perturbation = None
        total_random_edges = None
        node_remove_edges = None

        if self.perturbation_config is not None:
            edge_perturbation = self.perturbation_config.edge_perturbation
            total_random_edges = self.perturbation_config.total_random_edges
            node_remove_edges = self.perturbation_config.node_remove_edges

        return create_edge_index(
            graph_data=self.graph_data,
            edge_perturbation=edge_perturbation,
            total_random_edges=total_random_edges,
            node_idxs=node_remove_edges,
        )

    def create_node_tensors(self) -> torch.Tensor:
        """Create the node tensors for the graph."""
        if self.perturbation_config is not None:
            node_perturbation = get_node_perturbation(
                self.perturbation_config.node_perturbation
            )
        else:
            node_perturbation = None

        return create_node_tensors(
            graph_data=self.graph_data,
            perturbation=node_perturbation,
            positional_encoding=self.positional_encoding,
        )

    def instantiate_data_object(self) -> Data:
        """Create the PyG Data object."""
        edge_index = self.create_edge_index()
        node_tensors = self.create_node_tensors()
        return Data(x=node_tensors.contiguous(), edge_index=edge_index.contiguous())

    def create_masks(self, data: Data) -> Dict[str, torch.Tensor]:
        """Create masks for the Data object."""
        # get mask indexes for loss calculation
        train_loss_idxs, test_loss_idxs, val_loss_idxs, all_loss_idxs = _get_mask_idxs(
            graph_index=self.graph_indexes, split=self.split
        )

        # get mask indexes for training (masks include all nodes that fall on the chr)
        train_idxs, test_idxs, val_idxs = _assign_nodes_to_split(
            graph_data=self.graph_data,
            test_chrs=self.experiment_config.test_chrs,
            val_chrs=self.experiment_config.val_chrs,
        )

        return {
            "train_mask_loss": create_mask(data.num_nodes, train_loss_idxs),
            "test_mask_loss": create_mask(data.num_nodes, test_loss_idxs),
            "val_mask_loss": create_mask(data.num_nodes, val_loss_idxs),
            "all_mask_loss": create_mask(data.num_nodes, all_loss_idxs),
            "train_mask": create_mask(data.num_nodes, train_idxs),
            "test_mask": create_mask(data.num_nodes, test_idxs),
            "val_mask": create_mask(data.num_nodes, val_idxs),
        }

    def create_target_tensor(self, data: Data) -> torch.Tensor:
        """Create the target values tensor for the Data object."""
        # get values for the specified target type
        target_values = _get_target_values_for_mask(targets=self.targets)
        indices = _get_target_indices(self.regression_target)

        # align target value with graph index
        remapped = {
            self.graph_indexes[target]: target_values[target]
            for target in target_values
            if target in self.graph_indexes
        }

        # get list of masked tensors, one tensor for each target index
        y_tensors = [_get_masked_tensor(data.num_nodes) for _ in indices]

        # populate the masked tensors with the target values
        for idx, tensor in enumerate(y_tensors):
            for key, values in remapped.items():
                tensor[key] = values[indices[idx]]

        return torch.stack(y_tensors).view(len(y_tensors), -1)

    def make_data_object(self) -> Data:
        """Create the PyG Data object."""
        # create the data object with node features and edge index
        data = self.instantiate_data_object()
        masks = self.create_masks(data=data)
        targets = self.create_target_tensor(data=data)

        # add masks to the data object
        for mask_name, mask in masks.items():
            setattr(data, mask_name, mask.contiguous())

        # add target values to the data object
        data.y = targets.T.contiguous()

        return data

    @staticmethod
    def _load_data_object_prerequisites(
        experiment_config: ExperimentConfig,
        graph_type: str,
        split_name: str,
        scaled_targets: bool = False,
    ) -> Tuple[Any, Any, Any, Dict[str, Dict[str, np.ndarray]]]:
        """Prepare specific variables needed to make the PyG Data object."""

        def load_pickle(file_path: Path) -> Any:
            """Load pickle!"""
            with open(file_path, "rb") as f:
                return pickle.load(f)

        graph_dir = experiment_config.graph_dir
        experiment_name = experiment_config.experiment_name
        experiment_dir = graph_dir / split_name

        file_paths = {
            "index": graph_dir / f"{experiment_name}_{graph_type}_graph_idxs.pkl",
            "split": experiment_dir / TRAINING_SPLIT_FILE,
            "target": experiment_dir
            / (TARGET_FILE_SCALED if scaled_targets else TARGET_FILE),
            "graph_data": experiment_dir / f"{experiment_name}_{graph_type}_scaled.pkl",
        }

        return (
            load_pickle(file_paths["index"]),
            load_pickle(file_paths["split"]),
            load_pickle(file_paths["graph_data"]),
            load_pickle(file_paths["target"]),
        )


def _assign_nodes_to_split(
    graph_data: Dict[str, Any],
    test_chrs: List[str],
    val_chrs: List[str],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assign nodes to train, test, and validation sets according to their chr.

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
    """

    def get_tensor_for_genes(gene_list: List[str]) -> torch.Tensor:
        """Return a tensor of gene indices for the provided gene list."""
        return torch.tensor(
            [graph_index[gene] for gene in gene_list if gene in graph_index],
            dtype=torch.long,
        )

    # masks used for calculating loss, only contain gene nodes
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
    targets: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """Returns a collapsed dictionary of the targets for each split"""
    # load graph indexes
    return {
        k: v
        for split in ["train", "test", "validation"]
        for k, v in targets[split].items()
    }


def _get_masked_tensor(
    num_nodes: int,
    fill_value: int = -1,
) -> torch.Tensor:
    """Returns a tensor of size (num_nodes) with fill_value (default -1)"""
    return torch.full((num_nodes,), fill_value, dtype=torch.float)


def create_edge_index(
    graph_data: dict,
    edge_perturbation: Optional[str],
    total_random_edges: Optional[int],
    node_idxs: Optional[List[str]],
) -> torch.Tensor:
    """Create the edge index tensor for the graph, perturbing if necessary
    according to args.

    Args:
        graph_data (dict): The graph data dictionary.
        edge_perturbation (Optional[str]): The type of edge perturbation to apply.
        total_random_edges (Optional[int]): The total number of random edges to
        create. If not specified, amount of edges is equal to the amount of
        edges in the graph.
        node_idxs (Optional[List[str]]): The node indices to remove if
        perturbation type is `remove_specific_edges`.
    """
    if not edge_perturbation:
        return torch.tensor(graph_data["edge_index"], dtype=torch.long)
    else:
        return torch.tensor(
            perturb_edge_index(
                edge_index=graph_data["edge_index"],
                edge_perturbation=edge_perturbation,
                node_idxs=node_idxs,
                total_random_edges=total_random_edges,
            ),
            dtype=torch.long,
        )


def create_node_tensors(
    graph_data: dict,
    perturbation: Optional[NodePerturbation] = None,
    positional_encoding: bool = False,
) -> torch.Tensor:
    """Create the node tensors for the graph, perturbing if necessary according
    to input args.

    Args:
        graph_data (dict): The graph data dictionary.
        node_perturbation (Optional[str]): The type of node perturbation to
        apply. Default is None.
        zero_node_feats (bool): Flag indicating whether to zero out the node
        features. Default is False.
        randomize_feats (bool): Flag indicating whether to randomize the node
        features. Default is False.
    """
    if positional_encoding:
        return torch.tensor(
            np.concatenate(
                (graph_data["node_feat"], graph_data["node_positional_encoding"]),
                axis=1,
            ),
            dtype=torch.float,
        )
    elif perturbation is not None:
        return torch.tensor(
            perturb_node_features(perturbation, graph_data["node_feat"]),
            dtype=torch.float,
        )
    else:
        return torch.tensor(graph_data["node_feat"], dtype=torch.float)


def create_mask(num_nodes: int, indices: torch.Tensor) -> torch.Tensor:
    """Create a boolean mask tensor for the provided indices.

    Args:
        num_nodes (int): The number of nodes.
        indices (List[int]): The indices to set to True in the mask.
    """
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[indices] = True
    return mask


def _get_target_indices(regression_target: str) -> List[int]:
    """Return the indices of the target values to use for the regression
    target.
    """
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
