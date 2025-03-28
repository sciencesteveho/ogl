#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Perturb node features and measure impact on model output."""


from collections import defaultdict
import os
from typing import Dict, List, Tuple, Union

import joblib  # type: ignore
import torch
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import NeighborLoader  # type: ignore
from torch_geometric.utils import k_hop_subgraph  # type: ignore
from tqdm import tqdm  # type: ignore

from omics_graph_learning.interpret.interpret_utils import combine_masks
from omics_graph_learning.interpret.perturb_runner import PerturbRunner


def get_test_loader(
    data: Data,
    mask: str,
    batch_size: int = 64,
) -> NeighborLoader:
    """Create a neighborloader for test/validation/evaluation nodes."""
    data = combine_masks(data)
    return NeighborLoader(
        data,
        num_neighbors=[data.avg_edges] * 2,
        batch_size=batch_size,
        input_nodes=getattr(data, f"{mask}_mask"),
        shuffle=False,
    )


def get_test_k_hop_subgraph(
    data: Data,
    node_list: torch.Tensor,
    k: int = 2,
) -> List[Data]:
    """Return a list of k-hop subgraph Data objects, one per node in
    `node_list`, capturing all neighbors up to k hops.
    """
    subgraphs = []
    for node_id in node_list:
        node_subgraph, edge_subgraph, mapping, _ = k_hop_subgraph(
            node_idx=node_id.item(),
            num_hops=k,
            edge_index=data.edge_index,
            relabel_nodes=True,
            num_nodes=data.num_nodes,
        )
        sub_data = Data(
            x=data.x[node_subgraph],
            edge_index=edge_subgraph,
        )

        # attach global node IDs (for consistent indexing in code)
        sub_data.n_id = node_subgraph

        # attach the loss masks
        if hasattr(data, "all_mask_loss"):
            sub_data.all_mask_loss = data.all_mask_loss[node_subgraph]

        subgraphs.append(sub_data)
    return subgraphs


def compute_baseline_output(
    runner: PerturbRunner,
    batch: Data,
    mask_tensor: torch.Tensor,
) -> torch.Tensor:
    """Compute baseline model output for comparison."""
    with torch.no_grad():
        baseline_out, _ = runner.model(
            x=batch.x,
            edge_index=batch.edge_index,
            mask=mask_tensor,
        )
    return baseline_out[mask_tensor].cpu()


def compute_feature_perturbation(
    runner: PerturbRunner,
    batch: Data,
    mask_tensor: torch.Tensor,
    baseline_output: torch.Tensor,
    feature_indices: Union[List[int], List[Tuple[int, int]]],
    scalers: Dict[int, object] = None,
) -> Dict[Union[int, Tuple[int, int]], Dict[int, List[float]]]:
    """Zero out each single feature in feature_indices if it is a list of ints,
    or zero out both features if it's a list of (idx1, idx2) tuples.

    Returns:
        {
            feature_index_or_pair: {
                node_index: [differences]
            }
        }
    """

    def scaled_zero(feat: int) -> float:
        """Return the scaled zero value for a feature index."""
        return scalers[feat].transform([[0.0]])[0, 0]  # from shape (1,1)

    feature_node_differences = defaultdict(lambda: defaultdict(list))
    node_indices = batch.n_id[mask_tensor].cpu()

    if not feature_indices:
        return feature_node_differences

    first = feature_indices[0]
    if isinstance(first, int):
        for feat_idx in feature_indices:
            x_perturbed = batch.x.clone()
            x_perturbed[:, feat_idx] = scaled_zero(
                feat_idx
            )  # zero out the single feature

            with torch.no_grad():
                perturbed_out, _ = runner.model(
                    x=x_perturbed,
                    edge_index=batch.edge_index,
                    mask=mask_tensor,
                )
            perturbed_out = perturbed_out[mask_tensor].cpu()

            diff = baseline_output - perturbed_out
            for idx, d in zip(node_indices, diff):
                feature_node_differences[feat_idx][idx.item()].append(d.item())

    else:
        for feat_idx1, feat_idx2 in feature_indices:
            x_perturbed = batch.x.clone()
            x_perturbed[:, feat_idx1] = scaled_zero(feat_idx1)
            x_perturbed[:, feat_idx2] = scaled_zero(feat_idx2)

            with torch.no_grad():
                perturbed_out, _ = runner.model(
                    x=x_perturbed,
                    edge_index=batch.edge_index,
                    mask=mask_tensor,
                )
            perturbed_out = perturbed_out[mask_tensor].cpu()

            diff = baseline_output - perturbed_out
            for idx, d in zip(node_indices, diff):
                feature_node_differences[(feat_idx1, feat_idx2)][idx.item()].append(
                    d.item()
                )

    return feature_node_differences


def compute_average_node_differences(
    feature_node_differences: Dict[Union[int, Tuple[int, int]], Dict[int, List[float]]],
) -> Dict[Union[int, Tuple[int, int]], Dict[int, float]]:
    """Compute the average difference for each node, for each feature index or
    feature tuple.

    Returns:
        {feature_key: {node: average_difference}}
    """
    return {
        feature_key: {node: sum(vals) / len(vals) for node, vals in node_diffs.items()}
        for feature_key, node_diffs in feature_node_differences.items()
    }


def compute_fold_changes(
    avg_diffs: Dict[Union[int, Tuple[int, int]], Dict[int, float]],
) -> Dict[Union[int, Tuple[int, int]], float]:
    """Compute the overall average difference (fold change) per feature index.

    Args:
        avg_diffs: {feature_index: node_index: average difference}

    Returns:
        {feature_index: overall average difference}
    """
    feature_fold_changes = {}
    for feat_idx, node_dict in avg_diffs.items():
        total_diff = sum(node_dict.values())
        num_nodes = len(node_dict)
        avg_diff = total_diff / num_nodes if num_nodes else 0
        feature_fold_changes[feat_idx] = avg_diff
    return feature_fold_changes


def get_top_n_nodes(
    avg_diffs_for_feature: Dict[int, float],
    n: int = 100,
) -> List[Tuple[int, float]]:
    """Get the top N nodes by absolute difference.

    Args:
        avg_diffs_for_feature: {node_idx: difference}
        n: Number of top nodes to retrieve.

    Returns:
        List of (node_idx, diff) sorted in descending order of absolute
        difference.
    """
    sorted_node_diffs = sorted(
        avg_diffs_for_feature.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    return sorted_node_diffs[:n]


def map_nodes_to_symbols(
    node_diff_list: List[Tuple[int, float]],
    node_idx_to_gene_id: Dict[int, str],
    gencode_to_symbol: Dict[str, str],
) -> List[Tuple[str, float]]:
    """Map node indices to gene symbols for pairs of (node_idx, difference).

    Args:
        node_diff_list: (node_idx, difference) pairs.
        node_idx_to_gene_id: maps node index to gene id
        gencode_to_symbol: maps gencode ids to gene symbols.

    Returns:
        List of (gene_symbol_or_node_idx, difference)
    """
    result = []
    for node_idx, diff_value in node_diff_list:
        gene_id = node_idx_to_gene_id.get(node_idx)
        if gene_id and "ENSG" in gene_id:
            gencode_id = gene_id.split("_")[0]
            symbol = gencode_to_symbol.get(gencode_id, gene_id)
            result.append((symbol, diff_value))
        else:
            result.append((str(node_idx), diff_value))
    return result


def get_top_feature_genes(
    feature_indices: Union[List[int], List[Tuple[int, int]]],
    avg_diffs: Dict[Union[int, Tuple[int, int]], Dict[int, float]],
    top_n: int,
    node_idx_to_gene_id: Dict[int, str],
    gencode_to_symbol: Dict[str, str],
) -> Dict[Union[int, Tuple[int, int]], List[Tuple[str, float]]]:
    """Get top nodes by average difference for each feature index and map them
    to gene symbols.
    """
    if not feature_indices:
        return {}

    feature_top_genes = {}
    for feature_key in feature_indices:
        top_nodes = get_top_n_nodes(avg_diffs[feature_key], n=top_n)
        top_genes = map_nodes_to_symbols(
            top_nodes,
            node_idx_to_gene_id=node_idx_to_gene_id,
            gencode_to_symbol=gencode_to_symbol,
        )
        feature_top_genes[feature_key] = top_genes

    return feature_top_genes


def perturb_node_features(
    data: Data,
    runner: PerturbRunner,
    feature_indices: Union[List[int], List[Tuple[int, int]]],
    device: torch.device,
    node_idx_to_gene_id: Dict[int, str],
    gencode_to_symbol: Dict[str, str],
    mask: str = "all",
    scalers: Dict[int, object] = None,
) -> Tuple[
    Dict[Union[int, Tuple[int, int]], Dict[int, float]],
    Dict[Union[int, Tuple[int, int]], float],
    Dict[Union[int, Tuple[int, int]], List[Tuple[str, float]]],
]:
    """Zero out specified node features and measure the impact on model output.

    Computes the output per batch, zeroes out a selected feature, and computes
    the difference. Differences are then averaged per node.

    Args:
        data: torch_geometric Data object
        runner: Runner holding the model to test
        feature_indices: feature indices to zero out
        mask: Mask name (e.g., 'test', 'val', etc.)
        device: CPU or GPU device.
        node_idx_to_gene_id: maps node index to gene id
        output_prefix: Where to write output files
        sample (str): Sample name for file naming
        top_n (int): Number of top genes to save

    Returns:
        (feature_fold_changes, feature_top_genes) where
          feature_fold_changes = { feature_key -> overall average diff }
          feature_top_genes     = { feature_key -> list of (gene_symbol, diff) }

        For single-feature runs, keys = int
        For pair-feature runs, keys = tuple(int, int)
    """
    global_differences = defaultdict(lambda: defaultdict(list))

    gene_nodes = getattr(data, f"{mask}_mask_loss").nonzero(as_tuple=True)[0]
    subgraph_batches = get_test_k_hop_subgraph(data, gene_nodes)

    for sub_data in tqdm(subgraph_batches, desc="Node Feature Perturbation"):
        sub_data = sub_data.to(device)
        mask_tensor = getattr(sub_data, f"{mask}_mask_loss")

        # skip if no gene nodes present
        if mask_tensor.sum() == 0:
            continue

        baseline_out = compute_baseline_output(runner, sub_data, mask_tensor)

        # compute differences for each feature index
        batch_differences = compute_feature_perturbation(
            runner=runner,
            batch=sub_data,
            mask_tensor=mask_tensor,
            baseline_output=baseline_out,
            feature_indices=feature_indices,
            scalers=scalers,
        )

        # aggregate differences across batches
        for feat_idx, node_dict in batch_differences.items():
            for node_idx, diffs in node_dict.items():
                global_differences[feat_idx][node_idx].extend(diffs)

    # compute avg difference and fold changes
    avg_diffs = compute_average_node_differences(global_differences)
    feature_fold_changes = compute_fold_changes(avg_diffs)

    # get top nodes and map to gene symbols
    feature_top_genes = get_top_feature_genes(
        feature_indices=feature_indices,
        avg_diffs=avg_diffs,
        top_n=len(gene_nodes),
        node_idx_to_gene_id=node_idx_to_gene_id,
        gencode_to_symbol=gencode_to_symbol,
    )

    return avg_diffs, feature_fold_changes, feature_top_genes
